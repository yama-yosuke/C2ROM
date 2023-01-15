import torch
import os
import logging


class AltEnv(object):
    '''
    Generate a batch of environments of nodes with randomized features
    '''

    def __init__(self, rank, device, world_size, sample_num, global_batch_size, rep=1):
        """
        rank
        device: rank(int) or cpu
        world_size: total num of process
        sample_num: sample num (before scatter)
        global_batch_size: (globalB)sample num per batch (before scatter)
        """
        # env setting
        self.rank = rank
        self.device = device
        self.world_size = world_size
        self.sample_num = sample_num
        self.rep = rep  # for sampling
        self.global_batch_size = global_batch_size
        self.fraction = global_batch_size % world_size  # batchの端数はmaster(rank=0)が処理
        self.batch_size = ((global_batch_size // world_size) + self.fraction) if rank == 0 else (global_batch_size // world_size)  # batch size for each device
        self.calc_batch_size = self.batch_size * rep  # batch size for each device including repetition for sampling
        self.global_batch_size = global_batch_size
        assert sample_num % global_batch_size == 0, "sample_num cannnot be divided by batch_size"
        self.batch_num = sample_num // global_batch_size  # iteration times to finish (プロセスに依存しない)
        self.sample_num = sample_num

    def make_maps(self, n_custs, max_demand):
        self.n_nodes = n_custs + 1
        self.max_demand = max_demand
        init_demand = (torch.FloatTensor(self.batch_num, self.batch_size, self.n_nodes).uniform_(0, self.max_demand).int() + 1).float()
        init_demand[:, :, 0] = 0  # depotのdemandは0
        # on CPU
        self.dataset = {
            # [batch_num, localB, n_nodes, 2]
            "location": torch.FloatTensor(self.batch_num, self.batch_size, self.n_nodes, 2).uniform_(0, 1),
            # [batch_num, localB, n_nodes]
            "init_demand": init_demand
        }

        self.reindex()

    def load_maps(self, n_custs, max_demand, phase="val", seed="456"):
        program_dir = os.path.dirname(os.path.abspath(__file__))
        self.n_nodes = n_custs + 1
        self.max_demand = max_demand
        print(f"Process {self.rank}: Loading environment...\n")
        load_path = os.path.join(program_dir, "dataset", phase, 'C{}-MD{}-S{}-seed{}.pt'.format(n_custs, max_demand, self.sample_num, seed))
        load_data = torch.load(load_path)
        if self.rank == 0:
            idx_from = 0
            idx_to = self.batch_size
        else:
            idx_from = self.fraction + self.batch_size * self.rank
            idx_to = self.fraction + self.batch_size * (self.rank + 1)
        # on CPU
        logging.info(f"Process {self.rank}: Batch index={idx_from}-{idx_to}")
        self.dataset = {
            # [sample_num, n_nodes, 2] -> [batch_num, globalB, n_nodes, 2] -> [batch_num, B, n_nodes, 2]
            "location": load_data["location"].reshape(self.batch_num, -1, self.n_nodes, 2)[:, idx_from:idx_to],
            # [sample_num, n_nodes] -> [batch_num, globalB, n_nodes] -> [batch_num, B, n_nodes]
            "init_demand": load_data["init_demand"].reshape(self.batch_num, -1, self.n_nodes)[:, idx_from:idx_to]
        }

        self.reindex()

    def reindex(self):
        self.index = 0

    def next(self):
        """
        datasetの次のindexのdataを取得
        Returns:
            True if next data exists, else False
        """
        if self.index == self.batch_num:
            return False
        self.location = self.dataset["location"][self.index].to(self.device).repeat(self.rep, 1, 1)  # on CUDA, [b, n_nodes, 2]
        self.init_demand_ori = self.dataset["init_demand"][self.index].clone().repeat(self.rep, 1)  # on CPU, [b, n_nodes]
        self.index += 1
        return True

    def clone(self):
        return AltEnv(self.rank, self.device, self.world_size, self.sample_num, self.global_batch_size)

    def clone_static(self, env):
        self.n_nodes = env.n_nodes
        self.location = env.location
        self.init_demand_ori = env.init_demand_ori
        self.n_agents = env.n_agents
        self.speed = env.speed
        self.max_load = env.max_load

    def clone_dynamic(self, env):
        self.next_agent = env.next_agent.clone()
        self.position = env.position.clone()
        self.time_to_arrival = env.time_to_arrival.clone()
        self.load = env.load.clone()
        self.demand = env.demand.clone()

    def init_deploy(self, n_agents, speed, max_load, agent_id=0):
        """
        初期配置に戻す
        Args:
            agent_id: int, 最初に動かすagentのid
        Retruns:
            static["batch_size"]: int
            static["n_nodes"]: int
            staitc["n_agents"]: int
            static["location"] : [B, n_nodes, 2], location of nodes
            static["max_load"] : [B, n_agents], max load of agents(normalized)
            static["speed"] : [B, n_agents], speed
            static["init_demand]: [B, n_nodes], initial demand(normalized)

            dynamic["next_agent"] : [B, 1], next agent
            dynamic["position"] : [B, n_agents], position of agent
            dynamic["time_to_arrival"] : [B, n_agents], time to arrival
            dynamic["load"] : [B, n_agents], load(normalized, 初期値はstaticのmax loadと同じ)
            dynamic["demand]: [B, n_nodes], current demand(normalized, , 初期値はstaticのinit demandと同じ)
            dynamic["current_time"] : [B, 1], current time(init=0)
            dynamic["done"] : [B, 1], done(init=False)

            mask: [B, n_nodes], (init=1)
        """
        # agent settings
        self.n_agents = n_agents  # int
        base_of_norm = max(max_load)  # int(demand, loadはmax_loadの最大値で正規化する)

        # on CUDA
        # static
        self.speed = torch.tensor(speed, device=self.device).expand(self.calc_batch_size, -1)  # [B, n_agents]
        self.max_load = torch.tensor(max_load, device=self.device).expand(self.calc_batch_size, -1) / base_of_norm  # [B, n_agents]
        self.init_demand = (self.init_demand_ori / base_of_norm).to(self.device)  # [B, n_nodes]

        # dynamic
        self.next_agent = torch.full((self.calc_batch_size, 1), agent_id, dtype=int, device=self.device)
        self.position = torch.full((self.calc_batch_size, self.n_agents), 0, dtype=int, device=self.device)  # init position is depot
        self.time_to_arrival = torch.zeros(self.calc_batch_size, self.n_agents, device=self.device)
        self.load = self.max_load.clone()  # [B, n_agents]
        self.demand = self.init_demand.clone()  # [B, n_nodes]
        self.current_time = torch.zeros(self.calc_batch_size, 1, device=self.device)
        self.done = torch.full((self.calc_batch_size, 1), False, dtype=bool, device=self.device)

        # [B, n_agents] holding True if agent is next acting agent of the batch
        agent_mask = torch.arange(self.n_agents, device=self.device).expand(self.calc_batch_size, -1).eq(self.next_agent)
        mask = self.make_mask(agent_mask)

        static = {
            "batch_size": self.batch_size,
            "n_nodes": self.n_nodes,
            "n_agents": self.n_agents,
            "location": self.location,
            "max_load": self.max_load,
            "speed": self.speed,
            "init_demand": self.init_demand,
        }

        # Clone in necessary for the variables which will be in-placed afterward(in env.step)
        # otherwise, RuntimeError is raised in backpropagation : one of the variables needed for gradient computation has been modified by an inplace operation
        dynamic = {
            "next_agent": self.next_agent.clone(),
            "position": self.position.clone(),
            "time_to_arrival": self.time_to_arrival.clone(),
            "load": self.load.clone(),
            "demand": self.demand.clone(),
            "current_time": self.current_time.clone(),
            "done": self.done.clone(),
        }

        return static, dynamic, mask

    def step(self, action):
        """
        Update dynamic states based on current agent
        Update dynamic states based on next agent
        Update mask of next agent
        Reward are calculated along the way

        Static states: location, max_load, max_inventory, rate, speed, wait_time, end_time
        Dynamic states: next_agent, position, time_to_arrival, load, inventory, current_time, done

        Args:
            action: [B, 1]: all non-negative indices
        Returns:
            dynamic["next_agent"] : [B, 1], next agent
            dynamic["position"] : [B, n_agents], position of agent
            dynamic["time_to_arrival"] : [B, n_agents], time to arrival
            dynamic["load"] : [B, n_agents], load(normalized, 初期値はstaticのmax loadと同じ)
            dynamic["demand]: [B, n_nodes], current demand(normalized, , 初期値はstaticのinit demandと同じ)
            dynamic["current_time"] : [B, 1], current time(init=0)
            dynamic["done"] : [B, 1], done(init=False)

            mask: [B, n_nodes]

            additional_distance_oh: [B, n_agents], 各agentの移動距離(対象のagent以外は0)
        """
        # =======================PHASE1=======================
        # SET DESTINATION(update time_to_arrival and position)
        # [B, n_agents] holding True if agent is current acting agent of the batch
        agent_mask = torch.arange(self.n_agents, device=self.device).expand(self.calc_batch_size, -1).eq(self.next_agent)

        # calculate additional_distance
        # [B, 1] holding additional distance added on to tour
        # (input: [B, n_nodes, 2], dim=1, index: [B, 1, 2]) → [B, 1, 2] = coord from
        coord1 = torch.gather(self.location, 1,
                              self.position[agent_mask].view(-1, 1, 1).expand(-1, -1, self.location.size(2)))
        # (input: [B, n_nodes, 2], dim=1, index: [B, 1, 2]) → [B, 1, 2] = coord to
        coord2 = torch.gather(self.location, 1,
                              action.view(-1, 1, 1).expand(-1, -1, self.location.size(2)))
        additional_distance = (coord2 - coord1).pow(2).sum(2).sqrt()  # [B, 1]
        self.position[agent_mask] = action.squeeze(1)  # [B, n_agents] holding fiexed future position

        additional_distance_oh = torch.zeros(self.calc_batch_size, self.n_agents, device=self.device)  # [B, n_agents], agentごとの移動量
        additional_distance_oh[agent_mask] = additional_distance.squeeze()

        # =======================PHASE2=======================
        # FILL DEMAND(update vehicle load and node demand)

        # [B, n_agents] holding True if acting agent of episode is at a depot
        # position of greater than equal to n_custs means depot
        at_depot = action.eq(0).expand_as(agent_mask)
        # [B, n_nodes] holding True if the node is being visited
        at_node = torch.arange(self.n_nodes, device=self.device).expand(self.calc_batch_size, -1).eq(action)

        # Update load & demand
        # [B] (split delivery不可の場合は、deltaは常にdemandと等しい)
        # depotのdemandは0なので、delta=0で影響しない。
        delta = torch.min(torch.stack([self.demand[at_node], self.load[agent_mask]], dim=1), dim=1).values.squeeze()
        self.demand[at_node] -= delta
        self.load[agent_mask] -= delta
        # depotでは充填される
        self.load[agent_mask * at_depot] = self.max_load[agent_mask * at_depot].clone()

        # =======================PHASE3=======================
        # TIME STEP(update time_to_arrival based on next agent)
        # UPDATE NEXT_AGENT
        # [B, 1] holding index of next acting agent
        self.next_agent = (self.next_agent + 1) % self.n_agents

        # INITIALIZE SECOND AGENT BOOLEAN MASK
        # [B, n_agents] holding True if agent is next acting agent of the batch
        agent_mask = torch.arange(self.n_agents, device=self.device).expand(self.calc_batch_size, -1).eq(self.next_agent)

        # UPDATE MASK
        mask = self.make_mask(agent_mask)
        # define done and change masking rule
        self.done = (self.demand.sum(1).eq(0) * self.position.sum(1).eq(0)).unsqueeze(1)   # [B, 1] if all demands are satisfied and all vehicles are at depot

        dynamic = {
            "next_agent": self.next_agent.clone(),
            "position": self.position.clone(),
            "time_to_arrival": self.time_to_arrival.clone(),  # fixed 0
            "load": self.load.clone(),
            "demand": self.demand.clone(),
            "current_time": self.current_time.clone(),  # fixed 0
            "done": self.done.clone(),
        }

        return dynamic, mask, additional_distance_oh

    def make_mask(self, agent_mask):
        """
        maskを作成
        1. mask visited customer
        2. mask customer if demand > load
        3. mask depot to depot if demand is left(if demands are all satisfied, depot to depot is allowed to wait for other episodes)
        Args:
            agent_mask: [B, n_agents] holding True if agent is next acting agent of the batch
        Returns:
            mask: [B, n_nodes]
        """
        mask = torch.full((self.calc_batch_size, self.n_nodes), 1, device=self.device)
        """
        # 1. next_agentのloadが0の場合は、全customerをmask
        # empty [B, n_nodes]
        # next agentのloadが0のバッチはTrue, [B]
        empty = self.load[agent_mask].eq(0)
        # depot以外をmask
        mask[empty, 1:] = 0
        """

        # 2. Demandが0のcustomerをmask
        # [B, n_nodes]
        visited_cust = self.demand.eq(0)
        visited_cust[:, 0] = False  # Depotは除く
        mask[visited_cust] = 0

        # 3.  Demand > loadのcustomerをmask(split deliveryでは緩和する)
        mask[self.demand.gt(self.load[agent_mask].unsqueeze(1).expand_as(self.demand))] = 0

        # 4. DEPOTから他のDEPOTへの移動をmask for unifinished episode
        # [B, n_nodes], agentがいるところがFalse
        demand_left = self.demand.sum(1).ne(0)  # [B, 1], True if customer is left
        at_depot = self.position[agent_mask].eq(0)  # [B, 1], depotにいるならTrue
        mask[at_depot * demand_left, 0] = 0

        """
        # 6. (1-5を上書き)エピソードが終了している場合、Depot以外をmask
        # This rule overrides all other rules on finished episodes
        # [B, n_nodes]
        mask[self.done.squeeze(), 1:] = 0
        """
        return mask

    def sim_step(self, agent_id, action):
        """
        agent_idがactionをとった後のagent stateとnode stateを返す。
        simulationは、実際のactionをとる時間を起点に考えるので、時間はfix

        Args:
            agent_id: [B, 1]
            action: [B, 1]: all non-negative indices
        Returns:
            dynamic["next_agent"] : [B, 1], next agent = agent_id
            dynamic["position"] : [B, n_agents], position of agent
            dynamic["time_to_arrival"] : [B, n_agents], time to arrival, fixed
            dynamic["load"] : [B, n_agents], load(normalized, 初期値はstaticのmax loadと同じ)
            dynamic["demand]: [B, n_nodes], current demand(normalized, , 初期値はstaticのinit demandと同じ)

            mask: [B, n_nodes]
        """
        # [B, n_agents] holding True if agent is current acting agent of the batch
        agent_mask = torch.arange(self.n_agents, device=self.device).expand(self.batch_size, -1).eq(agent_id)
        self.position[agent_mask] = action.squeeze(1)  # [B, n_agents] holding fiexed future position

        # FILL DEMAND(update vehicle load and node demand)
        # [B, n_agents] holding True if acting agent of episode is at a depot
        # position of greater than equal to n_custs means depot
        at_depot = action.eq(0).expand_as(agent_mask)
        # [B, n_nodes] holding True if the node is being visited
        at_node = torch.arange(self.n_nodes, device=self.device).expand(self.batch_size, -1).eq(action)

        # Update load & demand
        # [B] (split delivery不可の場合は、deltaは常にdemandと等しい)
        # depotのdemandは0なので、delta=0で影響しない。
        delta = torch.min(torch.stack([self.demand[at_node], self.load[agent_mask]], dim=1), dim=1).values.squeeze()
        self.demand[at_node] -= delta
        self.load[agent_mask] -= delta
        # depotでは充填される
        self.load[agent_mask * at_depot] = self.max_load[agent_mask * at_depot].clone()

        # UPDATE MASK
        mask = self.make_mask(agent_mask)

        # time_to_arrivalは更新しない(実際のactionをとった時点を起点にsimulationを行うので)
        time_to_arrival_obs = self.time_to_arrival.clone()
        time_to_arrival_obs[time_to_arrival_obs == float("inf")] = -1

        dynamic = {
            "next_agent": agent_id,
            "position": self.position.clone(),
            "time_to_arrival": time_to_arrival_obs.clone(),
            "load": self.load.clone(),
            "demand": self.demand.clone(),
        }

        return dynamic, mask
