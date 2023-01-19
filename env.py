import torch
import os
import logging


class Env(object):

    def __init__(self, rank, device, world_size, instance_num, global_batch_size, rep=1):
        """
        Args:
            rank:
            device: rank(int) or cpu
            world_size: total num of process
            instance_num: number of instances per iteration
            global_batch_size: (globalB), batch size(before scattering to each device) 
            rep: num of sampling(1 if greedy) 
        """
        # env setting
        self.rank = rank
        self.device = device
        self.world_size = world_size
        self.instance_num = instance_num
        self.rep = rep  # for sampling
        self.global_batch_size = global_batch_size  # globalB
        assert instance_num % global_batch_size == 0, "instance_num cannnot be divided by batch_size"
        self.batch_num = instance_num // global_batch_size  # number of batches per iterarion(same in all devices)
        # calculate batch size in each device(localB)
        self.fraction = global_batch_size % world_size
        self.batch_size = ((global_batch_size // world_size) + self.fraction) if rank == 0 else (global_batch_size // world_size)  # localB
        self.calc_batch_size = self.batch_size * rep  # batch size for each device including repetition for sampling, # calcB
        

    def make_maps(self, n_custs, max_demand):
        self.n_nodes = n_custs + 1
        self.max_demand = max_demand
        init_demand = (torch.FloatTensor(self.batch_num, self.batch_size, self.n_nodes).uniform_(0, self.max_demand).int() + 1).float()
        init_demand[:, :, 0] = 0  # demand of depot is 0
        # on CPU
        self.dataset = {
            # [batch_num, localB(batch size in a device), n_nodes, 2]
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
        get next batch data
        Returns:
            True if next data exists, else False
        """
        if self.index == self.batch_num:
            return False
        self.location = self.dataset["location"][self.index].to(self.device).repeat(self.rep, 1, 1)  # on CUDA, [calcB, n_nodes, 2]
        self.init_demand_ori = self.dataset["init_demand"][self.index].clone().repeat(self.rep, 1)  # on CPU, [calcB, n_nodes]
        self.index += 1
        return True

    def init_deploy(self, n_agents, speed, max_load, agent_id=0):
        """
        set fleet and node state as the initial state
        Args:
            n_agents:
            speed:
            max_load:
            agent_id (int): the first agent to move
        Retruns:
            static["batch_size"] (int): 
            static["n_nodes"] (int):
            staitc["n_agents"] (int):
            static["location"] (Tensor): location of nodes, shape=[calcB, n_nodes, 2]
            static["max_load"] (Tensor) : max load of agents(normalized), shape=[calcB, n_agents]
            static["speed"] (Tensor) : speed, shape=[calcB, n_agents]
            static["init_demand] (Tensor): initial demand(normalized), shape=[calcB, n_nodes]

            dynamic["next_agent"] (Tensor): next agent, shape=[calcB, 1]
            dynamic["position"] (Tensor): position(node index) of agent, shape=[calcB, n_agents]
            dynamic["remaining_time"] (Tensor): remainingtime, shape=[calcB, n_agents]
            dynamic["load"] (Tensor): load(normalized), shape=[calcB, n_agents]
            dynamic["demand] (Tensor): current demand, shape=[calcB, n_nodes]
            dynamic["current_time"] (Tensor): current time(init=0), shape=[calcB, 1]
            dynamic["done"] (Tensor): done(init=False), shape=[calcB, 1]

            mask (Tensor): (init=1), shape=[calcB, n_nodes]
        """
        # agent settings
        self.n_agents = n_agents
        base_of_norm = max(max_load)

        # on CUDA
        # static
        self.speed = torch.tensor(speed, device=self.device).expand(self.calc_batch_size, -1)
        self.max_load = torch.tensor(max_load, device=self.device).expand(self.calc_batch_size, -1) / base_of_norm
        self.init_demand = (self.init_demand_ori / base_of_norm).to(self.device)

        # dynamic
        self.next_agent = torch.full((self.calc_batch_size, 1), agent_id, dtype=int, device=self.device)
        self.position = torch.full((self.calc_batch_size, self.n_agents), 0, dtype=int, device=self.device)  # init position is depot
        self.remaining_time = torch.zeros(self.calc_batch_size, self.n_agents, device=self.device)
        self.load = self.max_load.clone()
        self.demand = self.init_demand.clone()
        self.current_time = torch.zeros(self.calc_batch_size, 1, device=self.device)
        self.done = torch.full((self.calc_batch_size, 1), False, dtype=bool, device=self.device)

        # [calcB, n_agents] holding True if agent is next acting agent of the batch
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

        dynamic = {
            "next_agent": self.next_agent.clone(),
            "position": self.position.clone(),
            "remaining_time": self.remaining_time.clone(),
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
            dynamic["remaining_time"] : [B, n_agents], remainingtime
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

        # UPDATE TIME_TO_ARRIVAL PART 1
        # time_to_arrival: [B, n_agents] holding total elapsed time of vehicle
        # add time of travel if new node or predetermined waiting time if same node
        agent_speed = torch.gather(self.speed, 1, self.next_agent)  # [B, 1]
        additional_time = additional_distance / agent_speed   # save a_d for reward
        additional_time[additional_distance == 0] = float("inf")

        self.remaining_time[agent_mask] = additional_time.squeeze(1)  # [B, n_agents] holding remainingtime
        self.position[agent_mask] = action.squeeze(1)  # [B, n_agents] holding fiexed future position

        additional_distance_oh = torch.zeros(self.calc_batch_size, self.n_agents, device=self.device)  # [B, n_agents], agentごとの移動量
        additional_distance_oh[agent_mask] = additional_distance.squeeze()

        # [B, 1], 全ビークルがwait_time = infなら、エピソード終了
        self.done = self.remaining_time.isinf().all(dim=1, keepdim=True)

        # =======================PHASE2=======================
        # FILL DEMAND(update vehicle load and node demand)
        # additional time後の未来だが、確定した情報なので、次stepの判断に活用する=(fixed future state)

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
        self.next_agent = self.remaining_time.argmin(1).unsqueeze(1)

        # INITIALIZE SECOND AGENT BOOLEAN MASK
        # [B, n_agents] holding True if agent is next acting agent of the batch
        agent_mask = torch.arange(self.n_agents, device=self.device).expand(self.calc_batch_size, -1).eq(self.next_agent)

        # DETERMINE ELAPSED TIME UNTIL NEXT AGENT ACTION
        # [B, 1] holding the shortest remainingtime
        elapsed_time = self.remaining_time[agent_mask].unsqueeze(1)
        elapsed_time[elapsed_time == float("inf")] = 0  # 終了済みエピソード(全部INF)は時間を固定

        # TIME STEP
        self.remaining_time -= elapsed_time  # [B, n_agents] holding total elapsed time of vehicle
        self.current_time += elapsed_time  # [B, 1] holding total elapsed time of the episode

        # UPDATE MASK
        mask = self.make_mask(agent_mask)

        # infをモデルに流すとバグる
        # → agentのobservationとしては、帰還済みビークルを-1で表現
        time_to_arrival_obs = self.remaining_time.clone()
        time_to_arrival_obs[time_to_arrival_obs == float("inf")] = -1

        dynamic = {
            "next_agent": self.next_agent.clone(),
            "position": self.position.clone(),
            "remaining_time": time_to_arrival_obs.clone(),
            "load": self.load.clone(),
            "demand": self.demand.clone(),
            "current_time": self.current_time.clone(),
            "done": self.done.clone(),
        }

        return dynamic, mask, additional_distance_oh

    def make_mask(self, agent_mask):
        """
        maskを作成
        1. next_agentのloadが0の場合は、全customerをmask
        2. Demandが0のcustomerをmask
        3. Demand > loadのcustomerをmask(split deliveryでは緩和する)
        4. (Deleted)DEPOTから他のDEPOTへの移動をmask
        5. 残り一台&&DEPOTにいる&&demandが残っている →　depot to depotはmask
            (demandが充足されずにエピソード終了するのを回避)
        6. (Deleted, 重複)エピソードが終了している場合、Depotをmask
        Args:
            agent_mask: [B, n_agents] holding True if agent is next acting agent of the batch
        Returns:
            mask: [B, n_nodes]
        """
        mask = torch.full((self.calc_batch_size, self.n_nodes), 1, device=self.device)

        # 1. next_agentのloadが0の場合は、全customerをmask
        # empty [B, n_nodes]
        # next agentのloadが0のバッチはTrue, [B]
        empty = self.load[agent_mask].eq(0)
        # depot以外をmask
        mask[empty, 1:] = 0

        # 2. Demandが0のcustomerをmask
        # [B, n_nodes]
        visited_cust = self.demand.eq(0)
        visited_cust[:, 0] = False  # Depotは除く
        mask[visited_cust] = 0

        # 3.  Demand > loadのcustomerをmask(split deliveryでは緩和する)
        mask[self.demand.gt(self.load[agent_mask].unsqueeze(1).expand_as(self.demand))] = 0
        """
        # 4. DEPOTから他のDEPOTへの移動をmask
        # [B, n_nodes], agentがいるところがFalse
        other_depot = torch.arange(self.n_nodes).expand(self.batch_size, -1).ne(cur_pos)
        other_depot[:, :self.n_custs] = False  # agentがいないdepotはTrue
        at_depot = self.position[agent_mask].ge(self.n_custs).unsqueeze(1)  # [B, 1], depotにいるならTrue
        mask[at_depot * other_depot] = 0
        """

        # 5. 残り一台&&DEPOTにいる&&demandが残っている →　depot to depotはmask
        # →　残り一台&&DEPOTにいる&&demandが残っていない　→　depot to depotを選択し、次のループで終了
        # [B], time_to_arrivalのnot INFの要素数が1つならTrue
        is_last = (torch.count_nonzero(~self.remaining_time.isinf(), 1) == 1)
        # [B], depotにいるならTrue
        at_depot = self.position[agent_mask].eq(0)
        # [B], demandが残っている場合True
        not_end = self.demand.sum(1).ne(0)
        # [B, n_nodes]
        mask[is_last * at_depot * not_end, 0] = 0

        """
        # 6. (1-5を上書き)エピソードが終了している場合、Depot以外をmask
        # This rule overrides all other rules on finished episodes
        # [B, n_nodes]
        mask[self.done.squeeze(), 1:] = 0
        """
        return mask
