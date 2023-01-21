import torch
import os


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
        self.rep = rep  # instance repetition for sampling
        self.global_batch_size = global_batch_size  # globalB, batch size processed by the all device
        assert instance_num % global_batch_size == 0, "instance_num cannnot be divided by global_batch_size"
        self.batch_num = instance_num // global_batch_size  # number of batches per iterarion(same for all devices)
        # calculate batch size on each device(localB)
        self.fraction = global_batch_size % world_size
        self.local_batch_size = ((global_batch_size // world_size) + self.fraction) if rank == 0 else (global_batch_size // world_size)  # localB
        self.batch_size = self.local_batch_size * rep  # batch size for each device including repetition for sampling, # B
        

    def make_maps(self, n_custs, max_demand):
        self.n_nodes = n_custs + 1
        self.max_demand = max_demand
        init_demand = (torch.FloatTensor(self.batch_num, self.local_batch_size, self.n_nodes).uniform_(0, self.max_demand).int() + 1).float()
        init_demand[:, :, 0] = 0  # demand of depot is 0
        # on CPU
        self.dataset = {
            # [batch_num, localB, n_nodes, 2]
            "location": torch.FloatTensor(self.batch_num, self.local_batch_size, self.n_nodes, 2).uniform_(0, 1),
            # [batch_num, localB, n_nodes]
            "init_demand": init_demand
        }

        self.reindex()

    def load_maps(self, n_custs, max_demand, phase="val", seed="456"):
        program_dir = os.path.dirname(os.path.abspath(__file__))
        self.n_nodes = n_custs + 1
        self.max_demand = max_demand
        print(f"Process {self.rank}: Loading environment...\n")
        load_path = os.path.join(program_dir, "dataset", phase, 'C{}-MD{}-S{}-seed{}.pt'.format(n_custs, max_demand, self.instance_num, seed))
        load_data = torch.load(load_path)
        if self.rank == 0:
            idx_from = 0
            idx_to = self.local_batch_size
        else:
            idx_from = self.fraction + self.local_batch_size * self.rank
            idx_to = self.fraction + self.local_batch_size * (self.rank + 1)
        # on CPU
        self.dataset = {
            # [instance_num, n_nodes, 2] -> [batch_num, globalB, n_nodes, 2] -> [batch_num, localB, n_nodes, 2]
            "location": load_data["location"].reshape(self.batch_num, -1, self.n_nodes, 2)[:, idx_from:idx_to],
            # [instance_num, n_nodes] -> [batch_num, globalB, n_nodes] -> [batch_num, localB, n_nodes]
            "init_demand": load_data["init_demand"].reshape(self.batch_num, -1, self.n_nodes)[:, idx_from:idx_to]
        }

        self.reindex()

    def reindex(self):
        self.index = 0

    def next(self):
        """
        get next batch data
        Returns:
            bool: True if next data exists, else False
        """
        if self.index == self.batch_num:
            return False
        self.location = self.dataset["location"][self.index].to(self.device).repeat(self.rep, 1, 1)  # on CUDA, [B, n_nodes, 2]
        self.init_demand_ori = self.dataset["init_demand"][self.index].clone().repeat(self.rep, 1)  # on CPU, [B, n_nodes]
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
            dict:
                static["local_batch_size"] (int)
                static["n_nodes"] (int):
                staitc["n_agents"] (int):
                static["location"] (Tensor): location of nodes, shape=[B, n_nodes, 2]
                static["max_load"] (Tensor) : max load of agents(normalized), shape=[B, n_agents]
                static["speed"] (Tensor) : speed, shape=[B, n_agents]
                static["init_demand] (Tensor): initial demand(normalized), shape=[B, n_nodes]
            dict:
                dynamic["next_agent"] (Tensor): next agent, shape=[B, 1]
                dynamic["position"] (Tensor): position(node index) of agent, shape=[B, n_agents]
                dynamic["remaining_time"] (Tensor): remainingtime, shape=[B, n_agents]
                dynamic["load"] (Tensor): load(normalized), shape=[B, n_agents]
                dynamic["demand] (Tensor): current demand, shape=[B, n_nodes]
                dynamic["current_time"] (Tensor): current time(init=0), shape=[B, 1]
                dynamic["done"] (Tensor): done(init=False), shape=[B, 1]
            
            Tensor: mask for infeasible action, shape=[B, n_nodes]
        """
        # agent settings
        self.n_agents = n_agents
        base_of_norm = max(max_load)

        # on CUDA
        # static
        self.speed = torch.tensor(speed, device=self.device).expand(self.batch_size, -1)
        self.max_load = torch.tensor(max_load, device=self.device).expand(self.batch_size, -1) / base_of_norm
        self.init_demand = (self.init_demand_ori / base_of_norm).to(self.device)

        # dynamic
        self.next_agent = torch.full((self.batch_size, 1), agent_id, dtype=int, device=self.device)
        self.position = torch.full((self.batch_size, self.n_agents), 0, dtype=int, device=self.device)  # init position is depot
        self.remaining_time = torch.zeros(self.batch_size, self.n_agents, device=self.device)
        self.load = self.max_load.clone()
        self.demand = self.init_demand.clone()
        self.current_time = torch.zeros(self.batch_size, 1, device=self.device)
        self.done = torch.full((self.batch_size, 1), False, dtype=bool, device=self.device)

        # [B, n_agents] holding True if agent is next acting agent of the batch
        agent_mask = torch.arange(self.n_agents, device=self.device).expand(self.batch_size, -1).eq(self.next_agent)
        mask = self.make_mask(agent_mask)

        static = {
            "local_batch_size": self.local_batch_size,
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
        transition rule to push fleet and node state one step forward

        Args:
            action (Tensor): [B, 1], selected node index by the active vehicle
        Returns:
            dict: dynamic state
            Tensor: [B, n_agents], additional distance in ohe-hot format(set to 0 for non-active vehciles)
        """
        agent_mask = torch.arange(self.n_agents, device=self.device).expand(self.batch_size, -1).eq(self.next_agent)
        
        # CALC ADDITIONAL DISTANCE TO THE NEXT DESTINATION
        coord1 = torch.gather(self.location, 1,
                              self.position[agent_mask].view(-1, 1, 1).expand(-1, -1, self.location.size(2)))
        coord2 = torch.gather(self.location, 1,
                              action.view(-1, 1, 1).expand(-1, -1, self.location.size(2)))
        additional_distance = (coord2 - coord1).pow(2).sum(2).sqrt()  # [B, 1]
        # [B, n_agents], additional distance in ohe-hot format
        additional_distance_oh = torch.zeros(self.batch_size, self.n_agents, device=self.device)
        additional_distance_oh[agent_mask] = additional_distance.squeeze()

        # UPDATE REMAINING TIME AND POSITION OF ACTICE VEHCILE
        agent_speed = torch.gather(self.speed, 1, self.next_agent)  # [B, 1]
        additional_time = additional_distance / agent_speed
        additional_time[additional_distance == 0] = float("inf")  # OOS vehcile is represented by setting remaining time as infinity
        self.remaining_time[agent_mask] = additional_time.squeeze(1)  # [B, n_agents]
        self.position[agent_mask] = action.squeeze(1)  # [B, n_agents], destination
        # [B, 1], if all vehcile is OOS, the episode is done(terminated)
        self.done = self.remaining_time.isinf().all(dim=1, keepdim=True)

        # UPDATE DEMAND AND LOAD
        # [B, n_agents] holding True if active agent of episode is at a depot
        at_depot = action.eq(0).expand_as(agent_mask)
        # [B, n_nodes] holding True if the node is being visited
        at_node = torch.arange(self.n_nodes, device=self.device).expand(self.batch_size, -1).eq(action)
        delta = torch.min(torch.stack([self.demand[at_node], self.load[agent_mask]], dim=1), dim=1).values.squeeze()
        self.demand[at_node] -= delta  # satisfy demand
        self.load[agent_mask] -= delta  # use load
        self.load[agent_mask * at_depot] = self.max_load[agent_mask * at_depot].clone()  # refill at depot

        # TIME STEP AND DECIDE NEXT VEHICLE CHRONOLOGICALLY
        self.next_agent = self.remaining_time.argmin(1).unsqueeze(1)  # [B, 1] holding index of next acting agent
        # [B, n_agents] holding True if agent is next acting agent of the batch
        agent_mask = torch.arange(self.n_agents, device=self.device).expand(self.batch_size, -1).eq(self.next_agent)
        # [B, 1] holding the shortest remaining time
        time_delta = self.remaining_time[agent_mask].unsqueeze(1)
        time_delta[time_delta == float("inf")] = 0  # fix time for terminated episode
        self.remaining_time -= time_delta  # [B, n_agents]
        self.current_time += time_delta  # [B, 1] holding total elapsed time of the episode

        # UPDATE MASK
        mask = self.make_mask(agent_mask)

        # remaining time for OOS vehicle is masked with -1
        remaining_time_obs = self.remaining_time.clone()
        remaining_time_obs[remaining_time_obs == float("inf")] = -1

        dynamic = {
            "next_agent": self.next_agent.clone(),
            "position": self.position.clone(),
            "remaining_time": remaining_time_obs.clone(),
            "load": self.load.clone(),
            "demand": self.demand.clone(),
            "current_time": self.current_time.clone(),
            "done": self.done.clone(),
        }

        return dynamic, mask, additional_distance_oh

    def make_mask(self, agent_mask):
        """
        1. if the load of active vehcile is 0, mask all customers
        2. mask visited customer
        3. mask unsatisfiable customer(Demand > Load)
        4. prevent declaring last in-servive vehcile to be out-of-service before visiting all customers
        Args:
            agent_mask: [B, n_agents], holding True if agent is next acting agent of the batch
        Returns:
            Tensor: [B, n_nodes], 0 if unable to visit
        """
        mask = torch.full((self.batch_size, self.n_nodes), 1, device=self.device)

        # 1. if the load of active vehcile is 0, mask all customers
        # empty [B, n_nodes]
        empty = self.load[agent_mask].eq(0)
        mask[empty, 1:] = 0

        # 2. mask visited customer
        visited_cust = self.demand.eq(0)
        visited_cust[:, 0] = False  # exculde Depot 
        mask[visited_cust] = 0

        # 3. mask unsatisfiable customer(Demand > Load)
        mask[self.demand.gt(self.load[agent_mask].unsqueeze(1).expand_as(self.demand))] = 0

        # 4. prevent declaring last in-servive vehcile to be out-of-service before visiting all customers
        is_last = (torch.count_nonzero(~self.remaining_time.isinf(), 1) == 1)
        at_depot = self.position[agent_mask].eq(0)
        not_end = self.demand.sum(1).ne(0)
        mask[is_last * at_depot * not_end, 0] = 0

        return mask
