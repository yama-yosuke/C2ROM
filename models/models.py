import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import sort

import math

from models.modules import Embedder, simpleMHA, SelfAttention, AttentionLayer, MHA

class PolicyNetwork(nn.Module):
    """
    agent info in node(demand, time_to_node) + node MHA after casting
    v10.4のcontextにagent speedを追加
    Node Attention = Multi-Head Attention + FF
    Agent Attention = Source-Target Attention + FF
    Query: Next Agent
    Memory: Other Agent
    """

    def __init__(self, dim_embed, n_heads, n_layers_n, n_layers_a, norm_n, norm_a,
                 tanh_clipping, dropout, target, device, hidden_dim=512, dim_n=3, dim_a=4, n_agents=3):
        """
        Args:
            dim_embed: 埋め込み次元
            n_heads: MHAのヘッド数
            n_layers_n: node用SelfAttentionのlayer数
            n_layers_a: agent用SelfAttentionのlayer数
            n_norm_n: node用のNormalization(batch, layer, instance or None)
            n_norm_a: node用のNormalization(batch, layer, instance or None)
            tanh_clipping
            dropout: droput ratio
            target: "MS" or "MM"
            device
            hidden_dim: FFの隠れ層の次元(default=512)
            dim_n: nodeの次元(default=3, (x, y, demand))
            dim_a: agentの次元(default=6, (x, y, time_to_arrival, load, init_load, speed))

            b: batch size without repetion
            B: batch size with repetition
        """
        super(PolicyNetwork, self).__init__()

        self.n_heads = n_heads
        self.tanh_clipping = tanh_clipping
        self.device = device
        self.target = target
        self.n_agents = n_agents

        # fixed network
        self.embedder_node = Embedder(dim_n, dim_embed)
        self.embedder_depot = Embedder(2, dim_embed)
        self.mha_n = SelfAttention(dim_embed, n_heads, n_layers_n - 1, norm_n, dropout, hidden_dim)
        self.project_memory = Embedder(dim_embed, 3 * dim_embed)  # memory embedding
        self.project_graph = Embedder(dim_embed, dim_embed)  # graph embedding

        # active networks
        self.project_context = Embedder(dim_embed + 3, dim_embed)
        self.embedder_node_active = Embedder(2 * self.n_agents, dim_embed)
        self.project_memory_active = Embedder(dim_embed, 3 * dim_embed)
        self.mha_n_active = SelfAttention(dim_embed, n_heads, 1, "None", dropout, hidden_dim)
        self.project_glimpse = Embedder(dim_embed, dim_embed)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def precompute(self, node):
        """
        Precomute fixed data(node)
        Args:
            node: [b, dim_n=3, n_nodes] holding all node attributes(x, y, demand*)
        """
        # Embedding [b, dim_embed, n_nodes] & Attention [b, dim_embed, n_nodes]
        node_embed = self.mha_n(
            torch.cat((self.embedder_depot(node[:, :2, 0].unsqueeze(-1)), self.embedder_node(node[:, :, 1:])), dim=-1)
        )
        # [b, dim_embed, 1]
        depot_embed = node_embed[:, :, 0].unsqueeze(2)
        # [b, 3*dim_embed, n_nodes]
        memory = self.project_memory(node_embed)
        # [b, dim_embed, 1]
        graph_embed = self.project_graph(node_embed.mean(dim=2, keepdim=True))

        fixed = {
            "node_embed": node_embed,
            "depot_embed": depot_embed,
            "graph_embed": graph_embed,
            "memory": memory,
        }
        return fixed

    def calc_ttn(self, location, position, tta, speed):
        """
        Params:
            location: [B, n_nodes, 2]
            position: [B, n_agents]
            tta: [B, n_agents], time to arrival
            speed: [B, n_agents]
        Returns:
            time_to_node: [B, n_agents, n_nodes]
        """
        # [B, n_agents, 1, 2] holding destination coordinates of agents
        # input=[B, n_nodes, 1, 2], index = [B, n_agents, 2]
        agent_loc = torch.gather(location, 1, position.unsqueeze(2).expand(-1, -1, 2)).unsqueeze(2)
        # [B, 1, n_nodes, 2] holding destination coordinates of agents
        node_loc = location.unsqueeze(1)
        # [B, n_agents, n_nodes]
        dist = (agent_loc - node_loc).pow(2).sum(3).sqrt()
        tta[tta == -1] = float("inf")  # non-active agent
        ttn = (dist / speed.unsqueeze(2)) + tta.unsqueeze(2)
        ttn[ttn == float("inf")] = -1  # non-active agent
        return ttn

    def sort(self, feat, next_agent, n_agents):
        """
        Params:
            feat: [B, n_agents*3, n_nodes], features to bo sorted
            next_agent: [B, 1], index of next_agent
        Returns:
            sorted_feat: [B, n_agents*3, n_nodes], sorted features
        """
        all_agent = torch.arange(n_agents, device=self.device).unsqueeze(0).expand(next_agent.size(0), -1)  # [B, n_agents]
        other_agent = all_agent[all_agent != next_agent].reshape((next_agent.size(0), n_agents - 1))  # [B, n_agents-1]
        index_ = torch.cat([next_agent, other_agent], dim=1)  # [B, n_agents]
        index = torch.cat([index_, index_ + n_agents, ], dim=1)  # [B, 2*n_agents]
        return torch.gather(feat, dim=1, index=index.unsqueeze(2).expand_as(feat))

    def sample_step(self, static, dynamic, mask, fixed, rep):
        """
        Args:
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

            fixed["node_embed"]: [b, dim_embed, n_nodes]
            fixed["depot_embed"]: [b, dim_embed, 1]
            fixed["graph_embed"]: [b, dim_embed, 1]
            fixed["memory"]: [b, 3*dim_embed, n_nodes]
        }
        Returns:
            logprob: [B, n_nodes]
        """
        # agent input
        # [B, 1], (input: [B, n_agents], dim=1, index: [B, 1])
        agent_pos = torch.gather(dynamic["position"], dim=1, index=dynamic["next_agent"])
        # [B, dim_embed, 1]=(h_t-1), (input: [b*rep, dim_embed, n_nodes], dim=1, index: [B, dim_embed, 1])
        agent_dest = torch.gather(input=fixed["node_embed"].repeat(rep, 1, 1), dim=2, index=agent_pos.unsqueeze(1).expand(-1, fixed["node_embed"].size(1), -1))
        # [B, 1, 1], (input: [B, n_agents], dim=1, index: [B, 1])
        agent_load = torch.gather(input=dynamic["load"], dim=1, index=dynamic["next_agent"]).unsqueeze(2)
        # [B, 1, 1], (input: [B, n_agents], dim=1, index: [B, 1])
        agent_max_load = torch.gather(input=static["max_load"], dim=1, index=dynamic["next_agent"]).unsqueeze(2)
        # [B, 1, 1], (input: [B, n_agents], dim=1, index: [B, 1])
        agent_speed = torch.gather(input=static["speed"], dim=1, index=dynamic["next_agent"]).unsqueeze(2)
        agent_input = torch.cat((
            agent_dest.detach(),
            agent_load,
            agent_max_load,
            agent_speed), dim=1
        )

        # node input
        # [B, n_agents, n_nodes]
        demand_rel = dynamic["demand"].unsqueeze(1) / dynamic["load"].unsqueeze(2)
        demand_rel[demand_rel > 1.0] = -1.0  # avoid "inf" in case load = 0, set -1 to impossible node
        demand_rel.nan_to_num_()  # convert "nan" to 0 in case load=demand=0
        # [B, n_agents, n_nodes]
        time_to_node = self.calc_ttn(static["location"], dynamic["position"], dynamic["time_to_arrival"], static["speed"])

        # [B, 2*n_agents, n_nodes]
        node_input_ = torch.cat((
            demand_rel,
            time_to_node,
        ), dim=1
        )

        # [B, 2*n_agents, n_nodes]
        node_input = self.sort(node_input_, dynamic["next_agent"], self.n_agents)
        # sampling時の律速↓
        node_embed = self.mha_n_active(self.embedder_node_active(node_input))
        # Encoder(node)
        # [b*rep, 3*dim_embed, n_nodes] -> [B, dim_embed, n_nodes] for each
        glimpse_key, glimpse_val, logit_key = (fixed["memory"].repeat(rep, 1, 1) + self.project_memory_active(node_embed)).chunk(3, dim=1)

        # Decoder
        # [B, dim_embed, 1]
        context = fixed["depot_embed"].repeat(rep, 1, 1) + fixed["graph_embed"].repeat(rep, 1, 1) + self.project_context(agent_input)
        # [B, dim_embed, 1], next_agent
        glimse_q = self.project_glimpse(simpleMHA(context, glimpse_key, glimpse_val, self.n_heads))
        # (k^T * q)/(dk)^0.5
        # [B, n_nodes, dim_embed] * [B, dim_embed, 1] = [B, n_nodes, 1] → [B, n_nodes]
        logits = torch.bmm(logit_key.permute(0, 2, 1), glimse_q).squeeze(2) / math.sqrt(glimse_q.size(1))
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        logits = logits + mask.log()
        logprob = F.log_softmax(logits, 1)
        return logprob

    def forward(self, args, env, n_agents, speed, max_load, out_tour, sampling=False, rep=1):
        """
        Params:
            env: Environment
            speed: list of n_agnets
            max_load: list of n_agnets
            out_tour: If true, output tour index
        Returns:
            sum_logprobs:[B](cuda), 選んだactionに対するlogprobのsum
            rewards: [B](cuda), min-max of total time or min-sum of total time
            key_agents: [B](cpu), MMにおいて、最も時間のかかったactorのindex, MSならNone
            routes: list of [B, n_agents, n_visits](cpu)
        """
        # RESET ENVIRONMENT
        static, dynamic, mask = env.init_deploy(n_agents, speed, max_load)

        # [B, 3(x, y, demand), n_nodes] holding all node attributes
        # (location[B, 2, n_nodes], demand[B, 1, n_nodes])
        node_input = torch.cat((static["location"][:static["batch_size"]].transpose(1, 2), dynamic["demand"][:static["batch_size"]].unsqueeze(1)), 1)
        fixed = self.precompute(node_input)

        # SAMPLE LOOP (env.step)→observation→action→(env.step)
        actions = []
        logprobs = []
        additinal_distances = []
        while not dynamic["done"].all():
            # [B, n_nodes]
            logprob = self.sample_step(static, dynamic, mask, fixed, rep)
            if self.training or sampling:
                prob_dist = torch.distributions.Categorical(logprob.exp())
                # [B]
                action = prob_dist.sample()  # Stochastic policy
            else:
                # [B]
                action = torch.argmax(logprob, 1)  # Greedy policy
            
            logprob_selected = torch.gather(logprob, dim=1, index=action.unsqueeze(1))  # [B, 1]
            agent_id = dynamic["next_agent"]  # before env.step
            # this is still called skipped even though it's just self.done
            skipped = dynamic["done"]

            # Clone in necessary because action must not be in-placed(because it is used in calc of logprob_selected)
            # otherwise, RuntimeError is raised in backpropagation : one of the variables needed for gradient computation has been modified by an inplace operation
            action_cp = action.clone().detach()  # [B]

            dynamic, mask, additional_distance_oh = env.step(action_cp.unsqueeze(1))
            # APPEND RESULTS
            action_cp[skipped.squeeze(1)] = -1  # set skips to -1 [B]
            # target = [B, n_agents], index=[B, 1], src=[B, 1]
            # [B, n_agents], 動いたagentは行先ノード、他は-1
            action_agent_wise = torch.scatter(torch.full((static["batch_size"]*rep, static["n_agents"]), -1, device=self.device),
                                              dim=1, index=agent_id, src=action_cp.unsqueeze(1))
            actions.append(action_agent_wise)  # list of [B, n_agents]
            logprobs.append(logprob_selected)
            # additional_distance: [B, n_agents]
            additinal_distances.append(additional_distance_oh)
        # PROCESS RESULTS
        # [B]
        sum_logprobs = torch.cat(logprobs, 1).sum(1)
        # [B, n_agents]
        total_distance = torch.stack(additinal_distances, 1).sum(1).squeeze(1)
        # [B, n_agents]
        total_time = total_distance / static["speed"]
        if args.target == "MS":
            rewards = total_time.sum(1)  # [B], sum of agents' tour_length
            key_agents = None
        else:
            # min-max
            rewards, key_agents = total_time.max(1)  # [B], max of agents' tota_time, max()は、val, indexのtupleを返す

        if out_tour:
            # n_actionsは、ループの回数に等しい。n_visitsは、エピソード終了後を除いた訪問数
            # [B, n_agents, n_actions]
            routes = torch.stack(actions, 2).cpu()
            # [B, n_agents, 1 + n_actions]
            routes = torch.cat((torch.zeros_like(dynamic["position"], device=torch.device("cpu")).unsqueeze(2), routes), 2).tolist()
            # i=each batch, j=each agent, k=each action(position)
            # routes = list of [B, n_agents, n_visits]
            routes = [[[k for k in j if k != -1] for j in i] for i in routes]
        else:
            routes = None
        return sum_logprobs, rewards, key_agents.cpu(), routes




