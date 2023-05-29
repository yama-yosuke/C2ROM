import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import Embedder, SelfAttention, simpleMHA


class PolicyNetwork(nn.Module):
    def __init__(self, dim_embed, n_heads, tanh_clipping, dropout, target, device, hidden_dim=512, n_agents=3):
        super(PolicyNetwork, self).__init__()

        self.n_heads = n_heads  # number of heads in MHA
        self.tanh_clipping = tanh_clipping
        self.device = device
        self.target = target
        self.n_agents = n_agents

        # node encoder
        self.embedder_node = Embedder(3, dim_embed)
        self.embedder_depot = Embedder(2, dim_embed)
        self.mha_n = SelfAttention(dim_embed, n_heads, 2, "batch", dropout, hidden_dim)
        self.project_graph = Embedder(dim_embed, dim_embed)  # graph embedding

        # fleet encoder
        self.project_context = Embedder(dim_embed + 3, dim_embed)
        self.embedder_node_active = Embedder(2 * self.n_agents, dim_embed)
        self.mha_n_active = SelfAttention(dim_embed, n_heads, 1, "None", dropout, hidden_dim)

        # decoder
        self.project_memory = Embedder(dim_embed, 3 * dim_embed)  # memory embedding
        self.project_memory_active = Embedder(dim_embed, 3 * dim_embed)
        self.project_glimpse = Embedder(dim_embed, dim_embed)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def precompute(self, node):
        """
        Precomute fixed data(node encoder)
        Args:
            node (Tensor): holding all node attributes(x, y, demand*), shape=[B, 3, n_nodes]
        Retuns
            dict: dict of static embeddings
        """
        # [B, dim_embed, n_nodes]
        node_embed = self.mha_n(
            torch.cat((self.embedder_depot(node[:, :2, 0].unsqueeze(-1)), self.embedder_node(node[:, :, 1:])), dim=-1)
        )
        # [B, dim_embed, 1]
        depot_embed = node_embed[:, :, 0].unsqueeze(2)
        # [B, 3*dim_embed, n_nodes]
        memory = self.project_memory(node_embed)
        # [B, dim_embed, 1]
        graph_embed = self.project_graph(node_embed.mean(dim=2, keepdim=True))

        fixed = {
            "node_embed": node_embed,
            "depot_embed": depot_embed,
            "graph_embed": graph_embed,
            "memory": memory,
        }
        return fixed

    def calc_mrt(self, location, position, remaining_time, speed):
        """
        Args:
            location: [B, n_nodes, 2]
            position: [B, n_agents]
            remaining_time: [B, n_agents]
            speed: [B, n_agents]
        Returns:
            Tensor: min_reah_time, shape=[B, n_agents, n_nodes]
        """
        # [B, n_agents, 1, 2] destination coordinates of agents
        agent_loc = torch.gather(location, 1, position.unsqueeze(2).expand(-1, -1, 2)).unsqueeze(2)
        # [B, 1, n_nodes, 2] coordinates of nodes
        node_loc = location.unsqueeze(1)
        # [B, n_agents, n_nodes]
        dist = (agent_loc - node_loc).pow(2).sum(3).sqrt()
        remaining_time[remaining_time == -1] = float("inf")  # OOS agent
        min_reach_time = (dist / speed.unsqueeze(2)) + remaining_time.unsqueeze(2)
        min_reach_time[min_reach_time == float("inf")] = -1  # OOS agent
        return min_reach_time

    def sort(self, feat, next_agent, n_agents):
        """
        Args:
            feat: [B, n_agents*3, n_nodes], features to bo sorted
            next_agent: [B, 1], index of next_agent
        Returns:
            Tensor: [B, n_agents*3, n_nodes], sorted features
        """
        all_agent = (
            torch.arange(n_agents, device=self.device).unsqueeze(0).expand(next_agent.size(0), -1)
        )  # [B, n_agents]
        other_agent = all_agent[all_agent != next_agent].reshape((next_agent.size(0), n_agents - 1))  # [B, n_agents-1]
        index_ = torch.cat([next_agent, other_agent], dim=1)  # [B, n_agents]
        index = torch.cat(
            [
                index_,
                index_ + n_agents,
            ],
            dim=1,
        )  # [B, 2*n_agents]
        return torch.gather(feat, dim=1, index=index.unsqueeze(2).expand_as(feat))

    def calc_logp(self, static, dynamic, mask, fixed, rep):
        """
        execute fleet encoder and decoder
        Returns:
            Tensor: [B, n_nodes], logprob
        """
        # FLEET ENCODER
        # generate vehicle context vector
        # [B, 1], current pos
        agent_pos = torch.gather(dynamic["position"], dim=1, index=dynamic["next_agent"])
        # [B, dim_embed, 1]
        agent_dest = torch.gather(
            input=fixed["node_embed"].repeat(rep, 1, 1),
            dim=2,
            index=agent_pos.unsqueeze(1).expand(-1, fixed["node_embed"].size(1), -1),
        )
        # [B, 1, 1]
        agent_load = torch.gather(input=dynamic["load"], dim=1, index=dynamic["next_agent"]).unsqueeze(2)
        # [B, 1, 1]
        agent_max_load = torch.gather(input=static["max_load"], dim=1, index=dynamic["next_agent"]).unsqueeze(2)
        # [B, 1, 1]
        agent_speed = torch.gather(input=static["speed"], dim=1, index=dynamic["next_agent"]).unsqueeze(2)
        agent_input = torch.cat((agent_dest.detach(), agent_load, agent_max_load, agent_speed), dim=1)

        # generate interpretatve node feature
        # [B, n_agents, n_nodes]
        demand_rel = dynamic["demand"].unsqueeze(1) / dynamic["load"].unsqueeze(2)
        demand_rel[demand_rel > 1.0] = -1.0  # set -1 to unsatisfiable node
        demand_rel.nan_to_num_()  # convert "nan" to 0 in case load=demand=0
        # [B, n_agents, n_nodes]
        min_reach_time = self.calc_mrt(
            static["location"], dynamic["position"], dynamic["remaining_time"], static["speed"]
        )

        # [B], 2*n_agents, n_nodes]
        interpret_feature_ = torch.cat(
            (
                demand_rel,
                min_reach_time,
            ),
            dim=1,
        )

        # [B, 2*n_agents, n_nodes]
        interpret_feature = self.sort(interpret_feature_, dynamic["next_agent"], self.n_agents)
        interpret_embed = self.mha_n_active(self.embedder_node_active(interpret_feature))

        # DECODER
        # [B, dim_embed, n_nodes] for each
        glimpse_key, glimpse_val, logit_key = (
            fixed["memory"].repeat(rep, 1, 1) + self.project_memory_active(interpret_embed)
        ).chunk(3, dim=1)

        # context embedding, [B, dim_embed, 1]
        context = (
            fixed["depot_embed"].repeat(rep, 1, 1)
            + fixed["graph_embed"].repeat(rep, 1, 1)
            + self.project_context(agent_input)
        )
        # [B, dim_embed, 1], next_agent
        glimse_q = self.project_glimpse(simpleMHA(context, glimpse_key, glimpse_val, self.n_heads))

        # [B, n_nodes]
        logits = torch.bmm(logit_key.permute(0, 2, 1), glimse_q).squeeze(2) / math.sqrt(glimse_q.size(1))
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        logits = logits + mask.log()
        logprob = F.log_softmax(logits, 1)
        return logprob

    def forward(self, args, env, n_agents, speed, max_load, sampling=False, rep=1):
        """
        Returns:
            Tensor: sum of log probability for selected action, shpae=[B]
            Tensor: shape=[B]
            list: sequence of node index in visited order, shape=[B, n_agents, n_visits]
        """
        # reset env
        static, dynamic, mask = env.init_deploy(n_agents, speed, max_load)

        # NODE ENCODER
        node_input = torch.cat(
            (
                static["location"][: static["local_batch_size"]].transpose(1, 2),
                dynamic["demand"][: static["local_batch_size"]].unsqueeze(1),
            ),
            1,
        )
        fixed = self.precompute(node_input)

        # MDP loop (env.step)→observation→action→(env.step)
        actions = []
        logprobs = []
        additinal_distances = []
        while not dynamic["done"].all():
            # FLEET ENCODER and DECODER, [B, n_nodes]
            logprob = self.calc_logp(static, dynamic, mask, fixed, rep)
            if self.training or sampling:
                prob_dist = torch.distributions.Categorical(logprob.exp())
                action = prob_dist.sample()  # Stochastic policy [B]
            else:
                action = torch.argmax(logprob, 1)  # Greedy policy [B]

            logprob_selected = torch.gather(logprob, dim=1, index=action.unsqueeze(1))  # [B, 1]
            agent_id = dynamic["next_agent"]  # before env.step
            done = dynamic["done"]  # done=termination

            action_cp = action.clone().detach()  # [B]

            dynamic, mask, additional_distance_oh = env.step(action_cp.unsqueeze(1))
            # APPEND RESULTS
            action_cp[done.squeeze(1)] = -1  # set action in terminated episode as -1 [B]
            # [B, n_agents], set next node index to active vehciel, set -1 to other vehicles
            action_agent_wise = torch.scatter(
                torch.full((static["local_batch_size"] * rep, static["n_agents"]), -1, device=self.device),
                dim=1,
                index=agent_id,
                src=action_cp.unsqueeze(1),
            )
            actions.append(action_agent_wise)  # list of [B, n_agents]
            logprobs.append(logprob_selected)
            # additional_distance_oh: [B, n_agents]
            additinal_distances.append(additional_distance_oh)
        # PROCESS RESULTS
        sum_logprobs = torch.cat(logprobs, 1).sum(1)  # [B]
        total_distance = torch.stack(additinal_distances, 1).sum(1).squeeze(1)  # [B, n_agents]
        total_time = total_distance / static["speed"]  # [B, n_agents]
        if args.target == "MS":
            rewards = total_time.sum(1)  # [B]
        else:
            # min-max
            rewards, _ = total_time.max(1)  # [B]

        # [B, n_agents, n_actions]
        routes = torch.stack(actions, 2).cpu()
        # [B, n_agents, 1 + n_actions], add starting point(depot)
        routes = torch.cat(
            (torch.zeros_like(dynamic["position"], device=torch.device("cpu")).unsqueeze(2), routes), 2
        ).tolist()
        # i=each batch, j=each agent, k=each action(position)
        # routes = list of [B, n_agents, n_visits]
        routes = [[[k for k in j if k != -1] for j in i] for i in routes]

        return sum_logprobs, rewards, routes
