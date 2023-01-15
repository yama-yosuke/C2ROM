from train import sample_actor
import os
import datetime
import argparse
import json
import pprint as pp
import itertools
from tqdm import tqdm

from models.models import *

from env import Env
from env_alt import AltEnv
import csv
import torch
import matplotlib.pyplot as plt
from const import SPEED, MAX_LOAD, N_CUSTS


def get_options():
    parser = argparse.ArgumentParser(
        description="HCVRP by Reinforcement Learning")

    # Model Selection
    parser.add_argument("actor_path", type=str)

    # Environment parameters(overwrite loaded settings)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--all_cust", action="store_true")
    parser.add_argument("--n_custs", type=int, default=None, help="size of customers")
    parser.add_argument("--n_agents", type=int, default=None, help="size of agents")
    parser.add_argument("--speed_type", type=str, default=None, help="hom or het")
    parser.add_argument("--n_sampling", type=int, default=1, help="size of sampling")

    # dataset parameters
    parser.add_argument("--sample_num", type=int, default=1280)
    parser.add_argument("--batch_size", type=int, default=1280)
    parser.add_argument("--seed", type=str, default="=Li")
    parser.add_argument("--max_demand", type=int, default=9)
    parser.add_argument("--rendering", action="store_true")

    args = parser.parse_args()  # dict

    args.sampling = (args.n_sampling > 1)

    assert args.sample_num % args.batch_size == 0, "sample num must be divisible by batch size"

    with open(os.path.join(os.path.dirname(args.actor_path), "args.json"), "r") as f:
        args_load = json.load(f)  # dict

    # Actor parameters(fixed)
    args.actor_type = args_load["actor_type"]
    args.n_heads = args_load["n_heads"]
    args.n_layers_n = args_load["n_layers_n"]
    args.n_layers_a = args_load["n_layers_a"]
    args.norm_n = args_load["norm_n"]
    args.norm_a = args_load["norm_a"]
    args.dim_embed = args_load["dim_embed"]
    args.tanh_clipping = args_load["tanh_clipping"]
    args.dropout = args_load["dropout"]
    args.n_foresight = args_load["n_foresight"]
    args.veh_sel = args_load["veh_sel"]

    # fixed
    args.multi_gpus = False
    args.target = args_load["target"]

    # param(overwrite loaded settings)
    args.n_custs = args.n_custs or args_load["n_custs"]
    args.n_agents = args.n_agents or args_load["n_agents"]
    args.speed_type = args.speed_type or args_load.get("speed_type", "hom")  # for backward comapatibility

    args.speed = SPEED[args.speed_type][args.n_agents]
    args.max_load = MAX_LOAD[args.n_agents]

    args.title = 'V{}-C{}-{}'.format(args.n_agents, args.n_custs, args.target)

    return args


def load_checkpoint(args):
    checkpoint = {}
    print('Loading checkpoint...\n')
    checkpoint = torch.load(args.actor_path, map_location=args.device)
    return checkpoint


def set_actor(args, checkpoint):
    actor_class = {
        "MHAv8.1": ActorMHAv81,
        "MHAv8.6": ActorMHAv86,
        "MHAv8.7": ActorMHAv87,
        "MHAv9.1": ActorMHAv91,
        "MHAv9.2": ActorMHAv92,
        "MHAv10.1": ActorMHAv101,
        "MHAv10.2": ActorMHAv102,
        "MHAv10.3": ActorMHAv103,
        "MHAv10.4": ActorMHAv104,
        "MHAv10.4mini": ActorMHAv104mini,
        "MHAv10.5": ActorMHAv105,
        "MHAv10.6": ActorMHAv106,
        "MHAv10.6_no_coop": ActorMHAv106_no_coop,
        "MHAv11.1": ActorMHAv111,
    }.get(args.actor_type)
    if "9" in args.actor_type:
        actor = actor_class(dim_embed=args.dim_embed, n_heads=args.n_heads, n_layers_n=args.n_layers_n, n_layers_a=args.n_layers_a, norm_n=args.norm_n, norm_a=args.norm_a,
                            tanh_clipping=args.tanh_clipping, dropout=args.dropout, target=args.target, device=args.device, n_foresight=args.n_foresight).to(args.device)
    if "10" in args.actor_type or "11" in args.actor_type:
        actor = actor_class(dim_embed=args.dim_embed, n_heads=args.n_heads, n_layers_n=args.n_layers_n, n_layers_a=args.n_layers_a, norm_n=args.norm_n, norm_a=args.norm_a,
                            tanh_clipping=args.tanh_clipping, dropout=args.dropout, target=args.target, device=args.device, n_agents=args.n_agents).to(args.device)
    else:
        actor = actor_class(dim_embed=args.dim_embed, n_heads=args.n_heads, n_layers_n=args.n_layers_n, n_layers_a=args.n_layers_a,
                            norm_n=args.norm_n, norm_a=args.norm_a, tanh_clipping=args.tanh_clipping, dropout=args.dropout, target=args.target, device=args.device).to(args.device)
    if "actor" in checkpoint:
        print('Loading actor checkpoint...\n')
        actor.load_state_dict(checkpoint["actor"])
    return actor


def render(location, tours, rewards, key_agents, save_path):
    """
    location : torch.tensor [B, n_nodes, 2], location of nodes
    tours: list [B, n_agent, n_visits], n_visitsはagentにより異なる
    rewards;: [B], tour_length
    epoch: int
    n_custs: int, customerの数
    save path
    """
    # color_list = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
    cmap = plt.get_cmap("hsv")
    n_agents = len(tours[0])
    print(n_agents)
    _, axes = plt.subplots(nrows=3, ncols=3, sharex='col',
                           sharey='row', figsize=(10, 10))
    axes = [a for ax in axes for a in ax]  # 3*3 or 1*1を　1次元配列に変換
    for episode, ax in enumerate(axes):
        # node
        ax.scatter(location[episode, 0, 0], location[episode, 0, 1],
                   color="k", marker="*", s=200, zorder=2)  # depot
        ax.scatter(location[episode, 1:, 0], location[episode,
                   1:, 1], color="k", marker="o", zorder=2)  # customer

        # tour
        for agent_id, tour in enumerate(tours[episode]):
            tour_index = torch.tensor(tour).unsqueeze(
                1).expand(-1, 2)  # [n_visits]
            # input=[n_nodes, 2], dim=1, index=[n_visits, 2]
            tour_xy = torch.gather(location[episode], dim=0, index=tour_index)
            ax.plot(tour_xy[:, 0], tour_xy[:, 1], color=cmap(agent_id / n_agents),
                    linestyle="-", linewidth=0.8, zorder=1, label=f"agent{agent_id}")
        title = "time:{:.3f}".format(rewards[episode].item())
        if key_agents is not None:
            # min-maxの場合、rewardを決定するagent idを記録
            title += "(agent{})".format(key_agents[episode].item())
        ax.set_aspect('equal')  # xy比固定
        ax.set_title(title, y=-0.1)
        ax.axis([-0.05, 1.05, -0.05, 1.05])
        ax.set_xticklabels([])  # 目盛り消去
        ax.set_yticklabels([])
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1,), ncol=3)
    plt.tight_layout()
    plt.suptitle(f"test", y=0)
    plt.savefig(save_path, facecolor="white", edgecolor="black")
    plt.close()
    return


def validate(rewards, routes, locations, init_demamds, max_load, speed):
    """
    Args:
        rewards: [B]
        routes: list of [B, n_agents, n_visits]
        locations: [B, n_nodes, 2](cpu), xy
        init_demands: [B, n_nodes](cpu), initial demand 
        max_load: list of len n_agents
    """
    for reward, route, location, init_demand in zip(rewards, routes, locations, init_demamds):
        # each episode
        assert (init_demand[1:] > 0).all(), "some nodes had no demand"
        visited_nodes = set(itertools.chain.from_iterable(route))
        assert len(visited_nodes) == init_demand.size(0), "some nodes were left unvisited"
        distances = []
        for agent_idx, agent_load in enumerate(max_load):
            # each agent
            agent_route = route[agent_idx]  # list of n_visits
            used_load = 0
            dist = []
            assert agent_route[0] == 0, "did not start from depot"
            assert agent_route[-1] == 0, "did not return to depot"
            pre_loc = location[0]  # depot
            for node_idx in agent_route:
                # each step
                next_loc = location[node_idx]
                dist.append((next_loc - pre_loc).pow(2).sum().sqrt())
                pre_loc = next_loc
                if node_idx != 0:
                    # depotでない
                    used_load += init_demand[node_idx].item()
                else:
                    used_load = 0
                assert used_load <= agent_load, "some nodes ware left unsatisfied"
            distances.append(torch.tensor(dist).sum() / speed[agent_idx])
        max_dist = torch.tensor(distances).max()
        assert round(max_dist.item(), 1) == round(reward.item(), 1), "calculation of distance failed"


def test(args, actor, n_agents, n_custs, speed, max_load, img_path):

    # start testing
    print(f'Test: V{n_agents}-C{n_custs}')
    # initialize env
    # test_env
    if args.veh_sel == "chr":
        test_env = Env(rank=0, device=args.device, world_size=1, sample_num=args.sample_num, global_batch_size=args.batch_size, rep=args.n_sampling)
    else:
        test_env = AltEnv(rank=0, device=args.device, world_size=1, sample_num=args.sample_num, global_batch_size=args.batch_size, rep=args.n_sampling)
    test_env.load_maps(n_custs, args.max_demand, "test", args.seed)

    rewards_list = []
    process_time_list = []

    pbar = tqdm(total=test_env.batch_num)
    while(test_env.next()):
        _, rewards, key_agents, routes, process_time = sample_actor(args, test_env, actor, n_agents, speed, max_load, test=True, out_tour=args.rendering, sampling=args.sampling, rep=args.n_sampling, calc_time=True)
        rewards_list.append(rewards.reshape(args.n_sampling, -1).permute(1, 0))  # list of len sample_num/batch_size, ele=tensor[batch_size, n_sampling]
        process_time_list.append(process_time)
        pbar.update(1)
    pbar.close()
    rewards_all = torch.vstack(rewards_list)  # [sample_num, n_sampling]
    rewards_min = torch.min(rewards_all, dim=1)[0]  # [sample_num]
    ave_reward = rewards_min.mean().cpu().item()  # int
    process_time_per_batch = sum(process_time_list) / len(process_time_list)

    if args.rendering:
        validate(rewards.cpu(), routes, test_env.location.cpu(), test_env.init_demand_ori.cpu(), args.max_load, speed)
        render(test_env.location.cpu(), routes, rewards.cpu(), key_agents, img_path)
    print("test result:", ave_reward)
    print("test time per batch:", process_time_per_batch)

    return ave_reward, process_time_per_batch


def get_name(args):
    if args.all:
        problem = "all"
    elif args.all_cust:
        problem = "all-cust"
    else:
        problem = args.title
    if args.sampling:
        strategy = f"sampling{args.n_sampling}"
    else:
        strategy = "greedy"
    name = f"{problem}-{strategy}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    return name


def main(args):
    # make directories
    epoch = os.path.basename(args.actor_path).split(".")[0]
    model_dir = os.path.dirname(os.path.dirname(args.actor_path))
    dir_name = get_name(args)
    save_dir = os.path.join(model_dir, f'test_e{epoch}', dir_name)
    log_path = os.path.join(save_dir, "result.csv")
    os.makedirs(save_dir, exist_ok=True)
    print(save_dir)
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pp.pprint(vars(args))

    checkpoint = load_checkpoint(args)
    actor = set_actor(args, checkpoint)

    with open(log_path, "a") as f:
        writer = csv.writer(f)
        writer.writerow(["actor_type", "actor_path", "n_sampling", "batch_size", "max_demand", "seed"])
        writer.writerow([args.actor_type, args.actor_path, args.n_sampling, args.batch_size, args.max_demand, args.seed])
        writer.writerow(["problem", "reward", "time(sec/batch)"])

    if args.all:
        for n_agents, n_custs_list in N_CUSTS.items():
            writer.writerow([" "])
            for n_custs in n_custs_list:
                speed = SPEED[args.speed_type][n_agents]
                max_load = MAX_LOAD[n_agents]
                img_path = os.path.join(save_dir, f"V{n_agents}-C{n_custs}.png")
                reward, process_time_per_batch = test(args, actor, n_agents, n_custs, speed, max_load, img_path)
                with open(log_path, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([f"V{n_agents}-C{n_custs}", reward, process_time_per_batch])
    elif args.all_cust:
        for n_custs in N_CUSTS[args.n_agents]:
            speed = SPEED[args.speed_type][args.n_agents]
            max_load = MAX_LOAD[args.n_agents]
            img_path = os.path.join(save_dir, f"V{args.n_agents}-C{n_custs}.png")
            reward, process_time_per_batch = test(args, actor, args.n_agents, n_custs, speed, max_load, img_path)
            with open(log_path, "a") as f:
                writer = csv.writer(f)
                writer.writerow([f"V{args.n_agents}-C{n_custs}", reward, process_time_per_batch])

    else:
        speed = SPEED[args.speed_type][args.n_agents]
        max_load = MAX_LOAD[args.n_agents]
        img_path = os.path.join(save_dir, f"V{args.n_agents}-C{args.n_custs}.png")
        reward, process_time_per_batch = test(args, actor, args.n_agents, args.n_custs, speed, max_load, img_path)
        with open(log_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([f"V{args.n_agents}-C{args.n_custs}", reward, process_time_per_batch])


if __name__ == "__main__":
    args = get_options()
    main(args)
