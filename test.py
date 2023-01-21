from train import execute_routing
import os
import datetime
import argparse
import json
import pprint as pp
from tqdm import tqdm

from models.models import *

from env import Env
import csv
import torch
import matplotlib.pyplot as plt
from const import SPEED, MAX_LOAD


def get_options():
    parser = argparse.ArgumentParser(description="Test C2ROM")

    parser.add_argument("actor_path", type=str, default=None, help="path to model parameter")

    # Environment parameters
    parser.add_argument("--n_custs", type=int, default=None, help="size of customers")
    parser.add_argument("--n_sampling", type=int, default=1, help="size of sampling")

    # dataset parameters
    parser.add_argument("--instance_num", type=int, default=1280)
    parser.add_argument("--batch_size", type=int, default=1280)
    parser.add_argument("--seed", type=str, default="=TEST")
    parser.add_argument("--max_demand", type=int, default=9)

    args = parser.parse_args()  # dict

    assert args.instance_num % args.batch_size == 0, "sample num must be divisible by batch size"

    with open(os.path.join(os.path.dirname(args.actor_path), "args.json"), "r") as f:
        args_load = json.load(f)  # dict

    # fixed
    args.multi_gpus = False
    args.n_heads = args_load["n_heads"]
    args.dim_embed = args_load["dim_embed"]
    args.tanh_clipping = args_load["tanh_clipping"]
    args.dropout = args_load["dropout"]
    args.n_agents = args_load["n_agents"]
    args.target = args_load["target"]
    args.speed_type = args_load["speed_type"]
    args.sampling = (args.n_sampling > 1)  # flag for sampling

    # overwrite loaded settings
    args.n_custs = args.n_custs or args_load["n_custs"]

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
    actor = PolicyNetwork(dim_embed=args.dim_embed, n_heads=args.n_heads,
                          tanh_clipping=args.tanh_clipping, dropout=args.dropout, target=args.target, device=args.device, n_agents=args.n_agents).to(args.device)
    if "actor" in checkpoint:
        print('Loading actor checkpoint...\n')
        actor.load_state_dict(checkpoint["actor"])
    return actor


def render(location, tours, rewards, save_path):
    """
    render the routing result
    Args:
        location (Tensor): [B, n_nodes, 2], location of nodes
        tours (list): list [B, n_agent, n_visits], n_visits varies among vehicles
        rewards (tensor): [B], tour_length
        n_custs: int, customerの数
        save path
    """
    cmap = plt.get_cmap("hsv")
    n_agents = len(tours[0])
    _, axes = plt.subplots(nrows=3, ncols=3, sharex='col', sharey='row', figsize=(10, 10))
    axes = [a for ax in axes for a in ax]
    for episode, ax in enumerate(axes):
        # node
        ax.scatter(location[episode, 0, 0], location[episode, 0, 1],
                   color="k", marker="*", s=200, zorder=2)  # depot
        ax.scatter(location[episode, 1:, 0], location[episode,
                   1:, 1], color="k", marker="o", zorder=2)  # customer
        # tour
        for agent_id, tour in enumerate(tours[episode]):
            tour_index = torch.tensor(tour).unsqueeze(1).expand(-1, 2)  # [n_visits]
            # input=[n_nodes, 2], dim=1, index=[n_visits, 2]
            tour_xy = torch.gather(location[episode], dim=0, index=tour_index)
            ax.plot(tour_xy[:, 0], tour_xy[:, 1], color=cmap(agent_id / n_agents),
                    linestyle="-", linewidth=0.8, zorder=1, label=f"agent{agent_id}")
        title = "time:{:.3f}".format(rewards[episode].item())
        ax.set_aspect('equal')
        ax.set_title(title, y=-0.1)
        ax.axis([-0.05, 1.05, -0.05, 1.05])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1,), ncol=3)
    plt.tight_layout()
    plt.suptitle(f"test", y=0)
    plt.savefig(save_path, facecolor="white", edgecolor="black")
    plt.close()
    return


def test(args, actor, n_agents, n_custs, speed, max_load, img_path):

    # start testing
    print(f'Test: V{n_agents}-C{n_custs}')
    # initialize env
    # test_env
    test_env = Env(rank=0, device=args.device, world_size=1, instance_num=args.instance_num, global_batch_size=args.batch_size, rep=args.n_sampling)
    test_env.load_maps(n_custs, args.max_demand, "test", args.seed)

    rewards_list = []

    pbar = tqdm(total=test_env.batch_num)
    while(test_env.next()):
        _, rewards, routes = execute_routing(args, test_env, actor, n_agents, speed, max_load, test=True, sampling=args.sampling, rep=args.n_sampling)
        rewards_list.append(rewards.reshape(args.n_sampling, -1).permute(1, 0))  # list of len=instance_num/batch_size, ele=tensor[batch_size, n_sampling]
        pbar.update(1)
    pbar.close()
    rewards_all = torch.vstack(rewards_list)  # [sample_num, n_sampling]
    rewards_min = torch.min(rewards_all, dim=1)[0]  # [sample_num]  # retrieve the best solution in sampling strategy
    ave_reward = rewards_min.mean().cpu().item()  # int

    print("test result:", ave_reward)
    render(test_env.location.cpu(), routes, rewards.cpu(), img_path)

    return ave_reward


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
    strategy = f"sampling{args.n_sampling}" if args.n_sampling > 1 else "greedy"
    save_dir = os.path.join(model_dir, f'test_e{epoch}', f"{args.title}-{strategy}")
    log_path = os.path.join(save_dir, "result.csv")
    os.makedirs(save_dir, exist_ok=True)
    print(save_dir)
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pp.pprint(vars(args))

    checkpoint = load_checkpoint(args)
    actor = set_actor(args, checkpoint)

    with open(log_path, "a") as f:
        writer = csv.writer(f)
        writer.writerow(["problem", "reward"])

    speed = SPEED[args.speed_type][args.n_agents]
    max_load = MAX_LOAD[args.n_agents]
    img_path = os.path.join(save_dir, f"V{args.n_agents}-C{args.n_custs}.png")
    reward = test(args, actor, args.n_agents, args.n_custs, speed, max_load, img_path)
    with open(log_path, "a") as f:
        writer = csv.writer(f)
        writer.writerow([f"V{args.n_agents}-C{args.n_custs}", reward])


if __name__ == "__main__":
    args = get_options()
    main(args)
