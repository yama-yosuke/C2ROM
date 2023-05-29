# -*- coding: utf-8 -*-
import copy
import datetime
import json
import os
import pprint as pp

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from scipy.stats import ttest_rel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from tqdm import tqdm

from env import Env
from models.models import PolicyNetwork
from options import get_options
from utils import Logger


def setup_mp(rank, world_size, args):
    """
    set up for multi-processing
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_mp():
    dist.destroy_process_group()


def execute_routing(args, env, model, n_agents, speed, max_load, test, sampling=False, rep=1):
    """
    Args:
        env: Environment
        model: actor or baseline
        speed (list): list of speeds of all vehicles
        max_load (list): list of loads of all vehicles
        test (bool): model mode
        sampling (bool): If True, select action stochastically
        rep (int): sampling num in sampling strategy
    Returns:
        Tensor: sum of logprobs of selected action, shape=[B]
        Tensor: min-max(MM) of total time or min-sum(MS) of total time, shape=[B]
        list: routes(index of node in visited order), shape=[B, n_agents, n_visits]
    """
    if rep > 1:
        assert sampling and test, "reputation is only available in test sampling"
    model.train(mode=(not test))
    with torch.set_grad_enabled(mode=(not test)):
        sum_logprobs, rewards, routes = model(args, env, n_agents, speed, max_load, sampling, rep)

    return sum_logprobs, rewards, routes


def calc_ave(args, rewards, mv_ave):
    """
    Args:
        rewards (Tensor): shape=[B], train rewards
        mv_ave (Tensor): shape=[B], moving average of train reward
    Returns:
        Tensor: shape=[B], updated moving average of train reward
    """
    batch_mean = rewards.mean().expand_as(rewards)  # [B]
    if mv_ave is not None:
        return args.mv_beta * mv_ave + (1 - args.mv_beta) * batch_mean
    return batch_mean


def load_checkpoint(rank, device, args):
    checkpoint = {}
    if args.cp_path:
        print(f"{rank}: Loading checkpoint...\n")
        checkpoint = torch.load(args.cp_path, map_location=device)
    return checkpoint


def set_actor(rank, device, parallel, args, checkpoint):
    actor = PolicyNetwork(
        dim_embed=args.dim_embed,
        n_heads=args.n_heads,
        tanh_clipping=args.tanh_clipping,
        dropout=args.dropout,
        target=args.target,
        device=device,
        n_agents=args.n_agents,
    ).to(device)
    if checkpoint:
        print("Loading actor checkpoint...\n")
        actor.load_state_dict(checkpoint["actor"])
    if parallel:
        actor = DDP(actor, device_ids=[rank])
    actor_optimizer = Adam(actor.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.LambdaLR(actor_optimizer, lambda epoch: args.lr_decay**epoch)
    if checkpoint:
        actor_optimizer.load_state_dict(checkpoint["actor_opt"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    return actor, actor_optimizer, lr_scheduler


def set_baseline(parallel, actor, checkpoint):
    baseline = copy.deepcopy(actor)
    if checkpoint:
        print("Loading rollout baseline checkpoint...\n")
        if parallel:
            # rename to use DDP baseline
            state_dict = {}
            for key, val in checkpoint["rollout_baseline"].items():
                state_dict["module." + key] = val
            baseline.load_state_dict(state_dict)
        else:
            baseline.load_state_dict(checkpoint["rollout_baseline"])
    return baseline


def check_update(device, args, actor_rewards, baseline_rewards):
    """
    conduct TTestOneSidedPairedTTest
    Args:
        args:
        baseline_rewards (Tensor):  baseline rewards on rank0, shape=[ttest_samples]
        actor_rewards (Tensor): actor rewards on rank0, shape=[ttest_samples]
    Returns:
        Tensor: if True, updating baseline is needed
    """
    print("\n===== OneSidedPairedTTest =====\n")
    # paired t test
    t, p = ttest_rel(actor_rewards, baseline_rewards)
    p_val = p / 2  # one-sided
    print("t={:.3e}".format(t))
    if t >= 0:
        print("Warning!! t>=0")
        print("keep baseline (p_val={:.3e})\n".format(p_val))
        return torch.tensor(False, device=device)
    if p_val < args.ttest_alpha:
        print("update baseline (p_val={:.3e} < alpha={})\n".format(p_val, args.ttest_alpha))
        return torch.tensor(True, device=device)
    else:
        print("keep baseline (p_val={:.3e} > alpha={})\n".format(p_val, args.ttest_alpha))
        return torch.tensor(False, device=device)


def save(parallel, save_dir, epoch, actor, actor_optimizer, lr_scheduler, baseline):
    save_path = os.path.join(save_dir, "{}.pt".format(epoch))
    checkpoint = {}
    if parallel:
        checkpoint["actor"] = copy.deepcopy(actor).to("cpu").module.state_dict()
        checkpoint["actor_opt"] = actor_optimizer.state_dict()
        checkpoint["rollout_baseline"] = copy.deepcopy(baseline).to("cpu").module.state_dict()
    else:
        checkpoint["actor"] = copy.deepcopy(actor).to("cpu").state_dict()
        checkpoint["actor_opt"] = actor_optimizer.state_dict()
        checkpoint["rollout_baseline"] = copy.deepcopy(baseline).to("cpu").state_dict()
    checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
    torch.save(checkpoint, save_path)


def validate(args, val_env, actor, n_agents, speed, max_load):
    """
    Returns:
        (Tensor): mean of validaton rewards, on each device, shape=[1]
    """
    val_reward_list = []
    val_env.reindex()  # val_dataset is used repeatedly
    while val_env.next():
        _, val_rewards, _ = execute_routing(args, val_env, actor, n_agents, speed, max_load, test=True)
        val_reward_list.append(val_rewards.mean())
    val_reward_mean = torch.stack(val_reward_list, 0).mean()
    return val_reward_mean


def test(args, ttest_env, baseline, actor, n_custs, n_agents, speed, max_load):
    """
    sample solutions greedily on the same instances by baseline and actor and return thier rewards
    Args:
    Returns:
        Tensor: baseline rewards on each device, shape=[ttest_samples/world_size]
        Tensor: actor rewards on each device, shape=[ttest_samples/world_size]
    """
    baseline_list = []
    actor_list = []
    ttest_env.make_maps(n_custs, args.max_demand)
    while ttest_env.next():
        _, bl_rewards, _ = execute_routing(args, ttest_env, baseline, n_agents, speed, max_load, test=True)
        baseline_list.append(bl_rewards)
        _, actor_rewards, _ = execute_routing(args, ttest_env, actor, n_agents, speed, max_load, test=True)
        actor_list.append(actor_rewards)
    baseline_rewards = torch.cat(baseline_list, 0)  # torch.tensor [ttest_samples/world_size]
    actor_rewards = torch.cat(actor_list, 0)  # torch.tensor [ttest_samples/world_size]
    return baseline_rewards, actor_rewards


def gather_rewards(rank, args, baseline_rewards, actor_rewards):
    """
    gather test rewards on rank0
    Args:
        baseline_rewards (Tensor): baseline rewards on each device, shape=[ttest_samples/world_size]
        actor_rewards (Tensor): actor rewards on each device, shape=[ttest_samples/world_size]
    Returns:
        if rank == 0:
            Tensor:  baseline rewards on rank0, shape=[ttest_samples]
            Tensor: actor rewards on rank0, shape=[ttest_samples]
        else:
            None
            None
    """
    if rank == 0:
        baseline_rewards_list = [torch.zeros_like(baseline_rewards) for _ in range(args.world_size)]
        actor_rewards_list = [torch.zeros_like(actor_rewards) for _ in range(args.world_size)]
        dist.gather(baseline_rewards, gather_list=baseline_rewards_list)
        dist.gather(actor_rewards, gather_list=actor_rewards_list)
    else:
        dist.gather(baseline_rewards, dst=0)
        dist.gather(actor_rewards, dst=0)
    if rank == 0:
        baseline_rewards = torch.cat(baseline_rewards_list, 0)  # np.array [ttest_samples]
        actor_rewards = torch.cat(actor_rewards_list, 0)  # np.array [ttest_samples]
        return baseline_rewards, actor_rewards
    else:
        return None, None


def train_dist(rank, args, parallel, logger, cp_dir):
    device = torch.device("cpu") if (args.no_gpu or not torch.cuda.is_available()) else torch.device("cuda", rank)
    print(f"{rank}: Running train_dist on device {device}.")
    if parallel:
        setup_mp(rank, args.world_size, args)

    torch.manual_seed(args.seed + rank)

    # initialize env
    train_env = Env(
        rank=rank,
        device=device,
        world_size=args.world_size,
        instance_num=args.train_instance_num,
        global_batch_size=args.train_batch_size,
    )
    val_env = Env(
        rank=rank,
        device=device,
        world_size=args.world_size,
        instance_num=args.val_instance_num,
        global_batch_size=args.val_batch_size,
    )
    val_env.load_maps(args.n_custs, args.max_demand)  # validation data is fixed
    ttest_env = Env(
        rank=rank,
        device=device,
        world_size=args.world_size,
        instance_num=args.ttest_instance_num,
        global_batch_size=args.val_batch_size,
    )

    # initialize models
    checkpoint = load_checkpoint(rank, device, args)
    actor, actor_optimizer, lr_scheduler = set_actor(rank, device, parallel, args, checkpoint)
    baseline = set_baseline(parallel, actor, checkpoint)
    bl_rewards = None

    # start training
    for epoch in range(args.start_epoch + 1, args.end_epoch + 1):
        # calc offset
        trained_batch_num = (epoch - 1) * args.train_batch_num
        # generate train data on the fly
        train_env.make_maps(args.n_custs, args.max_demand)

        if rank == 0:
            pbar = tqdm(total=args.train_batch_num)
        while train_env.next():
            trained_batch_num += 1
            # execute actor
            sum_logprobs, train_rewards, _ = execute_routing(
                args, train_env, actor, args.n_agents, args.speed, args.load, test=False
            )
            train_reward_mean = train_rewards.mean()

            # execute baseline
            if epoch == 1:
                # at initial epoch, moving average of training reward is used instead of rollout baseline
                bl_rewards = calc_ave(args, train_rewards, bl_rewards)
            else:
                _, bl_rewards, _ = execute_routing(
                    args, train_env, baseline, args.n_agents, args.speed, args.load, test=True
                )

            # update actor parameter
            advantage = (train_rewards - bl_rewards).detach()  # [B]
            actor_loss = (sum_logprobs * advantage).mean()  # [1]
            actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
            actor_optimizer.step()

            # log
            if trained_batch_num % args.log_interval == 0:
                # execute validation on each device
                val_reward_mean = validate(args, val_env, actor, args.n_agents, args.speed, args.load)
                if parallel:
                    # reduce results to rank0
                    dist.reduce(train_reward_mean, dst=0, op=dist.ReduceOp.SUM)
                    dist.reduce(actor_loss, dst=0, op=dist.ReduceOp.SUM)
                    dist.reduce(val_reward_mean, dst=0, op=dist.ReduceOp.SUM)
                if rank == 0:
                    # output log in rank0
                    trained_instance_num = trained_batch_num * args.train_batch_size
                    train_reward_mean = (train_reward_mean / args.world_size).cpu().item()
                    actor_loss = (actor_loss / args.world_size).cpu().item()
                    val_reward_mean = (val_reward_mean / args.world_size).cpu().item()
                    logger.add_vals(
                        epoch,
                        trained_batch_num,
                        lr_scheduler.get_last_lr()[0],
                        trained_instance_num,
                        train_reward_mean,
                        actor_loss,
                        val_reward_mean,
                    )
                    logger.output()
                if parallel:
                    dist.barrier()
            if rank == 0:
                pbar.update(1)
        if rank == 0:
            pbar.close()
        lr_scheduler.step()

        # check if baseline needs updating
        # execute routing on test data on each device using both baseline and actor
        baseline_rewards, actor_rewards = test(
            args, ttest_env, baseline, actor, args.n_custs, args.n_agents, args.speed, args.load
        )
        if parallel:
            # gather rewards on rank0
            baseline_rewards, actor_rewards = gather_rewards(rank, args, baseline_rewards, actor_rewards)
        if rank == 0:
            # conduct TTest on rank0
            needs_update = check_update(device, args, actor_rewards.cpu().numpy(), baseline_rewards.cpu().numpy())
        else:
            # used to receice broadcasted flag
            needs_update = torch.tensor(True, device=device, dtype=torch.bool)
        if parallel:
            # broadcast TTest results to all devices
            dist.barrier()
            dist.broadcast(needs_update, 0)
        if needs_update or epoch == 1:
            # update baseline
            baseline = copy.deepcopy(actor)

        if rank == 0 and (epoch % args.cp_interval == 0):
            # save model
            save(parallel, cp_dir, epoch, actor, actor_optimizer, lr_scheduler, baseline)
        if parallel:
            dist.barrier()

    if parallel:
        cleanup_mp()


def main(args):
    # setup log
    start_time = datetime.datetime.now()
    save_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", args.title, start_time.strftime("%Y-%m-%d-%H-%M-%S")
    )
    cp_dir = os.path.join(save_dir, "checkpoints")
    log_path = os.path.join(save_dir, "log.csv")
    os.makedirs(cp_dir)
    logger = Logger(args, start_time, log_path)
    # record args
    pp.pprint(vars(args))
    with open(os.path.join(cp_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=True)

    if args.world_size > 1:
        print(f"\n===== Spawn {args.world_size} processes =====")
        mp.spawn(train_dist, args=(args, True, logger, cp_dir), nprocs=args.world_size, join=True)
    else:
        print(f"\n===== Spawn single process =====")
        train_dist(rank=0, args=args, parallel=False, logger=logger, cp_dir=cp_dir)


if __name__ == "__main__":
    args = get_options()
    main(args)
