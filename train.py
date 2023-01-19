# -*- coding: utf-8 -*-
import os
import datetime
import copy
import pprint as pp
import logging

import json
from tqdm import tqdm
from models.models import PolicyNetwork
from env import Env
from utils import Logger
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from scipy.stats import ttest_rel
from options import get_options


def setup(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def sample_actor(args, env, actor, n_agents, speed, max_load, test, out_tour=False, sampling=False, rep=1, calc_time=False):
    """
    Params:
        env: Environment
        actpr: actor
        speed: list of n_agnets
        max_load: list of n_agnets
        test: If True, greedy 
        out_tour: If true, output tour index
    Returns:
        sum_logprobs:[B](cuda), 選んだactionに対するlogprobのsum
        rewards: [B](cuda), min-max of total time or min-sum of total time
        key_agents: [B](cpu), MMにおいて、最も時間のかかったactorのindex, MSならNone
        routes: list of [B, n_agents, n_visits](cpu)
    """
    if rep > 1:
        assert sampling and test, "reputation is only available in test sampling"
    actor.train(mode=(not test))  # trainingならTureにする
    with torch.set_grad_enabled(mode=(not test)):
        start = time.time()
        sum_logprobs, rewards, key_agents, routes = actor(args, env, n_agents, speed, max_load, out_tour, sampling, rep)
        process_time = (time.time() - start)
    if calc_time:
        return sum_logprobs, rewards, key_agents, routes, process_time
    else:
        return sum_logprobs, rewards, key_agents, routes


def calc_ave(args, rewards, mv_ave):
    """
    rolloutのwarmup中に移動平均を計算
    Args:
        rewards: [B]
        mv_ave: [B], 移動平均(バッチ平均)
    Returns:
        mv_ave: [B]
    """
    batch_mean = rewards.mean().expand_as(rewards)  # [B]
    if mv_ave is not None:
        return (args.mv_beta * mv_ave + (1 - args.mv_beta) * batch_mean)
    return batch_mean


def load_checkpoint(rank, device, args):
    checkpoint = {}
    if args.cp_path:
        print(f'{rank}: Loading checkpoint...\n')
        checkpoint = torch.load(args.cp_path, map_location=device)
    return checkpoint


def set_actor(rank, device, parallel, args, checkpoint):
    actor = PolicyNetwork(dim_embed=args.dim_embed, n_heads=args.n_heads, n_layers_n=args.n_layers_n, n_layers_a=args.n_layers_a, norm_n=args.norm_n, norm_a=args.norm_a,
                            tanh_clipping=args.tanh_clipping, dropout=args.dropout, target=args.target, device=device, n_agents=args.n_agents).to(device)
    if checkpoint:
        print('Loading actor checkpoint...\n')
        actor.load_state_dict(checkpoint["actor"])
    if parallel:
        actor = DDP(actor, device_ids=[rank])
    actor_optimizer = Adam(actor.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.LambdaLR(actor_optimizer, lambda epoch: args.lr_decay ** epoch)
    if checkpoint:
        actor_optimizer.load_state_dict(checkpoint["actor_opt"])
        if args.start_epoch == int(os.path.basename(args.cp_path).split(".")[0]):
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    return actor, actor_optimizer, lr_scheduler


def set_baseline(parallel, actor, checkpoint):
    baseline = copy.deepcopy(actor)
    if checkpoint:
        print('Loading rollout baseline checkpoint...\n')
        if parallel:
            state_dict = {}
            for key, val in checkpoint["rollout_baseline"].items():
                # rename to use DDP
                state_dict["module." + key] = val
            baseline.load_state_dict(state_dict)
        else:
            baseline.load_state_dict(checkpoint["rollout_baseline"])
    return baseline


def check_update(device, args, actor_samples, baseline_samples):
    """
    OneSidedPairedTTestで有意ならbaselineを更新する
    Returns:
        bool: if True, updating baseline is needed
    """
    print('\n===== OneSidedPairedTTest =====\n')
    # paired t test
    t, p = ttest_rel(actor_samples, baseline_samples)  # sign(actor_samples - baseline_samples) = sign(t) → 学習しているならactor_samples < baseline_samples　→　t<0
    p_val = p / 2  # one-sided
    print("t={:.3e}".format(t))  # TODO: t>=0だと学習が進んでいない？？
    if t >= 0:
        print("Warning!! t>=0")
        print("keep baseline (p_val={:.3e})\n".format(p_val))
        return torch.tensor(False, device=device)
    if p_val < args.ttest_alpha:
        # 帰無仮説棄却
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
        val_reward_mean: [1](on each actor's device)
    """
    val_reward_list = []
    val_env.reindex()  # val_dataset is used repeatedly
    while(val_env.next()):
        _, val_rewards, _, _ = sample_actor(args, val_env, actor, n_agents, speed, max_load, test=True)
        val_reward_list.append(val_rewards.mean())
    val_reward_mean = torch.stack(val_reward_list, 0).mean()
    return val_reward_mean


def sample_test(args, ttest_env, baseline, actor, n_custs, n_agents, speed, max_load):
    """
    Returns:
        baseline_samples: [ttest_samples/world_size] on each device
        actor_samples: [ttest_samples/world_size] on each device
    """
    baseline_list = []
    actor_list = []
    ttest_env.make_maps(n_custs, args.max_demand)
    while(ttest_env.next()):
        _, bl_rewards, _, _ = sample_actor(args, ttest_env, baseline, n_agents, speed, max_load, test=True)
        baseline_list.append(bl_rewards)
        _, actor_rewards, _, _ = sample_actor(args, ttest_env, actor, n_agents, speed, max_load, test=True)
        actor_list.append(actor_rewards)
    baseline_samples = torch.cat(baseline_list, 0)  # torch.tensor [ttest_samples/world_size]
    actor_samples = torch.cat(actor_list, 0)  # torch.tensor [ttest_samples/world_size]
    return baseline_samples, actor_samples


def gather_samples(rank, args, baseline_samples, actor_samples):
    """
    Args:
        baseline_samples: [ttest_samples/world_size] on each device
        actor_samples: [ttest_samples/world_size] on each device
    Returns:
        if rank == 0:
            baseline_samples: torch.tensor[ttest_samples] on rank0
            actor_samples: torch.tensor[ttest_samples] on rank0
        else:
            None
            None
    """
    if rank == 0:
        baseline_samples_list = [torch.zeros_like(baseline_samples) for _ in range(args.world_size)]
        actor_samples_list = [torch.zeros_like(actor_samples) for _ in range(args.world_size)]
        dist.gather(baseline_samples, gather_list=baseline_samples_list)
        dist.gather(actor_samples, gather_list=actor_samples_list)
    else:
        dist.gather(baseline_samples, dst=0)
        dist.gather(actor_samples, dst=0)
    if rank == 0:
        logging.info(f"samples_list, len{len(baseline_samples_list)}, ele.shape{baseline_samples_list[0].shape}")
        baseline_samples = torch.cat(baseline_samples_list, 0)  # np.array [ttest_samples]
        actor_samples = torch.cat(actor_samples_list, 0)  # np.array [ttest_samples]
        return baseline_samples, actor_samples
    else:
        return None, None


def train_dist(rank, args, parallel, logger, cp_dir):
    device = torch.device("cpu") if (args.no_gpu or not torch.cuda.is_available()) else torch.device('cuda', rank)
    print(f"{rank}: Running train_dist on device {device}.")
    if parallel:
        setup(rank, args.world_size, args)

    torch.manual_seed(args.seed + rank)

    # initialize env
    train_env = Env(rank=rank, device=device, world_size=args.world_size, sample_num=args.train_samples, global_batch_size=args.train_batch_size)
    val_env = Env(rank=rank, device=device, world_size=args.world_size, sample_num=args.val_samples, global_batch_size=args.val_batch_size)
    val_env.load_maps(args.n_custs, args.max_demand)  # validation data is fixed
    ttest_env = Env(rank=rank, device=device, world_size=args.world_size, sample_num=args.ttest_samples, global_batch_size=args.val_batch_size)

    # initialize models
    checkpoint = load_checkpoint(rank, device, args)
    actor, actor_optimizer, lr_scheduler = set_actor(rank, device, parallel, args, checkpoint)
    baseline = set_baseline(parallel, actor, checkpoint)
    bl_rewards = None

    # start training
    for epoch in range(args.start_epoch + 1, args.end_epoch + 1):
        # calc offset
        step = (epoch - 1) * args.train_batches

        train_env.make_maps(args.n_custs, args.max_demand)

        if rank == 0:
            pbar = tqdm(total=args.train_batches)
        while(train_env.next()):
            step += 1
            # [B]
            sum_logprobs, train_rewards, _, _ = sample_actor(args, train_env, actor, args.n_agents, args.speed, args.load, test=False)
            train_reward_mean = train_rewards.mean()  # [1]
            # baseline
            if epoch == 1:
                # warm-up in rollout
                bl_rewards = calc_ave(args, train_rewards, bl_rewards)
            else:
                _, bl_rewards, _, _ = sample_actor(args, train_env, baseline, args.n_agents, args.speed, args.load, test=True)
            # ACTOR UPDATE
            # [B]
            advantage = (train_rewards - bl_rewards).detach()
            # [1]
            actor_loss = (sum_logprobs * advantage).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
            actor_optimizer.step()

            # log
            if step % args.log_interval == 0:
                val_reward_mean = validate(args, val_env, actor, args.n_agents, args.speed, args.load)
                if parallel:
                    dist.reduce(train_reward_mean, dst=0, op=dist.ReduceOp.SUM)
                    dist.reduce(actor_loss, dst=0, op=dist.ReduceOp.SUM)
                    dist.reduce(val_reward_mean, dst=0, op=dist.ReduceOp.SUM)
                if rank == 0:
                    sample = step * args.train_batch_size
                    train_reward_mean = (train_reward_mean / args.world_size).cpu().item()
                    actor_loss = (actor_loss / args.world_size).cpu().item()
                    val_reward_mean = (val_reward_mean / args.world_size).cpu().item()
                    logger.add_vals(epoch, step, lr_scheduler.get_last_lr()[0], sample, train_reward_mean, actor_loss, val_reward_mean)
                    logger.output()
                if parallel:
                    dist.barrier()
            if rank == 0:
                pbar.update(1)
        if rank == 0:
            pbar.close()
        lr_scheduler.step()

        # update baseline
        # torch.tensor([ttest_samples/world_size] on each device)
        baseline_samples, actor_samples = sample_test(args, ttest_env, baseline, actor, args.n_custs, args.n_agents, args.speed, args.load)
        if parallel:
            # torch.tensor([ttest_samples]) on rank 0, None on other ranks
            baseline_samples, actor_samples = gather_samples(rank, args, baseline_samples, actor_samples)
        if rank == 0:
            needs_update = check_update(device, args, actor_samples.cpu().numpy(), baseline_samples.cpu().numpy())
        else:
            # flag to use to receice broadcasted bool
            needs_update = torch.tensor(True, device=device, dtype=torch.bool)
        if parallel:
            # sync
            dist.barrier()
            dist.broadcast(needs_update, 0)
        if needs_update or epoch == 1:
            baseline = copy.deepcopy(actor)

        if rank == 0 and (epoch % args.cp_interval == 0):
            # save model
            save(parallel, cp_dir, epoch, actor, actor_optimizer, lr_scheduler, baseline)
        if parallel:
            dist.barrier()

    if rank == 0:
        logger.close()
    if parallel:
        cleanup()


def main(args):
    # setup log
    start_time = datetime.datetime.now()
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', args.title, start_time.strftime('%Y-%m-%d-%H-%M-%S'))
    cp_dir = os.path.join(save_dir, "checkpoints")
    log_path = os.path.join(save_dir, "log.csv")
    os.makedirs(cp_dir)
    logger = Logger(args, start_time, log_path)
    # record args
    pp.pprint(vars(args))
    with open(os.path.join(cp_dir, "args.json"), 'w') as f:
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

