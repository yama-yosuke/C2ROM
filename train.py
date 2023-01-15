# -*- coding: utf-8 -*-
import os
import datetime
import copy
import pprint as pp
import logging
import traceback

import json
from tqdm import tqdm
from models.models import *
from env import Env
from env_alt import AltEnv
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


def setup_logger(args):
    # make directories
    program_dir = os.path.dirname(os.path.abspath(__file__))
    start_time = datetime.datetime.now()
    model_name = args.actor_type
    if args.veh_sel == "alt":
        model_name += "-ALT"
    save_dir = os.path.join(program_dir, 'results', args.title, "{}-{}".format(model_name, start_time.strftime('%Y-%m-%d-%H-%M-%S')))
    img_dir = os.path.join(save_dir, "imgs")
    cp_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(img_dir)
    os.makedirs(cp_dir)
    # record args
    with open(os.path.join(cp_dir, "args.json"), 'w') as f:
        json.dump(vars(args), f, indent=True)
    logger = Logger(args, start_time, save_dir, img_dir)
    return logger, cp_dir


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
        "MHAv10.7mini": ActorMHAv107mini,
        "MHAv11.1": ActorMHAv111,
    }.get(args.actor_type)
    if "9" in args.actor_type:
        actor = actor_class(dim_embed=args.dim_embed, n_heads=args.n_heads, n_layers_n=args.n_layers_n, n_layers_a=args.n_layers_a, norm_n=args.norm_n, norm_a=args.norm_a,
                            tanh_clipping=args.tanh_clipping, dropout=args.dropout, target=args.target, device=device, n_foresight=args.n_foresight).to(device)
    if "10" in args.actor_type or "11" in args.actor_type:
        actor = actor_class(dim_embed=args.dim_embed, n_heads=args.n_heads, n_layers_n=args.n_layers_n, n_layers_a=args.n_layers_a, norm_n=args.norm_n, norm_a=args.norm_a,
                            tanh_clipping=args.tanh_clipping, dropout=args.dropout, target=args.target, device=device, n_agents=args.n_agents).to(device)
    else:
        actor = actor_class(dim_embed=args.dim_embed, n_heads=args.n_heads, n_layers_n=args.n_layers_n, n_layers_a=args.n_layers_a,
                            norm_n=args.norm_n, norm_a=args.norm_a, tanh_clipping=args.tanh_clipping, dropout=args.dropout, target=args.target, device=device).to(device)
    if checkpoint:
        print('Loading actor checkpoint...\n')
        actor.load_state_dict(checkpoint["actor"])
    if parallel:
        actor = DDP(actor, device_ids=[rank])
    actor_optimizer = Adam(actor.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.LambdaLR(actor_optimizer, lambda epoch: args.lr_decay ** epoch)
    if checkpoint:
        actor_optimizer.load_state_dict(checkpoint["actor_opt"])
        # issue with pytorch1.12.0(https://github.com/pytorch/pytorch/issues/80809)
        # resolved in 1.12.1
        # actor_optimizer.param_groups[0]["capturable"] = True
        if args.start_epoch == int(os.path.basename(args.cp_path).split(".")[0]):
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        print(f"lr:{lr_scheduler.get_last_lr()[0]}")
    return actor, actor_optimizer, lr_scheduler


def set_baseline(parallel, actor, checkpoint):
    # Note1. deepcopy must come after forward propagation with no grad or before training. Othrewise, RuntimeError occurs.
    # Note1. RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment
    # Note2. actor, baseline別でDDPを作るとエラーが出るので,copy
    baseline = copy.deepcopy(actor)
    if checkpoint:
        print('Loading rollout baseline checkpoint...\n')
        if parallel:
            state_dict = {}
            for key, val in checkpoint["rollout_baseline"].items():
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


def train_dist(rank, args, parallel):
    if args.debug:
        logging.basicConfig(level=logging.INFO)
    device = torch.device("cpu") if (args.no_gpu or not torch.cuda.is_available()) else torch.device('cuda', rank)
    print(f"{rank}: Running train_dist on device {device}.")
    if parallel:
        setup(rank, args.world_size, args)

    torch.manual_seed(args.seed + rank)
    # TODO: Is "+rank" necessary?
    # https://pytorch.org/docs/1.12/data.html#randomness-in-multi-process-data-loading
    logging.info(f"rank:{rank}, {torch.initial_seed()}")

    # initialize env
    if args.veh_sel == "chr":
        # chronological vehicle selection
        train_env = Env(rank=rank, device=device, world_size=args.world_size, sample_num=args.train_samples, global_batch_size=args.train_batch_size)
        val_env = Env(rank=rank, device=device, world_size=args.world_size, sample_num=args.val_samples, global_batch_size=args.val_batch_size)
        val_env.load_maps(args.n_custs, args.max_demand)  # validation data is fixed
        ttest_env = Env(rank=rank, device=device, world_size=args.world_size, sample_num=args.ttest_samples, global_batch_size=args.val_batch_size)
    else:
        # alternating vehicle selection
        train_env = AltEnv(rank=rank, device=device, world_size=args.world_size, sample_num=args.train_samples, global_batch_size=args.train_batch_size)
        val_env = AltEnv(rank=rank, device=device, world_size=args.world_size, sample_num=args.val_samples, global_batch_size=args.val_batch_size)
        val_env.load_maps(args.n_custs, args.max_demand)  # validation data is fixed
        ttest_env = AltEnv(rank=rank, device=device, world_size=args.world_size, sample_num=args.ttest_samples, global_batch_size=args.val_batch_size)

    # initialize models
    # TODO: env(seed=1)から再開すると、学習データが被る？？
    checkpoint = load_checkpoint(rank, device, args)
    actor, actor_optimizer, lr_scheduler = set_actor(rank, device, parallel, args, checkpoint)
    baseline = set_baseline(parallel, actor, checkpoint)
    bl_rewards = None

    if rank == 0:
        logger, cp_dir = setup_logger(args)

    # start training
    for epoch in range(args.start_epoch + 1, args.end_epoch + 1):
        # calc offset
        step = (epoch - 1) * args.train_batches

        train_env.make_maps(args.n_custs, args.max_demand)

        if rank == 0:
            if args.curriculum:
                print(f"training on V{args.n_agents}-C{args.n_custs}")
            pbar = tqdm(total=args.train_batches)
        while(train_env.next()):
            step += 1
            # logging.info(f"rank{rank} @step{step}: actor.param={actor.module.embedder_node.embed.weight[0]}")
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
            advantage = (train_rewards - bl_rewards).detach()  # bl_rewardsにcriticを用いる場合のためにdetach()
            # [1]
            # torch.autograd.set_detect_anomaly(True)
            actor_loss = (sum_logprobs * advantage).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            logging.info(f"rank:{rank}, after backward")
            # distributed synchronization points
            nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
            actor_optimizer.step()
            logging.info(f"rank:{rank}, after step")

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
                    logging.info(f"{rank}: after barrier val")
            if rank == 0:
                pbar.update(1)
        if rank == 0:
            pbar.close()
        lr_scheduler.step()

        # update baseline
        # torch.tensor([ttest_samples/world_size] on each device)
        baseline_samples, actor_samples = sample_test(args, ttest_env, baseline, actor, args.n_custs, args.n_agents, args.speed, args.load)
        logging.info(f"{rank}:after ttest sample")
        if parallel:
            # torch.tensor([ttest_samples]) on rank 0, None on other ranks
            baseline_samples, actor_samples = gather_samples(rank, args, baseline_samples, actor_samples)
        if rank == 0:
            needs_update = check_update(device, args, actor_samples.cpu().numpy(), baseline_samples.cpu().numpy())
        else:
            # 受け取る変数を作成
            needs_update = torch.tensor(True, device=device, dtype=torch.bool)
        if parallel:
            # sync
            dist.barrier()
            logging.info(f"{rank}:after barrier ttest")
            dist.broadcast(needs_update, 0)
            logging.info(f"{rank}: received flag {needs_update}")
        if needs_update or epoch == 1:
            baseline = copy.deepcopy(actor)
            logging.info(f"{rank}: actor cloned to baseline")

        if rank == 0:
            if epoch % args.render_interval == 0 or epoch == args.end_epoch:
                # render result
                val_env.reindex()
                val_env.next()
                _, val_rewards, key_agents, routes = sample_actor(args, val_env, actor, args.n_agents, args.speed, args.load, test=True, out_tour=True)
                logger.render(val_env.location.cpu(), routes, val_rewards.cpu(), key_agents)

            if epoch % args.cp_interval == 0 or epoch == args.end_epoch:
                # save model
                save(parallel, cp_dir, epoch, actor, actor_optimizer, lr_scheduler, baseline)
        if parallel:
            dist.barrier()
            logging.info(f"{rank}:after barrier @ end of epoch")

    if rank == 0:
        logger.close()
    if parallel:
        cleanup()


def main(args):
    pp.pprint(vars(args))
    if args.world_size > 1:
        print(f"\n===== Spawn {args.world_size} processes =====")
        mp.spawn(train_dist, args=(args, True), nprocs=args.world_size, join=True)
    else:
        print(f"\n===== Spawn single process =====")
        train_dist(rank=0, args=args, parallel=False)


if __name__ == "__main__":
    args = get_options()
    main(args)

