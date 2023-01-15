import argparse
import torch
import os
from const import SPEED, MAX_LOAD


def get_options():
    parser = argparse.ArgumentParser(description="C2ROM")

    # Environment parameters
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--n_custs", type=int, default=40, help="size of customers")
    parser.add_argument("--n_agents", type=int, default=3, help="size of fleet")
    parser.add_argument("--target", type=str, default="MM", help="Min-Max(MM) or Min-Sum(MS)")
    parser.add_argument("--speed_type", type=str, default="hom", choices=["hom", "het"], helo="hom: homogeneous speed / het: heterogeneous speed")
    parser.add_argument("--max_demand", type=int, default=9)
    parser.add_argument("--veh_sel", default="chr", choices=["chr", "alt"], help="chr: chronological strategy / alt: alternating strategy")

    # Actor parameters
    parser.add_argument("--n_heads", type=int, default=8, help="number of heads in MHA")
    parser.add_argument("--dim_embed", type=int, default=128, help="embedding dimension")
    parser.add_argument("--norm_n", type=str, default="batch", choices=["batch", "instance", "None"], help="Normalization in MHA in Node encoder")
    parser.add_argument("--norm_a", type=str, default="None", choices=["batch", "instance", "None"], help="Normalization in MHA in Fleet Encoder")

    # Baseline parameters
    parser.add_argument("--mv_beta", type=float, default=0.8, help="weight to calculate moving average(only used in initial epoch")
    parser.add_argument("--ttest_alpha", type=float, default=0.05, help="significance level in t-test")
    parser.add_argument("--ttest_samples", type=int, default=10240, help="sample size used in t-test")

    # Learning parameters
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--lr_decay", type=float, default=0.995, help="learning rate decay")
    parser.add_argument("--max_grad_norm", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--tanh_clipping", type=float, default=10.0)
    parser.add_argument("--end_epoch", type=int, default=50)
    parser.add_argument("--train_batch_size", type=int, default=512)
    parser.add_argument("--val_batch_size", type=int, default=512)
    parser.add_argument("--train_samples", type=int, default=1280000, help="number of training instances used per epoch")
    parser.add_argument("--val_samples", type=int, default=10240, help="number of training instances used for validation")

    # checkpoint(resume)
    parser.add_argument("--cp_path", type=str, default="", help="path to the parameter(.pt file) to resume from")
    parser.add_argument("--start_epoch", type=int, default=0, help="epoch to resume from")

    # misc
    parser.add_argument("--log_interval", type=int, default=-1, help="interval for logging (in batch num), default=five times per epoch")  # step(batch)
    parser.add_argument("--render_interval", type=int, default=10, help="interval to render validation result (in epoch)")  # epoch
    parser.add_argument("--cp_interval", type=int, default=1, help="interval to save parameters (in epoch)")  # epoch
    parser.add_argument("--port", type=str, default="49152", help="set different port to run multiple multi-GPU training on the same server")
    parser.add_argument("--no_gpu", action="store_true")

    args = parser.parse_args()

    # fixed settings
    if args.no_gpu:
        args.world_size = 1
    else:
        args.world_size = torch.cuda.device_count()

    # learning parameters
    assert args.train_samples % args.train_batch_size == 0, "train_samples is not divisible by batch_size"
    assert args.val_samples % args.val_batch_size == 0, "val_samples is not divisible by batch_size"
    args.train_batches = args.train_samples // args.train_batch_size
    if args.log_interval == -1:
        args.log_interval = int(args.train_batches / 5)

    args.title = 'V{}-C{}-{}'.format(args.n_agents, args.n_custs, args.target)
    if args.speed_type == "het":
        args.title += "-HS"  # heterogeneous speed

    args.speed = SPEED[args.speed_type][args.n_agents]
    args.load = MAX_LOAD[args.n_agents]

    return args
