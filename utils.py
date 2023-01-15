import torch
import matplotlib.pyplot as plt
import os
import csv
import datetime
import wandb

from collections import OrderedDict


class Logger():
    def __init__(self, args, start_time, save_dir, img_dir):
        """
        Logger
        Args:
            start_time: (datetime.datetime)logの起点
            save_dir
            img_dir: renderの保存先
        """
        self.start_time = start_time
        self.img_dir = img_dir
        self.save_dir = save_dir

        # set csv log
        log_name = '{}.csv'.format(args.actor_type)
        self.log_path = os.path.join(save_dir, log_name)

        # argsの記録
        settings = self.make_settings(args)
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=settings.keys())
            writer.writeheader()
            writer.writerow(settings)

        # 学習進捗のヘッダー
        self.header_csv = ['epoch', 'step', "lr", 'sample', 'time', 'train/actor_reward', 'train/actor_loss', 'val/actor_reward']
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.header_csv)
            writer.writeheader()

        # set wandb
        self.no_wandb = args.no_wandb
        if not self.no_wandb:
            wandb.init(
                project=f"HCVRP({args.title})",
                config=settings,
                name=settings["save dir"]
            )

        # logs
        self.logs = {}
        self.total_steps = args.end_epoch * args.train_batches
        self.start_step = args.start_epoch * args.train_batches

    def make_settings(self, args):
        """
        argsから記録したい情報をまとめる
        """
        settings = OrderedDict([
            ("title", args.title),
            ("save dir", os.path.basename(self.save_dir)),
            ("major actor type", args.actor_type.split(".")[0]),
            ("actor type", args.actor_type),
            ("train batch size", args.train_batch_size),
            ("num heads", args.n_heads),
            ("num layers(node)", args.n_layers_n),
            ("num layers(agent)", args.n_layers_a),
            ("normalization(node)", args.norm_n),
            ("normalization(agent)", args.norm_a),
            ("learning rate", args.lr),
            ("lr decay", args.lr_decay),
            ("max grad norm", args.max_grad_norm),
            ("dropout", args.dropout),
            ("end epochs", args.end_epoch),
            ("train samples", args.train_samples),
            ("n foresight", args.n_foresight)
        ])
        return settings

    def add_vals(self, epoch, step, lr, sample, train_reward, actor_loss, val_reward):
        self.logs["epoch"] = epoch
        self.logs["step"] = step
        self.logs["lr"] = lr
        self.logs["sample"] = sample
        self.logs["train/actor_reward"] = train_reward
        self.logs["train/actor_loss"] = actor_loss
        self.logs["val/actor_reward"] = val_reward

    def calc_hms(self, sec):
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return "{:.0f}:{:.0f}:{:.0f}".format(h, m, s)

    def record_time(self):
        elapsed_seconds = (datetime.datetime.now() - self.start_time).total_seconds()
        required_seconds = elapsed_seconds *  (self.total_steps - self.logs["step"]) / (self.logs["step"] - self.start_step)
        self.logs["time"] = self.calc_hms(elapsed_seconds)
        self.required_time = self.calc_hms(required_seconds)

    def write_console(self):
        print('\n===== Epoch: {}, Step: {}, Lr: {:.4e}, Time: {}, (Required Time: {}) ====='.format(self.logs["epoch"], self.logs["step"], self.logs["lr"], self.logs["time"], self.required_time))
        for key in ['train/actor_reward', 'train/actor_loss', 'val/actor_reward']:
            print(f"{key}:\t{round(self.logs[key], 3)} ")

    def write_wandb(self):
        if not self.no_wandb:
            header_wandb = ["epoch", "step", "lr", "sample", 'train/actor_reward', 'train/actor_loss', 'val/actor_reward']
            wandb.log({key: self.logs[key] for key in header_wandb})

    def write_csv(self):
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.header_csv)
            writer.writerow(self.logs)

    def output(self):
        self.record_time()
        self.write_console()
        self.write_csv()
        self.write_wandb()

    def render(self, location, tours, rewards, key_agents):
        """
        location : torch.tensor [B, n_nodes, 2], location of nodes
        tours: list [B, n_agent, n_visits], n_visitsはagentにより異なる
        rewards;: [B], tour_length
        epoch: int
        n_custs: int, customerの数
        save path
        """
        #color_list = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
        cmap = plt.get_cmap("hsv")
        n_agents = len(tours[0])
        _, axes = plt.subplots(nrows=3, ncols=3, sharex='col', sharey='row', figsize=(10, 10))
        axes = [a for ax in axes for a in ax]  # 3*3 or 1*1を　1次元配列に変換
        for episode, ax in enumerate(axes):
            # node
            ax.scatter(location[episode, 0, 0], location[episode, 0, 1], color="k", marker="*", s=200, zorder=2)  # depot
            ax.scatter(location[episode, 1:, 0], location[episode, 1:, 1], color="k", marker="o", zorder=2)  # customer

            # tour
            for agent_id, tour in enumerate(tours[episode]):
                tour_index = torch.tensor(tour).unsqueeze(1).expand(-1, 2)  # [n_visits]
                # input=[n_nodes, 2], dim=1, index=[n_visits, 2]
                tour_xy = torch.gather(location[episode], dim=0, index=tour_index)
                ax.plot(tour_xy[:, 0], tour_xy[:, 1], color=cmap(agent_id/n_agents), linestyle="-", linewidth=0.8, zorder=1, label=f"agent{agent_id}")
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
        plt.suptitle(f"Epoch: {self.logs['epoch']}", y=0)
        save_path = os.path.join(self.img_dir, f"Epoch_{self.logs['epoch']}.png")
        plt.savefig(save_path, facecolor="white", edgecolor="black")
        plt.close()
        return

    def close(self):
        if not self.no_wandb:
            wandb.finish()
        print('\n===== END ====')


def sort(feat, index):
    """
    featのdim_to_sort次元について、indexで示した列が最初に並ぶようにsortする
    Params:
        feat: [B, dim_feat, dim_to_sort], features to bo sorted
        index: [B, 1], index of next_agent
    Returns:
        sorted_feat: [B, dim_feat, dim_to_sort], sorted features
    """
    all_index = torch.arange(feat.size(2), device=feat.device).unsqueeze(0).expand(feat.size(0), -1)  # [B, dim_to_sort]
    other_index = all_index[all_index != index].reshape((index.size(0), feat.size(2) - 1))  # [B, dim_to_sort - 1]
    index = torch.cat([index, other_index], dim=1)  # [B, dim_to_sort]
    return torch.gather(feat, dim=2, index=index.unsqueeze(1).expand_as(feat))
