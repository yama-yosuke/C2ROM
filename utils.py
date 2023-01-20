import csv
import datetime


class Logger():
    def __init__(self, args, start_time, log_path):
        """
        Args:
            start_time (datetime.datetime): start time of training 
            log_path (str): path to log_file
        """
        self.start_time = start_time
        self.log_path = log_path

        # columns to save in log file
        self.header = ['epoch', 'step', "lr", 'trained_instances', 'time', 'train/actor_reward', 'train/actor_loss', 'val/actor_reward']
        with open(self.log_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.header)
            writer.writeheader()

        # logs
        self.logs = {}
        self.total_steps = args.end_epoch * args.train_batch_num
        self.start_step = args.start_epoch * args.train_batch_num

    def add_vals(self, epoch, step, lr, instance, train_reward, actor_loss, val_reward):
        self.logs["epoch"] = epoch
        self.logs["step"] = step
        self.logs["lr"] = lr
        self.logs["trained_instances"] = instance
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

    def write_csv(self):
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.header)
            writer.writerow(self.logs)

    def output(self):
        self.record_time()
        self.write_console()
        self.write_csv()
