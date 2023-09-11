import os
import math
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from core import logger
from tensorboardX import SummaryWriter
from copy import deepcopy
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from collections import OrderedDict
from collections import defaultdict
from collections.abc import Iterable
from core.runcontroler import listify, to_numpy, get_env_rank, AvgMeter, Timer, RunControler, get_env_local_rank
import torch
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import wandb
import torchinfo



class Callback(object):
    """
    begin
    ---epoch begin (one epoch - one run of every loader)
    ------loader begin
    ---------batch begin
    ---------on_after_backward
    ---------batch end
    ------loader end
    ---epoch end
    end
    """
    def __init__(self, *args, **kwargs):
        self.controler = None

    def set_controler(self, controler):
        self.controler = controler

    def on_batch_begin(self):
        pass

    def on_batch_end(self):
        pass

    def on_loader_begin(self):
        pass

    def on_loader_end(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

    def on_begin(self):
        pass

    def on_end(self):
        pass

    def on_after_backward(self):
        pass

class Callbacks(Callback):

    def __init__(self, callbacks):
        super().__init__()
        self.callbacks = listify(callbacks)

    def set_controler(self, controler):
        for callback in self.callbacks:
            callback.set_controler(controler)

    def on_batch_begin(self):
        for callback in self.callbacks:
            callback.on_batch_begin()

    def on_batch_end(self):
        for callback in self.callbacks:
            callback.on_batch_end()

    def on_loader_begin(self):
        for callback in self.callbacks:
            callback.on_loader_begin()

    def on_loader_end(self):
        for callback in self.callbacks:
            callback.on_loader_end()

    def on_epoch_begin(self):
        for callback in self.callbacks:
            callback.on_epoch_begin()

    def on_epoch_end(self):
        for callback in self.callbacks:
            callback.on_epoch_end()

    def on_begin(self):
        for callback in self.callbacks:
            callback.on_begin()

    def on_end(self):
        for callback in self.callbacks:
            callback.on_end()

    def on_after_backward(self):
        for callback in self.callbacks:
            callback.on_after_backward()


def rank_zero_only(cls: Callback) -> Callback:
    is_rank_zero = get_env_rank() == 0
    if isinstance(cls, type):  # cls is a class
        return cls if is_rank_zero else Callback
    else:  # cls is an instance of some Callback
        return cls if is_rank_zero else Callback()

@rank_zero_only
class Callbacker(Callback):
    """Prints training progress to console for monitoring."""
    def __init__(self,
                 logger,
                 tb_log_dir,
                 logging_step,
                 metrics,
                 model_save_dir,
                 model_save_include_monitor=True,
                 model_save_name="model_{ep}_{metric_name}_{metric:.4f}.pth",
                 monitor="mAP",
                 mode="min",
                 include_optimizer=True,
                 verbose=True,
                 use_wandb=False,
                 using_batch_metric=True,
                 using_tensorboard_logger=False,
                 save_tb_confusion_matrix=False,
                 has_printed_time=True):
        super(Callback, self).__init__()
        self.metrics = listify(metrics)
        self.metric_names = [m.name for m in self.metrics]
        self.using_batch_metric = using_batch_metric
        self.using_tensorboard_logger = using_tensorboard_logger
        if tb_log_dir is not None:
            self.using_tensorboard_logger = True
        self.save_tb_confusion_matrix = save_tb_confusion_matrix
        self.epoch_target = []
        self.epoch_output = []
        self.logger = logger
        self.use_wandb = use_wandb
        self.model_save_dir = model_save_dir
        self.model_save_name = model_save_name
        self.include_optimizer = model_save_include_monitor
        self.monitor = monitor
        self.logging_step = logging_step

        self.tb_log_dir = tb_log_dir
        self.has_printed_time = has_printed_time
        self.timer = Timer()
        if mode == "min":
            self.best = np.inf
            self.monitor_op = np.less
        elif mode == "max":
            self.best = -np.inf
            self.monitor_op = np.greater
        self.include_optimizer = include_optimizer
        self.verbose = verbose
        self.grad_norm = 0
        self.p_norm = 0

    def on_begin(self):
        for name in self.metric_names:
            self.controler.metric_meters[name] = AvgMeter(name=name)
        if self.using_tensorboard_logger and self.tb_log_dir is not None:
            os.makedirs(self.tb_log_dir, exist_ok=True)
            self.controler.tb_logger = SummaryWriter(self.tb_log_dir)
        os.makedirs(self.model_save_dir, exist_ok=True)
        self.logger.info(f"Compute Model Parameters {self.controler.total_compute_params}")
        self.logger.info(f"Total Model Parameters {self.controler.total_params}")

    def on_loader_begin(self):
        self.epoch_target = []
        self.epoch_output = []
        # self.controler.data_target = []
        self.timer.reset()
        self.timer.epoch_start()
        # if hasattr(tqdm, "_instances"):  # prevents many printing issues
        #     tqdm._instances.clear()
        stage_str = "Train" if self.controler.is_train else "Validate"
        desc = f"Epoch [{self.controler.epoch_log:2d}/{self.controler.num_epochs}] - {stage_str}"
        # self.pbar = tqdm(total=self.controler.epoch_size, desc=desc, ncols=0)
        self.logger.info(desc)

    def on_batch_begin(self):
        self.timer.batch_start()

    def on_after_backward(self):
        self.grad_norm, self.p_norm = self._log_grad_norm()

    def on_batch_end(self):

        if self.controler.output is not None:
            # self.epoch_output.append(self.controler.output.cpu().detach())
            if self.using_batch_metric and self.controler.is_train:
                _, target = deepcopy(self.controler.input)
                output = self.controler.output.cpu().detach()
                with amp.autocast(self.controler.use_fp16):
                    for metric, name in zip(self.metrics, self.metric_names):
                        self.controler.metric_meters[name].update(to_numpy(metric(output, target).squeeze()))
        self.timer.batch_end()

        desc = OrderedDict({name: f"{to_numpy(m.mean()):.6f}" for (name, m) in self.controler.allloss.items()}) \
            if self.controler.allloss is not None else OrderedDict()
        # desc = OrderedDict({"Loss": f"{self.controler.loss_meter.val:.4f}"})
        desc['loss_meter'] = f"{self.controler.loss_meter.val:.4f}"
        # desc['mse'] =
        if self.controler.quartile is not None:
            desc['quantile'] = self.controler.quartile
        if self.controler.metric_meters is not None:
            for name, metric in self.controler.metric_meters.items():
                desc[name] = f"{metric.val:.4f}"

        # desc["t1"] = self.controler.denoising_step
        # if self.controler.denoising_step2 is not None:
        #     desc["t2"] = self.controler.denoising_step2
        desc["lr"] = self.controler.optimizer.state_dict()['param_groups'][0]['lr']

        # desc["output"] = self.controler.output
        # desc["output_before"] = self.controler.outputbeforesigmoid

        if self.controler.scheduler is not None:
            desc["lr"] = self.controler.scheduler.get_last_lr()[0]
        # desc["Memory"] = pynvml.nvmlDevice
        if self.controler.is_train:
            if self.use_wandb:
                wandb.log({"grad_norm": self.grad_norm})
                wandb.log({"p_norm": self.p_norm})
                for name, metric in self.controler.metric_meters.items():
                    # wandb.log({name: round(metric.val.item(), 4)})
                    wandb.log({name: round(metric.val, 4)})
        if self.controler.is_train and (self.controler.step % self.logging_step== 0):
            if self.using_tensorboard_logger and self.tb_log_dir is not None:
                self.controler.tb_logger.add_scalar(
                    "train_/loss", self.controler.loss_meter.avg_smooth, self.controler.global_sample_step
                )
                for name, metric in self.controler.metric_meters.items():
                    self.controler.tb_logger.add_scalar(f"train_/{name}", metric.avg_smooth, self.controler.global_sample_step)

            # console print
            log = ''
            title = f"Epoch [{self.controler.epoch_log:2d}/{self.controler.num_epochs}] - Step [{self.controler.step:4d}/{self.controler.epoch_size}]   "
            log += title
            for name, num in desc.items():
                log += f"{name}: {num:}, "
            log += f"grad_norm: {self.grad_norm:.6f}, "

            if self.has_printed_time:
                d_time = self.timer.data_time.sum
                b_time = self.timer.batch_time.sum
                timer = f"Data Time: {d_time:.3f}s - Model Time: {b_time:.3f}s "
                log += timer
            self.timer.data_time.resum()
            self.timer.batch_time.resum()
            self.logger.info(log)

        # self.pbar.set_postfix(**desc)
        # self.pbar.update()

    @torch.no_grad()
    def on_loader_end(self):
        if self.controler.is_train:
            for metric in self.controler.metric_meters.values():
                metric.reset()
        if len(self.controler.data_target) > 0:
            for output, target in self.controler.data_target:
                self.epoch_output.append(output)
                self.epoch_target.append(target)
            # self.controler.data_target = []
            self.epoch_target = torch.cat(self.epoch_target)
            self.epoch_output = torch.cat(self.epoch_output)
            self.controler.data_target = []
            # print(len(self.epoch_target), len(self.epoch_output))

            with amp.autocast(self.controler.use_fp16):
                for metric, name in zip(self.metrics, self.metric_names):
                    self.controler.metric_meters[name].update(to_numpy(metric(self.epoch_output, self.epoch_target).squeeze()))
        if not self.controler.is_train:
            if self.use_wandb:
                for name, metric in self.controler.metric_meters.items():
                    wandb.log({'val_' + name: round(metric.val.item(), 4)})
        self.timer.epoch_end()
        log = ''
        # update to avg
        title = f"Epoch [{self.controler.epoch_log:2d}/{self.controler.num_epochs}] - Finished   "
        log += title
        desc = OrderedDict({"Epoch_Average_Loss": f"{self.controler.loss_meter.avg:.4f}"}) if self.controler.is_train else OrderedDict()
        # desc["lr"] = self.controler.optimizer.param_groups[0]['lr']
        # desc["Memory"] = pynvml.nvmlDevice
        # loss = f"Epoch_Average_Loss: {desc['Loss']}  "
        # log += loss
        desc.update({name: f"{m.avg:.3f}" for (name, m) in self.controler.metric_meters.items()})
        for name, num in desc.items():
            log += f"{name}: {num}, "
        if self.has_printed_time:
            b_time = self.timer.epoch_time.avg_smooth
            timer = f"Total_Epoch_Time: {b_time:.3f}s "
            log += timer

        self.logger.info(log)

    def on_val_start(self):
        if self.controler.need_val:
            self.logger.info(f"Epoch [{self.controler.epoch_log:2d}/{self.controler.num_epochs}] - Start Eval")

    def on_epoch_end(self):
        ls = [name for name, p in self.controler.model.named_parameters() if p.grad is None] if self.controler.is_train else None
        if self.controler.is_train:
            self.logger.info("unused parameters: {}".format(ls))
        current = self.get_monitor_value()
        ep = self.controler.epoch_log
        if self.monitor_op(current, self.best):

            if self.verbose:
                self.logger.info(f"Epoch {ep:2d}: best {self.monitor} improved from {self.best:.4f} to {current:.4f}")
            self.best = current
        if self.monitor_op(current, self.best) or ep % self.controler.save_per_epochs == 0:
            save_name = os.path.join(self.model_save_dir, self.model_save_name.format(ep=ep, metric_name=self.monitor, metric=current))
            # if self.controler.cur_epoch % self.controler.save_per_epochs == 0:
            if self.controler.ema_rate is not None:
                ema_save_name = os.path.join(self.model_save_dir,
                                             'ema' + self.model_save_name.format(ep=ep, metric_name=self.monitor, metric=current))
                self._save_checkpoint(save_name, ema_save_name)
            else:
                self._save_checkpoint(save_name)
        if self.using_tensorboard_logger and self.tb_log_dir is not None:
            self.controler.tb_logger.add_scalar("train/loss", self.controler.train_loss.avg, self.controler.global_sample_step)
            for name, metric in self.controler.metric_meters.items():
                self.controler.tb_logger.add_scalar(f"train/{name}", metric.avg, self.controler.global_sample_step)

            lr = sorted([pg["lr"] for pg in self.controler.optimizer.param_groups])[-1]  # largest lr
            self.controler.tb_logger.add_scalar("train_/lr", lr, self.controler.global_sample_step)
            self.controler.tb_logger.add_scalar("train/epoch", self.controler.epoch, self.controler.global_sample_step)
            if self.controler.val_loss is None:
                return
            self.controler.tb_logger.add_scalar("val/loss", self.controler.val_loss.avg, self.controler.global_sample_step)
            for name, metric in self.controler.val_metrics.items():
                self.controler.tb_logger.add_scalar(f"val/{name}", metric.avg, self.controler.global_sample_step)

    def on_end(self):
        if self.using_tensorboard_logger and self.tb_log_dir is not None:
            self.controler.tb_logger.close()

    def _save_checkpoint(self, path, ema_path=None):
        if hasattr(self.controler.model, "module"):  # used for saving DDP models
            state_dict = self.controler.model.module.state_dict()
        else:
            state_dict = self.controler.model.state_dict()

        for i, (name, _value) in enumerate(self.controler.model.module.named_parameters() if hasattr(self.controler.model, "module")
                                           else self.controler.model.named_parameters()):
            assert name in state_dict, "name: {} not in state_dict"
            state_dict[name] = self.controler.model_params[i]
        if ema_path is not None:
            if hasattr(self.controler.model, "module"):  # used for saving DDP models
                ema_state_dict = self.controler.model.module.state_dict()
            else:
                ema_state_dict = self.controler.model.state_dict()
            for rate, ema_params in zip(self.controler.ema_rate, self.controler.ema_params):
                for i, (name, _value) in enumerate(self.controler.model.module.named_parameters() if hasattr(self.controler.model, "module")
                                           else self.controler.model.named_parameters()):
                    assert name in ema_state_dict, "name: {} not in state_dict"
                    ema_state_dict[name] = ema_params[i]
            ema_save_dict = {"epoch": self.controler.epoch, "state_dict": state_dict}
            if self.include_optimizer:
                ema_save_dict["optimizer"] = self.controler.optimizer.state_dict()
                ema_save_dict["scheduler"] = self.controler.scheduler.state_dict()
            if self.controler.use_fp16:
                ema_save_dict["scaler"] = self.controler.grad_scaler.state_dict()
            if get_env_local_rank() == 0:
                self.logger.info(f"saving ema model ...")

                torch.save(ema_save_dict, ema_path)
        save_dict = {"epoch": self.controler.epoch, "state_dict": state_dict}

        if self.include_optimizer:
            save_dict["optimizer"] = self.controler.optimizer.state_dict()
            save_dict["scheduler"] = self.controler.scheduler.state_dict()
        if self.controler.use_fp16:
            save_dict["scaler"] = self.controler.grad_scaler.state_dict()

        if get_env_local_rank() == 0:
            self.logger.info(f"saving model ...")

            torch.save(save_dict, path)
        # dist.barrier()

    def get_monitor_value(self):
        value = None
        if self.monitor == "loss":
            value = self.controler.loss_meter.avg
        else:
            for name, metric_meter in self.controler.metric_meters.items():
                if name == self.monitor:
                    value = metric_meter.avg
        if value is None:
            raise ValueError(f"CheckpointSaver can't find {self.monitor} value to monitor")
        return value

    def _log_grad_norm(self):
        sqsum = 0.0
        p_norm = 0.0
        for p in self.controler.model_params:
            with torch.no_grad():
                p_norm += torch.norm(p, p=2, dtype=torch.float32).item() ** 2
                if p.grad is not None:
                    sqsum += torch.norm(p.grad, p=2, dtype=torch.float32).item() ** 2
        return np.sqrt(sqsum), np.sqrt(p_norm)



















