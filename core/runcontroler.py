import os
import time
import torch
from collections.abc import Iterable
import numpy as np
import torch.distributed as dist
from typing import Any
from torch.cuda.amp import GradScaler
from collections import defaultdict
from typing import Dict, Optional
import copy
from core.module.step_sampler import UniformSampler


class RunControler:

    def __init__(
        self,
        *,
        # generator,
        train_loader=None,
        save_per_epochs=20,
        ema_rate=None,
        model=None,
        optimizer=None,
        scheduler=None,
        criterion=None,
        use_fp16=False,
        schedule_sampler=None,
        grad_scaler=None,
        accumulate_steps=1,
    ):
        # base
        self.train_loader = train_loader
        self.val_loader = None
        # self.generator = generator
        self.ema_rate = ema_rate
        self.model = model
        self.total_compute_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.total_params = sum(p.numel() for p in model.parameters())
        self.model_params = list(self.model.parameters()) if self.model.parameters is not None else None
        self.ema_params = [
            copy.deepcopy(self.model_params) for _ in range(len(self.ema_rate))
        ]
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        # make state aware of fp16 and scale. if use_fp16 is False, grad_scaler is NoOp
        self.use_fp16 = use_fp16
        self.grad_scaler = grad_scaler
        self.need_val = False
        # data pipeline
        self.input = None
        self.allloss = {}
        self.output = None
        self.outputbeforesigmoid = None
        # self.tb_logger = None
        # counters
        # self.schedulesampler = schedule_sampler or UniformSampler(generator)
        self.model_kwargs = {}
        self.num_classes = None
        self.num_epochs = 1
        self.epoch = 0
        self.train_loss = AvgMeter("train_loss")
        self.loss_meter = AvgMeter("loss")
        self.loss_meter_list = []
        self.train_metrics: Dict[str, AvgMeter] = defaultdict(AvgMeter())
        self.metric_meters: Dict[str, AvgMeter] = defaultdict(AvgMeter())
        self.val_loss: Optional[AvgMeter] = None
        self.val_metrics: Optional[Dict[str, AvgMeter]] = None
        self.is_train = True
        self.epoch_size = None
        self.batch_size = 0
        # number of steps performed. resets each epoch!
        self.step = None
        self.quartile = None
        self.denoising_step2 = None
        # total number of samples seen. usefull to log independentely of batch_size or world_size
        self.global_sample_step = 0
        self.total_sample_step = 0
        # number of steps to accumulate
        self.accumulate_steps = accumulate_steps
        # dict for communication between callbacks
        self.communication_dict = dict()
        self.tb_logger = None
        self.work_dir = None
        self.save_per_epochs = save_per_epochs
        self.cur_epoch = 0
        # for DDP
        self.rank = get_env_rank()
        self.world_size = get_env_world_size()
        self.data_target = None

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)

    @property
    def epoch_log(self):
        return self.epoch + 1

    def reduce_meters(self):
        """aggregate loss and metrics from all processes"""
        # meters = list(self.train_metrics.values()) + [self.train_loss]
        # meters = list(self.metric_meters.values()) + [self.loss_meter]
        meters = [self.loss_meter]
        # print(meters)
        # if self.val_loss is not None:
        #     meters = meters + list(self.val_metrics.values()) + [self.val_loss]
        for meter in meters:
            reduce_meter(meter)  # NoOp if world_size == 1

def get_env_rank():
    return int(os.environ.get("RANK", 0))

def get_env_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))

def get_env_world_size():
    return int(os.environ.get("WORLD_SIZE", 1))

def listify(p):
    if not isinstance(p, Iterable):
        p = [p]
    elif p is None:
        p = []
    return p

def to_float(x):
    if isinstance(x, float):
        return x
    elif isinstance(x, torch.Tensor):
        return x.item()
    elif isinstance(x, np.ndarray):
        return x.item()
    elif isinstance(x, (list, tuple)):
        return [to_float(v) for v in x]
    else:
        raise ValueError("Unsupported type")

def to_numpy(x):

    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple, int, float)):
        return np.array(x)
    else:
        raise ValueError("Unsupported type")


def to_tensor(x: Any, dtype=None) -> torch.Tensor:

    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    elif isinstance(x, (list, tuple)):
        x = np.array(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    else:
        raise ValueError("Unsupported input type" + str(type(x)))


class AvgMeter:
    """Computes and stores the average and current value
    Attributes:
        val - last value
        avg - true average
        avg_smooth - smoothed average"""

    def __init__(self, name="Meter", avg_mom=0.95):
        self.avg_mom = avg_mom
        self.name = name
        self.sum = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.avg_smooth = 0
        self.count = 0

    def resum(self):
        self.sum = 0

    def update(self, val):
        self.val = val
        if self.count == 0:
            self.avg_smooth = self.avg_smooth + val
        else:
            self.avg_smooth = self.avg_smooth * self.avg_mom + val * (1 - self.avg_mom)
        self.count += 1
        self.avg *= (self.count - 1) / self.count
        self.avg += val / self.count

    def add(self, val):
        self.sum += val

    def __call__(self, val):
        return self.update(val)

    def __repr__(self):
        return f"AvgMeter(name={self.name}, avg={self.avg:.3f}, count={self.count})"
        # return f"{self.name}: {self.avg:.3f}" # maybe use this version for easier printing?

class AvgMeters(AvgMeter):

    def __init__(self, avgmeters):
        super().__init__()
        self.avgmeters = listify(avgmeters)


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


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.batch_time = AvgMeter()
        self.data_time = AvgMeter()
        self.start = time.time()
        self.epoch_time = AvgMeter()

    def batch_start(self):
        self.data_time.update(time.time() - self.start)
        self.data_time.add(time.time() - self.start)

    def batch_end(self):
        self.batch_time.update(time.time() - self.start)
        self.batch_time.add(time.time() - self.start)
        self.start = time.time()

    def epoch_start(self):
        self.epoch_start_time = time.time()

    def epoch_end(self):
        self.epoch_time.update(time.time() - self.epoch_start_time)


def reduce_meter(meter: AvgMeter) -> AvgMeter:
    """Args: meter (AvgMeter): meter to reduce"""
    if get_env_world_size() == 1:
        return meter
    # can't reduce AvgMeter so need to reduce every attribute separately
    reduce_attributes = ["val", "avg", "avg_smooth", "count"]
    for attr in reduce_attributes:
        old_value = to_tensor([getattr(meter, attr)]).float().cuda()
        setattr(meter, attr, reduce_tensor(old_value).cpu().numpy()[0])

def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return sum_tensor(tensor) / get_env_world_size()


def sum_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt

def update_ema(target_params, source_params, rate=0.99):

    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)