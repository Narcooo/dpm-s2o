import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR, ReduceLROnPlateau, _LRScheduler
import torch

from core import optimizers
from core import config
from torch.cuda.amp import GradScaler
from core.diffusion.script_util import create_diffusion


def Model(*args, **kwargs):
    import core.module as cm
    cfgs = args[0]
    assert cfgs.model.name in cm.__dict__, "make sure that the model.name in your config file {} is the right name and you already import it in the __init__.py".format(
        cfgs.model.name)
    modelclass = cm.__dict__[cfgs.model.name].__dict__[cfgs.model.name]
    model = modelclass(**kwargs)
    return model

def Dataloader(dataset, cfgs, sampler, mode='train'):
    data_str = mode + '_dataset'
    data_attr = getattr(cfgs.data, data_str)
    loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=data_attr.batch_size,
                                               shuffle=(sampler is None),
                                               pin_memory=data_attr.pin_memory,
                                               sampler=sampler,
                                               num_workers=data_attr.num_worker,
                                               drop_last=data_attr.drop_last
                                               )
    return loader

def Generator(*args, **kwargs):
    import core.diffusion as cd
    cfgs = args[0]
    assert cfgs.generator.name in cd.__dict__, "make sure that the generator.name in your config file {} is the right name and you already import it in the __init__.py".format(
        cfgs.generator.name)
    modelclass = cd.__dict__[cfgs.generator.name].__dict__[cfgs.generator.name]
    if cfgs.generator.name == 'SpacedDiffuser':
        model = create_diffusion(timestep_respacing="")
    else:
        model = modelclass(**kwargs)
    return model

def Dataset(cfg):
    import dataset
    assert cfg.data.name in dataset.__dict__, "make sure that the data.name in your config file {} is the right name and you already import it in the __init__.py".format(config.data.name)
    dataset_class = dataset.__dict__[cfg.data.name]
    # if hasattr(dataset_class, config.data.name)
    transform = getattr(dataset.transform, cfg.data.train_dataset.transform)
    dataset = dataset_class(root_path=cfg.data.root_path, img_prefix=cfg.data.images_path,
                            label_prefix=cfg.data.labels_path, transforms=transform)
    return dataset

def Optimizer(cfg, model_params):
    from core import optimizers
    assert hasattr(optimizers, cfg.training.optimizer.name), "make sure that the training.optimizer.name in your config file {} is the right name and you wrote it in optimizers.py"
    optim_class = optimizers.__dict__[cfg.training.optimizer.name]
    init_params = {}
    config.convert_to_config(cfg.training.optimizer.init_params, init_params)
    optimizer = optim_class(model_params, **init_params)
    return optimizer

def Loss(cfg, type):
    from core import losses
    types = {}
    config.convert_to_config(cfg.training.loss, types)
    assert type in types, "you can only choose these types"
    attr_type = getattr(cfg.training.loss, type)
    init_params = attr_type.init_params or {}
    loss_class = losses.__dict__[attr_type.name] or None
    return loss_class(**init_params)

def Scheduler(cfg, optimizer):
    from core import schedulers
    assert hasattr(schedulers,
                   cfg.training.scheduler.name), "make sure that the training.optimizer.name in your config file {} is the right name and you wrote it in optimizers.py"
    sche_class = schedulers.__dict__[cfg.training.scheduler.name]
    init_params = {}
    if cfg.training.scheduler.init_params is not None:
        config.convert_to_config(cfg.training.scheduler.init_params, init_params)
    scheduler = sche_class(optimizer, **init_params)
    return scheduler


def Metrics(cfg, type):

    from core import metrics
    types = {}
    config.convert_to_config(cfg.training.metrics, types)
    assert type in types, "you can only choose these types"
    attr_type = getattr(cfg.training.metrics, type)
    # init_params = attr_type.init_params or {}
    # if attr_type.name is not None:
    assert isinstance(attr_type.name, list), "the name of metrics must be a list"
    metrics_class = []
    metrics_class.extend([metrics.__dict__[names]() for names in attr_type.name])

    return metrics_class

def Scaler(cfgs):
    return GradScaler(init_scale=2. ** cfgs.init_scale, enabled=cfgs.use_fp16, growth_interval=1000)