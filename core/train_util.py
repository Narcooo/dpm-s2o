import torch
from core.runcontroler import get_env_local_rank

def load_pretrained_parameters(logger, cfgs, model):
    if get_env_local_rank() == 0:
        logger.info("Use Pretrained Model")
        logger.info("Model Load From -> {}".format(cfgs.model.pretrained_path))
    state_dict = torch.load(cfgs.model.pretrained_path, map_location=torch.device('cpu'))["state_dict"]
    a = {}
    for name, param in model.named_parameters():
        a[name] = param
    model.encoder.load_state_dict({k: v for k, v in state_dict.items()}, strict=True)

def load_resume_parameters(logger, cfgs, model, optimizer, scheduler, scaler):
    state_all = torch.load(cfgs.model.resume_path, map_location=torch.device('cpu'))
    state_dict = state_all["state_dict"]
    # print(state_dict.keys())
    init_epoch = state_all["epoch"] + 1
    a = {}
    for name, param in model.named_parameters():
        # print(name)
        a[name] = param

    if get_env_local_rank() == 0:
        logger.info(f"loading model from checkpoint: {cfgs.model.resume_path}...")
    if "optimizer" in state_all:
        optimizer_dict = state_all["optimizer"]
        optimizer.load_state_dict(optimizer_dict)
    if "scheduler" in state_all:
        scheduler_dict = state_all["scheduler"]
        scheduler.load_state_dict(scheduler_dict)
    if "scaler" in state_all:
        scaler_dict = state_all["scaler"]
        scaler.load_state_dict(scaler_dict)
    model.load_state_dict({k: v for k, v in state_dict.items()}, strict=True)
    return init_epoch