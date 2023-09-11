import torch


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)

def cosine_beta_schedule_enhance(timesteps, s=0.0):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    t_min = torch.atan(torch.exp(torch.tensor(-0.5 * 15)))
    t_max = torch.atan(torch.exp(torch.tensor(-0.5 * -15)))
    l = -2 * torch.log(torch.tan(t_min + x[0:] / 1000 * (t_max - t_min)))
    lshift = l + 2 * torch.log(torch.tensor(64 / 256))
    beta = torch.sigmoid(-1 * lshift)
    # alpha = torch.sigmoid(lshift)
    return beta

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def sigmoid_beta_enhance_schedule(timesteps, tau=0.5, clip_min = 1e-4):
    steps = timesteps
    x = torch.linspace(timesteps, 1, steps) / 1000
    start = torch.tensor(0)
    end = torch.tensor(3)
    v_start = torch.sigmoid(start / torch.tensor(tau))
    v_end = torch.sigmoid(end / torch.tensor(tau))
    output = torch.sigmoid((x * (end - start) + start) / tau)
    output = (v_end - output) / (v_end - v_start)
    return torch.clip(output, clip_min, 0.9999)
