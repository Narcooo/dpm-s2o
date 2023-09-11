# Sample from the base model.
from .script_util import create_diffusion
import torch as th
# @th.inference_mode()
def sample(
        model,
        # glide_options,
        image_size,
        context,
        batch_size=1,
        channel=4,
        guidance_scale=2,
        device=None,
        use_context=False,
        use_label=True,
        y=None,
        y_null=46,
        prediction_respacing="250",
        upsample_enabled=False,
        upsample_temp=0.997,
        mode='',
):
    eval_diffusion = create_diffusion(
        timestep_respacing=prediction_respacing
    )
    if device is None:
        device = next(model.parameters()).device
    # Create the classifier-free guidance tokens (empty)

    def cfg_model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :channel], model_out[:, channel:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)

        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)

        eps = th.cat([half_eps, half_eps], dim=0)
        return th.cat([eps, rest], dim=1)

    model_fn = cfg_model_fn  # so we use CFG for the base model.

    if use_context:
        noise = th.randn((batch_size, channel, image_size, image_size), device=device)
        noise = th.cat([noise, noise], 0)
        label_null = th.tensor([y_null] * batch_size * 2, device=device)
        full_batch_size = batch_size * 2
        cond_ref = context[:batch_size]
        uncond_ref = th.ones_like(cond_ref)

        model_kwargs = {}
        model_kwargs['y'] = label_null
        model_kwargs['context'] = th.cat([cond_ref, uncond_ref], 0).to(device)
        samples = eval_diffusion.ddim_sample_loop(
            model_fn,
            (full_batch_size, channel, image_size, image_size),  # only thing that's changed
            noise=noise,
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
    elif use_label:
        noise = th.randn((batch_size, channel, image_size, image_size), device=device)
        noise = th.cat([noise, noise], 0)
        label_null = th.tensor([y_null] * batch_size, device=device)
        full_batch_size = batch_size * 2
        label = th.cat([y[:batch_size], label_null], 0)
        model_kwargs = {}
        model_kwargs['y'] = label
        samples = eval_diffusion.ddim_sample_loop(
            model_fn,
            (full_batch_size, channel, image_size, image_size),  # only thing that's changed
            noise=noise,
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
    else:
        model_kwargs = {}
        noise = th.randn((batch_size, channel, image_size, image_size), device=device)
        samples = eval_diffusion.ddim_sample_loop(
            model,
            (batch_size, channel, image_size, image_size),  # only thing that's changed
            noise=noise,
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]

    return samples