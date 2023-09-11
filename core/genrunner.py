import copy
import os
import numpy as np
import torch
import random
from core.diffusion.Consistencer import Consistencer
import wandb
from torch.cuda import amp
from torchvision.utils import save_image
from .runcontroler import RunControler
from .callbacks import Callbacks
from .runcontroler import to_numpy
from .runcontroler import get_env_local_rank, get_env_rank, update_ema
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from core.diffusion.sampler import sample
from diffusers.models import AutoencoderKL
from tools.load_pretrained import fix_vae_state_dict
# torch.autograd.set_detect_anomaly(True)


class GenerativeRunner:

    def __init__(
        self,
        generator,
        model,
        device,
        is_distributed,
        train_sampler,
        optimizer,
        scheduler,
        cfgs,
        criterion,
        callbacks,
        use_ddim,
        work_dir,
        grad_scaler,
        use_wandb=False,
        use_context=False,
        save_per_epochs=50,
        accumulate_steps=1, #waiting for running accumulate_steps * batchsize batches update gradient
        use_fp16=False,
        grad_norm_clip=0,
        ema_rate=0.9999,

    ):
        super().__init__()
        self.controler = RunControler(
            model=model, optimizer=optimizer,
            scheduler=scheduler, criterion=criterion,
            use_fp16=use_fp16, accumulate_steps=accumulate_steps,
            save_per_epochs=save_per_epochs, grad_scaler=grad_scaler,
            ema_rate=([ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")])
        )
        self.use_wandb = use_wandb
        self.use_label = True
        self.use_context = use_context
        self.callbacks = Callbacks(callbacks)
        self.callbacks.set_controler(self.controler)
        # self.generator = generator
        # self.cgen = Consistencer()
        self.model = model
        self.generator = generator
        self.is_distributed = is_distributed
        self.train_sampler = train_sampler
        self.grad_norm_clip = grad_norm_clip
        self.device = device
        self.use_ddim = use_ddim
        self.controler.work_dir = work_dir
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.controler.num_classes = cfgs.num_classes if cfgs.generator.use_class_guide else None
        self.criterion = nn.MSELoss()
        # self.vae = AutoencoderKL(latent_channels=4, layers_per_block=2, block_out_channels=(128, 256, 512, 512),
        #             up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        #             down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        #             norm_num_groups=32, scaling_factor=0.18215, sample_size=256, in_channels=3, out_channels=3).to("cuda")
        #
        # state_dict = torch.load("models/vae-ft-mse-840000-ema-pruned.ckpt")
        # sd = state_dict["state_dict"]
        # sd = fix_vae_state_dict(sd)
        # self.vae.load_state_dict(state_dict=sd, strict=True)
        # self.vae.eval()
        # self.cur_step = 0

    def fit(
        self,
        train_loader,
        image_size,
        batches,
        channels,
        steps_per_epoch=None,
        val_loader=None,
        val_steps=None,
        check_val_every_n_epoch=1,
        check_gen_every_n_epoch=1,
        epochs=1,
        start_epoch=0,
    ):
        self.controler.num_epochs = epochs
        self.controler.train_loader = train_loader
        self.controler.val_loader = val_loader if val_loader is not None else None
        self.controler.total_sample_step = len(train_loader) * epochs
        self.controler.batch_size = getattr(train_loader, "batch_size", 1) if val_loader is None else getattr(
            val_loader, "batch_size", 1)
        self.callbacks.on_begin()
        # print(self.controler.model.named_parameters)
        # for name, param in self.controler.model.named_parameters():
        #     print(f"{name}: contiguous={param.grad.is_contiguous()}")

        if self.use_wandb and get_env_rank() == 0:
            wandb.watch(self.controler.model, log="all", log_freq=500)
        for epoch in range(start_epoch, epochs):
            self.controler.cur_epoch += 1
            self.controler.epoch = epoch
            if self.is_distributed:
                self.train_sampler.set_epoch(self.controler.epoch)
            self.controler.is_train = True

            self.callbacks.on_epoch_begin()
            self.controler.model.train()
            self._running_epoch(train_loader, steps=steps_per_epoch)

            # need_val = ((epoch + 1) % check_val_every_n_epoch) == 0
            need_gen = ((epoch + 1) % check_gen_every_n_epoch) == 0
            has_val_loader = val_loader is not None
            # if has_val_loader and need_val:
            #     self.evaluate(val_loader, steps=val_steps)
            if need_gen:
                self.generate(image_size=image_size, channels=channels, work_dir=self.controler.work_dir, batches=batches, npernode=torch.cuda.device_count())
            self.controler.reduce_meters()
            # self.controler.need_val = ((epoch + 1) % check_val_every_n_epoch) == 0
            # if self.controler.need_val:
            #     self.generate(image_size=image_size, channels=channels, work_dir= self.controler.work_dir, batches=batches, npernode=torch.cuda.device_count())
            self.callbacks.on_epoch_end()
        self.callbacks.on_end()

    def evaluate(self, loader, steps=None):
        self.controler.is_train = False
        self.controler.model.eval()
        self._running_epoch(loader, steps=steps)
        self.controler.reduce_meters()
        return self.controler.loss_meter.avg, [m.avg for m in self.controler.metric_meters.values()]

    def num_to_groups(self, num, divisor):
        groups = num // divisor
        remainder = num % divisor
        arr = [divisor] * groups
        if remainder > 0:
            arr.append(remainder)
        return arr

    def generate(self, image_size, channels, work_dir, batches, npernode=2):
        # if self.controler.is_train:
        #     self.controler.loss_meter.reset()
        self.controler.data_target = []
        for metric in self.controler.metric_meters.values():
            metric.reset()
        # self.controler.epoch_size = steps or len(loader)  # steps overwrites len
        self.controler.is_train = False
        self.controler.model.eval()
        self.callbacks.on_loader_begin()


        def split_list(lst, n):
            avg_len = len(lst) // n
            remain = len(lst) % n
            result = []
            start = 0

            for i in range(n):
                end = start + avg_len
                if remain > 0:
                    end += 1
                    remain -= 1
                result.append(lst[start:end])
                start = end

            return result

        # val_data = self.controler.train_loader if self.controler.val_loader is None else self.controler.val_loader
        train_data = self.controler.train_loader
        # val_data.sampler.set_epoch(self.controler.epoch)
        model_kwargs = {}
        images_list = []
        os.makedirs(os.path.join(work_dir, 'samples'), exist_ok=True)
        os.makedirs(os.path.join(work_dir, 'oris'), exist_ok=True)
        with amp.autocast(self.controler.use_fp16):
            for idx, (data, target) in enumerate(train_data):
                model_kwargs['context'] = data.to(self.device) if self.use_context else None
                model_kwargs['y'] = target.to(self.device) if self.use_label else None
                images_list.append((list(
                    map(lambda n: sample(model=self.controler.model, image_size=image_size, channel=data.shape[1], batch_size=n,
                                         **model_kwargs, use_label=self.use_label, use_context=self.use_context), [self.controler.batch_size]))))
                if self.is_distributed:
                    torch.distributed.barrier()
                if idx >= batches - 1:
                    break

        for i, (sample_images_list) in enumerate(images_list):
            sample_images = sample_images_list[0]
            decoded_images_list = []

            for i in range(sample_images.shape[0]):
                single_image = sample_images[i:i + 1]
                # single_image = self.vae.decode(single_image / 0.23908).sample
                decoded_images_list.append(single_image)

            sample_images = torch.cat(decoded_images_list, dim=0)
            # sample_images = self.vae.decode(sample_images / 0.18215).sample
            # torch.cat()
            sample_images = (sample_images + 1) * 0.5
            save_image(sample_images,
                       str(work_dir + f'/samples/sample-e{self.controler.epoch_log}-b{i}-g{get_env_rank()}.png'),
                       nrow=int(self.controler.batch_size))
        self.callbacks.on_loader_end()

    def _running_step(self):
        data, target = self.controler.input
        data, target = data.to(self.device), target.to(self.device)
        # print(torch.mean(data))
        pred = None
        context = None

        if self.controler.is_train:
        # noise = torch.randn_like(data)
            if self.controler.step % self.controler.accumulate_steps == 0:
                self.controler.optimizer.zero_grad()
            if get_env_local_rank() == 0:
                self.controler.global_sample_step += 1

        t = torch.randint(0, self.generator.num_timesteps, (data.shape[0],), device=self.device).long()
        with amp.autocast(self.controler.use_fp16, dtype=torch.float16):
            # self.controler.model_kwargs['y'] = target
            self.controler.model_kwargs['context'] = data if self.use_context else None
            self.controler.model_kwargs['y'] = target if self.use_label else None
            # self.controler.model_kwargs['target'] = target
            output = self.generator.p_losses(self.controler.model, data, t, model_kwargs=self.controler.model_kwargs)
            loss = output['loss'].mean()
            if "context" in output:
                if output["context"] is not None:
                    context = output["context"]
                    output["context"] = context.mean()
                    output["c_max"] = context.max()
                else:
                    del output["context"]


            if "model_output" in output:
                pred = output["model_output"]
                output["model_output"] = pred.mean()
                output["o_max"] = pred.max()

            if "recon" in output:
                recon = output["recon"]
                output["recon"] = recon.mean()
                output["r_max"] = recon.max()

            self.controler.quartile = int(4 * torch.mean(t.float(), dim=0) / self.generator.num_timesteps)
        self.controler.allloss = output


        if self.controler.is_train:
            # with torch.autograd.detect_anomaly():
            #     self.controler.grad_scaler.scale(loss / self.controler.accumulate_steps).backward()
            self.controler.grad_scaler.scale(loss / self.controler.accumulate_steps).backward()
            # for name, param in self.controler.model.named_parameters():
            #     if param.grad is None:
            #         print(name)

            self._clip_grad_norm()
            self.callbacks.on_after_backward()

            if self.controler.step % self.controler.accumulate_steps == 0:
                self.controler.grad_scaler.step(self.controler.optimizer)
                self.controler.grad_scaler.update()
                if self.controler.scheduler is not None:
                    self.controler.scheduler.step()
            for rate, params in zip(self.ema_rate, self.controler.ema_params):
                update_ema(params, self.controler.model_params, rate=rate)
            # torch.cuda.synchronize()
            if self.use_wandb and get_env_local_rank() == 0:
                wandb.log({'loss': loss})
                wandb.log({'lr': self.controler.optimizer.state_dict()['param_groups'][0]['lr']})
                wandb.log({"step": self.controler.global_sample_step})
                if pred is not None:
                    wandb.log({
                        "pred_mean": pred.mean().item(),
                        "pred_std": pred.std().item(),
                        "pred_max": pred.max().item(),
                        "pred_min": pred.min().item()
                    })
                    # wandb.log({
                    #     "mean_before": self.controler.outputbeforesigmoid.mean().item(),
                    #     "std_before": self.controler.outputbeforesigmoid.std().item(),
                    #     "max_before": self.controler.outputbeforesigmoid.max().item(),
                    #     "min_before": self.controler.outputbeforesigmoid.min().item()
                    # })
                    wandb.log({"model_output_hist": wandb.Histogram(pred.cpu().detach().numpy())})

                if context is not None:
                    wandb.log({"context_hist": wandb.Histogram(context.cpu().detach().numpy())})
                    wandb.log({
                        "context_mean": context.mean().item(),
                        "context_std": context.std().item(),
                        "context_max": context.max().item(),
                        "context_min": context.min().item()
                    })
                    wandb.log({"context_hist": wandb.Histogram(context.cpu().detach().numpy())})

        # Update loss
        # print(self.controler.loss_meter)
        # print(loss)
        # print(self.controler.metric_meters)
            self.controler.loss_meter.update(to_numpy(loss))
        # self.controler.metric_meters[name].update(to_numpy(metric(output, target).squeeze()))
        # Metrics are now updated inside callbacks

    def _running_epoch(self, loader, steps=None):
        if self.controler.is_train:
            self.controler.loss_meter.reset()
        self.controler.data_target = []
        for metric in self.controler.metric_meters.values():
            metric.reset()
        self.controler.epoch_size = steps or len(loader)  # steps overwrites len
        self.callbacks.on_loader_begin()
        with torch.set_grad_enabled(self.controler.is_train):
            for i, (data, target) in enumerate(loader):

                self.controler.step = i
                # self.controler.global_sample_step += self.controler.batch_size * self.controler.world_size
                self.controler.input = [data, target]
                self.callbacks.on_batch_begin()
                self._running_step()
                self.callbacks.on_batch_end()

        self.callbacks.on_loader_end()
        return


    def _clip_grad_norm(self):
        params = [
            p for p in self.controler.model_params if p.grad is not None
        ]
        if self.grad_norm_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, self.grad_norm_clip)

    def sample(self, image_size, channels, work_dir, val_loader):
        self.controler.is_train = False
        self.controler.model.eval()

        resize_transform = transforms.Resize((256, 256))
        model_kwargs = {}
        # os.makedirs(os.path.join(work_dir, 'targets'), exist_ok=True)
        img_name_list = []
        os.makedirs(os.path.join(work_dir, 'evals_ema8'), exist_ok=True)
        with amp.autocast(self.controler.use_fp16):
            for idx, (data, target, img_names) in enumerate(val_loader):
                model_kwargs['context'] = data

                img_name_list.append((self.generator.final_sample(model=self.controler.model, image_size=image_size, batch_size=data.shape[0],
                                                        use_ddim=self.use_ddim,
                                                        channels=3,
                                                        model_kwargs=model_kwargs), img_names))
                if self.is_distributed:
                    torch.distributed.barrier()
                if get_env_rank() == 0:
                    for i, (sample_images, img_names) in enumerate(img_name_list):
                        # print(len(sample_images))
                        # print(sample_images.shape)
                        for img, img_name in zip(sample_images, img_names):
                            # print(img.shape)
                            img = img.unsqueeze(0)
                            img = (img + 1) * 0.5
                            save_image(img,
                                       str(work_dir + f"/evals_ema8/{img_name}.png"))
                            # img = resize_transform(img)
                            # img = (img + 1) * 127.5
                            # img = img.byte()
                            # img = transforms.ToPILImage()(img)
                            # img.save(work_dir + f"/evals_ema8/{img_name}.png")


    def eval(
        self,
        val_loader,
        image_size,
        channels,
    ):

        self.sample(image_size=image_size, channels=channels, val_loader=val_loader, work_dir= self.controler.work_dir)


def forward_hook(module, input, output):
    if torch.isnan(output).any():
        print(f'NaN detected in forward pass in module {module}')

def backward_hook(module, grad_input, grad_output):
    for grad in grad_output:
        if grad is not None and torch.isnan(grad).any():
            print(f'NaN detected in backward pass in module {module}')
    return grad_output


