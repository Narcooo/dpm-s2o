import math
import torch
from functools import partial
from torch import nn, einsum
from einops import rearrange, repeat
from utils import default, exists, torch_checkpoint
import utils as U
from abc import abstractmethod
import torch.nn.functional as F
import random
import numpy as np
from torchvision.models import vgg16
from flash_attn.modules.mha import FlashCrossAttention
from torch.cuda.amp import autocast


def trunc_normal_(tensor, mean=0., std=1.):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def pad_dims_like(x, target):
    while len(x.shape) < len(target.shape):
        x = x[..., None]
    return x

def drop_path(x, drop_prob: float = 0., scale_by_keep: bool = True):

    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.training and self.drop_prob > 0:
            x = drop_path(x, self.drop_prob, self.scale_by_keep)
        return x

def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-5, affine=True)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x.float()).type(x.dtype), **kwargs)


class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                 x, context = layer(x, context)
            else:
                x = layer(x)
        return x, context


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


class NearestUpsample(nn.Module):

    def __init__(self, channels, use_conv=False):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.conv = nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1) if use_conv else None

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')

        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):

    def __init__(self, channels, use_conv=False):
        super().__init__()
        self.use_conv = use_conv
        if self.use_conv:
            self.downsample = nn.Conv2d(channels,
                                  channels,
                                  kernel_size=3,
                                  stride=2,
                                  padding=1)
        else:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.downsample(x)
        return x


class Swish(nn.Module):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        h = x.type(torch.float32)
        h = h * torch.sigmoid(self.beta * h)
        x = h.type(x.dtype)
        return x


class SinusoidalPositionEmbeddings(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, timesteps):
        # device = time.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class ResnetBlock(TimestepBlock):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, in_channels, out_channels, *, dropout, up=False, down=False,
                 use_conv_shortcut=False, time_emb_dim=None, use_checkpoint=True,
                 ext_emb_dim=None):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        # self.use_conv_shortcut = use_conv_shortcut
        # self.use_scale_shift_norm = use_scale_shift_norm
        # self.ext_emb_dim = ext_emb_dim
        self.mlp = (
            nn.Sequential(Swish(), nn.Linear(time_emb_dim,
                                             2 * out_channels, ))
            if U.exists(time_emb_dim)
            else None
        )
        self.in_layers = nn.Sequential(
            Normalize(in_channels),
            Swish(),
            nn.Conv2d(in_channels, self.out_channels, 3, padding=1),
        )
        self.out_layers = nn.Sequential(
            Normalize(self.out_channels),
            Swish(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
            ),
        )
        # self.block1 = Block(in_channels, out_channels)
        # self.block2 = Block(out_channels, out_channels,
        #                     dropout=dropout, use_zero_module=True,
        #                     is_end=True)
        self.updown = up or down
        self.is_end = True

        if up:
            self.h_upd = NearestUpsample(in_channels, False)
            self.x_upd = NearestUpsample(in_channels, False)
        elif down:
            self.h_upd = Downsample(in_channels, False)
            self.x_upd = Downsample(in_channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        # self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        if self.out_channels == self.in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv_shortcut:
            self.skip_connection = nn.Conv2d(
                self.in_channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = nn.Conv2d(self.in_channels, self.out_channels, 1)

    def _forward(self, x, time_emb=None):

        # h = self.block1(x)
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        if U.exists(self.mlp) and U.exists(time_emb):
            time_emb = self.mlp(time_emb).type(x.dtype)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(time_emb, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
        else:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            h = out_norm(h)
        h = out_rest(h)
        return h + self.skip_connection(x)

    def forward(self, x, time_emb=None):
        return torch_checkpoint(self._forward, (x, time_emb), self.use_checkpoint)


class ConvNextBlock(nn.Module):

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if U.exists(time_emb_dim)
            else None
        )

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if U.exists(self.mlp) and U.exists(time_emb):
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")

        h = self.net(h)
        return h + self.res_conv(x)

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, encoder_kv=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        if encoder_kv is not None:
            assert encoder_kv.shape[1] == self.n_heads * ch * 2
            ek, ev = encoder_kv.reshape(bs * self.n_heads, ch * 2, -1).split(ch, dim=1)
            k = torch.cat([ek, k], dim=-1)
            v = torch.cat([ev, v], dim=-1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

class FlashAttention(nn.Module):
    def __init__(
        self,
        dim, heads=12, dim_head=64,
        batch_first=True,
        attention_dropout=0.0,
        causal=False,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        from einops import rearrange
        from flash_attn.modules.mha import FlashSelfAttention

        assert batch_first
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = dim
        self.num_heads = heads
        self.causal = causal
        self.inner_dim = dim_head * heads
        self.project_out = not (heads == 1 and dim_head == dim)
        assert (
            self.inner_dim % heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = dim_head
        assert self.head_dim in [16, 32, 48, 64], "Only support head_dim == 16, 32, 48 or 64"

        # self.inner_attn = FlashAttention(
        #     attention_dropout=attention_dropout, **factory_kwargs
        # )
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=True)
        self.inner_attn = FlashSelfAttention(
            attention_dropout=attention_dropout
        )
        self.to_out = zero_module(nn.Linear(self.inner_dim, dim) if self.project_out else nn.Identity())

        self.rearrange = rearrange

    def forward(self, qkv, attn_mask=None, key_padding_mask=None, need_weights=False):
        qkv = self.to_qkv(qkv)
        qkv = self.rearrange(
            qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads
        )
        qkv = self.inner_attn(
            qkv
        )
        return self.to_out(self.rearrange(qkv, "b s h d -> b s (h d)"))

class LegacyAttention(nn.Module):
    def __init__(self, *, dim, heads=12, dim_head=64, dropout=0., use_zero_module=False, qkv_bias=False):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=qkv_bias)
        self.to_out = nn.Linear(self.inner_dim, dim)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, dropout=0.):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.q_scale = False
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (heads c) h w -> b h c (h w)", heads=self.heads), qkv
        )
        k = k.softmax(dim=-1)
        if self.q_scale:
            q = q.softmax(dim=-2)
            q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w)
        out = self.to_out(out)
        return out


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class FCrossAttention(nn.Module):

    def __init__(self, *, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.project_out = not (heads == 1 and dim_head == query_dim)
        self.context_dim = context_dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        # self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, self.inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, query_dim) if self.project_out else nn.Identity(),
            nn.Dropout(dropout)
        )
        self.inner_attn = FlashCrossAttention()

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=h), (q, k, v))
        kv = torch.stack((k, v), dim=2)
        out = self.inner_attn(q, kv)
        out = rearrange(out, 'b n h d -> b n (h d)', h=h).contiguous()
        return self.to_out(out)


class CrossAttention(nn.Module):

    def __init__(self, *, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.project_out = not (heads == 1 and dim_head == query_dim)
        self.context_dim = context_dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, self.inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, query_dim) if self.project_out else nn.Identity(),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(dots.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            dots.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = self.attend(dots)
        # attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, context_dim=640, skip=False,
                 dropout=0., use_geglu=False, is_end=False, use_flash_attn=True, use_checkpoint=False):
        super(Transformer, self).__init__()
        self.checkpoint = use_checkpoint

        self.attn1 = LegacyAttention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout) if not use_flash_attn else \
            FlashAttention(dim=dim, heads=heads, dim_head=dim_head, attention_dropout=dropout)
        self.attn2 = LegacyAttention(dim=dim, heads=heads, dim_head=dim_head,
                                     dropout=dropout) if not use_flash_attn else \
            FlashAttention(dim=dim, heads=heads, dim_head=dim_head, attention_dropout=dropout)
        self.ff1 = FeedForward(dim, dropout=dropout, glu=use_geglu)
        self.ff2 = FeedForward(dim, dropout=dropout, glu=use_geglu)

        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        if context_dim is not None:
            self.attn3 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                                heads=heads, dim_head=dim_head,
                                                dropout=dropout) if not use_flash_attn else \
                    FCrossAttention(query_dim=dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout)
            self.ff3 = FeedForward(dim, dropout=dropout, glu=use_geglu)
            self.norm5 = nn.LayerNorm(dim)
            self.norm6 = nn.LayerNorm(dim)
            if not is_end:
                self.attn4 = LegacyAttention(dim=context_dim, heads=heads, dim_head=dim_head,
                                             dropout=dropout) if not use_flash_attn else \
                    FlashAttention(dim=context_dim, heads=heads, dim_head=dim_head, attention_dropout=dropout)

                self.ff4 = FeedForward(context_dim, dropout=dropout, glu=use_geglu)
                self.norm7 = nn.LayerNorm(context_dim)
                self.norm8 = nn.LayerNorm(context_dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)

        self.is_end = is_end

    def forward(self, x, skip=None, context=None):
        return torch_checkpoint(self._forward, (x, skip, context), self.checkpoint)

    def _forward(self, x, skip=None, context=None):

        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = self.attn1(self.norm1(x)) + x
        x = self.ff1(self.norm2(x)) + x
        x = self.attn2(self.norm3(x)) + x
        x = self.ff2(self.norm4(x)) + x
        if context is not None:
            x = self.attn3(self.norm5(x), context=context) + x
            x = self.ff3(self.norm6(x)) + x
            if not self.is_end:
                context = self.attn4(self.norm7(context), context=x)
                context = self.ff4(self.norm8(context))
        return x, context


class PercepModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.vgg = vgg16(pretrained=True)
        self.fmap_layers = (3, 8, 15, 22, 29)  # layer indexes which contain perceptual feature map info

        # expects inputs to be between [-1, 1]
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

        blocks = []
        e_i = 0
        for layer_idx in self.fmap_layers:
            s_i, e_i = e_i, layer_idx
            blocks.append(self.vgg.features[s_i:e_i])
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        out = []
        x = (x - self.shift) / self.scale
        for block in self.blocks:
            x = block(x)
            out.append(x)
        return out

class TransformerSeq(nn.Module):
    def __init__(self, dim, depth=6, heads=8, dim_head=64, dropout=0.,
                 use_flash_attn=True, use_checkpoint=True):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.checkpoint = use_checkpoint
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, LegacyAttention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout) if not use_flash_attn else \
                        FlashAttention(dim=dim, heads=heads, dim_head=dim_head, attention_dropout=dropout)),
                PreNorm(dim, FeedForward(dim, dim, dropout=dropout))
            ]))

    def _forward(self, x):
        for norm_attn, norm_ffn in self.layers:
            x = x + norm_attn(x)
            x = x + norm_ffn(x)

        return x

    def forward(self, x):
        return torch_checkpoint(self._forward, (x,), self.checkpoint)


class ConnectTransformerBlockEnd(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0.1, context_dim=None, gated_ff=False, use_flash_attn=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout) if not use_flash_attn else \
            FCrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        # is a self-attention
        self.ff1 = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # self.ff2 = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # self.attn2 = CrossAttention(query_dim=context_dim, context_dim=dim,
        #                             heads=n_heads, dim_head=d_head, dropout=dropout) if not use_flash_attn else \
        #     FCrossAttention(query_dim=context_dim, context_dim=dim,
        #                    heads=n_heads, dim_head=d_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # self.norm3 = nn.LayerNorm(dim)
        # self.norm4 = nn.LayerNorm(dim)


    def forward(self, x, context=None):

        x = self.attn1(self.norm1(x), context=context) + x
        x = self.ff1(self.norm2(x)) + x
        return x, context

class ConnectTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0.1, context_dim=None, gated_ff=False, use_flash_attn=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout) if not use_flash_attn else \
            FCrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        # is a self-attention
        self.ff1 = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # self.ff2 = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # self.attn2 = CrossAttention(query_dim=context_dim, context_dim=dim,
        #                             heads=n_heads, dim_head=d_head, dropout=dropout) if not use_flash_attn else \
        #     FCrossAttention(query_dim=context_dim, context_dim=dim,
        #                    heads=n_heads, dim_head=d_head, dropout=dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # self.norm3 = nn.LayerNorm(dim)
        # self.norm4 = nn.LayerNorm(dim)



    def forward(self, x, context=None):

        x = self.attn1(self.norm1(x), context=self.norm6(context)) + x
        x = self.ff1(self.norm2(x)) + x

        return x, context


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 use_checkpoint=False, is_end=False, use_flash_attn=False):

        super().__init__()
        self.in_channels = in_channels
        self.checkpoint = use_checkpoint
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.is_end = is_end
        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [ConnectTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, use_flash_attn=use_flash_attn)
                for d in range(depth)] if not is_end else
            [ConnectTransformerBlockEnd(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, use_flash_attn=use_flash_attn)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        return torch_checkpoint(self._forward, (x, context), self.checkpoint)

    def _forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x

        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x, context = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        x = self.proj_out(x)

        return x + x_in, context


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 heads=12,
                 dim_head=64,
                 attention_dropout=0.0,
                 causal=False,
                 device=None,
                 dtype=None,
                 use_flash_attn=True,
                 use_checkpoint=False,
                 ):
        super().__init__()
        from einops import rearrange
        # from flash_attn.flash_attention import FlashAttention
        from flash_attn.modules.mha import FlashSelfAttention

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = dim
        self.num_heads = heads
        self.use_checkpoint = use_checkpoint
        self.causal = causal
        self.inner_dim = dim_head * heads
        self.project_out = not (heads == 1 and dim_head == dim)
        assert (
            self.inner_dim % heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = dim_head

        assert dim_head in [16, 32, 48, 64], "Only support head_dim == 16, 32, or 64 not {}".format(self.head_dim)

        self.use_flash_attn = use_flash_attn
        self.attention = FlashSelfAttention(
            attention_dropout=attention_dropout
        ) if use_flash_attn else QKVAttention(self.num_heads)
        # self.to_out = nn.Linear(self.inner_dim, dim) if self.project_out else nn.Identity()

        self.rearrange = rearrange
        self.norm = Normalize(dim)
        # print(heads, dim, self.inner_dim)
        self.proj = zero_module(nn.Conv1d(dim, dim, 1)) if not use_flash_attn else zero_module(nn.Conv1d(self.inner_dim, dim, 1))
        self.to_qkv = nn.Conv1d(dim, dim * 3, 1) if not use_flash_attn else nn.Conv1d(dim, self.inner_dim * 3, 1)


    def forward(self, x):
        return torch_checkpoint(self._forward, (x,), self.use_checkpoint)


    def _forward(self, x):
        # b, c, *spatial = x.shape
        x = self.norm(x)
        b, c, h, w = x.shape
        qkv = self.to_qkv(x.view(b, c, -1))
        # qkv = self.rearrange(qkv, "b c h w -> b c (h w)")
        if self.use_flash_attn:
            qkv = self.rearrange(
                qkv, "b (three h d) s -> b s three h d", three=3, h=self.num_heads
            )
        qkv = self.attention(
            qkv
        )
        if self.use_flash_attn:
            qkv = self.rearrange(qkv, "b s h d -> b (h d) s")
        # _h = self.rearrange(_h, "b c (h w) -> b c h w", h=h).contiguous()
        _h = self.proj(qkv)
        return x + _h.reshape(b, c, h, w)