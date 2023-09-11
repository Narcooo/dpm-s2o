import torch
import torch.nn as nn
import utils as U
from einops import repeat, rearrange
from utils import pair, exists, default
from core.module.extractor import UnetExtractor
from einops.layers.torch import Rearrange
import einops
from core.module.autoencoder import ViTEncoder
from core.module.layers import Transformer
# from core.module.autoencoder import AutoEncoder
from core.module.layers import  Swish, SinusoidalPositionEmbeddings, \
    trunc_normal_, timestep_embedding


class gViT(nn.Module):
    def __init__(self, init_weight=True, channels=4, embed_dim=1024, out_channels=3,
                 image_size=32, patch_size=2, ext_patch_size=16, ext_depth=4,
                 num_classes=None, dim=1024, depth=10, heads=16,
                 ext_dim=768, use_fp16=False, context_dim=768,
                 dropout=0., mlp_time_embed=False, use_encoder=False, use_context=False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        # assert depth % 4 == 0, 'Depth must be divisible by 4.'
        self.use_fp16 = use_fp16
        self.depth = depth
        self.image_size = image_size
        self.ext_dim = ext_dim
        self.dim = dim
        self.num_classes = num_classes
        # extractor_shape = (int(image_size / patch_size), int(image_size / patch_size))
        self.encoder = ViTEncoder(dim=ext_dim, depth=ext_depth, patch_size=ext_patch_size) if use_encoder else None
        self.extractor = UnetExtractor(out_shape=2, context_dim=context_dim) if use_context else None

        self.channels = channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            Swish(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()
        self.extra_dim = 1
        if self.num_classes is not None:
            self.label_embed = nn.Sequential(
                nn.Embedding(num_classes, embed_dim),
            )
            self.extra_dim += 1
        self.dtype = torch.float32 if not use_fp16 else torch.float16
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c p1 p2 -> b (p1 p2) c', p1=image_height // patch_height, p2=image_width // patch_width),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.extra_dim, embed_dim))
        dim_head = dim // heads
        self.dim_head = dim_head
        self.in_blocks = nn.ModuleList([
            Transformer(dim=dim, context_dim=context_dim if use_context else None,
                        heads=heads, dim_head=dim_head, dropout=dropout, use_flash_attn=use_fp16)
            for _ in range((depth // 2) - 1)])
        self.in_blocks.append(Transformer(dim=dim, context_dim=context_dim if use_context else None,
                        heads=heads, dim_head=dim_head, dropout=dropout, use_flash_attn=use_fp16, is_end=True))
        self.mid_block = Transformer(dim=dim, context_dim=context_dim if use_context else None,
                        heads=heads, dim_head=dim_head, dropout=dropout, use_flash_attn=use_fp16)

        self.out_blocks = nn.ModuleList([
            Transformer(dim=dim, context_dim=context_dim if use_context else None, skip=True,
                        heads=heads, dim_head=dim_head, dropout=dropout, use_flash_attn=use_fp16)
            for _ in range(depth // 2)])

        self.to_latent = nn.Identity()
        self.patch_dim = patch_size ** 2 * out_channels
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True)

        self.final_layer = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        trunc_normal_(self.pos_embed, std=.02)
        if init_weight:
            self.apply(self._initialize_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def extract_scale(self, x):
        context = self.extractor(x)
        return {'context': context}

    def extract_seg(self, x):
        seg_cond = self.extractor(x)
        return seg_cond

    def forward(self, x, time, context=None, y=None):
        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        x = self.patch_embed(x)
        b, l, _ = x.shape
        time_emb = self.time_embed(timestep_embedding(time, self.embed_dim))

        out_dict = self.extract_seg(context) if context is not None else None
        if out_dict is not None:
            context = out_dict['out']
            con_proj = out_dict['proj']
            context = rearrange(context, 'b c n -> b n c')
            con_proj = self.transformer_proj(con_proj)
            time_emb = time_emb + con_proj.to(time_emb)
        time_emb = time_emb.unsqueeze(dim=1)
        x = torch.cat((time_emb, x), dim=1)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            label_emb = self.label_embed(y)
            label_emb = label_emb.unsqueeze(dim=1)
            x = torch.cat((label_emb, x), dim=1)

        x += self.pos_embed
        skips = []
        for blk in self.in_blocks:
            x, _ = blk(x, context=context)
            skips.append(x)
        x, _ = self.mid_block(x)

        for blk in self.out_blocks:
            x, _ = blk(x, skip=skips.pop())
        x = self.norm(x)
        x = self.decoder_pred(x)
        assert x.size(1) == self.extra_dim + l
        x = x[:, self.extra_dim:, :]
        x = unpatchify(x, self.out_channels)
        x = self.final_layer(x)
        return x


    def _initialize_weights(self, m):
        # Initialize transformer layers:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def patchify(imgs, patch_size):
        x = einops.rearrange(imgs, 'B C (h p1) (w p2) -> B (h w) (p1 p2 C)', p1=patch_size, p2=patch_size)
        return x


def unpatchify(x, channels=3):
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
    x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
    return x