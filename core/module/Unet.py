import torch
import math
import torch.nn as nn
import numpy as np
import utils as U
from einops import rearrange, repeat
from core.module.extractor import UnetExtractor
# from core.module.autoencoder import AutoEncoder
from core.module.layers import ResnetBlock, ConvNextBlock, Swish, SinusoidalPositionEmbeddings, \
    Attention, NearestUpsample, Downsample, TimestepEmbedSequential, zero_module, Normalize, trunc_normal_, \
    SpatialTransformer, timestep_embedding
from functools import partial
from torch.cuda.amp import autocast


class Unet(nn.Module):
    def __init__(
            self,
            attention_resolutions=[32, 16, 8],
            init_weight=True,
            channel_mults=(1, 1, 2, 2, 4, 8),
            channels=3,
            num_classes=None,
            attn_type='vanilla',
            num_res_blocks=[1, 1, 2, 2, 2, 2],
            out_channels=6,
            model_channels=128,
            num_head_channels=32,
            num_heads=-1,
            use_linear_attention=False,
            num_heads_upsample=1,
            use_block_upsample=True,
            use_block_downsample=True,
            transformer_depth=1,  # custom transformer support
            context_dim=768,
            use_convnext=True,
            convnext_mult=2,
            dropout=0.1,
            use_cheackpoint=True,
            time_embedding_channels=768,
            use_fp16=False,
            use_context=False,
    ):
        super().__init__()

        # determine dimensions
        self.use_cheackpoint = use_cheackpoint
        self.extractor = UnetExtractor(context_dim=context_dim, use_fp16=use_fp16, out_shape=2) if use_context else None
        self.dropout = dropout
        self.channels = channels
        self.num_classes = num_classes
        self.num_stages = len(channel_mults)
        self.out_channels = out_channels
        self.dtype = torch.float32 if not use_fp16 else torch.float16
        use_flash_attn = use_fp16

        time_dim = time_embedding_channels
        self.model_channels = model_channels
        self.ext_emb_dim = model_channels * channel_mults[-1]
        self.time_mlp = nn.Sequential(
            # SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(time_dim, 2 * time_dim),
            Swish(),
            nn.Linear(2 * time_dim, time_dim),
        )
        self.time_dim = time_dim
        if self.num_classes is not None:
            self.label_emb = nn.Sequential(
                nn.Embedding(num_classes, time_dim),
                # nn.SiLU(),
                # nn.Linear(time_dim, time_dim),
            )
        self._feature_size = model_channels

        ch = model_channels
        input_block_chans = [model_channels]
        ds = 1
        self.down_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv2d(channels, model_channels, 3, padding=1))]
        )
        # self.down_blocks.append(TimestepEmbedSequential(self.init_conv))
        for level, mult in enumerate(channel_mults):
            for _ in range(num_res_blocks[level]):
                layers = [
                    ResnetBlock(in_channels=ch, out_channels=mult * model_channels,
                                time_emb_dim=time_dim, dropout=dropout,
                                ext_emb_dim=self.ext_emb_dim,
                                )]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    layers.append(
                        Attention(
                            ch, heads=num_heads, dim_head=dim_head, use_flash_attn=use_flash_attn
                        )
                    )
                    if use_context:
                        layers.append(SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                            use_flash_attn=use_flash_attn
                            )
                        )
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mults) - 1:
                out_ch = ch
                self.down_blocks.append(TimestepEmbedSequential(
                                          ResnetBlock(
                                              ch, time_emb_dim=time_dim, dropout=dropout,
                                              out_channels=out_ch,
                                              down=True, ext_emb_dim=self.ext_emb_dim,
                                          ) if use_block_downsample else
                                        Downsample(ch)))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        self.middle_block = nn.ModuleList([])
        self.middle_block.append(TimestepEmbedSequential(
            ResnetBlock(
                ch, out_channels=ch, time_emb_dim=time_dim, dropout=dropout,
                ext_emb_dim=self.ext_emb_dim,
            )))
        self.middle_block.append(TimestepEmbedSequential(
            Attention(
                ch, heads=num_heads, dim_head=dim_head, use_flash_attn=use_flash_attn
            )
        ))
        if use_context:
            self.middle_block.append(SpatialTransformer(
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                use_flash_attn=use_flash_attn
            ))
        self.middle_block.append(TimestepEmbedSequential(
            ResnetBlock(
                ch, out_channels=ch, time_emb_dim=time_dim, dropout=dropout,
                ext_emb_dim=self.ext_emb_dim,
            )
        ))
        # self.middle_block = TimestepEmbedSequential(
        #     ResnetBlock(
        #         ch, out_channels=ch, time_emb_dim=time_dim, dropout=dropout,
        #         ext_emb_dim=self.ext_emb_dim,
        #         use_adaptive_norm=self.use_adaptive_norm
        #     ),
        #     Attention(
        #         ch, heads=num_heads, dim_head=dim_head,
        #     ) if not self.use_spatial_transformer else SpatialTransformer(
        #         ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
        #     ),
        #     ResnetBlock(
        #         ch, out_channels=ch, time_emb_dim=time_dim, dropout=dropout,
        #         ext_emb_dim=self.ext_emb_dim,
        #         use_adaptive_norm=self.use_adaptive_norm
        #     ))
        self._feature_size += ch
        middle_ch = ch
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mults))[::-1]:
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResnetBlock(
                        ch + ich, time_emb_dim=time_dim, dropout=dropout,
                        out_channels=model_channels * mult, ext_emb_dim=self.ext_emb_dim,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if ds == min(attention_resolutions):
                        if num_head_channels == -1:
                            dim_head = ch // num_heads
                        else:
                            num_heads = ch // num_head_channels
                            dim_head = num_head_channels
                        layers.append(
                            Attention(
                                ch, heads=num_heads_upsample, dim_head=dim_head, use_flash_attn=use_flash_attn
                            )
                        )
                        if use_context:
                            layers.append(
                                SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth,
                                                   is_end=True, context_dim=context_dim, use_flash_attn=use_flash_attn
                                                   )
                            )
                    else:
                        if num_head_channels == -1:
                            dim_head = ch // num_heads
                        else:
                            num_heads = ch // num_head_channels
                            dim_head = num_head_channels
                        layers.append(
                            Attention(
                                ch, heads=num_heads_upsample, dim_head=dim_head, use_flash_attn=use_flash_attn
                            )
                        )
                        if use_context:
                            layers.append(
                                SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth,
                                                   context_dim=context_dim, use_flash_attn=use_flash_attn
                                                   )
                            )

                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResnetBlock(
                            ch, time_emb_dim=time_dim, dropout=dropout, out_channels=out_ch,
                            up=True, ext_emb_dim=self.ext_emb_dim,
                        )
                        if use_block_upsample
                        else NearestUpsample(ch)
                    )
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            Normalize(ch),
            Swish(),
            zero_module(nn.Conv2d(model_channels, out_channels, 3, padding=1)),
        )
        self.transformer_proj = nn.Linear(context_dim, time_embedding_channels) if use_context else None
        if init_weight:
            self._initialize_weights()

    def extract_seg(self, x):
        seg_cond = self.extractor(x)
        return seg_cond

    def forward(self, x, time, context=None, y=None):

        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        out_dict = self.extract_seg(context) if context else None

        emb = self.time_mlp(timestep_embedding(time, self.time_dim))
        # print(context.max())
        if out_dict is not None:
            context = out_dict['out']
            con_proj = out_dict['proj']
            context = rearrange(context, 'b c n -> b n c')
        # time = time / 1000
            con_proj = self.transformer_proj(con_proj)
            emb = emb + con_proj.to(emb)
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            label_emb = self.label_emb(y)
            emb += label_emb
        hs = []
        # a=0
        # downsample
        # TODO add multi-scale layers to the model
        # h = x.type(self.dtype)
        h = x

        for module in self.down_blocks:
            # a+=1
            # print(a)
            h, context = module(h, emb, context)
            hs.append(h)
        # h = self.middle_block(h, emb, context)
        for module in self.middle_block:
            h, context = module(h, emb, context)
        for module in self.up_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h, context = module(h, emb, context)
        # h = h.type(x.dtype)
        # if self.predict_codebook_ids:
        #     return self.id_predictor(h)
        # else:
        # with autocast(enabled=False):
        h = self.out(h)
        # h = self.out(h)
        return h

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight.data, std=.02)

                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=1.0)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

