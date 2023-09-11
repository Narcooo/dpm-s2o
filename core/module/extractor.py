import torch
import math
import torch.nn as nn
import numpy as np
import utils as U
from core.module.layers import ResnetBlock, ConvNextBlock, Swish, SinusoidalPositionEmbeddings, \
    Attention, NearestUpsample, Downsample, TimestepEmbedSequential, zero_module, Normalize, TransformerSeq
from functools import partial
from torch.cuda.amp import autocast

class UnetExtractor(nn.Module):

    def __init__(
            self,
            context_dim,
            use_time_embedding=False,
            attention_resolutions=[8, 4],
            init_weight=True,
            channel_mults=(1, 2, 3, 6),
            channels=3,
            num_classes=None,
            attn_type='vanilla',
            num_res_blocks=2,
            out_channels=3,
            model_channels=96,
            resnet_block_groups=32,
            num_head_channels=32,
            num_heads=-1,
            use_linear_attention=False,
            num_heads_upsample=-1,
            use_block_upsample=False,
            use_block_downsample=True,
            use_spatial_transformer=False,
            transformer_depth=1,  # custom transformer support
            # context_dim=None,
            use_convnext=True,
            convnext_mult=2,
            time_embedding_channels=512,
            dropout=0.,
            use_fp16=False,
            out_shape=2,
            extractor_shape=None,
    ):
        super().__init__()

        # determine dimensions
        self.dropout = dropout
        self.channels = channels
        self.num_classes = num_classes
        self.num_stages = len(channel_mults)
        self.out_channels = out_channels
        self.dtype = torch.float32 if not use_fp16 else torch.float16
        self.init_conv = nn.Conv2d(channels, model_channels, 7, padding=3) \
            if not use_convnext else nn.Conv2d(channels, model_channels, 3, padding=1)

        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = time_embedding_channels
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels + 1, time_dim),
            Swish(),
            nn.Linear(time_dim, time_dim),
        ) if use_time_embedding else None

        if self.num_classes is not None:
            self.label_emb = nn.Sequential(
                nn.Embedding(num_classes, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )
        # layers
        # self.downs = nn.ModuleList([])
        # self.ups = nn.ModuleList([])
        self._feature_size = model_channels
        use_flash_attn = use_fp16
        ch = model_channels
        input_block_chans = [model_channels]
        ds = 1
        self.down_blocks = nn.ModuleList([])
        self.down_blocks.append(TimestepEmbedSequential(self.init_conv))
        for level, mult in enumerate(channel_mults):
            for _ in range(num_res_blocks):
                layers = [
                    ResnetBlock(in_channels=ch, out_channels=mult * model_channels,
                                dropout=dropout,
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
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mults) - 1:
                out_ch = ch
                self.down_blocks.append(TimestepEmbedSequential(
                                          ResnetBlock(
                                              ch, dropout=dropout,
                                              out_channels=out_ch, down=True,
                                          )
                                        if use_block_downsample else
                                        Downsample(ch)))# Downsample(ch)

                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels

        self.middle_blocks = TimestepEmbedSequential(
            ResnetBlock(
                ch, out_channels=ch, dropout=dropout,
            ),
            Attention(
                ch, heads=num_heads, dim_head=dim_head, use_flash_attn=use_flash_attn
            ) if not use_spatial_transformer else SpatialTransformer(
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
            ),
            ResnetBlock(
                ch, out_channels=ch, dropout=dropout,
            ))
        self._feature_size += ch
        middle_ch = ch
        # self.up_blocks = nn.ModuleList([])
        # for level, mult in list(enumerate(channel_mults))[::-1]:
        #     for i in range(num_res_blocks + 1):
        #         ich = input_block_chans.pop()
        #         layers = [
        #             ResnetBlock(
        #                 ch + ich, dropout=dropout,
        #                 out_channels=model_channels * mult
        #             )
        #         ]
        #         ch = model_channels * mult
        #         if ds in attention_resolutions:
        #             if num_head_channels == -1:
        #                 dim_head = ch // num_heads
        #             else:
        #                 num_heads = ch // num_head_channels
        #                 dim_head = num_head_channels
        #             layers.append(
        #                 Attention(
        #                     ch, heads=num_heads, dim_head=dim_head,
        #                 ) if not use_spatial_transformer else SpatialTransformer(
        #                     ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
        #                 )
        #             )
        #         if level and i == num_res_blocks:
        #             out_ch = ch
        #             layers.append(
        #                 ResnetBlock(
        #                     ch, dropout=dropout, out_channels=out_ch,
        #                     up=True
        #                 )
        #                 if use_block_upsample
        #                 else NearestUpsample(ch, use_conv=True)
        #             )
        #             ds //= 2
        #         self.up_blocks.append(TimestepEmbedSequential(*layers))
        #         self._feature_size += ch
        #
        # self.out_recon = nn.Sequential(
        #     Normalize(ch),
        #     Swish(),
        #     zero_module(nn.Conv2d(model_channels, out_channels, 3, padding=1)),
        # )
        self.out = nn.Sequential(
                Normalize(middle_ch),
                Swish(),
                nn.AdaptiveAvgPool2d((1, 1)) if extractor_shape is None else nn.AdaptiveAvgPool2d(extractor_shape),
                nn.Conv2d(middle_ch, context_dim, 1),
                nn.Flatten(),
            ) if out_shape == 1 else nn.Sequential(
            Normalize(middle_ch),
            Swish(),
            # nn.AdaptiveAvgPool2d((1, 1)) if extractor_shape is None else nn.AdaptiveAvgPool2d(extractor_shape),
            zero_module(nn.Conv2d(middle_ch, context_dim, 1)),
            nn.Flatten(start_dim=2),
            # nn.Linear(ch * channel_mults[-1], context_dim)
        )
        self.final_ln = nn.LayerNorm(context_dim)
        self.transformer = TransformerSeq(dim=context_dim, use_flash_attn=use_fp16)
        self.cls_token = nn.Parameter(torch.empty(1, 1, context_dim, dtype=torch.float32))
        self.positional_embedding = nn.Parameter(torch.empty(1, 1 + int((256 / (2 ** (len(channel_mults) - 1))) ** 2), context_dim, dtype=torch.float32))
        # self.context_proj = nn.Linear(ch * channel_mults[-1], context_dim)
            # i
        if init_weight:
            self._initialize_weights()

    def forward(self, x, time=None, y=None):

        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        # x = self.init_conv(x)

        emb = self.time_mlp(time) if U.exists(self.time_mlp) else None
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = self.label_emb(y)
            emb += emb
        hs = []
        # a=0
        # downsample
        # h = x.type(self.dtype)
        h = x
        for module in self.down_blocks:

            h, context = module(h, emb)
            hs.append(h)
        h, context = self.middle_blocks(h, emb)
        h_context = h.type(x.dtype)
        # with autocast(enabled=False):
        h_context = self.out(h_context)
        # for module in self.up_blocks:
        #     h = torch.cat([h, hs.pop()], dim=1)
        #     h, context = module(h, emb)
        # h = h.type(x.dtype)
        # with autocast(enabled=False):
        #     h = self.out_recon(h)
        h_context = h_context.transpose(1, 2)
        h_context = h_context + self.positional_embedding[:, 1:, :]
        cls_token = self.cls_token + self.positional_embedding[:, :1, :]
        cls_tokens = cls_token.expand(h_context.shape[0], -1, -1)
        h_context = torch.cat((h_context, cls_tokens), dim=1)

        h_context = self.transformer(h_context)
        if self.final_ln is not None:
            h_context = self.final_ln(h_context)

        proj = h_context[:, -1]
        out = h_context[:, :-1].permute(0, 2, 1)  # NLC -> NCL

        outputs = dict(proj=proj, out=out)
        return outputs
        # return h_context, h

    def _initialize_weights(self):
        '''
        Initialize the weights of the model.
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=1.0)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

