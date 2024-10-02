import torch
import torch.nn as nn

from layers.residual import ResBlock
from layers.residual import zero_module
from layers.residual import TimestepBlock
from layers.pooling import Pool2d, UnPool2d
from layers.t_emb import timestep_embedding
from layers.resize import Upsample, Downsample
from layers.attention import SpatialSelfAttention


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class EfficientUNet(nn.Module):
    def __init__(
            self,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 3, 4),
            conv_resample=True,
            dim_head=64,
            num_heads=4,
            use_linear_attn=True,
            use_scale_shift_norm=True,
            pool_factor=-1,
    ):
        """
        2D UNet model with attention. It includes down- and up-
        sampling blocks to train an end-to-end high resolution
        diffusion model.

        Args:
            in_channels: channels in the input Tensor.
            model_channels: base channel count for the model.
            out_channels: channels in the output Tensor.
            num_res_blocks: number of residual blocks per downsample.
            attention_resolutions: a collection of downsample rates at which
                attention will take place. Maybe a set, list, or tuple.
                For example, if this contains 4, then at 4x downsampling, attention
                will be used.
            dropout: the dropout probability.
            channel_mult: channel multiplier for each level of the UNet.
            conv_resample: if True, use learned convolutions for upsampling and
                downsampling.
            num_heads: the number of attention heads in each attention layer.
            dim_head: the dimension of each attention head.
            use_linear_attn: If true, applies linear attention in the encoder/decoder.
            use_scale_shift_norm: If True, use ScaleShiftNorm instead of LayerNorm.
            pool_factor: Down-sampling factor for spatial dimensions (w. conv layer)
        """
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.pool_factor = pool_factor

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # if ds factors > 1, it will down-sample the input, otherwise its just identity
        if pool_factor > 1:
            self.pool = Pool2d(in_channels, model_channels, pool_factor=pool_factor)
            starting_channels = model_channels
        else:
            self.pool = nn.Identity()
            starting_channels = in_channels

        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                nn.Conv2d(starting_channels, model_channels, 3, padding=1)
            )
        ])
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        out_channels=mult * model_channels,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        SpatialSelfAttention(
                            dim=ch,
                            heads=num_heads,
                            dim_head=dim_head,
                            use_linear=use_linear_attn,
                            use_efficient_attn=True
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                use_scale_shift_norm=use_scale_shift_norm
            ),
            SpatialSelfAttention(
                ch,
                heads=num_heads,
                dim_head=dim_head,
                use_linear=False,
                use_efficient_attn=True,
            ),
            ResBlock(
                ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        emb_channels=time_embed_dim,
                        out_channels=model_channels * mult,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        SpatialSelfAttention(
                            ch,
                            heads=num_heads,
                            dim_head=dim_head,
                            use_linear=use_linear_attn,
                            use_efficient_attn=True,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # check whether we first down-sampled the input
        if pool_factor > 1:
            self.out = nn.Sequential(
                nn.GroupNorm(32, ch),
                nn.SiLU(),
                nn.Conv2d(model_channels, model_channels, 3, padding=1)
            )
            self.un_pool = UnPool2d(model_channels, out_channels, pool_factor=pool_factor)
        else:
            self.out = nn.Sequential(
                nn.GroupNorm(32, ch),
                nn.SiLU(),
                zero_module(
                    nn.Conv2d(model_channels, out_channels, 3, padding=1))
            )
            self.un_pool = nn.Identity()

    def forward(self, x, t, context=None, context_ca=None, **kwargs):
        """
        Apply the model to an input batch.

        Args:
            x: an [N x C x ...] Tensor of inputs.
            t: a 1-D batch of timesteps.
            context: an optional [N x C x ...] Tensor of context that gets
                concatenated to the inputs.
        Returns:
            an [N x C x ...] Tensor of outputs.
        """
        if context_ca is not None:
            raise NotImplementedError("Cross-attn conditioning not supported yet.")
        
        # TODO: add support for kwargs (cross-attn conditioning)
        emb = self.time_embed(timestep_embedding(t, self.model_channels))

        if context is not None:
            x = torch.cat([x, context], dim=1)

        x = self.pool(x)

        hs = []
        h = x

        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        h = self.middle_block(h, emb)

        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)

        h = self.out(h)
        h = self.un_pool(h)

        return h


if __name__ == "__main__":
    unet = EfficientUNet(
        in_channels=3,
        model_channels=64,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=[4],
        dropout=0,
        channel_mult=(1, 2, 4),
        conv_resample=True,
        dim_head=64,
        num_heads=4,
        use_linear_attn=True,
        use_scale_shift_norm=True,
        pool_factor=1,
    )
    print(f"Params: {sum(p.numel() for p in unet.parameters() if p.requires_grad):,}")

    ipt = torch.randn((2, 3, 64, 64))
    cont = torch.randn((2, 3, 64, 64))
    t_ = torch.rand((2,))
    out = unet(ipt, t_)
    print("Input:", ipt.shape)                      # (bs, c, h, w)
    print("Output:", out.shape)                     # (bs, c, h, w)
