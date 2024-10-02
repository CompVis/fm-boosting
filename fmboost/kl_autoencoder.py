import os
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
from inspect import isfunction

try:
    _ = nn.functional.scaled_dot_product_attention
except AttributeError:
    print("[AutoencoderKL] Please update PyTorch to 2.0 or higher to use efficient attention.")

try:
    import natten # type: ignore
    NATTEN_IS_AVAILBLE = True
except:
    NATTEN_IS_AVAILBLE = False

# scale factor for normalising the latent space (default value in SD)
LATENT_SCALE = 0.18215


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def nonlinearity(x):
    # this incurs additional memory: return x*torch.sigmoid(x)
    # so we switch to pytorch's in-built function and perform silu in-place:
    return nn.SiLU(inplace=True)(x)


def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class DiagonalGaussianDistribution:
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean
    

""" Resnet blocks """


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels,
                                               out_channels,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels,
                                              out_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


""" Attention """


class MemoryEfficientAttnBlock(nn.Module):
    """
    Uses xformers efficient implementation,
    see https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    Note: this is a single-head self-attention operation
    """
    def __init__(self, in_channels, natten_kernel_size=-1, use_null_attention=False):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        conv_kwargs = dict(kernel_size=1, stride=1, padding=0)
        self.q = nn.Conv2d(in_channels, in_channels, **conv_kwargs)
        self.k = nn.Conv2d(in_channels, in_channels, **conv_kwargs)
        self.v = nn.Conv2d(in_channels, in_channels, **conv_kwargs)
        self.proj_out = nn.Conv2d(in_channels, in_channels, **conv_kwargs)

        if natten_kernel_size > -1:
            assert NATTEN_IS_AVAILBLE, "natten_kernel_size > -1 but natten is not available"
        assert (natten_kernel_size % 2) == 1, 'natten_kernel_size must be odd'
        self.natten_kernel_size = natten_kernel_size
        self.use_null_attention = use_null_attention

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute null attention by discarding q,k
        if self.use_null_attention:
            out = self.proj_out(v)
            return x+out

        # compute attention
        if NATTEN_IS_AVAILBLE and self.natten_kernel_size > -1:
            q, k, v = map(lambda x: rearrange(x, 'b c h w -> b 1 h w c'), (q, k, v))
            qk = natten.functional.natten2dqk(q, k, self.natten_kernel_size, 1)
            a = torch.softmax(qk, dim=-1)
            out = natten.functional.natten2dav(a, v, self.natten_kernel_size, 1)
            out = rearrange(out, 'b 1 h w c -> b c h w')
        
        else:
            b,c,h,w = q.shape
            q, k, v = map(lambda x: rearrange(x, 'b c h w -> b (h w) c'), (q, k, v))
            out = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)

        out = self.proj_out(out)
        return x+out


def make_attn(in_channels, attn_type="vanilla", natten_kernel_size=-1, use_null_attention=False):
    if attn_type == "vanilla":
        assert not (use_null_attention and natten_kernel_size > -1), "use_null_attention and natten_kernel_size > -1 are mutually exclusive"
        if natten_kernel_size > -1 and not NATTEN_IS_AVAILBLE:
            raise ValueError("natten_kernel_size > -1 but natten is not available")
        if use_null_attention:
            print(f"Using null attention to save memory and compute...")
        return MemoryEfficientAttnBlock(in_channels, natten_kernel_size=natten_kernel_size, use_null_attention=use_null_attention)
    
    elif attn_type == "none":
        return nn.Identity(in_channels)
    
    else:
        raise NotImplementedError(f"attn_type {attn_type} not implemented")


""" Encoder and Decoder """


class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, attn_type="vanilla",
                 natten_kernel_size=-1, use_null_attention=False,
                 **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type, natten_kernel_size=natten_kernel_size, use_null_attention=use_null_attention))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type, natten_kernel_size=natten_kernel_size, use_null_attention=use_null_attention)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False,
                 natten_kernel_size=-1, use_null_attention=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.natten_kernel_size = natten_kernel_size
        self.use_null_attention = use_null_attention

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type, natten_kernel_size=natten_kernel_size, use_null_attention=use_null_attention)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type, natten_kernel_size=natten_kernel_size, use_null_attention=use_null_attention))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


""" KL-regularized Autoencoder """


DEFAULT_DDCONFIG = dict(
    attn_type           = "vanilla",
    double_z            = True,
    z_channels          = 4,
    resolution          = 256,
    in_channels         = 3,
    out_ch              = 3,
    ch                  = 128,
    ch_mult             = [1, 2, 4, 4],
    num_res_blocks      = 2,
    attn_resolutions    = [ ],
    dropout             = 0.0
)


class AutoencoderKL(nn.Module):
    def __init__(
            self,
            ckpt_path: str = None,
            ddconfig=DEFAULT_DDCONFIG,
            embed_dim: int = 4,
            use_null_attention: bool = False,
            **kwargs
        ):
        super().__init__()
        ddconfig['use_null_attention'] = use_null_attention
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

        if exists(ckpt_path):
            assert os.path.exists(ckpt_path), f'[AutoencoderKL] Checkpoint {ckpt_path} not found!'
            print(f'[AutoencoderKL] Loading checkpoint from {ckpt_path}')
            if torch.cuda.is_available():
                self.load_state_dict(torch.load(ckpt_path))
            else:
                self.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
        else:
            import warnings
            warnings.warn(f'[AutoencoderKL] No checkpoint provided. Random initialization.')

    def encode(self, x, normalize=False):
        """
        Args:
            x: input tensor (B, C, H, W) in range [-1, 1]
            normalize: if True, normalizes the latent code with SDs
                LATENT_SCALE before returning z with (B, C, H, W).
                Otherwise, returns the DiagonalGaussianDistribution
                object (default).
        """
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        if normalize:
            return posterior.mode() * LATENT_SCALE
        return posterior

    def decode(self, z, denorm=False):
        """
        Args:
            z: latent code tensor (B, C, H, W)
            denorm: if True, denormalizes the latent code
                with SDs LATENT_SCALE before decoding.
        """
        if denorm:
            z = z / LATENT_SCALE
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input, normalize=False)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z, denorm=False)
        return dec, posterior


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoencoderKL().to(dev)
    ipt = torch.randn(1, 3, 256, 256).to(dev)
    out, posterior = model(ipt)
    z = posterior.sample()

    print(f"{'Params':<10}: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'Input':<10}: {ipt.shape}")
    print(f"{'Latent':<10}: {z.shape}")
    print(f"{'Output':<10}: {out.shape}")
