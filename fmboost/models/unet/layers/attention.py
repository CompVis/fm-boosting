import math
import torch
import torch.nn as nn


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """
    def __init__(self, efficient_attn: bool = True, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        self.efficient_attn = efficient_attn
        if self.efficient_attn:
            try:
                _ = nn.functional.scaled_dot_product_attention
            except AttributeError:
                print("Please update PyTorch to 2.0 or higher to use efficient attention.")
                self.efficient_attn = False

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Args:
            q, k, v: (n, ..., l, c) tensors of Queries, Keys, Values. The ...
                can be any number of batch dimensions (e.g. heads).
        Returns:
            res: (n, ..., l, c) tensor after attention.
        """
        if self.efficient_attn:
            res = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
        else:
            ch = q.shape[-1]
            scale = 1. / math.sqrt(ch)
            dot = torch.einsum('...td, ...kd -> ...tk', q, k) * scale
            weight = torch.softmax(dot, dim=-1)
            if self.dropout > 0.0:
                weight = torch.dropout(weight, p=self.dropout, train=self.training)
            res = torch.einsum('...dt, ...tv -> ...dv', weight, v)
        return res


class LinearQKVAttention(nn.Module):
    """
    A module which performs linear QKV attention.
    (https://arxiv.org/abs/1812.01243)
    """
    def __init__(self, l2_norm_v: bool = False):
        super().__init__()
        self.l2_norm_v = l2_norm_v

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Args:
            q, k, v: (n, ..., l, c) tensors of Queries, Keys, Values. The ...
                can be any number of batch dimensions (e.g. heads).
        Returns:
            res: (n, ..., l, c) tensor after attention.
        """
        ch = q.shape[-1]
        scale = 1. / math.sqrt(ch)
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)
        q = q * scale
        if self.l2_norm_v:
            v = torch.nn.functional.normalize(v, dim=-1)
        context = torch.einsum('...nd, ...ne -> ...de', k, v)
        res = torch.einsum('...nd, ...de -> ...ne', q, context)
        return res


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class SpatialSelfAttention(nn.Module):
    """
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 64,
                 use_linear: bool = False, use_efficient_attn: bool = True):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads

        self.norm = nn.GroupNorm(32, dim)
        self.qkv = nn.Conv1d(dim, self.inner_dim * 3, 1)
        self.attention = LinearQKVAttention() if use_linear else QKVAttention(efficient_attn=use_efficient_attn)
        self.proj_out = zero_module(nn.Conv1d(self.inner_dim, self.dim, 1))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape (b, c, *spatial), where spatial can be (f, h, w) or (h, w).
        Returns:
            x: Tensor after attention, MHSA(x) + residual.
        """
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)                                     # (b, c, f * h * w)
        qkv = self.qkv(self.norm(x))                                # (b, 3 * c * nh, f * h * w)
        qkv = qkv.reshape(b, self.heads, qkv.shape[-1], -1)         # (b, nh, f * h * w, 3 * c)
        q, k, v = qkv.chunk(3, dim=-1)                              # (b, nh, f * h * w, c) each
        h = self.attention(q, k, v)                                 # (b, nh, f * h * w, c)
        h = h.reshape(b, self.inner_dim, -1)                        # (b, nh * c, f * h * w)
        h = self.proj_out(h)                                        # (b, c, f * h * w)
        return (x + h).reshape(b, c, *spatial)


if __name__ == "__main__":
    ipt = torch.randn((1, 32, 8, 8))
    print("Input:", ipt.shape)
    attn = SpatialSelfAttention(32, heads=4, dim_head=64, use_linear=True)
    print(f"Params: {sum(p.numel() for p in attn.parameters()):,}")
    out = attn(ipt)
    print("Output:", out.shape)
