import torch
import torch.nn as nn


class Pool2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool_factor: int = 2):
        super().__init__()
        if pool_factor > 1 and pool_factor not in [2, 4, 8]:
            raise ValueError(f"Only 1, 2, 4, and 8 are allowed as down-sampling "
                             f"factor, got {pool_factor}")
        if pool_factor <= 1:
            self.down_conv = nn.Identity()
        else:
            k = (pool_factor, pool_factor)
            self.down_conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=k)

    def forward(self, x: torch.Tensor):
        return self.down_conv(x)


class UnPool2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool_factor: int = 2):
        super().__init__()
        if pool_factor > 1 and pool_factor not in [2, 4, 8]:
            raise ValueError(f"Only 1, 2, 4, and 8 are allowed as down-sampling "
                             f"factor, got {pool_factor}")
        self.ds_factor = pool_factor
        if pool_factor <= 1:
            self.up_conv = nn.Identity()
        else:
            k = (pool_factor, pool_factor)
            self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=k, stride=k)

    def forward(self, x: torch.Tensor):
        return self.up_conv(x)


if __name__ == "__main__":
    chs = 128
    ipt = torch.randn((2, chs, 64, 64))
    print("Input:", ipt.shape)

    sp_ds = 8

    patchify = Pool2d(chs, chs, sp_ds)
    out_patch = patchify(ipt)
    print("Patchify:", out_patch.shape)         # (2, 3, 10, 16, 16)

    unpatchify = UnPool2d(chs, chs, sp_ds)
    out_unpatch = unpatchify(out_patch)
    print("UnPatchify:", out_unpatch.shape)     # (2, 3, 10, 64, 64)
