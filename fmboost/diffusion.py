import torch
import numpy as np
import torch.nn as nn


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def cosine_log_snr(t, eps=0.00001):
    """
    Returns log Signal-to-Noise ratio for time step t and image size 64
    eps: avoid division by zero
    """
    return -2 * np.log(np.tan((np.pi * t) / 2) + eps)


def shifted_cosine_log_snr(t, im_size: int, ref_size: int = 64):
    return cosine_log_snr(t) + 2 * np.log(ref_size / im_size)


def log_snr_to_alpha_bar(t):
    return sigmoid(cos_log_snr(t))


def shifted_cosine_alpha_bar(t, im_size: int, ref_size: int = 64):
    return sigmoid(shifted_cosine_log_snr(t, im_size, ref_size))


class ForwardDiffusion(nn.Module):
    def __init__(self,
                 im_size: int = 64,
                 n_diffusion_timesteps: int = 1000):
        super().__init__()
        self.n_diffusion_timesteps = n_diffusion_timesteps
        cos_alpha_bar_t = shifted_cosine_alpha_bar(
            np.linspace(0, 1, n_diffusion_timesteps),
            im_size=im_size
        ).astype(np.float32)
        self.register_buffer("alpha_bar_t", torch.from_numpy(cos_alpha_bar_t))
        
    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps. In other
        words sample from q(x_t | x_0).

        Args:
            x_start: The initial data batch.
            t: The diffusion time-step (must be a single t).
            noise: If specified, the noise to use for the diffusion.
        Returns:
            A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        alpha_bar_t = self.alpha_bar_t[t]

        return torch.sqrt(alpha_bar_t) * x_start + torch.sqrt(1 - alpha_bar_t) * noise


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    d = ForwardDiffusion(im_size=64)

    plt.plot(d.alpha_bar_t)
    plt.plot(1 - d.alpha_bar_t)
    plt.savefig("cosine.png")
    plt.close()

    im = torch.ones((1, 3, 128, 128))

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        xt = d.q_sample(im, i * 100)
        print(f"t={i*100}: {d.alpha_bar_t[i*100]}")
        ax.imshow(xt[0].permute(1, 2, 0).detach().numpy())
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("diffusion.png")
    plt.close()
