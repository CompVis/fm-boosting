import time
import wandb
import einops
import warnings
from PIL import Image

import torch
import torch.nn as nn
from torch import Tensor
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger

from fmboost.ema import EMA
from fmboost.diffusion import ForwardDiffusion

from fmboost.helpers import freeze
from fmboost.helpers import resize_ims
from fmboost.helpers import exists, default
from fmboost.helpers import un_normalize_ims
from fmboost.helpers import instantiate_from_config
from fmboost.helpers import load_partial_from_config


def hres_lres_pred_grid(hr_ims, lr_ims, hr_pred):
    # resize lr_ims if necessary
    if lr_ims.shape[-1] != hr_ims.shape[-1]:
        lr_ims = resize_ims(lr_ims, hr_ims.shape[-1], mode="bilinear")
    hr_ims = einops.rearrange(hr_ims, "b c h w -> (b h) w c")
    lr_ims = einops.rearrange(lr_ims, "b c h w -> (b h) w c")
    hr_pred = einops.rearrange(hr_pred, "b c h w -> (b h) w c")
    grid = torch.cat([hr_ims, lr_ims, hr_pred], dim=1)
    # normalize to [0, 255]
    grid = un_normalize_ims(grid).cpu().numpy()
    return grid


class TrainerFMBoost(LightningModule):
    def __init__(
            self,
            fm_cfg: dict,
            low_res_size: int,
            high_res_size: int = None,              # unused
            upsampling_mode: str = "bilinear",
            upsampling_mode_context: str = None,
            upsampling_mode_ca_context: str = None,
            start_from_noise: bool = False,
            noising_step: int = -1,
            concat_context: bool = False,
            ca_context: bool = False,
            first_stage_cfg: dict = None,
            scale_factor: int = 1.0,
            lr: float = 1e-4,
            weight_decay: float = 0.,
            n_images_to_vis: int = 16,
            ema_rate: float = 0.99,
            ema_update_every: int = 100,
            ema_update_after_step: int = 1000,
            use_ema_for_sampling: bool = True,
            metric_tracker_cfg: dict = None,
            lr_scheduler_cfg: dict = None,
            log_grad_norm: bool = False,
        ):
        """
        Args:
            fm_cfg: Flow matching model config.
            low_res_size: Size of low-res images.
            upsampling_mode: Mode for up-sampling (bilinear, nearest, psu).
            upsampling_mode_context: Mode for up-sampling the concatenated
                context (if None, same as upsampling_mode).
            upsampling_mode_ca_context: Mode for up-sampling the cross-attention
                context (if None, same as upsampling_mode).
            start_from_noise: Whether to start from noise with low-res image as
                conditioning (FM) or directly from low-res image (IC-FM).
            noising_step: Forward diffusion noising step with linear schedule
                of Ho et al. Set to -1 to disable.
            concat_context: Whether to concatenate the low-res images as conditioning.
            ca_context: Whether to use cross-attention context.
            first_stage_cfg: First stage config, if None, identity is used.
            scale_factor: Scale factor for the latent space (normalize the 
                latent space, default value for SD: 0.18215).
            lr: Learning rate.
            weight_decay: Weight decay.
            n_images_to_vis: Number of images to visualize.
            ema_rate: EMA rate.
            ema_update_every: EMA update rate (every n steps).
            ema_update_after_step: EMA update start after n steps.
            use_ema_for_sampling: Whether to use the EMA model for sampling.
            metric_tracker_cfg: Metric tracker config.
            lr_scheduler_cfg: Learning rate scheduler config.
            log_grad_norm: Whether to log the gradient norm.
        """
        super().__init__()
        self.model = instantiate_from_config(fm_cfg)
        # self.model = torch.compile(self.model)            # TODO haven't fully debugged yet
        self.ema_model = EMA(
            self.model, beta=ema_rate,
            update_after_step=ema_update_after_step,
            update_every=ema_update_every,
            power=3/4.,                     # recommended for trainings < 1M steps
            include_online_model=False      # we have the online model stored here

        )
        self.use_ema_for_sampling = use_ema_for_sampling

        assert low_res_size % 8 == 0, "Low-res size must be divisible by 8 (AE)"
        self.low_res_size = low_res_size

        self.upsampling_mode = upsampling_mode
        self.upsampling_mode_context = default(upsampling_mode_context, upsampling_mode)
        self.upsampling_mode_ca_context = default(upsampling_mode_ca_context, upsampling_mode)
        
        self.start_from_noise = start_from_noise
        self.concat_context = concat_context
        self.ca_context = ca_context

        # forward diffusion of image
        self.noise_image = noising_step > 0
        self.noising_step = noising_step
        if self.start_from_noise and self.noise_image:
            raise ValueError("Cannot use noising step with start_from_noise=True")
        if self.noising_step > 0:
            if self.noising_step > 1 and isinstance(self.noising_step, int):
                self.diffusion = ForwardDiffusion()
            else:
                raise ValueError("Invalid noising step")
        else:
            self.diffusion = None

        # first stage encoding
        self.scale_factor = scale_factor
        if exists(first_stage_cfg):
            self.first_stage = instantiate_from_config(first_stage_cfg)
            freeze(self.first_stage)
            self.first_stage.eval()
            if self.scale_factor == 1.0:
                warnings.warn("Using first stage with scale_factor=1.0")
        else:
            if self.scale_factor != 1.0:
                raise ValueError("Cannot use scale_factor with identity first stage")
            self.first_stage = None
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler_cfg = lr_scheduler_cfg
        self.log_grad_norm = log_grad_norm

        self.vis_samples = None
        self.metric_tracker = instantiate_from_config(metric_tracker_cfg) if exists(metric_tracker_cfg) else None

        self.n_images_to_vis = n_images_to_vis
        self.val_epochs = 0

        self.save_hyperparameters()

        # flag to make sure the signal is not handled at an incorrect state, e.g. during weights update
        self.stop_training = False

    def stop_training_method(self):
        # dummy function to be compatible
        pass

    def upsample_latent(self, lres_z: Tensor, z_size: int, lres_ims: Tensor = None, im_size: int = None, mode: str = "bilinear", **kwargs):
        """
        Args:
            lres_z: low-res latent code (in low-res space)
            z_size: size of the high-res latent code
            lres_ims: low-res images in pixel space
            im_size: size of the high-res images in pixel space
            mode: up-sampling mode (bilinear, nearest, conv, noise_upsampling, decoder_features, psu, identity)
        """
        if mode == "identity":
            return lres_z
        
        elif mode == "psu":
            # if latent code is already in high-res space, we just return it (pre-computed)
            if lres_z.shape[-1] == z_size:
                return lres_z
            # otherwise we do pixel space up-sampling and then encode the image to latent space
            assert exists(im_size), "Need to provide im_size for psu"
            if not exists(lres_ims):
                lres_ims = self.decode_first_stage(lres_z)
            lres_ims_in_hres = resize_ims(lres_ims, im_size, mode="bilinear")
            return self.encode_first_stage(lres_ims_in_hres)
        
        elif mode == "noise_upsampling":
            h, w = lres_z.shape[-2:]
            assert h == w, f"Mode noise_upsampling only works for square images. Got {h}x{w}"
            # if we reduce size, we just interpolate with nearest neighbor
            if z_size <= h:
                return nn.functional.interpolate(lres_z, size=z_size, mode="nearest", **kwargs)
            # for upsampling, we fill the new pixels with noise
            assert z_size % h == 0, f"New size not divisible by old size, got {z_size} and {h}"
            resized = resize_ims(lres_z, z_size, mode="nearest")
            scale = z_size // h
            mask = torch.zeros_like(resized)
            mask[..., ::scale, ::scale] = 1
            return mask * resized + (1 - mask) * torch.randn_like(resized)
        
        return nn.functional.interpolate(lres_z, size=z_size, mode=mode, **kwargs)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        out = dict(optimizer=opt)
        if exists(self.lr_scheduler_cfg):
            sch = load_partial_from_config(self.lr_scheduler_cfg)
            sch = sch(optimizer=opt)
            out["lr_scheduler"] = sch
        return out

    def forward(self, x_target: Tensor, x_source: Tensor, **kwargs):
        return self.model.training_losses(x1=x_target, x0=x_source, **kwargs)

    @torch.no_grad()
    def encode_first_stage(self, x):
        if not exists(self.first_stage):
            return x
        x = self.first_stage.encode(x)
        if not isinstance(x, torch.Tensor): # hack for posterior of original VAE
            x = x.mode()
        return x * self.scale_factor

    @torch.no_grad()
    def decode_first_stage(self, z):
        if not exists(self.first_stage):
            return z
        return self.first_stage.decode(z / self.scale_factor)
    
    def extract_from_batch(self, batch):
        """
        Takes batch and extracts high-res and low-res images and latent codes.

        Returns:  
            hres_ims: high-res images
            hres_z: high-res latent codes (if identity first stage, this is hres_ims)
            lres_ims: low-res images
            lres_z: low-res latent codes (if identity first stage, this is lres_ims)
        """
        hres_ims = batch["image"]

        # check if we have a pre-computed latent code for the high-res image
        if "latent" in batch:
            hres_z = batch["latent"]
            hres_z = hres_z * self.scale_factor
        else:
            hres_z = self.encode_first_stage(hres_ims)

        lres_ims = resize_ims(hres_ims, self.low_res_size, mode='bilinear')

        # check if we have a pre-computed latent code for the low-res image
        if "latent_lowres" in batch:
            lres_z = batch["latent_lowres"]
            lres_z = lres_z * self.scale_factor
        else:
            # encode to latent space (if no first stage, this is identity)
            lres_z = self.encode_first_stage(lres_ims)

        return hres_ims.float(), hres_z.float(), lres_ims.float(), lres_z.float()
        
    def training_step(self, batch, batch_idx):
        """ extract high-res and low-res images from batch """
        hres_ims, hres_z, lres_ims, lres_z = self.extract_from_batch(batch)

        """ context & conditioning information """
        x_source, context, context_ca = self.get_source_and_context(
            lres_z=lres_z,
            z_size=hres_z.shape[-1],
            lres_ims=lres_ims,
            im_size=hres_ims.shape[-1]
        )

        """ loss """
        loss = self.forward(x_target=hres_z, x_source=x_source, context=context, context_ca=context_ca)
        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=x_source.shape[0])

        """ misc """
        self.ema_model.update()
        if exists(self.lr_scheduler_cfg): self.lr_schedulers().step()
        if self.stop_training: self.stop_training_method()
        if self.log_grad_norm:
            grad_norm = get_grad_norm(self.model)
            self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False)

        return loss

    def get_source_and_context(self, lres_z: Tensor, z_size: int, lres_ims: Tensor = None, im_size: int = None):
        """
        Args:
            lres_z: low-res latent code (in low-res space)
            z_size: size of the high-res latent code
            lres_ims: low-res images in pixel space
            im_size: size of the high-res images in pixel space
        Returns:
            x_source: x0 for the flow matching model
            context: context for the flow matching model
        """
        # up-sample latent code, s.t. we have the same size as the high-res latents
        lres_z_hr = self.upsample_latent(lres_z, z_size=z_size, im_size=im_size, lres_ims=lres_ims, mode=self.upsampling_mode)

        # define x0
        if self.start_from_noise:
            x_source = torch.randn_like(lres_z_hr)
        else:
            x_source = lres_z_hr

        # noise the start
        if self.noise_image:
            x_source = self.diffusion.q_sample(x_start=x_source, t=self.noising_step)
        
        # define context (concatenated with image latent codes)
        if self.concat_context or self.start_from_noise:
            if self.upsampling_mode_context == self.upsampling_mode:
                context = lres_z_hr
            else:
                context = self.upsample_latent(lres_z, z_size=z_size, im_size=im_size, lres_ims=lres_ims, mode=self.upsampling_mode_context)
        else:
            context = None

        # cross-attention context
        if self.ca_context:
            if self.upsampling_mode_ca_context == self.upsampling_mode:
                context_ca = lres_z_hr
            else:
                context_ca = self.upsample_latent(lres_z, z_size=z_size, im_size=im_size, lres_ims=lres_ims, mode=self.upsampling_mode_ca_context)
        else:
            context_ca = None

        return x_source, context, context_ca

    def predict_high_res_z(self, lres_z: Tensor, z_hr_size: int, lres_ims: Tensor = None, im_size: int = None, sample_kwargs=None):
        """
        Decode from x0 -> x1 (low-res -> high-res latent code)
        Args:
            lres_z: low-res latent codes (in low-resolution)
            z_hr_size: size of the high-res latent code
        Returns:
            hr_pred_z: high-res latent codes
        """
        z, context, context_ca = self.get_source_and_context(lres_z, z_hr_size, lres_ims=lres_ims, im_size=im_size)

        # up-sample with flow matching
        if not exists(sample_kwargs):
            # default during training
            sample_kwargs = dict(num_steps=40, method="rk4")
        
        fn = self.ema_model.model.generate if self.use_ema_for_sampling else self.model.generate
        hr_pred_z = fn(x=z, context=context, context_ca=context_ca, sample_kwargs=sample_kwargs)
        
        return hr_pred_z
    
    def predict_high_res_img(self, lres_z: Tensor, z_hr_size: int, lres_ims: Tensor = None, im_size: int = None, sample_kwargs=None):
        """
        Decode from x0 -> x1 -> VAE-decode (low-res z -> high-res image)
        Args:
            lres_z: low-res latent codes (in low-resolution)
        Returns:
            hr_pred: high-res images (already decoded with first stage)
        """
        hr_pred_z = self.predict_high_res_z(lres_z, z_hr_size, lres_ims=lres_ims, im_size=im_size, sample_kwargs=sample_kwargs)
        # decode with first stage (if no first stage, this is identity)
        hr_pred = self.decode_first_stage(hr_pred_z)
        return hr_pred

    def validation_step(self, batch, batch_idx):
        hr_ims, hr_z, lr_ims, lr_z = self.extract_from_batch(batch)
        hr_pred = self.predict_high_res_img(
            lres_z=lr_z,
            z_hr_size=hr_z.shape[-1],
            lres_ims=lr_ims,
            im_size=hr_ims.shape[-1],
        )

        # track metrics
        if exists(self.metric_tracker): self.metric_tracker(hr_ims, hr_pred)

        if self.stop_training: self.stop_training_method()
        
        # store samples for visualization
        if self.vis_samples is None:
            self.vis_samples = {'hr': hr_ims, 'lr': lr_ims, 'pred': hr_pred}
        elif self.vis_samples['hr'].shape[0] < self.n_images_to_vis:
            self.vis_samples['hr'] = torch.cat([self.vis_samples['hr'], hr_ims], dim=0)
            self.vis_samples['lr'] = torch.cat([self.vis_samples['lr'], lr_ims], dim=0)
            self.vis_samples['pred'] = torch.cat([self.vis_samples['pred'], hr_pred], dim=0)

    def on_validation_epoch_end(self):
        # log low-res images, high-res images, and up-sampled images
        out_img = hres_lres_pred_grid(self.vis_samples['hr'], self.vis_samples['lr'], self.vis_samples['pred'])
        self.log_image(out_img, "val")
        self.vis_samples = None

        # compute metrics
        if exists(self.metric_tracker):
            metrics = self.metric_tracker.aggregate()
            for k, v in metrics.items():
                self.log(f"val/{k}", v, sync_dist=True)
            self.metric_tracker.reset()
        
        self.val_epochs += 1
        self.print(f"Val epoch {self.val_epochs} | Optimizer step {self.global_step}")

        torch.cuda.empty_cache()

    def log_image(self, img, name):
        """
        Args:
            ims: torch.Tensor or np.ndarray of shape (h, w, c) in range [0, 255]
            name: str
        """
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if isinstance(self.logger, WandbLogger):
            img = Image.fromarray(img)
            img = wandb.Image(img)
            self.logger.experiment.log({f"{name}/samples": img}, step=self.global_step)
        else:
            img = einops.rearrange(img, "h w c -> c h w")
            self.logger.experiment.add_image(f"{name}/samples", img, global_step=self.global_step)


def get_grad_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm
