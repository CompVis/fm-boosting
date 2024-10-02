""" Adapted from
- https://github.com/atong01/conditional-flow-matching
- https://github.com/willisma/SiT
Thanks for open-sourcing! :)
"""
import math
import torch
import einops
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch import Tensor
from typing import Union
from functools import partial
from torchdiffeq import odeint

from fmboost.helpers import instantiate_from_config


_ATOL = 1e-6
_RTOL = 1e-3


def pad_v_like_x(v_, x_):
    """
    Function to reshape the vector by the number of dimensions
    of x. E.g. x (bs, c, h, w), v (bs) -> v (bs, 1, 1, 1).
    """
    if isinstance(v_, float):
        return v_
    return v_.reshape(-1, *([1] * (x_.ndim - 1)))


def forward_with_cfg(x, t, model, cfg_scale=1.0, uc_cond=None, cond_key="y", **model_kwargs):
    """ Function to include sampling with Classifier-Free Guidance (CFG) """
    if cfg_scale == 1.0:                                # without CFG
        model_output = model(x, t, **model_kwargs)

    else:                                               # with CFG
        assert cond_key in model_kwargs, f"Condition key '{cond_key}' for CFG not found in model_kwargs"
        assert uc_cond is not None, "Unconditional condition not provided for CFG"
        kwargs = model_kwargs.copy()
        c = kwargs[cond_key]
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        if uc_cond.shape[0] == 1:
            uc_cond = einops.repeat(uc_cond, '1 ... -> bs ...', bs=x.shape[0])
        c_in = torch.cat([uc_cond, c])
        kwargs[cond_key] = c_in
        model_uc, model_c = model(x_in, t_in, **kwargs).chunk(2)
        model_output = model_uc + cfg_scale * (model_c - model_uc)

    return model_output


""" Schedules """


class LinearSchedule:
    def alpha_t(self, t):
        return t
    
    def alpha_dt_t(self, t):
        return 1
    
    def sigma_t(self, t):
        return 1 - t
    
    def sigma_dt_t(self, t):
        return -1

    """ Legacy functions to work with SiT Sampler """

    def compute_alpha_t(self, t):
        return self.alpha_t(t), self.alpha_dt_t(t)
    
    def compute_sigma_t(self, t):
        """Compute the noise coefficient along the path"""
        return self.sigma_t(t), self.sigma_dt_t(t)
    
    def compute_d_alpha_alpha_ratio_t(self, t):
        """Compute the ratio between d_alpha and alpha"""
        return 1 / t
    
    def compute_drift(self, x, t):
        """We always output sde according to score parametrization; """
        t = pad_v_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift = alpha_ratio * x
        diffusion = alpha_ratio * (sigma_t ** 2) - sigma_t * d_sigma_t

        return -drift, diffusion
    
    def compute_diffusion(self, x, t, form="constant", norm=1.0):
        """Compute the diffusion term of the SDE
        Args:
          x: [batch_dim, ...], data point
          t: [batch_dim,], time vector
          form: str, form of the diffusion term
          norm: float, norm of the diffusion term
        """
        t = pad_v_like_x(t, x)
        choices = {
            "constant": norm,
            "SBDM": norm * self.compute_drift(x, t)[1],
            "sigma": norm * self.compute_sigma_t(t)[0],
            "linear": norm * (1 - t),
            "decreasing": 0.25 * (norm * torch.cos(np.pi * t) + 1) ** 2,
            "increasing-decreasing": norm * torch.sin(np.pi * t) ** 2,
        }

        try: diffusion = choices[form]
        except KeyError: raise NotImplementedError(f"Diffusion form {form} not implemented")
        
        return diffusion
    
    def get_score_from_velocity(self, velocity, x, t):
        """Wrapper function: transfrom velocity prediction model to score
        Args:
            velocity: [batch_dim, ...] shaped tensor; velocity model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = pad_v_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * velocity - mean) / var
        return score
    
    def get_noise_from_velocity(self, velocity, x, t):
        """Wrapper function: transfrom velocity prediction model to denoiser
        Args:
            velocity: [batch_dim, ...] shaped tensor; velocity model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = pad_v_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = reverse_alpha_ratio * d_sigma_t - sigma_t
        noise = (reverse_alpha_ratio * velocity - mean) / var
        return noise

    def get_velocity_from_score(self, score, x, t):
        """Wrapper function: transfrom score prediction model to velocity
        Args:
            score: [batch_dim, ...] shaped tensor; score model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = pad_v_like_x(t, x)
        drift, var = self.compute_drift(x, t)
        velocity = var * score - drift
        return velocity
    

class GVPSchedule(LinearSchedule):
    def alpha_t(self, t):
        return torch.sin(t * math.pi / 2)
    
    def alpha_dt_t(self, t):
        return 0.5 * math.pi * torch.cos(t * math.pi / 2)
    
    def sigma_t(self, t):
        return torch.cos(t * math.pi / 2)
    
    def sigma_dt_t(self, t):
        return - 0.5 * math.pi * torch.sin(t * math.pi / 2)
    
    def compute_d_alpha_alpha_ratio_t(self, t):
        """Special purposed function for computing numerical stabled d_alpha_t / alpha_t"""
        return np.pi / (2 * torch.tan(t * np.pi / 2))


""" SDE Sampler """


class StepSDE:
    """SDE solver class"""
    def __init__(self, dt, drift, diffusion, sampler_type):
        self.dt = dt
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type
        self.sampler_dict = {
            "euler": self.__Euler_Maruyama_step,
            "heun": self.__Heun_step,
        }

        try: self.sampler = self.sampler_dict[sampler_type]
        except: raise NotImplementedError(f"Sampler type '{sampler_type}' not implemented.")

    def __Euler_Maruyama_step(self, x, mean_x, t, model, **model_kwargs):
        w_cur = torch.randn(x.size()).to(x)
        t = torch.ones(x.size(0)).to(x) * t
        dw = w_cur * torch.sqrt(self.dt)
        drift = self.drift(x, t, model, **model_kwargs)
        diffusion = self.diffusion(x, t)
        mean_x = x + drift * self.dt
        x = mean_x + torch.sqrt(2 * diffusion) * dw
        return x, mean_x
    
    def __Heun_step(self, x, mean_x, t, model, **model_kwargs):
        w_cur = torch.randn(x.size()).to(x)
        dw = w_cur * torch.sqrt(self.dt)
        t_cur = torch.ones(x.size(0)).to(x) * t
        diffusion = self.diffusion(x, t_cur)
        xhat = x + torch.sqrt(2 * diffusion) * dw
        K1 = self.drift(xhat, t_cur, model, **model_kwargs)
        xp = xhat + self.dt * K1
        K2 = self.drift(xp, t_cur + self.dt, model, **model_kwargs)
        return xhat + 0.5 * self.dt * (K1 + K2), xhat # at last time point we do not perform the heun step

    def __call__(self, x, mean_x, t, model, **model_kwargs):
        return self.sampler(x, mean_x, t, model, **model_kwargs)
    

class FlowSDE:
    def __init__(self, schedule, sample_eps=0):
        """ Sampler class for the FlowModel """
        self.sample_eps = sample_eps        # velocity & [GVP, LINEAR] is stable everywhere, hence 0
        self.schedule = schedule

    def drift(self, x, t, model, **model_kwargs):
        model_output = model(x, t, **model_kwargs)
        assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
        return model_output
    
    def score(self, x, t, model, **model_kwargs):
        # we only train velocity, hence only need to compute score from velocity
        score_out = self.schedule.get_score_from_velocity(model(x, t, **model_kwargs), x, t)
        return score_out

    def check_interval(self, diffusion_form="sigma", reverse=False, last_step_size=0.04):
        t0 = 0
        t1 = 1
        eps = self.sample_eps
        if (isinstance(self.schedule, GVPSchedule) or isinstance(self.schedule, LinearSchedule)):
            # avoid numerical issue by taking a first semi-implicit step
            t0 = eps if diffusion_form == "SBDM" else 0
            t1 = 1 - eps if last_step_size == 0 else 1 - last_step_size
        
        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    def __get_sde_diffusion_and_drift(
        self,
        diffusion_form="SBDM",
        diffusion_norm=1.0,
    ):

        def diffusion_fn(x, t):
            diffusion = self.schedule.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)
            return diffusion
        
        sde_drift = \
            lambda x, t, model, **kwargs: \
                self.drift(x, t, model, **kwargs) + diffusion_fn(x, t) * self.score(x, t, model, **kwargs)
    
        sde_diffusion = diffusion_fn

        return sde_drift, sde_diffusion
    
    def last_step(
        self,
        x,
        t,
        model,
        sde_drift,
        last_step,
        last_step_size,
        **model_kwargs
    ):
        """Get the last step function of the SDE solver"""
    
        if last_step is None:
            return x
        
        elif last_step == "Mean":
            return x + sde_drift(x, t, model, **model_kwargs) * last_step_size
        
        elif last_step == "Tweedie":
            alpha = self.schedule.compute_alpha_t # simple aliasing; the original name was too long
            sigma = self.schedule.compute_sigma_t
            # return x / alpha(t)[0] + (sigma(t)[0] ** 2) / alpha(t)[0] * self.score(x, t, model, **model_kwargs)
            raise NotImplementedError("Tweedie last step seems weird (alpha(t) is indexed twice?!?)")
        
        elif last_step == "Euler":
            return x + self.drift(x, t, model, **model_kwargs) * last_step_size
        
        else:
            raise NotImplementedError(f"Last step '{last_step}' not implemented.")
    
    def sample(
        self,
        init,
        model,
        sampling_method="euler",
        diffusion_form="sigma",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
        progress=True,
        return_intermediates=False,
        cfg_scale=1.0,
        uc_cond=None,
        cond_key="y",
        **model_kwargs
    ):
        """
        Args:
        - sampling_method: type of sampler used in solving the SDE; default to be Euler-Maruyama
        - diffusion_form: function form of diffusion coefficient; default to be matching SBDM
        - diffusion_norm: function magnitude of diffusion coefficient; default to 1
        - last_step: type of the last step; default to identity
        - last_step_size: size of the last step; default to match the stride of 250 steps over [0,1]
        - num_steps: total integration step of SDE
        """
        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form, diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.check_interval(diffusion_form=diffusion_form, reverse=False, last_step_size=last_step_size)
        ts = torch.linspace(t0, t1, num_steps).to(init.device)
        dt = ts[1] - ts[0]

        # enable classifier-free guidance 
        model_forward_fn = partial(forward_with_cfg, model=model, cfg_scale=cfg_scale, uc_cond=uc_cond, cond_key=cond_key)

        """ forward loop of sde """
        sampler = StepSDE(dt=dt, drift=sde_drift, diffusion=sde_diffusion, sampler_type=sampling_method)
        
        # sample
        x = init
        mean_x = init
        xs = []
        for ti in tqdm(ts[:-1], disable=not progress, desc="SDE sampling", total=num_steps, initial=1):
            with torch.no_grad():
                x, mean_x = sampler(x, mean_x, ti, model_forward_fn, **model_kwargs)
                xs.append(x)
        
        # make last step
        t_last = torch.ones(x.size(0), device=x.device) * t1
        x = self.last_step(
            x=xs[-1], t=t_last,
            model=model_forward_fn,
            sde_drift=sde_drift,
            last_step=last_step,
            last_step_size=last_step_size,
            **model_kwargs
        )
        xs.append(x)

        assert len(xs) == num_steps, "Samples does not match the number of steps"

        if return_intermediates:
            return xs
        return xs[-1]
    

""" Flow Model """


class FlowModel(nn.Module):
    def __init__(
            self,
            net_cfg: Union[dict, nn.Module],
            schedule: str = "linear",
            sigma_min: float = 0.0
        ):
        """
        Flow Matching, Stochastic Interpolants, or Rectified Flow model. :)
        
        Args:
            net: Neural network that takes in x and t and outputs the vector
                field at that point in time and space with the same shape as x.
            schedule: str, specifies the schedule for the flow. Currently
                supports "linear" and "gvp" (Generalized Variance Path) [3].
            sigma_min: a float representing the standard deviation of the
                Gaussian distribution around the mean of the probability
                path N(t * x1 + (1 - t) * x0, sigma), as used in [1].

        References:
            [1] Lipman et al. (2023). Flow Matching for Generative Modeling.
            [2] Tong et al. (2023). Improving and generalizing flow-based
                generative models with minibatch optimal transport.
            [3] Ma et al. (2024). SiT: Exploring flow and diffusion-based
                generative models with scalable interpolant transformers.
        """
        super().__init__()
        if isinstance(net_cfg, nn.Module):
            self.net = net_cfg
        else:
            self.net = instantiate_from_config(net_cfg)
        self.sigma_min = sigma_min

        if schedule == "linear":
            self.schedule = LinearSchedule()
        elif schedule == "gvp":
            assert sigma_min == 0.0, "GVP schedule does not support sigma_min."
            self.schedule = GVPSchedule()
        else:
            raise NotImplementedError(f"Schedule {schedule} not implemented.")
        
        self.sde_sampler = FlowSDE(schedule=self.schedule)

    def forward(self, x: Tensor, t: Tensor, cfg_scale=1.0, uc_cond=None, cond_key="y", **kwargs):
        if t.numel() == 1:
            t = t.expand(x.size(0))
        # _pred = self.net(x=x, t=t, **kwargs)
        _pred = forward_with_cfg(x, t, self.net, cfg_scale=cfg_scale, uc_cond=uc_cond, cond_key=cond_key, **kwargs)
        return _pred

    def ode_fn(self, t, x, **kwargs):
        return self(x=x, t=t, **kwargs)

    def generate(self, x: Tensor, sample_kwargs=None, reverse=False, return_intermediates=False, **kwargs):
        """
        Args:
            x: source minibatch (bs, *dim)
            sample_kwargs: dict, additional sampling arguments for the solver
                num_steps: int, number of steps to take
                cfg_scale: float, scale for the classifier-free guidance
                uc_cond: torch.Tensor, unconditional conditioning information (1, *dim) or (bs, *dim)
                cond_key: str, key for the conditional information
                intermediate_freq: int, frequency of intermediate outputs
                use_sde: if true, use SDE sampling instead of ODE
                __ ODE Sampler __:
                    method: str, method for the ODE solver (see torchdiffeq)
                    atol/rtol: float, absolute and relative tolerance for the ODE solver
                __ SDE Sampler __:
                    method: str, method for the SDE solver (euler, heun)
                    diffusion_form: str, form of the diffusion coefficient (sigma, SBDM, ...)
                    diffusion_norm: float, magnitude of the diffusion coefficient (default 1.0)
                    last_step: str, type of the last step (Mean, Tweedie, Euler)
                    last_step_size: float, size of the last step (default 0.04)
                    progress: bool, whether to show a progress bar
            reverse: bool, whether to reverse the direction of the flow. If True,
                we map from x1 -> x0, otherwise we map from x0 -> x1.
            n_intermediates: int, number of intermediate points to return.
            kwargs: additional arguments for the network (e.g. conditioning information).
        """
        sample_kwargs = sample_kwargs or {}

        # timesteps
        num_steps = sample_kwargs.get("num_steps", 50)
        t = torch.linspace(0, 1, num_steps, dtype=x.dtype).to(x.device)
        t = 1 - t if reverse else t

        # include classifier-free guidance
        cfg_kwargs = dict(
            cfg_scale=sample_kwargs.get("cfg_scale", 1.0),
            uc_cond=sample_kwargs.get("uc_cond", None),
            cond_key=sample_kwargs.get("cond_key", "y"),
        )

        # SDE sampling
        if sample_kwargs.get("use_sde", False):
            results = self.sde_sampler.sample(
                init=x,
                model=self.net,                         # sde_sampler already includes CFG
                sampling_method=sample_kwargs.get("method", "euler"),
                diffusion_form=sample_kwargs.get("diffusion_form", "sigma"),
                diffusion_norm=sample_kwargs.get("diffusion_norm", 1.0),
                last_step=sample_kwargs.get("last_step", "Mean"),
                last_step_size=sample_kwargs.get("last_step_size", 0.04),
                num_steps=num_steps,
                progress=sample_kwargs.get("progress", False),
                return_intermediates=True,
                **cfg_kwargs,
                **kwargs
            )

        # ODE sampling
        else:
            ode_fn = partial(self.ode_fn, **kwargs, **cfg_kwargs)
            results = odeint(
                ode_fn,
                x,
                t,
                method=sample_kwargs.get("method", "euler"),
                atol=sample_kwargs.get("atol", _ATOL),
                rtol=sample_kwargs.get("rtol", _RTOL)
            )

        if return_intermediates:
            intermediate_freq = sample_kwargs.get("intermediate_freq", 5)
            results = torch.stack([results[0], *results[1:-1:intermediate_freq], results[-1]], 0)
            return results
        return results[-1]

    """ Training """

    def compute_xt(self, x0: Tensor, x1: Tensor, t: Tensor):
        """
        Sample from the time-dependent density p_t
            xt ~ N(alpha_t * x1 + sigma_t * x0, sigma_min * I),
        according to Eq. (1) in [3] and for the linear schedule Eq. (14) in [2].

        Args:
            x0 : shape (bs, *dim), represents the source minibatch (noise)
            x1 : shape (bs, *dim), represents the target minibatch (data)
            t  : shape (bs,) represents the time in [0, 1]
        Returns:
            xt : shape (bs, *dim), sampled point along the time-dependent density p_t
        """
        t = pad_v_like_x(t, x0)
        alpha_t = self.schedule.alpha_t(t)
        sigma_t = self.schedule.sigma_t(t)
        xt = alpha_t * x1 + sigma_t * x0
        if self.sigma_min > 0:
            xt += self.sigma_min * torch.randn_like(xt)
        return xt

    def compute_ut(self, x0: Tensor, x1: Tensor, t: Tensor):
        """
        Compute the time-dependent conditional vector field
            ut = alpha_dt_t * x1 + sigma_dt_t * x0,
        see Eq. (7) in [3].

        Args:
            x0 : Tensor, shape (bs, *dim), represents the source minibatch (noise)
            x1 : Tensor, shape (bs, *dim), represents the target minibatch (data)
            t  : FloatTensor, shape (bs,) represents the time in [0, 1]
        Returns:
            ut : conditional vector field
        """
        t = pad_v_like_x(t, x0)
        alpha_dt_t = self.schedule.alpha_dt_t(t)
        sigma_dt_t = self.schedule.sigma_dt_t(t)
        return alpha_dt_t * x1 + sigma_dt_t * x0

    def training_losses(self, x1: Tensor, x0: Tensor = None, **cond_kwargs):
        """
        Args:
            x1: shape (bs, *dim), represents the target minibatch (data)
            x0: shape (bs, *dim), represents the source minibatch, if None
                we sample x0 from a standard normal distribution.
            cond_kwargs: additional arguments for the conditional flow
                network (e.g. conditioning information)
        Returns:
            loss: scalar, the training loss for the flow model
        """
        if x0 is None:
            x0 = torch.randn_like(x1)

        bs, dev, dtype = x1.shape[0], x1.device, x1.dtype

        # Sample time t from uniform distribution U(0, 1)
        t = torch.rand(bs, device=dev, dtype=dtype)

        # sample xt and ut
        xt = self.compute_xt(x0=x0, x1=x1, t=t)
        ut = self.compute_ut(x0=x0, x1=x1, t=t)
        vt = self.forward(x=xt, t=t, **cond_kwargs)

        return (vt - ut).square().mean()
