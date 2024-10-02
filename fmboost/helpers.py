import torch
import random
import warnings
import importlib
import numpy as np
import torch.nn as nn
from functools import partial
from inspect import isfunction


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def un_normalize_ims(ims):
    """ Convert from [-1, 1] to [0, 255] """
    ims = ((ims * 127.5) + 127.5).clip(0, 255).to(torch.uint8)
    return ims


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def load_partial_from_config(config):
    return partial(get_obj_from_str(config['target']),**config.get('params',dict()))


def load_model_from_config(config, ckpt, verbose=False, ignore_keys=[]):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"] if 'state_dict' in pl_sd else pl_sd
    keys = list(sd.keys())
    for k in keys:
        for ik in ignore_keys:
            if ik and k.startswith(ik):
                print("Deleting key {} from state_dict.".format(k))
                del sd[k]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    print(f'Missing {len(m)} keys and unexpecting {len(u)} keys')
    # model.cuda()
    # model.eval()
    return model

def load_model_from_ckpt(model, ckpt, verbose=False, ignore_keys=[]):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"] if 'state_dict' in pl_sd else pl_sd
    keys = list(sd.keys())
    for k in keys:
        for ik in ignore_keys:
            if ik and k.startswith(ik):
                print("Deleting key {} from state_dict.".format(k))
                del sd[k]
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    print(f'Missing {len(m)} keys and unexpecting {len(u)} keys')
    # model.cuda()
    # model.eval()
    return model


def load_model_weights(model, ckpt_path, strict=True, verbose=True):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if verbose:
        print("-" * 40)
        print(f"{'Loading weights':<16}: {ckpt_path}")
        print(f"{'Model':<16}: {model.__class__.__name__}")
        if "global_step" in ckpt:
            print(f"{f'Global Step':<16}: {ckpt['global_step']:,}")
        print(f"{'Strict':<16}: {'True' if strict else 'False'}")
        print("-" * 40)
    sd = ckpt["state_dict"] if 'state_dict' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if len(missing) > 0:
        warnings.warn(f"Load model weights - missing keys: {len(missing)}")
        if verbose:
            print(missing)
    if len(unexpected) > 0:
        warnings.warn(f"Load model weights - unexpected keys: {len(unexpected)}")
        if verbose:
            print(unexpected)
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def bool2str(b):
    return "True" if b else "False"


def resize_ims(x: torch.Tensor, size: int, mode: str = "bilinear", **kwargs):
    # for the sake of backward compatibility
    if mode in ["conv", "noise_upsampling", "decoder_features"]:
        return nn.functional.interpolate(x, size=size, mode="bilinear", **kwargs)
    # idea: blur image before down-sampling
    return nn.functional.interpolate(x, size=size, mode=mode, **kwargs)


def freeze(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False
