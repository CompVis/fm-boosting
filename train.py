import os
import sys
import wandb
import torch
import signal
import argparse
import datetime
from omegaconf import OmegaConf

from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# ddp stuff
from pytorch_lightning.strategies import DDPStrategy
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

from fmboost.helpers import load_model_weights
from fmboost.helpers import count_params, exists
from fmboost.helpers import instantiate_from_config

torch.set_float32_matmul_precision('high')


def parse_args():
    parser = argparse.ArgumentParser("FMBoost")
    parser.add_argument("--config", type=str, default=None, required=True)
    parser.add_argument("--name", type=str, default="debug")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--use_wandb_offline", action="store_true")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Resumes training from a checkpoint")
    parser.add_argument("--load_weights", type=str, default=None,
                        help="Only loads the weights from a checkpoint")
    parser.add_argument("--num_nodes", type=int, default=1)
    # if -1, it uses all available GPUs
    parser.add_argument("--devices", type=int, default=-1)
    parser.add_argument("--find_unused_parameters", action="store_true")
    parser.add_argument("--p2p-disable", action="store_true")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--tqdm_refresh_rate", type=int, default=1)

    known, unknown = parser.parse_known_args()

    if exists(known.resume_checkpoint) and exists(known.load_weights):
        raise ValueError("Can't resume checkpoint and load weights at the same time.")

    # check for mistakes
    for arg in unknown:
        if arg.startswith("-"):
            raise ValueError(f"Unknown argument: {arg}")

    return known, unknown


def main():
    """ parse args """
    args, unknown = parse_args()

    """ Set seed """
    seed_everything(args.seed)
   
    """ Load config """
    cli = OmegaConf.from_dotlist(unknown)
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, cli)

    """ Setup Logging """
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_name = f"{args.name}_{now}" if exists(args.name) else now
    log_dir = os.path.join("logs", exp_name)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    use_wandb_logging = args.use_wandb or args.use_wandb_offline
    
    # setup loggers
    if use_wandb_logging:
        usr_name = os.environ.get('USER', os.environ.get('USERNAME'))
        mode = "offline" if args.use_wandb_offline else "online"
        online_logger = WandbLogger(
            dir=log_dir,
            save_dir=log_dir,
            name=exp_name,
            project="fmboost",
            tags=[usr_name, *cfg.get("tags", [])],
            config=OmegaConf.to_object(cfg),
            mode=mode,
            group="DDP"
        )
    else:
        online_logger = TensorBoardLogger(
            save_dir=log_dir,
            name="",
            version="",
            log_graph=False,
            default_hp_metric=False,
        )
    csv_logger = CSVLogger(
        log_dir,
        name="",
        version="",
        prefix="",
        flush_logs_every_n_steps=500
    )
    csv_logger.log_hyperparams(OmegaConf.to_container(cfg))
    logger = [online_logger, csv_logger]

    """ Setup dataloader """
    data = instantiate_from_config(cfg.data)

    """ Setup model """
    module = instantiate_from_config(cfg.model)

    """ Setup callbacks """
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="step{step:06d}",
        # from config
        **cfg.train.checkpoint_callback_params
    )
    callbacks = [checkpoint_callback]
    
    # add tqdm progress bar callback
    if args.tqdm_refresh_rate != 1:
        from pytorch_lightning.callbacks import TQDMProgressBar
        tqdm_callback = TQDMProgressBar(refresh_rate=args.tqdm_refresh_rate)
        callbacks.append(tqdm_callback)

    # other callbacks from config
    callbacks_cfg = cfg.train.get("callbacks", None)
    if exists(callbacks_cfg):
        for cb_cfg in callbacks_cfg:
            cb = instantiate_from_config(cb_cfg)
            callbacks.append(cb)
    
    """ Setup trainer """
    if torch.cuda.is_available():
        print("Using GPU")
        gpu_kwargs = {
            'accelerator': 'gpu',
            'strategy': ('ddp_find_unused_parameters_true' if args.find_unused_parameters else "ddp")
        }
        if args.devices > 0:
            gpu_kwargs["devices"] = args.devices
        else:       # determine automatically
            gpu_kwargs["devices"] = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
        gpu_kwargs["num_nodes"] = args.num_nodes
        if args.num_nodes >= 2:
            # multi-node hacks from
            # https://lightning.ai/docs/pytorch/stable/advanced/ddp_optimizations.html
            gpu_kwargs["strategy"] = DDPStrategy(
                gradient_as_bucket_view=True,
                ddp_comm_hook=default_hooks.fp16_compress_hook
            )
        if args.p2p_disable:
            # multi-gpu hack for heidelberg servers
            os.environ["NCCL_P2P_DISABLE"] = "1"
    else:
        print("Using CPU")
        gpu_kwargs = {'accelerator': 'cpu'}

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        **gpu_kwargs,
        # from config
        **OmegaConf.to_container(cfg.train.trainer_params)
    )
    
    """ Setup signal handler """

    # hacky way to avoid define this in the traininer module
    def stop_training_method():
        module.stop_training = False
        print("-" * 40)
        print("Try to save checkpoint to {}".format(ckpt_dir))
        module.trainer.save_checkpoint(os.path.join(ckpt_dir, "interrupted.ckpt"))
        module.trainer.should_stop = True
        module.trainer.limit_val_batches = 0
        print("Saved checkpoint.")
        print("-" * 40)

    module.stop_training_method = stop_training_method

    # once the signal was sent, the stop_training flag tells
    # the pl module get ready for save checkpoint
    def signal_handler(sig, frame):
        module.stop_training = True

    signal.signal(signal.SIGUSR1, signal_handler)

    """ Log some information """
    # compute global batchsize
    bs = cfg.data.params.batch_size
    bs = bs * gpu_kwargs.get("devices", 1)
    bs = bs * gpu_kwargs.get("num_nodes", 1)
    bs = bs * cfg.train.trainer_params.get("accumulate_grad_batches", 1)
    # log info
    some_info = {
        'Config': args.config,
        'Name': exp_name,
        'Log dir': log_dir,
        'Logging': "Wandb" if use_wandb_logging else "Tensorboard",
        'Params': count_params(module),
        'Trainer': cfg.model.get("target", "not-specified"),
        'Dataset': cfg.data.get("name", "not-specified"),
        'Batchsize': cfg.data.params.batch_size,
        'Devices': gpu_kwargs.get("devices", 1),
        'Num nodes': gpu_kwargs.get("num_nodes", 1),
        'Gradient accum': cfg.train.trainer_params.get("accumulate_grad_batches", 1),
        'Global batchsize': bs,
        'Resume ckpt': args.resume_checkpoint,
        'Load weights': args.load_weights,
        'Seed': args.seed,
        # training specific
        'Low-Res': cfg.model.params.get('low_res_size', 'not-specified'),
        'High-Res': cfg.model.params.get('high_res_size', 'not-specified'),
        'Upsampling': cfg.model.params.get('upsampling_mode', 'bilinear'),
        'Start w. Noise': cfg.model.params.get('start_from_noise', False),
        'Noising Step': cfg.model.params.get('noising_step', -1),
        'CA context': cfg.model.params.get('ca_context', False),
        'CAT context': cfg.model.params.get('concat_context', False),
    }
    
    # Make sure we don't log multiple times
    if trainer.global_rank == 0:
        print("-" * 40)
        for k, v in gpu_kwargs.items():
            print(f"{k:<16}: {v}")
        print("-" * 40)
        for k, v in some_info.items():
            if use_wandb_logging:
                online_logger.experiment.summary[k] = v
            if isinstance(v, float):
                print(f"{k:<16}: {v:.5f}")
            elif isinstance(v, int):
                print(f"{k:<16}: {v:,}")
            elif isinstance(v, bool):
                print(f"{k:<16}: {'True' if v else 'False'}")
            else:
                print(f"{k:<16}: {v}")
        print("-" * 40)
        # log called command
        if use_wandb_logging:
            online_logger.experiment.summary["command"] = " ".join(["python"] + sys.argv)
        
        # save config file
        OmegaConf.save(cfg, f"{log_dir}/config.yaml")

    """ Train """
    ckpt_path = args.resume_checkpoint if exists(args.resume_checkpoint) else None
    if exists(args.load_weights):
        module = load_model_weights(module, args.load_weights, strict=False)
    trainer.fit(module, data, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
