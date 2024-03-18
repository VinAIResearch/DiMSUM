# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""

import datetime
import math
import sys
from pathlib import Path
import gc

import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from datasets_prep import get_dataset

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

from tqdm import tqdm

from create_model import create_model
from ptflops import get_model_complexity_info
from transport import create_transport, Sampler

eval_import_path = (Path(__file__).parent.parent / "eval_toolbox").resolve().as_posix()
sys.path.append(eval_import_path)
import dnnlib
from pytorch_fid import metric_main, metric_utils

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    experiment_index = args.exp
    model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    experiment_dir = f"{args.results_dir}/{experiment_index}-{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    sample_dir = f"{experiment_dir}/samples"
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = create_model(args) # mamba_models[args.model]()
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
        path_args={"diffusion_form": args.diffusion_form},
    )  # default: velocity; 
    transport_sampler = Sampler(transport)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"iDiM Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=1e-5, verbose=True)

    if args.resume or os.path.exists(os.path.join(checkpoint_dir, "content.pth")):
        checkpoint_file = os.path.join(checkpoint_dir, "content.pth")
        checkpoint = torch.load(checkpoint_file, map_location=torch.device(f'cuda:{device}'))
        init_epoch = checkpoint["epoch"]
        epoch = init_epoch
        model.module.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["opt"])
        ema.load_state_dict(checkpoint["ema"])
        train_steps = checkpoint["train_steps"]

        logger.info("=> resume checkpoint (epoch {})".format(checkpoint["epoch"]))
        del checkpoint

    elif args.model_ckpt and os.path.exists(args.model_ckpt):
        checkpoint = torch.load(args.model_ckpt, map_location=torch.device(f'cuda:{device}'))
        epoch = int(os.path.split(args.model_ckpt)[-1].split(".")[0])
        init_epoch = epoch
        model.module.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        opt.load_state_dict(checkpoint["opt"])
        train_steps = 0

        logger.info("=> loaded checkpoint (epoch {})".format(epoch))
        del checkpoint
    else:
        init_epoch = 0
        train_steps = 0
    requires_grad(ema, False)

    dataset = get_dataset(args)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.datadir})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()
    use_label = True if "imagenet" in args.dataset else False

    # Create sampling noise & label
    sample_bs = 4
    zs = torch.randn(sample_bs, 4, latent_size, latent_size, device=device)
    ys = None if not use_label else torch.randint(args.num_classes, size=(sample_bs,), device=device)
    use_cfg = args.cfg_scale > 1.0
    # Setup classifier-free guidance:
    if use_cfg:
        zs = torch.cat([zs, zs], 0)
        y_null = torch.tensor([args.num_classes] * sample_bs, device=device)
        ys = torch.cat([ys, y_null], 0)
        sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
        model_fn = ema.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = ema.forward

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(init_epoch, args.epochs+1):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in tqdm(loader):
            x = x.to(device)
            y = None if not use_label else y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            model_kwargs = dict(y=y)
            loss_dict = transport.training_losses(model, x, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

        # if not args.no_lr_decay:
        #     scheduler.step()

        if rank == 0:
            # latest checkpoint
            if epoch % args.save_content_every == 0:
                logger.info("Saving content.")
                content = {
                    "epoch": epoch + 1,
                    "train_steps": train_steps,
                    "args": args,
                    "model": model.module.state_dict(),
                    "opt": opt.state_dict(),
                    "ema": ema.state_dict(),
                }
                torch.save(content, os.path.join(checkpoint_dir, "content.pth"))

            # Save DiT checkpoint:
            if epoch % args.ckpt_every == 0 and epoch > 0:
                checkpoint = {
                    "epoch": epoch + 1,
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/{epoch:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            dist.barrier()

        if rank == 0 and epoch % args.plot_every == 0:
            logger.info("Generating EMA samples...")
            with torch.no_grad():
                sample_fn = transport_sampler.sample_ode() # default to ode sampling
                samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
                dist.barrier()
                if use_cfg: #remove null samples
                    samples, _ = samples.chunk(2, dim=0)
                samples = vae.decode(samples / 0.18215).sample

            # Save and display images:
            save_image(samples, f"{sample_dir}/image_{epoch:07d}.jpg", nrow=4, normalize=True, value_range=(-1, 1))
            del samples

        if epoch % args.eval_every == 0 and epoch > 0 or epoch == args.epochs - 1:
            ref_dir = Path(args.eval_refdir)
            if ref_dir.exists():
                n = args.eval_bs
                using_cfg = args.eval_cfg_scale > 1.0
                global_batch_size = n * dist.get_world_size()
                total_samples = int(
                    math.ceil(args.eval_nsamples / global_batch_size)
                    * global_batch_size
                )
                samples_needed_this_gpu = int(total_samples // dist.get_world_size())
                iterations = int(samples_needed_this_gpu // n)
                pbar = range(iterations)
                pbar = tqdm(pbar) if rank == 0 else pbar
                total = 0
                p = Path(experiment_dir) / f"fid{args.eval_nsamples}"
                p.mkdir(exist_ok=True, parents=True)
                model.eval()
                for _ in pbar:
                    # Sample inputs:
                    z = torch.randn(
                        n, 4, latent_size, latent_size, device=device
                    )
                    y = None if not use_label else torch.randint(args.num_classes, size=(sample_bs,), device=device)
                    # Setup classifier-free guidance:
                    if use_cfg:
                        z = torch.cat([z, z], 0)
                        y_null = torch.tensor([args.num_classes] * n, device=device)
                        y = torch.cat([y, y_null], 0)
                        sample_model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                        model_eval_fn = model.forward_with_cfg
                    else:
                        sample_model_kwargs = dict(y=y)
                        model_eval_fn = model.forward

                    # Sample images:
                    with torch.no_grad():
                        sample_fn = transport_sampler.sample_ode() # default to ode sampling
                        samples = sample_fn(z, model_eval_fn, **sample_model_kwargs)[-1]

                    if using_cfg:
                        samples, _ = samples.chunk(
                            2, dim=0
                        )  # Remove null class samples

                    samples = vae.decode(samples / 0.18215).sample
                    samples = (
                        torch.clamp(127.5 * samples + 128.0, 0, 255)
                        .permute(0, 2, 3, 1)
                        .to("cpu", dtype=torch.uint8)
                        .numpy()
                    )

                    # Save samples to disk as individual .png files
                    for i, sample in enumerate(samples):
                        index = i * dist.get_world_size() + rank + total
                        if index >= args.eval_nsamples:
                            break
                        pp = p / f"{index:06d}.jpg"
                        Image.fromarray(sample).save(pp.as_posix())
                    total += global_batch_size

                model.train()
                eval_args = dnnlib.EasyDict()
                eval_args.dataset_kwargs = dnnlib.EasyDict(
                    class_name="training.dataset.ImageFolderDataset",
                    path=ref_dir.as_posix(),
                    xflip=True,
                )
                eval_args.gen_dataset_kwargs = dnnlib.EasyDict(
                    class_name="training.dataset.ImageFolderDataset",
                    path=p.resolve().as_posix(),
                    xflip=True,
                )
                progress = metric_utils.ProgressMonitor(verbose=True)
                if rank == 0:
                    print("Calculating FID...")
                result_dict = metric_main.calc_metric(metric="fid2k_full", 
                                                    dataset_kwargs=eval_args.dataset_kwargs,
                                                    num_gpus=dist.get_world_size(),
                                                    rank=rank, 
                                                    device=device,
                                                    progress=progress,
                                                    gen_dataset_kwargs=eval_args.gen_dataset_kwargs,
                                                    cache=True)
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=p.as_posix(), snapshot_pkl=p.as_posix())
                del result_dict, samples
                gc.collect()
                torch.cuda.empty_cache()
            else:
                print(f"Reference directory {ref_dir} does not exist, skip eval")
            dist.barrier()
        

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    logger.info("Done!")
    cleanup()


def none_or_str(value):
    if value == 'None':
        return None
    return value


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, default="MambaDiffV1_XL_2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-in-channels", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=1.)
    parser.add_argument("--label-dropout", type=float, default=-1)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=25)
    parser.add_argument("--save-content-every", type=int, default=5)
    parser.add_argument("--plot-every", type=int, default=5)
    parser.add_argument("--model-ckpt", type=str, default='')
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--learn-sigma", action="store_true")
    parser.add_argument("--bimamba-type", type=str, default="v2", choices=['v2', 'none'])
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pe-type", type=str, default="ape", choices=["ape", "cpe", "rope"])
    parser.add_argument("--block-type", type=str, default="linear", choices=["linear", "raw"])
    parser.add_argument("--no-lr-decay", action='store_true', default=False)


    group = parser.add_argument_group("Eval")
    group.add_argument("--eval-every", type=int, default=100)
    group.add_argument("--eval-refdir", type=str, default=None)
    group.add_argument("--eval-nsamples", type=int, default=1000)
    group.add_argument("--eval-bs", type=int, default=4)
    group.add_argument("--eval-cfg-scale", type=float, default=1.0)

    group = parser.add_argument_group("MoE arguments")
    group.add_argument("--num-moe-experts", type=int, default=8)
    group.add_argument("--mamba-moe-layers", type=none_or_str, nargs="*", default=None)
    group.add_argument("--is-moe", action="store_true")
    group.add_argument("--routing-mode", type=str, choices=['sinkhorn', 'top1', 'top2', 'sinkhorn_top2'], default='top1')
    group.add_argument("--gated-linear-unit", action="store_true")

    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    group.add_argument("--sample-eps", type=float)
    group.add_argument("--train-eps", type=float)
    group.add_argument("--diffusion-form", type=str, default="none", \
                            choices=["none", "constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing", "log"],\
                            help="form of diffusion coefficient in the SDE")

    args = parser.parse_args()
    main(args)
