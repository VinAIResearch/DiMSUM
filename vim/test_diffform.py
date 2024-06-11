# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
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
from transport import create_transport, Sampler

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
    # dist.init_process_group("nccl")
    # assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    # rank = dist.get_rank()
    # device = rank % torch.cuda.device_count()
    device = 0
    # seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(args.global_seed)
    torch.cuda.set_device(device)
    # print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    # model = create_model(args) # mamba_models[args.model]()
    # # Note that parameter initialization is done within the DiT constructor
    # ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    # model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
        path_args={"diffusion_form": args.diffusion_form},
    )  # default: velocity; 
    transport_sampler = Sampler(transport)
    vae = AutoencoderKL.from_pretrained(f"./vim/stabilityai/sd-vae-ft-{args.vae}").to(device)
    # print(f"iDiM Parameters: {sum(p.numel() for p in model.parameters()):,}")


    # Setup data:
    breakpoint()

    x = torch.zeros(10, 4, 64, 64).to(device)
    t = torch.linspace(0, 1, steps=10).to(device)
    diffusion = transport.path_sampler.compute_diffusion(x, t, form=args.diffusion_form)
    print(diffusion.flatten())


def none_or_str(value):
    if value == 'None':
        return None
    return value


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=25)
    parser.add_argument("--save-content-every", type=int, default=5)
    parser.add_argument("--plot-every", type=int, default=5)
    parser.add_argument("--model-ckpt", type=str, default='')
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--learn-sigma", action="store_true")
    parser.add_argument("--bimamba-type", type=str, default="v2", choices=['v2', 'none'])


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
