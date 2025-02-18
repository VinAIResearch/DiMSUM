# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
import logging
import os
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time

import numpy as np
import torch.distributed as dist
from diffusers.models import AutoencoderKL, ConsistencyDecoderVAE
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder, ImageNet
from torchvision.io import read_image
from tqdm import tqdm

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
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
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
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


#################################################################################
#                                  Training Loop                                #
#################################################################################


class DualTransform:
    def __init__(self, image_size, transform=None):
        self.image_size = image_size
        self.transform = transform

    def __call__(self, x):
        cropped_image = center_crop_arr(x, self.image_size)

        original_transformed = (
            self.transform(cropped_image) if self.transform else cropped_image
        )
        flipped_transformed = (
            self.transform(cropped_image.transpose(Image.FLIP_LEFT_RIGHT))
            if self.transform
            else cropped_image.transpose(Image.FLIP_LEFT_RIGHT)
        )

        return torch.stack([original_transformed, flipped_transformed], dim=0)


# Define the transformations you want to apply to both the original and flipped images
common_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ]
)


def custom_collate_fn(batch):
    # 'batch' is a list of tensors returned from the dataset
    # Each tensor is of shape [2, C, H, W] where '2' represents the original and flipped images

    # We concatenate along the first dimension to merge all the batches into one
    # This operation will result in a tensor of shape [2*N, C, H, W], where N is the batch size
    x = [item[0] for item in batch]  # This will be a list of [2, C, H, W] tensors
    y = [item[1] for item in batch]  # This will be a list of labels

    x = torch.cat(x, dim=0)
    labels = torch.tensor(y)

    return x, labels


def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:

    device = "cuda:0"
    torch.cuda.set_device(device)

    # Setup a feature folder:

    os.makedirs(args.features_path, exist_ok=True)
    os.makedirs(
        os.path.join(args.features_path, "imagenet256_feature_flip"), exist_ok=True
    )
    os.makedirs(
        os.path.join(args.features_path, "imagenet256_label_flip"), exist_ok=True
    )

    # Create model:
    assert (
        args.image_size % 8 == 0
    ), "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.requires_grad_(False)

    transform = DualTransform(transform=common_transforms, image_size=args.image_size)
    dataset = ImageFolder(args.data_path, transform=transform)
    N = len(dataset)
    n_processes = args.total_batch  # Number of processes
    batch_size = len(dataset) // n_processes
    if args.batch_idx == n_processes - 1:
        start_idx_dataset = args.batch_idx * batch_size
        end_idx_dataset = len(dataset)
    else:
        start_idx_dataset = args.batch_idx * batch_size
        end_idx_dataset = (args.batch_idx + 1) * batch_size
    print("start_idx:", start_idx_dataset)
    print("end_idx:", end_idx_dataset)
    subset = Subset(dataset, range(start_idx_dataset, end_idx_dataset))

    feature_path = Path(f"{args.features_path}/imagenet256_feature_flip.dat")
    label_path = Path(f"{args.features_path}/imagenet256_label_flip.dat")
    feature_path.parent.mkdir(exist_ok=True, parents=True)
    total_pairs = len(dataset)
    if feature_path.exists():
        mode = "r+"
    else:
        mode = "w+"
    print("memmap mode:", mode)
    feature_array = np.memmap(
        feature_path, dtype=np.float32, mode=mode, shape=(total_pairs * 2, 4, 32, 32)
    )
    label_array = np.memmap(label_path, dtype=int, mode=mode, shape=(total_pairs * 2))

    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        drop_last=False,
    )
    print(f"Saved at: {feature_path}")
    for i, (x, y) in enumerate(tqdm(loader)):
        if i % 1000:
            feature_array.flush()  # Ensure all data is written to disk
            label_array.flush()  # Ensure all data is written to disk

        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        start_idx = start_idx_dataset + i
        feature_array[start_idx] = x[0]
        feature_array[start_idx + N] = x[1]
        label_array[start_idx] = y[0]
        label_array[start_idx + N] = y[0]
    feature_array.flush()  # Ensure all data is written to disk
    label_array.flush()  # Ensure all data is written to disk
    del feature_array  # Optionally delete the memmap object to release the file handle
    del label_array  # Optionally delete the memmap object to release the file handle


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--features-path", type=str, default="features")
    parser.add_argument("--total-batch", type=int, default="#batch")
    parser.add_argument("--batch-idx", type=int, default="batch_idx")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument(
        "--vae", type=str, choices=["ema", "mse"], default="ema"
    )  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    main(args)
