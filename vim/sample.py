# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from create_model import create_model
import argparse
from time import time


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # Load model:
    latent_size = args.image_size // 8
    model = create_model(args).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt # or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps), learn_sigma=args.learn_sigma)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    use_label = True if args.num_classes > 1 else False
    use_cfg = args.cfg_scale > 1.0
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    n = len(class_labels) if use_label else args.global_batch_size

    # Create sampling noise:
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = None if not use_label else torch.tensor(class_labels, device=device)
    
    # Setup classifier-free guidance:
    if use_cfg:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([args.num_classes] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        model_fn = model.forward_with_cfg 
    else:
        model_kwargs = dict(y=y)
        model_fn = model.forward 

    # Sample images:
    start_time = time()
    samples = diffusion.p_sample_loop(
        model_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    if use_cfg: #remove null samples
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample
    print(f"Sampling took {time() - start_time:.2f} seconds.")

    # Save and display images:
    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


def none_or_str(value):
    if value == 'None':
        return None
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DiM-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-in-channels", type=int, default=4)
    parser.add_argument("--label-dropout", type=float, default=-1)
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--learn-sigma", action="store_true")
    parser.add_argument("--bimamba-type", type=str, default="v2", choices=['v2', 'none'])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--global-batch-size", type=int, default=256)

    group = parser.add_argument_group("MoE arguments")
    group.add_argument("--num-moe-experts", type=int, default=8)
    group.add_argument("--mamba-moe-layers", type=none_or_str, nargs="*", default=None)
    group.add_argument("--is-moe", action="store_true")
    group.add_argument("--routing-mode", type=str, choices=['sinkhorn', 'top1', 'top2', 'sinkhorn_top2'], default='top1')
    group.add_argument("--gated-linear-unit", action="store_true")

    args = parser.parse_args()
    main(args)
