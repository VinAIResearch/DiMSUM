# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained SiT.
"""
import torch
from torch import nn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from download import find_model
from create_model import create_model
from transport import create_transport, Sampler
import argparse
import sys
from time import time
import numpy as np 
from tqdm import tqdm


class NFECount(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("nfe", torch.tensor(0.0))

    def forward(self, x, t, *args, **kwargs):
        self.nfe += 1.0
        return self.model(x, t, *args, **kwargs)

    def forward_with_cfg(self, x, t, *args, **kwargs):
        self.nfe += 1.0
        return self.model.forward_with_cfg(x, t, *args, **kwargs)

    def forward_with_adacfg(self, x, t, *args, **kwargs):
        self.nfe += 1.0
        return self.model.forward_with_adacfg(x, t, *args, **kwargs)
    
    def reset_nfe(self):
        self.nfe = torch.tensor(0.0)


def main(mode, args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    latent_size = args.image_size // 8
    model = create_model(args).to(device)
    # Auto-download a pre-trained model or load a custom SiT checkpoint from train.py:
    ckpt_path = args.ckpt # or f"SiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!

    if args.compute_nfe:
        # model.count_nfe = True
        model = NFECount(model).to(device) # count wrapper

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )
    sampler = Sampler(transport)
    if mode == "ODE":
        if args.likelihood:
            assert args.cfg_scale == 1, "Likelihood is incompatible with guidance"
            sample_fn = sampler.sample_ode_likelihood(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
            )
        else:
            sample_fn = sampler.sample_ode(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
                reverse=args.reverse
            )
            
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(
            sampling_method=args.sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            # last_step_size: 1/num_steps by default
            last_step_size=args.last_step_size,
            num_steps=args.num_sampling_steps,
        )
    

    vae = AutoencoderKL.from_pretrained(f"../stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    use_label = True if args.num_classes > 1 else False
    if use_label:
        real_num_classes = args.num_classes - 1 # not count uncond cls
    else:
        real_num_classes = args.num_classes
    
    use_cfg = args.cfg_scale > 1.0
    class_labels = [0] * args.global_batch_size  # [207, 360, 387, 974, 88, 979, 417, 279]
    n = len(class_labels) if use_label else args.global_batch_size

    # Create sampling noise:
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = None if not use_label else torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    if use_cfg:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([real_num_classes] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        model_fn = model.forward_with_cfg if not args.ada_cfg else model.forward_with_adacfg
    else:
        model_kwargs = dict(y=y)
        model_fn = model.forward 
    
    if args.compute_nfe:
        print("Compute nfe")
        average_nfe = 0.0
        num_trials = 30
        for i in tqdm(range(num_trials)):
            z = torch.randn(n, 4, latent_size, latent_size, device=device)
            y = None if not use_label else torch.tensor(class_labels, device=device)
            if use_cfg:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([real_num_classes] * n, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                model_fn = model.forward_with_cfg 
            else:
                model_kwargs = dict(y=y)
                model_fn = model.forward 
            _ = sample_fn(z, model_fn, **model_kwargs)[-1]
            average_nfe += model.nfe / num_trials
            model.reset_nfe()
        print(f"Average NFE over {num_trials} trials: {int(average_nfe)}")
        exit(0)

    if args.measure_time:
        print("Measure time")
        # INIT LOGGERS
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 30
        timings = np.zeros((repetitions, 1))
        # GPU-WARM-UP
        for _ in range(10):
            _ = model_fn(z, torch.ones((n), device=device), **model_kwargs)
        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in tqdm(range(repetitions)):
                starter.record()
                _ = sample_fn(z, model_fn, **model_kwargs)[-1]
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        print("Inference time: {:.2f}+/-{:.2f}ms".format(mean_syn, std_syn))
        exit(0)

    # Sample images:
    start_time = time()
    samples = sample_fn(z, model_fn, **model_kwargs)[-1]
    if use_cfg: #remove null samples
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample
    print(f"Sampling took {time() - start_time:.2f} seconds.")

    # Save and display images:
    if use_cfg:
        save_image(samples, f"sit_{class_labels[0]}_sample_cfg{args.cfg_scale}.png", nrow=3, normalize=True, value_range=(-1, 1), pad_value=1.)
    else:
        save_image(samples, "sit_sample.png", nrow=8, normalize=True, value_range=(-1, 1), pad_value=1.)


def none_or_str(value):
    if value == 'None':
        return None
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    if len(sys.argv) < 2:
        print("Usage: program.py <mode> [options]")
        sys.exit(1)
    
    mode = sys.argv[1]

    assert mode[:2] != "--", "Usage: program.py <mode> [options]"
    assert mode in ["ODE", "SDE"], "Invalid mode. Please choose 'ODE' or 'SDE'"

    # parser.add_argument("--sampler-type", type=str, default="ODE", choices=["ODE", "SDE"])
    
    parser.add_argument("--model", type=str, default="MambaDiffV1_XL_2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a SiT checkpoint (default: auto-download a pre-trained SiT-XL/2 model).")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--learn-sigma", action="store_true")
    parser.add_argument("--num-in-channels", type=int, default=4)
    parser.add_argument("--label-dropout", type=float, default=-1)
    parser.add_argument("--use-final-norm", action="store_true")
    parser.add_argument("--use-attn-every-k-layers", type=int, default=-1,)
    parser.add_argument("--not-use-gated-mlp", action="store_true")
    parser.add_argument("--ada-cfg", action="store_true", help="Use adaptive cfg as MDT")

    parser.add_argument("--bimamba-type", type=str, default="v2", choices=['v2', 'none', 'zigma_8', 'sweep_8', 'jpeg_8', 'sweep_4'])
    parser.add_argument("--pe-type", type=str, default="ape", choices=["ape", "cpe", "rope"])
    parser.add_argument("--block-type", type=str, default="linear", choices=["linear", "raw", "wave", 
        "combined", "window", "combined_fourier", "combined_einfft"])
    parser.add_argument("--cond-mamba", action="store_true")
    parser.add_argument("--scanning-continuity", action="store_true")
    parser.add_argument("--enable-fourier-layers", action="store_true")
    parser.add_argument("--rms-norm", action="store_true")
    parser.add_argument("--fused-add-norm", action="store_true")
    parser.add_argument("--drop-path", type=float, default=0.)
    parser.add_argument("--learnable-pe", action="store_true")
    parser.add_argument("--measure-time", action="store_true")
    parser.add_argument("--compute-nfe", action="store_true")

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

    if mode == "ODE":
        group = parser.add_argument_group("ODE arguments")
        group.add_argument("--sampling-method", type=str, default="dopri5", help="blackbox ODE solver methods; for full list check https://github.com/rtqichen/torchdiffeq")
        group.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
        group.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
        group.add_argument("--reverse", action="store_true")
        group.add_argument("--likelihood", action="store_true")
        # Further processing for ODE
    elif mode == "SDE":
        group = parser.add_argument_group("SDE arguments")
        group.add_argument("--sampling-method", type=str, default="Euler", choices=["Euler", "Heun"])
        group.add_argument("--diffusion-form", type=str, default="none", \
                            choices=["none", "constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing", "log"],\
                            help="form of diffusion coefficient in the SDE")
        group.add_argument("--diffusion-norm", type=float, default=1.0)
        group.add_argument("--last-step", type=none_or_str, default="Mean", choices=[None, "Mean", "Tweedie", "Euler"],\
                            help="form of last step taken in the SDE")
        group.add_argument("--last-step-size", type=float, default=-1, \
                            help="size of the last step taken")
        # Further processing for SDE
    
    args = parser.parse_known_args()[0]
    main(mode, args)
