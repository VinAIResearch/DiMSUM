import argparse
from PIL import Image

import torch as th
import torchvision.transforms as transforms
import torchvision

from diffusers.models import AutoencoderKL

from vim.transport import create_transport, Sampler


def none_or_str(value):
    if value == 'None':
        return None
    return value

parser = argparse.ArgumentParser()
group = parser.add_argument_group("Transport arguments")
group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
group.add_argument("--sample-eps", type=float)
group.add_argument("--train-eps", type=float)
group.add_argument("--diffusion-form", type=str, default="none", \
                        choices=["none", "constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing", "log"],\
                        help="form of diffusion coefficient in the SDE")
group.add_argument("--t-sample-mode", type=str, default="uniform")
group.add_argument("--use-blurring", action="store_true")
group.add_argument("--blur-sigma-max", type=int, default=3)
group.add_argument("--blur-upscale", type=int, default=4)

args = parser.parse_args()

args.use_blurring = False
args.blur_upscale = 4
args.blur_sigma_max = 2

device = 'cuda:0'
vae = AutoencoderKL.from_pretrained(f"../stabilityai/sd-vae-ft-ema").to(device)
transport = create_transport(
    args.path_type,
    args.prediction,
    args.loss_weight,
    args.train_eps,
    args.sample_eps,
    path_args={
        'diffusion_form': args.diffusion_form, 
        'use_blurring': args.use_blurring, 
        'blur_sigma_max': args.blur_sigma_max, 
        'blur_upscale': args.blur_upscale},
    t_sample_mode=args.t_sample_mode,
)  # default: velocity; 

img = Image.open("real_samples/celeba_256/1070.jpg")
transform = transforms.Compose([ 
    transforms.PILToTensor() 
]) 
to_0_1 = lambda x: (x+1)/2.
x1 = (transform(img).to(device)[None,] / 255 - 0.5) / 0.5
with th.no_grad():
    # Map input images to latent space + normalize latents:
    x1 = vae.encode(x1).latent_dist.sample().mul_(0.18215)

xt_list = []
sample_list = []
for i in range(10):
    t = th.tensor([i/9], device=device)[None,]
    x0 = th.randn_like(x1)
    t, xt, ut = transport.path_sampler.plan(t, x0, x1)
    # xt_list.append(to_0_1(xt[0]))
    xt_list.append(xt[0, :3])
    print(xt.shape, i)
    sample = vae.decode(xt / 0.18215).sample
    sample_list.append(to_0_1(sample[0, :3]))
torchvision.utils.save_image(xt_list, f"xt_latent_blurup{args.blur_upscale}_blurmax{args.blur_sigma_max}.jpg", normalize=False, nrow=10)
torchvision.utils.save_image(sample_list, f"xt_latent2image_blurup{args.blur_upscale}_blurmax{args.blur_sigma_max}.jpg", normalize=False, nrow=10)