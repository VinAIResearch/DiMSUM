#!/bin/sh

OUTPUT=samples/dimxl2_celeb256_softsnr4-DiM-XL-2/DiM-XL-2-0000300-size-256-vae-mse-cfg-1.0-seed-0

# --------------- FID/PR ------------------
python eval_toolbox/calc_metrics.py --metrics=fid50k_full,pr50k3_full --data=/lustre/scratch/client/scratch/research/group/anhgroup/haopt12/real_samples/celeba_256/ --mirror=1 --gen_data=$OUTPUT --img_resolution=256
# --------------- FLOPS ------------------
# python eval_toolbox/compute_flops.py --batch-size 1 --model DiM-B/2 --image-size 256 --learn-sigma