#!/bin/sh

OUTPUT=/lustre/scratch/client/vinai/users/haopt12/zigma/samples/myceleba256_uncond_zigzagN8_b1_pe2_0140000_llh0_bs128_ODE_250_dopri5

# --------------- FID/PR ------------------
python eval_toolbox/calc_metrics.py --metrics=fid50k_full,pr50k3_full --data=/lustre/scratch/client/scratch/research/group/anhgroup/haopt12/real_samples/celeba_256/ --mirror=1 --gen_data=$OUTPUT --img_resolution=256 --run_dir /lustre/scratch/client/vinai/users/haopt12/zigma/samples/myceleba256_uncond_zigzagN8_b1_pe2_0140000_llh0_bs128_ODE_250_dopri5
# --------------- FLOPS ------------------
# python eval_toolbox/compute_flops.py --batch-size 1 --model DiM-B/2 --image-size 256 --learn-sigma
