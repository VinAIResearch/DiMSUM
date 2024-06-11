#!/bin/sh

# OUTPUT=samples-50k/latent_imagenet_256_tuned_from_epoch173/DiM-L-2-0000015-cfg-1.0-128-ODE-250-dopri5
# REAL=real_samples/imagenet_256_crop/

# OUTPUT=samples-50k/idiml2_linear_alterorders_celeb256_GVP_condmamba_zigmasetting_nd4_attnevery4/DiM-L-2-0000225-cfg-1.0-128-ODE-250-dopri5/
REAL=/lustre/scratch/client/scratch/research/group/anhgroup/haopt12/real_samples/celeba_256/
# 
# # ,pr50k3_full
# # --------------- FID/PR ------------------
# python eval_toolbox/calc_metrics.py --metrics=fid50k_full --data=$REAL --mirror=1 --gen_data=$OUTPUT --img_resolution=256 --run_dir $OUTPUT

 
for i in '0120000'
do
    echo "Model $i"
    OUTPUT=/lustre/scratch/client/vinai/users/haopt12/zigma/samples/myceleba256_uncond_zigzagN8_b1_pe2_${i}_llh0_bs128_ODE_250_dopri5

    python eval_toolbox/calc_metrics.py --metrics=fid50k_full --data=$REAL --mirror=1 --gen_data=$OUTPUT --img_resolution=256 --run_dir $OUTPUT

done
