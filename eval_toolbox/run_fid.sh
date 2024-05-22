GEN_PATH=samples-50k/latent_imagenet_256_tuned_from_epoch173/DiM-L-2-0000020-cfg-1.0-128-ODE-250-dopri5
REAL_PATH=eval_toolbox/pytorch_fid/imagenet_stat.npy

python eval_toolbox/pytorch_fid/fid_score.py $GEN_PATH $REAL_PATH --batch-size 1024
