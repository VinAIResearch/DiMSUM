import os
import time
import subprocess

import numpy as np
import pandas as pd

slurm_template = """#!/bin/bash -e
#SBATCH --job-name={job_name}
#SBATCH --output={slurm_output}/slurm_%A.out
#SBATCH --error={slurm_output}/slurm_%A.err
#SBATCH --gpus={num_gpus}
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=36G
#SBATCH --cpus-per-gpu=8
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.haopt12@vinai.io
#SBATCH --ntasks=1

# module purge
# module load python/miniconda3/miniconda3
# eval "$(conda shell.bash hook)"
# conda activate ../envs/mamba

export MASTER_PORT={master_port}
export WORLD_SIZE=1

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

export PYTHONPATH=$(pwd):$PYTHONPATH

export MODEL_TYPE={model_type}
export EPOCH_ID={epoch}
export EXP={exp}
export OUTPUT_LOG={output_log}

echo "----------------------------"
echo $MODEL_TYPE $EPOCH_ID $EXP {method} {num_steps}
echo "----------------------------"

CUDA_VISIBLE_DEVICES={device} torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node={num_gpus} vim/sample_sit_ddp.py {sampler} \
    --model $MODEL_TYPE \
    --per-proc-batch-size 64 \
    --image-size {image_size} \
    --ckpt {ckpt_root}/{epoch:07d}.pt \
    --num-fid-samples {num_fid_samples} \
    --path-type GVP \
    --num-classes 1001 \
    --sampling-method {method} \
    --num-sampling-steps {num_steps} \
    --diffusion-form {diff_form} \
    --sample-dir {sample_dir} \
    --block-type combined \
    --bimamba-type none \
    --eval-refdir {eval_refdir} \
    --eval-metric {eval_metric} \
    --rms-norm \
    --fused-add-norm \
    --learnable-pe \
    --cond-mamba \
    --use-attn-every-k-layers 4 \
    --cfg-scale {cfg_scale} \
    --image-ext png \
    # --use-final-norm \
    # --enable-fourier-layers \
    # --scanning-continuity \


# CUDA_VISIBLE_DEVICES=0 python eval_toolbox/calc_metrics.py \
#     --metrics=fid10k_full,pr10k3_full 
#     --data={real_data} 
#     --mirror=1 
#     --gen_data=samples/{exp}/
#     --img_resolution=256
#     --run_dir={slurm_output}

"""

###### ARGS
model_type = ["DiM-L/2", "DiM-L/4"][0] # or "DiT-L/2" or "adm"
exp = "imnet256"
ckpt_root = f"results/{exp}/" # f"results/{exp}/checkpoints/"
real_data = ["real_samples/celeba_256/", "../data/data1024x1024/", "../MambaDiff/real_samples/imagenet_256_crop/"][-1]
image_size = [256, 1024][0]
num_fid_samples = 50_000
eval_metric = "fid{num_samples}k_full,pr{num_samples}k3_full".format(num_samples=num_fid_samples//1000)
sample_dir = f"samples-{num_fid_samples//1000}k-png/{exp}"
BASE_PORT = 18036
num_gpus = 2
device = "0,1"

config = pd.DataFrame({
    "epochs": [510]*3,
    "num_steps": [250]*3,
    "methods": ['dopri5']*3,
    "cfg_scale": [1, 1.5, 1.4],
    "diff_form": ["none"]*3,
    "sampler": ['ODE']*3,
})
print(config)

###################################
slurm_file_path = f"/lustre/scratch/client/vinai/users/haopt12/DiMSUMv1/slurm_scripts/{exp}/run2.sh"
slurm_output = f"/lustre/scratch/client/vinai/users/haopt12/DiMSUMv1/slurm_scripts/{exp}/"
output_log = f"{slurm_output}/log"
os.makedirs(slurm_output, exist_ok=True)
job_name = "test"

for idx, row in config.iterrows():
    # device = str(idx % 2)
    # slurm_file_path = f"/lustre/scratch/client/vinai/users/haopt12/cnf_flow/slurm_scripts/{exp}/run{device}.sh"
    slurm_command = slurm_template.format(
        job_name=job_name,
        model_type=model_type,
        exp=exp,
        epoch=row.epochs,
        master_port=str(BASE_PORT+idx),
        slurm_output=slurm_output,
        num_gpus=num_gpus,
        output_log=output_log,
        method=row.methods,
        num_steps=row.num_steps,
        device=device,
        cfg_scale=row.cfg_scale,
        diff_form=row.diff_form,
        sampler=row.sampler,
        real_data=real_data,
        ckpt_root=ckpt_root,
        eval_refdir=real_data,
        eval_metric=eval_metric,
        image_size=image_size,
        num_fid_samples=num_fid_samples,
        sample_dir=sample_dir,
    )
    mode = "w" if idx == 0 else "a"
    # mode = "a"
    with open(slurm_file_path, mode) as f:
        f.write(slurm_command)
print("Slurm script is saved at", slurm_file_path)

# print(f"Summited {slurm_file_path}")
# subprocess.run(['sbatch', slurm_file_path])
