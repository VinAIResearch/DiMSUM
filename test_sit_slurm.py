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
#SBATCH --cpus-per-gpu=16
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.haopt12@vinai.io
#SBATCH --ntasks=1

# module purge
# module load python/miniconda3/miniconda3
# eval "$(conda shell.bash hook)"
# conda activate /lustre/scratch/client/vinai/users/ngocbh8/quan/envs/flow
# cd /lustre/scratch/client/vinai/users/ngocbh8/quan/cnf_flow

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
    --per-proc-batch-size 100 \
    --image-size 256 \
    --ckpt {ckpt_root}/{epoch:07d}.pt \
    --num-fid-samples 50_000 \
    --path-type GVP \
    --num-classes 1 \
    --sampling-method {method} \
    --num-sampling-steps {num_steps} \
    --diffusion-form {diff_form} \
    --sample-dir samples/{exp} \
    --block-type linear \


# CUDA_VISIBLE_DEVICES=0 python eval_toolbox/calc_metrics.py \
#     --metrics=fid10k_full,pr10k3_full 
#     --data={real_data} 
#     --mirror=1 
#     --gen_data=samples/{exp}/
#     --img_resolution=256

"""

###### ARGS
model_type = "DiM-L/2" # or "DiT-L/2" or "adm"
exp = "idiml2_linear_celeb256_gvp-DiM-L-2"
ckpt_root = f"results/{exp}/checkpoints/"
BASE_PORT = 18016
num_gpus = 2
device = "0,1"
real_data = "/lustre/scratch/client/scratch/research/group/anhgroup/haopt12/real_samples/celeba_256/"

config = pd.DataFrame({
    "epochs": [200]*3,
    "num_steps": [250]*3,
    "methods": ['dopri5', 'Euler', 'Euler'],
    "cfg_scale": [1.]*3,
    "diff_form": ["none", "increasing-decreasing", "log"],
    "sampler": ['ODE', 'SDE', 'SDE']
})
print(config)

###################################
slurm_file_path = f"/lustre/scratch/client/vinai/users/haopt12/MambaDiff/slurm_scripts/{exp}/run2.sh"
slurm_output = f"/lustre/scratch/client/vinai/users/haopt12/MambaDiff/slurm_scripts/{exp}/"
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
        real_data=real_data,
        ckpt_root=ckpt_root,
        sampler=row.sampler,
    )
    mode = "w" if idx == 0 else "a"
    with open(slurm_file_path, mode) as f:
        f.write(slurm_command)
print("Slurm script is saved at", slurm_file_path)

# print(f"Summited {slurm_file_path}")
# subprocess.run(['sbatch', slurm_file_path])
