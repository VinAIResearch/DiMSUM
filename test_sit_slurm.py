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

CUDA_VISIBLE_DEVICES={device} torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node={num_gpus} vim/sample_sit_ddp.py ODE \
    --model $MODEL_TYPE \
    --per-proc-batch-size 100 \
    --image-size 256 \
    --ckpt results/idimxl2_celeb256_gvp-DiM-XL-2/checkpoints/{epoch:07d}.pt \
    --num-fid-samples 10_000 \
    --path-type GVP \
    --num-classes 0 \
    --sampling-method {method} \
    --num-sampling-steps {num_steps} \
    --diffusion-form sigma \
    --sample-dir samples/{exp} \

"""

###### ARGS
model_type = "DiM-XL/2" # or "DiT-L/2" or "adm"
exp = "idimxl2_celeb256_gvp-DiM-XL-2"
ckpt_root = f"results/{exp}/checkpoints/"
BASE_PORT = 18016
num_gpus = 1
device = "0,"

config = pd.DataFrame({
    "epochs": list(range(200, 600, 25)),
    "num_steps": [250]*len(range(200, 600, 25)),
    "methods": ['dopri5']*len(range(200, 600, 25)),
    "cfg_scale": [1.]*len(range(200, 600, 25)),
})
print(config)

###################################
slurm_file_path = f"/lustre/scratch/client/vinai/users/haopt12/vimdiff/slurm_scripts/{exp}/run2.sh"
slurm_output = f"/lustre/scratch/client/vinai/users/haopt12/vimdiff/slurm_scripts/{exp}/"
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
    )
    mode = "w" if idx == 0 else "a"
    with open(slurm_file_path, mode) as f:
        f.write(slurm_command)
print("Slurm script is saved at", slurm_file_path)

# print(f"Summited {slurm_file_path}")
# subprocess.run(['sbatch', slurm_file_path])
