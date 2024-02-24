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


echo "----------------------------"
echo {gen_data}
echo "----------------------------"

python eval_toolbox/calc_metrics.py --metrics=fid50k_full,pr50k3_full --data={real_data} --mirror=1 --gen_data={gen_data} --img_resolution=256

"""

###### ARGS
real_data = "/lustre/scratch/client/scratch/research/group/anhgroup/haopt12/real_samples/celeba_256/"

BASE_PORT = 18015
num_gpus = 1
device = "0,"

config = pd.DataFrame({
    "gen_data": [f"samples/idimxl2_celeb256_gvp-DiM-XL-2/DiM-XL-2-0000475-cfg-1.0-100-SDE-250-Euler-sigma-Mean--1" for e in range(500, 626, 25)],
})
print(config)

###################################
slurm_file_path = "/lustre/scratch/client/vinai/users/haopt12/vimdiff/slurm_sample_scripts/{exp}/run.sh"
slurm_output = "/lustre/scratch/client/vinai/users/haopt12/vimdiff/slurm_sample_scripts/{exp}/"
os.makedirs(slurm_output, exist_ok=True)
job_name = "test"

for idx, row in config.iterrows():
    # device = str(idx % 2)
    exp = row.gen_data.split('/')[1]
    slurm_file_path = slurm_file_path.format(exp=exp)
    os.makedirs(slurm_output.format(exp=exp), exist_ok=True)
    slurm_command = slurm_template.format(
        job_name=job_name,
        real_data=real_data,
        gen_data=row.gen_data,
        slurm_output=slurm_output.format(exp=exp),
        num_gpus=num_gpus,
        device=device,
        master_port=str(BASE_PORT+idx),
    )
    mode = "w" if idx == 0 else "a"
    with open(slurm_file_path, mode) as f:
        f.write(slurm_command)
print("Slurm script is saved at", slurm_file_path)

# print(f"Summited {slurm_file_path}")
# subprocess.run(['sbatch', slurm_file_path])