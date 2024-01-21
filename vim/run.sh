#!/bin/sh
#SBATCH --job-name=one # create a short name for your job
#SBATCH --output=/lustre/scratch/client/vinai/users/haopt12/swiftbrush-code/slurms/slurm_%A.out # create a output file
#SBATCH --error=/lustre/scratch/client/vinai/users/haopt12/swiftbrush-code/slurms/slurm_%A.err # create a error file
#SBATCH --partition=research # choose partition
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16 # 80
#SBATCH --mem-per-gpu=32GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10-00:00          # total run time limit (DD-HH:MM)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail          # send email when job fails
#SBATCH --mail-user=v.haopt12@vinai.io

set -x
set -e

export MASTER_PORT=10112
export WORLD_SIZE=1

export SLURM_JOB_NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')
export SLURM_NODELIST=$SLURM_JOB_NODELIST
master_address=$(echo $SLURM_JOB_NODELIST | cut -d' ' -f1)
export MASTER_ADDRESS=$master_address

echo MASTER_ADDRESS=${MASTER_ADDRESS}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${WORLD_SIZE}
echo "NODELIST="${SLURM_NODELIST}

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
#--rdzv_endpoint 0.0.0.0:8000

CUDA_VISIBLE_DEVICES=3 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:8000 --nproc_per_node=1 vim/train.py \
                                                                                            --exp layernorm \
                                                                                            --model MambaDiffV1_XL_2 \
                                                                                            --datadir ./vim/dataset/celeba-lmdb/ \
                                                                                            --dataset celeba_256 \
                                                                                            --global-batch-size 128 \