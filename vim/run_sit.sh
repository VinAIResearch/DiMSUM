#!/bin/sh
#SBATCH --job-name=ff # create a short name for your job
#SBATCH --output=/lustre/scratch/client/vinai/users/haopt12/vimdiff/slurms/slurm_%A.out # create a output file
#SBATCH --error=/lustre/scratch/client/vinai/users/haopt12/vimdiff/slurms/slurm_%A.err # create a error file
#SBATCH --partition=research # choose partition
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8 # 80
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

export MASTER_PORT=10119
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

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 vim/train_sit.py \
        --exp idimxl2_celeb256_gvp \
        --model DiM-XL/2 \
        --datadir ./vim/data/celeba-lmdb/ \
        --dataset celeba_256 \
        --num-classes 1 \
        --global-batch-size 32 \
        --epochs 800 \
        --path-type GVP \
        # --loss-weighting-gamma 4. \
        # --learn-sigma \
#         # --model-ckpt results/diml2_moe_celeb256-DiM-L-2/checkpoints/0000100.pt \

# CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 vim/train.py \
#         --exp diml2_moe_celeb256 \
#         --model DiM-L/2 \
#         --datadir ./vim/data/celeba-lmdb/ \
#         --dataset celeba_256 \
#         --num-classes 1 \
#         --global-batch-size 32 \
#         --epochs 800 \
#         --routing-mode top1 \
#         --model-ckpt results/diml2_moe_celeb256-DiM-L-2/checkpoints/0000100.pt \
#         --is-moe \
#         # --gated-linear-unit \

# torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 vim/train.py \
#         --exp diml2_moe_ffhq256 \
#         --model DiM-L/2 \
#         --datadir vim/data/ffhq-lmdb/ \
#         --dataset ffhq_256 \
#         --num-classes 1 \
#         --global-batch-size 32 \
#         --epochs 800 \
#         --routing-mode top1 \
#         --is-moe \
#         # --gated-linear-unit \
