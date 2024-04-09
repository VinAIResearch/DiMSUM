#!/bin/sh
#SBATCH --job-name=clip # create a short name for your job
#SBATCH --output=/lustre/scratch/client/vinai/users/haopt12/MambaDiff/slurms/slurm_%A.out # create a output file
#SBATCH --error=/lustre/scratch/client/vinai/users/haopt12/MambaDiff/slurms/slurm_%A.err # create a error file
#SBATCH --partition=research # choose partition
#SBATCH --gpus-per-node=2
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

# export NCCL_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=^docker0,lo
export MASTER_PORT=10132
export WORLD_SIZE=2
NUM_GPUs=1

SLURM_JOB_NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')
SLURM_NODELIST=$SLURM_JOB_NODELIST
master_address=$(echo $SLURM_JOB_NODELIST | cut -d' ' -f1)
export MASTER_ADDRESS=$master_address

echo MASTER_ADDRESS=${MASTER_ADDRESS}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${WORLD_SIZE}
echo "NODELIST="${SLURM_NODELIST}

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=$NUM_GPUs vim/train_sit.py \
        --exp idiml2_gatedmlp_alterorders_celeb256_GVP_condmamba_gradclip2_zigmasetting \
        --model DiM-L/2 \
        --datadir ../data/celeba_256/celeba-lmdb/ \
        --dataset celeba_256 \
        --num-classes 1 \
        --global-batch-size 32 \
        --epochs 300 \
        --path-type GVP \
        --diffusion-form none \
        --lr 1e-4 \
        --block-type linear \
        --bimamba-type none \
        --cond-mamba \
        --eval-every 50 \
        --eval-nsamples 2_000 \
        --eval-bs 4 \
        --eval-refdir real_samples/celeba_256/ \
        --rms-norm \
        --fused-add-norm \
        --drop-path 0.1 \
        --learnable-pe \
        # --use-blurring \
        # --blur-upscale 4 \
        # --blur-sigma-max 1 \
        # --enable-fourier-layers \
        # --t-sample-mode logitnormal \
        # --scanning-continuity \
        # --model-ckpt results/idiml2_gatedmlp_alterorders_celeb256_gvp_logitnormalsample/checkpoints/0000025.pt \


# torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=$NUM_GPUs vim/train_sit.py \
#         --exp idiml2_gatedmlp_alterorders_imnet256_10percent_GVP_condmamba \
#         --model DiM-L/2 \
#         --datadir ../data/imagenet1k_10percent_simclrv2 \
#         --dataset imagenet_256 \
#         --num-classes 1_000 \
#         --global-batch-size 64 \
#         --epochs 1_000 \
#         --path-type GVP \
#         --diffusion-form none \
#         --lr 1e-4 \
#         --block-type linear \
#         --eval-every 25 \
#         --eval-nsamples 2_000 \
#         --eval-bs 4 \
#         --eval-refdir ./real_samples/imagenet1k_10percent_simclrv2_256_samples \
#         --bimamba-type none \
#         --label-dropout 0.15 \
#         --cond-mamba \
#         # --t-sample-mode logitnormal \
#         # --scanning-continuity \
#         # --model-ckpt results/idiml2_gatedmlp_alterorders_celeb256_gvp_logitnormalsample/checkpoints/0000025.pt \


# CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 vim/train_sit.py \
#         --exp idimxl2_church_GVP_lr1e-5 \
#         --model DiM-XL/2 \
#         --datadir ./vim/data/lsun/ \
#         --dataset lsun_church \
#         --num-classes 1 \
#         --global-batch-size 32 \
#         --epochs 800 \
#         --path-type GVP \
#         --lr 1e-5 \
#         --resume \
#         # --bimamba-type none \
#         # --learn-sigma \
# #         # --model-ckpt results/diml2_moe_celeb256-DiM-L-2/checkpoints/0000100.pt \

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
