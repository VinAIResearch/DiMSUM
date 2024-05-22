export MASTER_PORT=10182

echo MASTER_PORT=${MASTER_PORT}

module purge
module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"
conda activate dimsum

### CelebA 256
# torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=2 vim/train_sit.py \
#         --exp dimsum_celeb256 \
#         --model DiM-L/2 \
#         --datadir ../data/celeba_256/celeba-lmdb/ \
#         --dataset celeba_256 \
#         --num-classes 1 \
#         --global-batch-size 64 \
#         --epochs 250 \
#         --path-type GVP \
#         --diffusion-form none \
#         --lr 1e-4 \
#         --block-type combined \
#         --bimamba-type none \
#         --eval-every 9999 \
#         --eval-nsamples 2_000 \
#         --eval-bs 4 \
#         --eval-refdir real_samples/celeba_256/ \
#         --rms-norm \
#         --fused-add-norm \
#         --drop-path 0.1 \
#         --learnable-pe \
#         --cond-mamba \
#         --use-attn-every-k-layers 4 \

### CelebA 512
# torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=4 vim/train_sit.py \
#         --exp dimsum_celeb512 \
#         --model DiM-L/2 \
#         --datadir ../data/celeba_512/celeba-lmdb/ \
#         --dataset celeba_512 \
#         --image-size 512 \
#         --num-classes 1 \
#         --global-batch-size 32 \
#         --epochs 250 \
#         --path-type GVP \
#         --diffusion-form none \
#         --lr 1e-4 \
#         --block-type combined \
#         --bimamba-type none \
#         --eval-every 9999 \
#         --eval-nsamples 2_000 \
#         --eval-bs 4 \
#         --eval-refdir real_samples/celeba_256/ \
#         --rms-norm \
#         --fused-add-norm \
#         --drop-path 0.1 \
#         --learnable-pe \
#         --cond-mamba \
#         --use-attn-every-k-layers 4 \


## Church
# torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=4 vim/train_sit.py \
#         --exp dimsum_church \
#         --model DiM-XL/2 \
#         --datadir ../data/lsun/ \
#         --dataset lsun_church \
#         --num-classes 1 \
#         --global-batch-size 128 \
#         --epochs 400 \
#         --path-type GVP \
#         --diffusion-form none \
#         --lr 5e-5 \
#         --block-type combined \
#         --bimamba-type none \
#         --eval-every 9999 \
#         --eval-nsamples 2_000 \
#         --eval-bs 4 \
#         --eval-refdir real_samples/lsun_church/ \
#         --rms-norm \
#         --fused-add-norm \
#         --drop-path 0.1 \
#         --learnable-pe \
#         --cond-mamba \
#         --use-attn-every-k-layers 4 \
        
### ImageNet1k 256
# torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=8 vim/train_sit.py \
#         --exp dimsum_imagenet256 \
#         --model DiM-L/2 \
#         --datadir ../data/imagenet \
#         --dataset latent_memmap_imagenet \
#         --num-classes 1000 \
#         --global-batch-size 704 \
#         --image-size 256 \
#         --epochs 500 \
#         --path-type GVP \
#         --diffusion-form none \
#         --lr 1e-4 \
#         --block-type combined \
#         --bimamba-type none \
#         --cond-mamba \
#         --eval-every 1_000 \
#         --eval-nsamples 2_000 \
#         --eval-bs 4 \
#         --eval-refdir real_samples/imagenet_256/ \
#         --rms-norm \
#         --fused-add-norm \
#         --drop-path 0.1 \
#         --label-dropout 0.15 \
#         --learnable-pe \
#         --use-attn-every-k-layers 4 \
#         --max-grad-norm 1 \
#         --ckpt-every 2 \
#         --save-content-every 2 \