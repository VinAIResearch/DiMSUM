MASTER_PORT=18036

### CelebA 256
# CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 dimsum/sample_ddp.py ODE \
#     --model DiM-L/2 \
#     --per-proc-batch-size 128 \
#     --image-size 256 \
#     --ckpt celeb256_225ep.pt \
#     --num-fid-samples 50_000 \
#     --path-type GVP \
#     --num-classes 1 \
#     --sampling-method dopri5 \
#     --num-sampling-steps 250 \
#     --diffusion-form none \
#     --sample-dir samples-50k \
#     --block-type combined \
#     --bimamba-type none \
#     --eval-refdir real_samples/celeba_256 \
#     --eval-metric fid50k_full,pr50k3_full \
#     --rms-norm \
#     --fused-add-norm \
#     --learnable-pe \
#     --cond-mamba \
#     --use-attn-every-k-layers 4 \

### CelebA 512
# CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 dimsum/sample_ddp.py ODE \
#     --model DiM-L/2 \
#     --per-proc-batch-size 64 \
#     --image-size 512 \
#     --ckpt celeb512_195ep.pt \
#     --num-fid-samples 50_000 \
#     --path-type GVP \
#     --num-classes 1 \
#     --sampling-method dopri5 \
#     --num-sampling-steps 250 \
#     --diffusion-form none \
#     --sample-dir samples-50k \
#     --block-type combined \
#     --bimamba-type none \
#     --eval-refdir real_samples/celeba_512 \
#     --eval-metric fid50k_full,pr50k3_full \
#     --rms-norm \
#     --fused-add-norm \
#     --learnable-pe \
#     --cond-mamba \
#     --use-attn-every-k-layers 4 \
    
### Church 256
# CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 dimsum/sample_ddp.py ODE \
#     --model DiM-L/2 \
#     --per-proc-batch-size 128 \
#     --image-size 256 \
#     --ckpt church_395ep.pt \
#     --num-fid-samples 50_000 \
#     --path-type GVP \
#     --num-classes 1 \
#     --sampling-method dopri5 \
#     --num-sampling-steps 250 \
#     --diffusion-form none \
#     --sample-dir samples-50k \
#     --block-type combined \
#     --bimamba-type none \
#     --eval-refdir real_samples/lsun_church \
#     --eval-metric fid50k_full,pr50k3_full \
#     --rms-norm \
#     --fused-add-norm \
#     --learnable-pe \
#     --cond-mamba \
#     --use-attn-every-k-layers 4 \

### ImageNet1k 256
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 dimsum/sample_ddp.py ODE \
    --model DiM-L/2 \
    --per-proc-batch-size 64 \
    --image-size 256 \
    --ckpt imnet256_510ep.pt \
    --num-fid-samples 50_000 \
    --path-type GVP \
    --num-classes 1001 \
    --sampling-method dopri5 \
    --num-sampling-steps 250 \
    --diffusion-form none \
    --sample-dir samples-50k \
    --block-type combined \
    --bimamba-type none \
    --eval-refdir real_samples/imagenet_256 \
    --eval-metric fid50k_full,pr50k3_full \
    --rms-norm \
    --fused-add-norm \
    --learnable-pe \
    --cond-mamba \
    --use-attn-every-k-layers 4 \
    --cfg-scale 1.4 \
    --image-ext png