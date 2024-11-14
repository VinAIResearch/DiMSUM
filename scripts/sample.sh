# ## CelebA 256 
# python dimsum/sample.py ODE \
#     --model DiM-L/2 \
#     --image-size 256 \
#     --ckpt celeb256_225ep.pt \
#     --global-batch-size 32 \
#     --path-type GVP \
#     --num-classes 1 \
#     --sampling-method dopri5 \
#     --diffusion-form none \
#     --num-sampling-steps 250 \
#     --block-type combined \
#     --bimamba-type none \
#     --rms-norm \
#     --fused-add-norm \
#     --learnable-pe \
#     --cond-mamba \
#     --use-attn-every-k-layers 4 \
#     # --compute-nfe \
#     # --measure-time \

# ## Church
# python dimsum/sample.py ODE \
#     --model DiM-L/2 \
#     --image-size 256 \
#     --ckpt church_395ep.pt \
#     --global-batch-size 32 \
#     --path-type GVP \
#     --num-classes 1 \
#     --sampling-method dopri5 \
#     --diffusion-form none \
#     --num-sampling-steps 250 \
#     --block-type combined \
#     --bimamba-type none \
#     --rms-norm \
#     --fused-add-norm \
#     --learnable-pe \
#     --cond-mamba \
#     --use-attn-every-k-layers 4 \
#     # --compute-nfe \
#     # --measure-time \

python dimsum/sample.py ODE \
    --model DiM-L/2 \
    --image-size 256 \
    --ckpt imnet256_510ep.pt \
    --global-batch-size 64 \
    --path-type GVP \
    --num-classes 1001 \
    --sampling-method dopri5 \
    --num-sampling-steps 250 \
    --diffusion-form none \
    --sample-dir  \
    --block-type combined \
    --bimamba-type none \
    --eval-refdir real_samples/imagenet_256 \
    --eval-metric fid50k_full,pr50k3_full \
    --rms-norm \
    --fused-add-norm \
    --learnable-pe \
    --cond-mamba \
    --use-attn-every-k-layers 4 \
    --cfg-scale 4.0 \