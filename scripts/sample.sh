## CelebA 256 
python vim/sample_sit.py ODE \
    --model DiM-L/2 \
    --image-size 256 \
    --ckpt celeb256_225ep.pt \
    --global-batch-size 32 \
    --path-type GVP \
    --num-classes 1 \
    --sampling-method dopri5 \
    --diffusion-form none \
    --num-sampling-steps 250 \
    --block-type combined \
    --bimamba-type none \
    --rms-norm \
    --fused-add-norm \
    --learnable-pe \
    --cond-mamba \
    --use-attn-every-k-layers 4 \
    # --compute-nfe \
    # --measure-time \

# ## Church
# python vim/sample_sit.py ODE \
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