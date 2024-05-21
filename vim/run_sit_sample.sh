export MASTER_PORT=11121

python vim/sample_sit.py ODE \
    --model DiM-L/2 \
    --image-size 256 \
    --ckpt results/idiml2_combinedxcrossattn_alterorders_celeb256_GVP_condmamba_zigmasetting_wscanlrandtb_attnevery4/checkpoints/0000250.pt \
    --global-batch-size 100 \
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
    # --measure-time \

# torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 vim/sample_sit_ddp.py SDE \
#     --model DiM-XL/2 \
#     --image-size 256 \
#     --ckpt results/idimxl2_celeb256_gvp-DiM-XL-2/checkpoints/0000775.pt \
#     --per-proc-batch-size 4 \
#     --num-fid-samples 10_000 \
#     --path-type GVP \
#     --num-classes 0 \
#     --sampling-method Euler \
#     --diffusion-form sigma \
