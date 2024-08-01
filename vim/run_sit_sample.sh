export MASTER_PORT=11121

python vim/sample_sit.py ODE \
    --model DiM-L/2 \
    --image-size 256 \
    --ckpt results/imnet256/0000510.pt \
    --global-batch-size 27 \
    --path-type GVP \
    --num-classes 1001 \
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
    --cfg-scale 4. \
    # --compute-nfe \
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
