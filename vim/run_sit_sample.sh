export MASTER_PORT=11121

python vim/sample_sit.py SDE \
    --model DiM-XL/2 \
    --image-size 256 \
    --ckpt results/idimxl2_celeb256_gvp_difflog-DiM-XL-2/checkpoints/0000200.pt \
    --global-batch-size 4 \
    --path-type GVP \
    --num-classes 0 \
    --sampling-method Euler \
    --diffusion-form log \
    --num-sampling-steps 250 \
# results/idimxl2_celeb256_gvp_difflog-DiM-XL-2/checkpoints/0000100.pt \


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
