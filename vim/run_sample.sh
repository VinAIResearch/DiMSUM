export MASTER_PORT=11122

# python vim/sample.py \
#     --model DiM-XL/2 \
#     --image-size 256 \
#     --ckpt results/dimxl2_celeb256_softsnr4-DiM-XL-2/checkpoints/0000775.pt \
#     --global-batch-size 4 \
#     --num-classes 1 \
#     --learn-sigma \
#     --seed 5 \
#     # --routing-mode top1 \
#     # --is-moe \
#     # --gated-linear-unit \
#     # results/diml2_moe_celeb256-DiM-L-2/checkpoints/0000775.pt \


torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 vim/sample_ddp.py \
    --model DiM-XL/2 \
    --image-size 256 \
    --ckpt results/dimxl2_celeb256_softsnr4-DiM-XL-2/checkpoints/0000300.pt \
    --per-proc-batch-size 100 \
    --num-fid-samples 10_000 \
    --num-classes 1 \
    --sample-dir samples/dimxl2_celeb256_softsnr4-DiM-XL-2/ \
    --learn-sigma \
    # --routing-mode top1 \
    # --is-moe \
#     # --gated-linear-unit \
