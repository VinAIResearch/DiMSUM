MODEL_TYPE=DiM-L/2

CUDA_VISIBLE_DEVICES=0 python eval_toolbox/compute_flops.py \
    --model ${MODEL_TYPE} \
    --batch-size 1 \
    --image-size 256 \
    --num-in-channels 4 \
    --num-classes 1 \
    --block-type combined \
    --bimamba-type none \
    --rms-norm \
    --fused-add-norm \
    --drop-path 0.1 \
    --learnable-pe \
    --cond-mamba \
    --use-attn-every-k-layers 4 \
    # --not-use-gated-mlp \


