#!/bin/sh

OUTPUT=samples-50k/celeba_256_gen/
REAL=real_samples/celeba_256/

# 
# --------------- FID/PR ------------------
python eval_toolbox/calc_metrics.py --metrics=fid50k_full,pr50k3_full --data=$REAL --mirror=1 --gen_data=$OUTPUT --img_resolution=256 --run_dir $OUTPUT
