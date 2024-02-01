#!/bin/sh

# --------------- Celeb ------------------
python eval_toolbox/calc_metrics.py --metrics=pr50k3_full --data=./real_samples/celeba_256/ --mirror=1 --gen_data=sample/DiM-B-2-0001200-size-256-vae-ema-cfg-1-seed-0/ --img_resolution=256
python eval_toolbox/calc_metrics.py --metrics=fid50k3_full --data=./real_samples/celeba_256/ --mirror=1 --gen_data=sample/DiM-B-2-0001200-size-256-vae-ema-cfg-1-seed-0/ --img_resolution=256

