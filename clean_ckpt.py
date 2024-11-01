import os
import glob
import torch

ckpt_path = "results/imnet256/"
outdir = "results/imnet256/"

os.makedirs(outdir, exist_ok=True)

for path in glob.glob(f"{ckpt_path}/*.pt"):
        print(path)
        ckpt = torch.load(path, map_location=torch.device('cpu'))
        tmp = {
                'epoch': ckpt['epoch'],
                'ema': ckpt['ema'],
                'args': ckpt['args'],
        }
        torch.save(tmp, os.path.join(outdir, "clean_" + os.path.basename(path)))
