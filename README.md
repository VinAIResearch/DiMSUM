<div align="center">
<h3>DiMSUM: Diffusion Mamba - A Scalable and Unified
Spatial-Frequency Method for Image Generation</h3>
</div>


## Overview
<div align="center">
<img src="assets/dim_arch.png" />
</div>

## Installation

- Python 3.10.13

  - `conda create -n dimsum python=3.10.13`

- torch 2.1.1 + cu118
  - `pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118`

- Requirements:
  - `pip install -r requirements.txt`

- Install ``causal_conv1d`` and ``mamba``
  - `conda install conda-forge::cudatoolkit-dev`
  - `cd causal_conv1d && pip install -e . && cd ..`
  - `cd mamba && pip install -e . && cd ..`

## Data
For CelebA HQ (256) and LSUN, please follow [this repo](https://github.com/NVlabs/NVAE.git) for dataset preparation.

For evaluation, please extract "jpeg" images from data first by using this command:
```bash
python eval_toolbox/resize.py main input_data_dir real_samples/dataname
```

## Pretrained Weights
We provide pretrained checkpoints of CelebA 256 & LSUN Church for quick testing at [here](https://drive.google.com/drive/folders/1SAE3uSrXv1FNoC92KhFefy24ZSkxjRY5?usp=sharing).

## Train
Comment/Uncomment command lines for desired dataset, then run:
`bash scripts/train.sh`

## Evaluation
To sampe images from pretrained checkpoints, run:
```bash
bash scripts/sample.sh
```

To evaluate, select a relevant command and run:
```bash
bash scripts/eval.sh
```