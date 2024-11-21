import argparse
import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


sys.path.append("./dimsum")
from dimsum.datasets_prep.lmdb_datasets import LMDBDataset
from dimsum.datasets_prep.lsun import LSUN


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extract dataset")
    parser.add_argument("--dataset", default="cifar10", help="name of dataset")
    parser.add_argument("--datadir", default="./data")
    parser.add_argument("--save_dir", default="real_samples/")

    parser.add_argument("--image_size", type=int, default=32, help="size of image")
    parser.add_argument("--batch_size", type=int, default=100, help="size of image")

    args = parser.parse_args()

    device = "cpu"

    if args.dataset == "lsun_church":
        train_transform = transforms.Compose(
            [
                transforms.Resize(args.image_size),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_data = LSUN(root=args.datadir, classes=["church_outdoor_train"], transform=train_transform)
        subset = list(range(0, 120000))
        dataset = torch.utils.data.Subset(train_data, subset)
    elif args.dataset == "celeba_256":
        train_transform = transforms.Compose(
            [
                transforms.Resize(args.image_size),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = LMDBDataset(root=args.datadir, name="celeba", train=True, transform=train_transform)

    save_dir = "./{}/{}/".format(args.save_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,  # cpu_count(),
    )
    for i, (x, _) in enumerate(tqdm(dataloader)):
        x = x.to(device, non_blocking=True)
        x = (x + 1.0) / 2.0  # move to 0 - 1
        assert 0 <= x.min() and x.max() <= 1
        for j, x in enumerate(x):
            index = i * args.batch_size + j
            torchvision.utils.save_image(x, "{}/{}.jpg".format(save_dir, index))
        print("Generate batch {}".format(i))
    print("Save images in {}".format(save_dir))
