import torch
import torchvision.transforms as transforms
from datasets_prep.data_transforms import center_crop_arr
from datasets_prep.latent_datasets import LatentDataset, LatentMemmapDataset
from datasets_prep.lmdb_datasets import LMDBDataset
from datasets_prep.lsun import LSUN
from torchvision.datasets import CIFAR10, ImageNet


def get_dataset(args):
    if args.dataset == "cifar10":
        dataset = CIFAR10(
            args.datadir,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )

    elif args.dataset == "imagenet_256":
        dataset = ImageNet(
            args.datadir,
            split="train",
            transform=transforms.Compose(
                [
                    transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )

    elif args.dataset.startswith("latent_imagenet"):
        dataname = ["imagenet256", "imagenet512"]["512" in args.dataset]
        dataset = LatentDataset(
            "imagenet",
            features_dir=f"{args.datadir}/{dataname}_features",
            labels_dir=f"{args.datadir}/{dataname}_labels",
        )

    elif args.dataset.startswith("latent_memmap_imagenet"):
        dataname = ["imagenet256", "imagenet512"]["512" in args.dataset]
        dataset = LatentMemmapDataset(
            "imagenet",
            features_dir=f"{args.datadir}/{dataname}_feature_flip.dat",
            labels_dir=f"{args.datadir}/{dataname}_label_flip.dat",
        )

    elif args.dataset.startswith("latent_celeba"):
        dataname = ["celebahq512", "celebahq1024"]["1024" in args.dataset]
        feat_dir = f"{args.datadir}/{dataname}_features"
        dataset = LatentDataset("celebhq", features_dir=feat_dir, labels_dir=None)

    elif args.dataset == "lsun_church":
        train_transform = transforms.Compose(
            [
                transforms.Resize(args.image_size),
                transforms.CenterCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_data = LSUN(root=args.datadir, classes=["church_outdoor_train"], transform=train_transform)
        subset = list(range(0, 120000))
        dataset = torch.utils.data.Subset(train_data, subset)

    elif args.dataset == "lsun_bedroom":
        train_transform = transforms.Compose(
            [
                transforms.Resize(args.image_size),
                transforms.CenterCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_data = LSUN(root=args.datadir, classes=["bedroom_train"], transform=train_transform)
        subset = list(range(0, 120000))
        dataset = torch.utils.data.Subset(train_data, subset)

    elif args.dataset == "celeba_256":
        train_transform = transforms.Compose(
            [
                transforms.Resize(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = LMDBDataset(root=args.datadir, name="celeba", train=True, transform=train_transform)

    elif args.dataset == "celeba_512":
        from torchtoolbox.data import ImageLMDB

        train_transform = transforms.Compose(
            [
                transforms.Resize(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = ImageLMDB(db_path=args.datadir, db_name="celeba_512", transform=train_transform, backend="pil")

    elif args.dataset == "celeba_1024":
        from torchtoolbox.data import ImageLMDB

        train_transform = transforms.Compose(
            [
                transforms.Resize(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = ImageLMDB(db_path=args.datadir, db_name="celeba_1024", transform=train_transform, backend="pil")

    elif args.dataset == "ffhq_256":
        train_transform = transforms.Compose(
            [
                transforms.Resize(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = LMDBDataset(root=args.datadir, name="ffhq", train=True, transform=train_transform)
    return dataset
