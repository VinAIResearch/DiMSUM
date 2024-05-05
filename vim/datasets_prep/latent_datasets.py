import numpy as np
import lmdb
import os
import io
from glob import glob
import torch
import torch.utils.data as data


# class LatentDataset(data.Dataset):
#     def __init__(self, root, train=True, transform=None):
#         self.train = train
#         self.transform = transform
#         if self.train:
#             latent_paths = glob(f'{root}/train/*.npy')
#         else:
#             latent_paths = glob(f'{root}/val/*.npy')
#         self.data = latent_paths

#     def __getitem__(self, index):
#         sample = np.load(self.data[index]).item()
#         target = torch.from_numpy(sample["label"])
#         x = torch.from_numpy(sample["input"])
#         if self.transform is not None:
#             x = self.transform(x)

#         return x, target

#     def __len__(self):
#         return len(self.data)


class LatentDataset(data.Dataset):
    def __init__(self, dataset, features_dir, labels_dir=None):
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        self.dataset = dataset

    def __len__(self):
        if self.dataset == "imagenet":
            return 1281167
        elif self.dataset == "celebhq":
            return 30000

    def __getitem__(self, idx):
        file_id = f"{str(idx).zfill(9)}.npy"
        features = np.load(os.path.join(self.features_dir, file_id))
        if self.labels_dir is not None:
            labels = np.load(os.path.join(self.labels_dir, file_id))
        else:
            return torch.from_numpy(features), torch.tensor(0)
        return torch.from_numpy(features), torch.from_numpy(labels)
