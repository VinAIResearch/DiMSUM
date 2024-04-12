import torch
import numpy as np
from vim.scanning_orders import sweep_path, zigma_path, jpeg_zigzag, reverse_permut_np

scan_fn = jpeg_zigzag
paths = scan_fn(4)
paths = [torch.from_numpy(np.array(x)) for x in paths]
print(paths)
a = torch.arange(16).reshape(1, 1, 16)
print(a.reshape(1, 4, 4))
# a_ = a[:, :, paths[0]]
a_ = torch.gather(a, 2, paths[-1][None, None, :].expand_as(a))
print(a_)

rev_paths = [torch.from_numpy(reverse_permut_np(x)) for x in paths]
# a_recon2 = a_[:, rev_indices]
a_recon2 = torch.gather(a_, 2, rev_paths[-1][None, None, :].expand_as(a))
print("A recon2: ", a_recon2.reshape(4, 4))

