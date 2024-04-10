import torch
import numpy as np
from vim.scanning_orders import sweep_path, zigma_path, jpeg_zigzag

paths = sweep_path(4)
paths = [torch.from_numpy(np.array(x)) for x in paths]
print(paths)
a = torch.arange(32).reshape(1, 2, 16)
print(a.reshape(2, 4, 4))
# a_ = a[:, :, paths[0]]
a_ = torch.gather(a, 2, paths[-1][None, None, :].expand_as(a))
print(a_)

# a = torch.arange(16).reshape(1, 16)
# print(a.reshape(4, 4))
# indices = jpeg_zigzag(4, 4)
# a_ = a[:, indices]
# print(a_.reshape(4, 4))
# print(indices)
# # a_recon = inverse_jpeg_zigzag(a_, *a.shape)
# # print(a_recon)

# rev_indices = reverse_permut_np(indices)
# a_recon2 = a_[:, rev_indices]
# print("A recon2: ", a_recon2.reshape(4, 4))

