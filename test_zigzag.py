import torch
import numpy as np
from vim.scanning_orders import sweep_path, zigma_path, jpeg_zigzag, reverse_permut_np, local_scan, local_reverse

scan_fn = jpeg_zigzag
N = 4
paths = scan_fn(N)
paths = [torch.from_numpy(np.array(x)) for x in paths]
paths_list = [x.tolist() for x in paths]
print(paths_list)
a = torch.arange(N*N).reshape(1, 1, N*N)
print(a.reshape(1, N, N))
# a_ = a[:, :, paths[0]]
a_ = torch.gather(a, 2, paths[-1][None, None, :].expand_as(a))
print(a_)

rev_paths = [torch.from_numpy(reverse_permut_np(x)) for x in paths]
# a_recon2 = a_[:, rev_indices]
a_recon2 = torch.gather(a_, 2, rev_paths[-1][None, None, :].expand_as(a))
print("A recon2: ", a_recon2.reshape(N, N))

# N = 8
# x = torch.arange(N*N).reshape(1,N*N,1)
# col = True
# print(x.reshape(N,N))
# x_out = local_scan(x, w=2, H=8, W=8, column_first=col)
# print(x_out.reshape(N,N))
# print(x_out.reshape(-1).tolist())
# x_recon = local_reverse(x_out, w=2, H=8, W=8, column_first=col)
# print(x_recon.reshape(N,N))



