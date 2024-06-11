import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_dct_kernel(in_ch, ksize=8, rsize=2):
    """[init a conv2d kernel for dct]

    Args:
        in_ch ([int]): [input dims]
        ksize (int, optional): [kernel size for dct]. Defaults to 8.
        rsize (int, optional): [reserve size for dct kernel]. Defaults to 2.

    Returns:
        [nn.Conv2d]: []
    """
    DCT_filter_n = np.zeros([ksize, ksize, 1, rsize**2])
    XX, YY = np.meshgrid(range(ksize), range(ksize))
    # DCT basis as filters
    C = np.ones(ksize)
    C[0] = 1 / np.sqrt(2)
    for v in range(rsize):
        for u in range(rsize):
            kernel = (
                (2 * C[v] * C[u] / ksize)
                * np.cos((2 * YY + 1) * v * np.pi / (2 * ksize))
                * np.cos((2 * XX + 1) * u * np.pi / (2 * ksize))
            )
            DCT_filter_n[:, :, 0, u + v * rsize] = kernel
    DCT_filter_n = np.transpose(DCT_filter_n, (3, 2, 0, 1))
    DCT_filter = torch.tensor(DCT_filter_n).float()

    DCT_filters = [DCT_filter for i in range(0, in_ch)]
    DCT_filters = torch.cat(DCT_filters, 0)

    dct_conv = nn.Conv2d(
        in_ch, rsize**2 * in_ch, kernel_size=(ksize, ksize), stride=ksize, padding=0, groups=in_ch, bias=False
    )
    dct_conv.weight = torch.nn.Parameter(DCT_filters)
    dct_conv.weight.requires_grad = False
    dct_conv.requires_grad = False

    return dct_conv


def init_idct_kernel(out_ch, ksize=8, rsize=2):
    """[init a conv2d kernel for idct]

    Args:
        out_ch ([int]): [output dims]
        ksize (int, optional): [kernel size for idct]. Defaults to 8.
        rsize (int, optional): [reserve size for idct kernel]. Defaults to 2.

    Returns:
        [nn.Conv2d]: []
    """
    IDCT_filter_n = np.zeros([1, 1, rsize**2, ksize**2])
    # IDCT basis as filters
    C = np.ones(ksize)
    C[0] = 1 / np.sqrt(2)
    for v in range(rsize):
        for u in range(rsize):
            for j in range(ksize):
                for i in range(ksize):
                    kernel = (
                        (2 * C[v] * C[u] / ksize)
                        * np.cos((2 * j + 1) * v * np.pi / (2 * ksize))
                        * np.cos((2 * i + 1) * u * np.pi / (2 * ksize))
                    )
                    IDCT_filter_n[0, 0, u + v * rsize, i + j * ksize] = kernel

    IDCT_filter_n = np.transpose(IDCT_filter_n, (3, 2, 0, 1))
    IDCT_filter = torch.tensor(IDCT_filter_n).float()
    IDCT_filters = [IDCT_filter for i in range(0, out_ch)]
    IDCT_filters = torch.cat(IDCT_filters, 0)

    idct_conv = nn.Conv2d(
        rsize**2 * out_ch, ksize**2 * out_ch, kernel_size=(1, 1), stride=1, padding=0, groups=out_ch, bias=False
    )
    idct_conv.weight = torch.nn.Parameter(IDCT_filters)
    idct_conv.weight.requires_grad = False
    idct_conv.requires_grad = False

    return idct_conv