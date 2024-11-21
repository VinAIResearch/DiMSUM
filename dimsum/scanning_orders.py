import math

import numpy as np
from torch.nn import functional as F


def sweep_path(N):
    """
    Mamba's sweep scan
    """

    def zigzag_path_lr(N, start_row=0, start_col=0, dir_row=1, dir_col=1):
        path = []
        for i in range(N):
            for j in range(N):
                col = j
                path.append((start_row + dir_row * i) * N + start_col + dir_col * col)
        return path

    def zigzag_path_tb(N, start_row=0, start_col=0, dir_row=1, dir_col=1):
        path = []
        for j in range(N):
            for i in range(N):
                row = i
                path.append((start_row + dir_row * row) * N + start_col + dir_col * j)
        return path

    paths = []
    for start_row, start_col, dir_row, dir_col in [
        (0, 0, 1, 1),
        (0, N - 1, 1, -1),
        (N - 1, 0, -1, 1),
        (N - 1, N - 1, -1, -1),
    ]:
        paths.append(zigzag_path_lr(N, start_row, start_col, dir_row, dir_col))
        paths.append(zigzag_path_tb(N, start_row, start_col, dir_row, dir_col))

    for _index, _p in enumerate(paths):
        paths[_index] = np.array(_p)
    return paths


def zigma_path(N):
    """
    Zigma's continuity scan
    """

    def zigzag_path_lr(N, start_row=0, start_col=0, dir_row=1, dir_col=1):
        path = []
        for i in range(N):
            for j in range(N):
                # If the row number is even, move right; otherwise, move left
                col = j if i % 2 == 0 else N - 1 - j
                path.append((start_row + dir_row * i) * N + start_col + dir_col * col)
        return path

    def zigzag_path_tb(N, start_row=0, start_col=0, dir_row=1, dir_col=1):
        path = []
        for j in range(N):
            for i in range(N):
                # If the column number is even, move down; otherwise, move up
                row = i if j % 2 == 0 else N - 1 - i
                path.append((start_row + dir_row * row) * N + start_col + dir_col * j)
        return path

    paths = []
    for start_row, start_col, dir_row, dir_col in [
        (0, 0, 1, 1),
        (0, N - 1, 1, -1),
        (N - 1, 0, -1, 1),
        (N - 1, N - 1, -1, -1),
    ]:
        paths.append(zigzag_path_lr(N, start_row, start_col, dir_row, dir_col))
        paths.append(zigzag_path_tb(N, start_row, start_col, dir_row, dir_col))

    for _index, _p in enumerate(paths):
        paths[_index] = np.array(_p)
    return paths


def jpeg_zigzag(N):
    """
    Jpeg's zigzag scan
    Modified from: https://github.com/getsanjeev/compression-DCT/blob/master/zigzag.py
    """

    def zigzag_path_lr(N, start_row, start_col, dir_row, dir_col):
        # initializing the variables
        # ----------------------------------
        h = 0
        v = 0

        vmin = 0
        hmin = 0

        vmax = N  # input.shape[0]
        hmax = N  # input.shape[1]
        # print(vmax, hmax)

        i = 0

        # output = np.zeros(( vmax * hmax))
        indices = []
        # ----------------------------------

        while (v < vmax) and (h < hmax):
            # indices.append(v*vmax + h)
            indices.append((start_row + dir_row * v) * vmax + start_col + dir_col * h)
            # output[i] = input[v, h]        # if we got to the first line

            if ((h + v) % 2) == 0:  # going up

                if v == vmin:
                    # print(1)

                    if h == hmax:
                        v = v + 1
                    else:
                        h = h + 1

                elif (h == hmax - 1) and (v < vmax):  # if we got to the last column
                    # print(2)

                    v = v + 1

                elif (v > vmin) and (h < hmax - 1):  # all other cases
                    # print(3)
                    v = v - 1
                    h = h + 1

            else:  # going down

                if (v == vmax - 1) and (h <= hmax - 1):  # if we got to the last line
                    # print(4)
                    h = h + 1

                elif h == hmin:  # if we got to the first column
                    # print(5)

                    if v == vmax - 1:
                        h = h + 1
                    else:
                        v = v + 1

                elif (v < vmax - 1) and (h > hmin):  # all other cases
                    # print(6)
                    v = v + 1
                    h = h - 1

            i = i + 1
            if (v == vmax - 1) and (h == hmax - 1):  # bottom right element
                # print(7)
                # output[i] = input[v, h]
                # indices.append(v*vmax + h)
                indices.append((start_row + dir_row * v) * vmax + start_col + dir_col * h)
                break

        return np.array(indices)

    def zigzag_path_tb(N, start_row, start_col, dir_row, dir_col):
        # initializing the variables
        # ----------------------------------
        h = 0
        v = 0

        vmin = 0
        hmin = 0

        vmax = N  # input.shape[0]
        hmax = N  # input.shape[1]
        # print(vmax, hmax)

        i = 0

        # output = np.zeros(( vmax * hmax))
        indices = []
        # ----------------------------------

        while (v < vmax) and (h < hmax):
            indices.append((start_row + dir_row * v) * vmax + start_col + dir_col * h)
            # indices.append(v*vmax + h)
            # output[i] = input[v, h]        # if we got to the first line

            if ((h + v) % 2) == 0:  # going up

                if h == hmin:  # if we got to the first column
                    # print(5)

                    if v == vmax - 1:
                        h = h + 1
                    else:
                        v = v + 1

                elif (v == vmax - 1) and (h <= hmax - 1):  # if we got to the last line
                    # print(4)
                    h = h + 1

                elif (v < vmax - 1) and (h > hmin):  # all other cases
                    # print(6)
                    v = v + 1
                    h = h - 1

            else:  # going down

                if (h == hmax - 1) and (v < vmax):  # if we got to the last column
                    # print(2)

                    v = v + 1

                elif v == vmin:
                    # print(1)

                    if h == hmax:
                        v = v + 1
                    else:
                        h = h + 1

                elif (v > vmin) and (h < hmax - 1):  # all other cases
                    # print(3)
                    v = v - 1
                    h = h + 1

            i = i + 1
            if (v == vmax - 1) and (h == hmax - 1):  # bottom right element
                # print(7)
                # output[i] = input[v, h]
                indices.append((start_row + dir_row * v) * vmax + start_col + dir_col * h)
                # indices.append(v*vmax + h)
                break

        return np.array(indices)

    paths = []
    for start_row, start_col, dir_row, dir_col in [
        (0, 0, 1, 1),
        (0, N - 1, 1, -1),
        (N - 1, 0, -1, 1),
        (N - 1, N - 1, -1, -1),
    ]:
        paths.append(zigzag_path_lr(N, start_row, start_col, dir_row, dir_col))
        paths.append(zigzag_path_tb(N, start_row, start_col, dir_row, dir_col))

    for _index, _p in enumerate(paths):
        paths[_index] = np.array(_p)
    return paths


def reverse_permut_np(permutation):
    n = len(permutation)
    reverse = np.array([0] * n)
    for i in range(n):
        reverse[permutation[i]] = i
    return reverse


# Inverse zigzag scan of a matrix
# Arguments are: a 1-by-m*n array,
# where m & n are vertical & horizontal sizes of an output matrix.
# Function returns a two-dimensional matrix of defined sizes,
# consisting of input array items gathered by a zigzag method.
#
# Matlab Code:
# Alexey S. Sokolov a.k.a. nICKEL, Moscow, Russia
# June 2007
# alex.nickel@gmail.com


def inverse_jpeg_zigzag(input, vmax, hmax):

    # print input.shape

    # initializing the variables
    # ----------------------------------
    h = 0
    v = 0

    vmin = 0
    hmin = 0

    output = np.zeros((vmax, hmax))

    i = 0
    # ----------------------------------

    while (v < vmax) and (h < hmax):
        # print ('v:',v,', h:',h,', i:',i)
        if ((h + v) % 2) == 0:  # going up

            if v == vmin:
                # print(1)
                output[v, h] = input[i]  # if we got to the first line

                if h == hmax:
                    v = v + 1
                else:
                    h = h + 1

                i = i + 1

            elif (h == hmax - 1) and (v < vmax):  # if we got to the last column
                # print(2)
                output[v, h] = input[i]
                v = v + 1
                i = i + 1

            elif (v > vmin) and (h < hmax - 1):  # all other cases
                # print(3)
                output[v, h] = input[i]
                v = v - 1
                h = h + 1
                i = i + 1

        else:  # going down

            if (v == vmax - 1) and (h <= hmax - 1):  # if we got to the last line
                # print(4)
                output[v, h] = input[i]
                h = h + 1
                i = i + 1

            elif h == hmin:  # if we got to the first column
                # print(5)
                output[v, h] = input[i]
                if v == vmax - 1:
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1

            elif (v < vmax - 1) and (h > hmin):  # all other cases
                output[v, h] = input[i]
                v = v + 1
                h = h - 1
                i = i + 1

        if (v == vmax - 1) and (h == hmax - 1):  # bottom right element
            # print(7)
            output[v, h] = input[i]
            break

    return output


"""PyTorch code for local scan and local reverse"""


def local_scan(x, w=7, H=14, W=14, flip=False, column_first=False):
    """Local windowed scan in LocalMamba
    Input:
        x: [B, L, C]
        H, W: original width and height before padding
        column_first: column-wise scan first (the additional direction in VMamba)
    Return: [B, L, C]
    """
    B, L, C = x.shape
    x = x.view(B, H, W, C)
    Hg, Wg = math.ceil(H / w), math.ceil(W / w)
    if H % w != 0 or W % w != 0:
        newH, newW = Hg * w, Wg * w
        x = F.pad(x, (0, 0, 0, newW - W, 0, newH - H))
    if column_first:
        x = x.view(B, Hg, w, Wg, w, C).permute(0, 3, 1, 4, 2, 5).reshape(B, -1, C)
    else:
        x = x.view(B, Hg, w, Wg, w, C).permute(0, 1, 3, 2, 4, 5).reshape(B, -1, C)
    if flip:
        x = x.flip([1])
    return x


def local_scan_bchw(x, w=7, H=14, W=14, flip=False, column_first=False):
    """Local windowed scan in LocalMamba
    Input:
        x: [B, C, H, W]
        H, W: original width and height before padding
        column_first: column-wise scan first (the additional direction in VMamba)
    Return: [B, C, L]
    """
    B, C, _, _ = x.shape
    x = x.view(B, C, H, W)
    Hg, Wg = math.ceil(H / w), math.ceil(W / w)
    if H % w != 0 or W % w != 0:
        newH, newW = Hg * w, Wg * w
        x = F.pad(x, (0, newW - W, 0, newH - H))
    if column_first:
        x = x.view(B, C, Hg, w, Wg, w).permute(0, 1, 4, 2, 5, 3).reshape(B, C, -1)
    else:
        x = x.view(B, C, Hg, w, Wg, w).permute(0, 1, 2, 4, 3, 5).reshape(B, C, -1)
    if flip:
        x = x.flip([-1])
    return x


def local_reverse(x, w=7, H=14, W=14, flip=False, column_first=False):
    """Local windowed scan in LocalMamba
    Input:
        x: [B, L, C]
        H, W: original width and height before padding
        column_first: column-wise scan first (the additional direction in VMamba)
    Return: [B, L, C]
    """
    B, L, C = x.shape
    Hg, Wg = math.ceil(H / w), math.ceil(W / w)
    if flip:
        x = x.flip([1])
    if H % w != 0 or W % w != 0:
        if column_first:
            x = x.view(B, Wg, Hg, w, w, C).permute(0, 2, 4, 1, 3, 5).reshape(B, C, Hg * w, Wg * w)
        else:
            x = x.view(B, Hg, Wg, w, w, C).permute(0, 1, 3, 2, 4, 5).reshape(B, C, Hg * w, Wg * w)
        x = x[:, :H, :W, :].reshape(B, -1, C)
    else:
        if column_first:
            x = x.view(B, Wg, Hg, w, w, C).permute(0, 2, 4, 1, 3, 5).reshape(B, L, C)
        else:
            x = x.view(B, Hg, Wg, w, w, C).permute(0, 1, 3, 2, 4, 5).reshape(B, L, C)
    return x


SCAN_ZOO = {
    "sweep": sweep_path,
    "zigma": zigma_path,
    "jpeg": jpeg_zigzag,
}
