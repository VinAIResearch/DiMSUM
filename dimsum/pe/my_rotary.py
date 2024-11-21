import numpy as np
import torch


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_rotary_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    emb_sin, emb_cos = get_2d_sincos_rotary_embed_from_grid(embed_dim, grid)
    # if cls_token and extra_tokens > 0:
    #     pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return emb_sin, emb_cos


def get_2d_sincos_rotary_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_sin_h, emb_cos_h = get_1d_sincos_rotary_embed_from_grid(embed_dim // 2, grid[0])
    emb_sin_w, emb_cos_w = get_1d_sincos_rotary_embed_from_grid(embed_dim // 2, grid[1])

    emb_sin = np.concatenate([emb_sin_h, emb_sin_w], axis=1)  # (H*W, D/2) ([1,2,3])
    emb_cos = np.concatenate([emb_cos_h, emb_cos_w], axis=1)  # (H*W, D/2)
    emb_sin = emb_sin.repeat(2, axis=1)  # (H*W, D) ([1,1,2,2,3,3])
    emb_cos = emb_cos.repeat(2, axis=1)  # (H*W, D)
    return emb_sin, emb_cos


def get_1d_sincos_rotary_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    return emb_sin, emb_cos


def rotate_half(x):
    x_r = torch.empty_like(x)
    x_r[..., 0::2] = -x[..., 1::2]
    x_r[..., 1::2] = x[..., 0::2]
    return x_r


def apply_rotary(x, emb_sin, emb_cos):
    x_r = rotate_half(x)
    return x * emb_cos + x_r * emb_sin


if __name__ == "__main__":
    emb_sin, emb_cos = get_2d_sincos_rotary_embed(128, 16)  # grid size
    print(emb_cos.shape, emb_sin.shape)
    x = torch.rand((2, 256, 128))  # B x L x D
    x_r = apply_rotary(x, emb_sin, emb_cos)
    print(x_r.shape)
