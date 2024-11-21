import torch
from sinusoidal import position_encoding_1d


def apply_rotary_position_embeddings(sinusoidal: torch.Tensor, *tensors):
    assert len(tensors) > 0, "at least one input tensor"

    cos_pos = sinusoidal[..., 1::2].repeat_interleave(2, 1)
    sin_pos = sinusoidal[..., 0::2].repeat_interleave(2, 1)

    cos_pos = cos_pos.expand_as(tensors[0])
    sin_pos = sin_pos.expand_as(tensors[0])

    outputs = []
    for t in tensors:
        t_r = torch.empty_like(t)
        t_r[..., 0::2] = -t[..., 1::2]
        t_r[..., 1::2] = t[..., 0::2]
        outputs.append(t * cos_pos + t_r * sin_pos)

    return outputs if len(tensors) > 1 else outputs[0]


class Rotary2D:
    def __init__(self, dim: int, base: float = 10000):
        self.dim = dim
        self.base = base
        self.pos_cached = None
        self.w_size_cached = None
        self.h_size_cached = None

    def forward(self, x: torch.Tensor):
        H, W = x.size(2), x.size(3)
        assert H % 2 == 0
        assert W % 2 == 0
        if self.pos_cached is None or self.w_size_cached != W or self.h_size_cached != H:
            self.h_size_cached = H
            self.w_size_cached = W

            position_x = position_encoding_1d(H, self.dim // 2, self.base)
            position_y = position_encoding_1d(W, self.dim // 2, self.base)

            position_x = position_x.reshape(H, -1, 2)
            position_y = position_y.reshape(W, -1, 2)

            self.pos_cached = torch.empty(H * W, self.dim, dtype=x.dtype, device=x.device)
            for i in range(H):
                for j in range(W):
                    emb = torch.cat(
                        [position_x[i, 0::2], position_y[j, 0::2], position_x[i, 1::2], position_y[j, 1::2]], 0
                    ).flatten(-2)
                    self.pos_cached[i * W + j] = emb.to(x.dtype).to(x.device)
        return self.pos_cached


# rotary 1D
class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
        return self.cos_cached, self.sin_cached


# rotary pos emb helpers:


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in torch < 1.8.0


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
