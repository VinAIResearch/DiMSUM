import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from timm.layers import use_fused_attn
from torch.jit import Final


class CrossAttentionFusion(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        swap_k=False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // 2 // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()
        self.swap_k = swap_k

        self.qkv1 = nn.Linear(dim // 2, dim // 2 * 3, bias=qkv_bias)
        self.q_norm1 = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm1 = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.qkv2 = nn.Linear(dim // 2, dim // 2 * 3, bias=qkv_bias)
        self.q_norm2 = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm2 = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _compute_attention(self, q, k, v):
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        return x

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        B, N, C = x1.shape

        qkv1 = self.qkv1(x1).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv1.unbind(0)
        q1, k1 = self.q_norm1(q1), self.k_norm1(k1)

        qkv2 = self.qkv2(x2).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q2, k2, v2 = qkv2.unbind(0)
        q2, k2 = self.q_norm2(q2), self.k_norm2(k2)

        if not self.swap_k:
            x12 = self._compute_attention(q1, k2, v2)
            x21 = self._compute_attention(q2, k1, v1)
        else:
            x12 = self._compute_attention(q2, k1, v2)
            x21 = self._compute_attention(q1, k2, v1)

        x12 = x12.transpose(1, 2).reshape(B, N, C)
        x21 = x21.transpose(1, 2).reshape(B, N, C)

        x = self.proj(torch.cat((x12, x21), dim=-1))
        x = self.proj_drop(x)
        return x
