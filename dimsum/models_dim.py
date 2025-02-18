import math
from functools import partial
from typing import Optional

from huggingface_hub import PyTorchModelHubMixin


import numpy as np
import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import CondMamba, Mamba
from pe.cpe import AdaInPosCNN
from pe.my_rotary import apply_rotary, get_2d_sincos_rotary_embed
from rope import *
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed
from torch import Tensor
from torch.nn import functional as F


try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from attention_fusion import CrossAttentionFusion
from dct_layer import init_dct_kernel, init_idct_kernel
from einops import rearrange
from mlp import GatedMLP
from scanning_orders import SCAN_ZOO, local_reverse, local_scan, reverse_permut_np
from switch_mlp import SwitchMLP
from wavelet_layer import DWT_2D, IDWT_2D


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
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
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
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

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.x_embedder.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0  # 1
        self.in_channels = num_classes + use_cfg_embedding  # 1001
        self.embedding_table = nn.Embedding(self.in_channels, hidden_size)
        self.num_classes = num_classes  # 1000
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)  # 1000 or labels
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

    def get_in_channels(self):
        return self.in_channels  # 1001


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiMBlock(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
        drop_path=0.0,
        reverse=False,
        transpose=False,
        scanning_continuity=False,
        skip=False,
        use_gated_mlp=True,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.reverse = reverse
        self.transpose = transpose
        self.scanning_continuity = scanning_continuity

        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)

        # w/o FFN
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

        self.norm_2 = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        mlp_hidden_dim = int(dim * 4)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = GatedMLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        c: Optional[Tensor] = None,
        inference_params=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
            c: (N, D)
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )

        l = hidden_states.shape[1]
        h = w = int(np.sqrt(l))
        if self.transpose:
            hidden_states = rearrange(hidden_states, "n (h w) c -> n (w h) c", h=h, w=w)

        if self.scanning_continuity:
            hidden_states = rearrange(hidden_states.clone(), "n (w h) c -> n c w h", h=h, w=w)
            hidden_states[:, :, 1::2] = hidden_states[:, :, 1::2].flip(-1)
            hidden_states = rearrange(hidden_states, "n c w h -> n (w h) c", h=h, w=w)

        if self.reverse:
            hidden_states = hidden_states.flip(1)

        shift_ssm, scale_ssm, gate_ssm, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        hidden_states = hidden_states + gate_ssm.unsqueeze(1) * self.mixer(
            modulate(hidden_states, shift_ssm, scale_ssm), c, inference_params=inference_params
        )
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm_2(hidden_states), shift_mlp, scale_mlp)
        )

        # transform back
        if self.reverse:
            hidden_states = hidden_states.flip(1)

        if self.scanning_continuity:
            hidden_states = rearrange(hidden_states.clone(), "n (w h) c -> n c w h", h=h, w=w)
            hidden_states[:, :, 1::2] = hidden_states[:, :, 1::2].flip(-1)
            hidden_states = rearrange(hidden_states, "n c w h -> n (w h) c", h=h, w=w)

        if self.transpose:
            hidden_states = rearrange(hidden_states, "n (h w) c -> n (w h) c", h=h, w=w)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class DiMBlockWindow(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
        drop_path=0.0,
        reverse=False,
        transpose=False,
        scanning_continuity=False,
        skip=False,
        shift_window=False,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.reverse = reverse
        self.transpose = transpose
        self.scanning_continuity = scanning_continuity
        self.shift_window = shift_window

        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)

        # w/o FFN
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

        self.norm_2 = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        mlp_hidden_dim = int(dim * 4)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = GatedMLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        c: Optional[Tensor] = None,
        inference_params=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
            c: (N, D)
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = (
                self.norm(residual)
                if isinstance(self.norm, nn.Identity)
                else self.norm(residual.to(dtype=self.norm.weight.dtype))
            )
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )

        l = hidden_states.shape[1]
        h = w = int(np.sqrt(l))
        column_first = True if self.transpose else False
        hidden_states = local_scan(hidden_states, w=4, H=h, W=w, column_first=column_first).contiguous()

        if self.shift_window:
            hidden_states = rearrange(hidden_states, "b (h w) c -> b h w c", h=h)
            hidden_states = torch.roll(hidden_states, shifts=(-1, -1), dims=(1, 2)).reshape(
                -1, h * w, hidden_states.size(-1)
            )

        if self.reverse:
            hidden_states = hidden_states.flip(1)

        shift_ssm, scale_ssm, gate_ssm, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        hidden_states = hidden_states + gate_ssm.unsqueeze(1) * self.mixer(
            modulate(hidden_states, shift_ssm, scale_ssm), c, inference_params=inference_params
        )
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm_2(hidden_states), shift_mlp, scale_mlp)
        )

        # transform back
        if self.reverse:
            hidden_states = hidden_states.flip(1)

        if self.shift_window:
            hidden_states = rearrange(hidden_states, "b (h w) c -> b h w c", h=h)
            hidden_states = torch.roll(hidden_states, shifts=(1, 1), dims=(1, 2)).reshape(
                -1, h * w, hidden_states.size(-1)
            )

        hidden_states = local_reverse(hidden_states, w=4, H=h, W=w, column_first=column_first)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class WaveDiMBlock(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
        drop_path=0.0,
        reverse=False,
        transpose=False,
        scanning_continuity=False,
        skip=False,
        no_ffn=False,
        c_dim=None,
        window_scan=True,
        num_wavelet_lv=2,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.reverse = reverse
        self.transpose = transpose
        self.scanning_continuity = scanning_continuity
        self.no_ffn = no_ffn
        self.window_scan = window_scan
        self.num_wavelet_lv = num_wavelet_lv
        c_dim = dim if c_dim is None else c_dim

        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)

        self.dwt = DWT_2D(wave="haar")
        self.idwt = IDWT_2D(wave="haar")

        # w/o FFN
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(c_dim, 6 * dim if not self.no_ffn else 3 * dim, bias=True)
        )
        if not self.no_ffn:
            self.norm_2 = norm_cls(dim)

            mlp_hidden_dim = int(dim * 4)
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = GatedMLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def _dwt_fast(self, x):
        # support only two-levels wavelet
        x = rearrange(x, "b (h w) c -> b c h w", h=int(np.sqrt(x.size(1))))
        subbands = self.dwt(x)  # xll, xlh, xhl, xhh where each has shape of [b, c, h/2, w/2]
        scale = 2**self.num_wavelet_lv
        patch_size = scale
        if self.num_wavelet_lv > 1:
            out = (self.dwt(subbands) / scale).chunk(patch_size * patch_size, dim=1)
            indices = []
            for i in range(patch_size * patch_size):
                indices.append(i % 4 * patch_size + i // 4)
            out = torch.cat([out[i] for i in indices], dim=1)
        else:
            out = subbands / scale
        return rearrange(out, "b (c p1 p2) h w -> b (h p1 w p2) c", p1=patch_size, p2=patch_size)  # [b, c, h, w]

    def _idwt_fast(self, x):
        scale = 2**self.num_wavelet_lv
        patch_size = scale
        lowest_size = int(np.sqrt(x.size(1))) // patch_size
        subbands = rearrange(
            x * scale, "b (h p1 w p2) c -> b (c p1 p2) h w", p1=patch_size, p2=patch_size, h=lowest_size
        ).chunk(patch_size * patch_size, dim=1)
        if self.num_wavelet_lv > 1:
            indices = []
            for i in range(patch_size * patch_size):
                indices.append(i % 4 * patch_size + i // 4)
            subbands = torch.cat([subbands[i] for i in indices], dim=1)
            out = self.idwt(subbands)
            out = self.idwt(out)
        else:
            out = self.idwt(torch.cat(subbands, dim=1))
        return rearrange(out, "b c h w -> b (h w) c")  # [b, c, h, w]

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        c: Optional[Tensor] = None,
        inference_params=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
            c: (N, D)
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = (
                self.norm(residual)
                if isinstance(self.norm, nn.Identity)
                else self.norm(residual.to(dtype=self.norm.weight.dtype))
            )
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )

        l = hidden_states.shape[1]
        h = w = int(np.sqrt(l))
        hidden_states = self._dwt_fast(hidden_states).contiguous()
        patch_size = int(2**self.num_wavelet_lv)  # old: 4
        if self.window_scan:
            column_first = True if self.transpose else False
            hidden_states = local_scan(
                hidden_states, w=w // patch_size, H=h, W=w, column_first=column_first
            ).contiguous()  # Resolve fixed window size
        else:
            if self.transpose:
                hidden_states = rearrange(hidden_states, "n (h w) c -> n (w h) c", h=h, w=w)

        if self.scanning_continuity:
            hidden_states = rearrange(hidden_states.clone(), "n (w h) c -> n c w h", h=h, w=w)
            hidden_states[:, :, 1::2] = hidden_states[:, :, 1::2].flip(-1)
            hidden_states = rearrange(hidden_states, "n c w h -> n (w h) c", h=h, w=w)

        if self.reverse:
            hidden_states = hidden_states.flip(1)

        if not self.no_ffn:
            shift_ssm, scale_ssm, gate_ssm, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            hidden_states = hidden_states + gate_ssm.unsqueeze(1) * self.mixer(
                modulate(hidden_states, shift_ssm, scale_ssm), c, inference_params=inference_params
            )
            hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.mlp(
                modulate(self.norm_2(hidden_states), shift_mlp, scale_mlp)
            )
        else:
            shift_ssm, scale_ssm, gate_ssm = self.adaLN_modulation(c).chunk(3, dim=1)
            hidden_states = hidden_states + gate_ssm.unsqueeze(1) * self.mixer(
                modulate(hidden_states, shift_ssm, scale_ssm), c, inference_params=inference_params
            )

        # transform back
        if self.reverse:
            hidden_states = hidden_states.flip(1)

        if self.scanning_continuity:
            hidden_states = rearrange(hidden_states.clone(), "n (w h) c -> n c w h", h=h, w=w)
            hidden_states[:, :, 1::2] = hidden_states[:, :, 1::2].flip(-1)
            hidden_states = rearrange(hidden_states, "n c w h -> n (w h) c", h=h, w=w)

        if self.window_scan:
            hidden_states = local_reverse(hidden_states, w=w // patch_size, H=h, W=w, column_first=column_first)
        else:
            if self.transpose:
                hidden_states = rearrange(hidden_states, "n (h w) c -> n (w h) c", h=h, w=w)
        hidden_states = self._idwt_fast(hidden_states).contiguous()

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class EinFFT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.hidden_size = dim
        self.num_blocks = 4
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0
        self.sparsity_threshold = 0.01
        self.scale = 0.02

        self.complex_weight_1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * self.scale
        )
        self.complex_weight_2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * self.scale
        )
        self.complex_bias_1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * self.scale
        )
        self.complex_bias_2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * self.scale
        )

    def multiply(self, input, weights):
        return torch.einsum("...bd,bdk->...bk", input, weights)

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B, N, self.num_blocks, self.block_size)

        x = torch.fft.fft2(x, dim=(1, 2), norm="ortho")  # FFT on N dimension

        x_real_1 = F.relu(
            self.multiply(x.real, self.complex_weight_1[0])
            - self.multiply(x.imag, self.complex_weight_1[1])
            + self.complex_bias_1[0]
        )
        x_imag_1 = F.relu(
            self.multiply(x.real, self.complex_weight_1[1])
            + self.multiply(x.imag, self.complex_weight_1[0])
            + self.complex_bias_1[1]
        )
        x_real_2 = (
            self.multiply(x_real_1, self.complex_weight_2[0])
            - self.multiply(x_imag_1, self.complex_weight_2[1])
            + self.complex_bias_2[0]
        )
        x_imag_2 = (
            self.multiply(x_real_1, self.complex_weight_2[1])
            + self.multiply(x_imag_1, self.complex_weight_2[0])
            + self.complex_bias_2[1]
        )

        x = torch.stack([x_real_2, x_imag_2], dim=-1).float()
        x = F.softshrink(x, lambd=self.sparsity_threshold) if self.sparsity_threshold else x
        x = torch.view_as_complex(x)

        x = torch.fft.ifft2(x, dim=(1, 2), norm="ortho")

        # RuntimeError: "fused_dropout" not implemented for 'ComplexFloat'
        x = x.to(torch.float32)
        x = x.reshape(B, N, C)
        return x


class DCTBlock(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        fused_add_norm=False,
        residual_in_fp32=False,
        drop_path=0.0,
        norm_cls=nn.LayerNorm,
        dct_size=2,
        reverse=False,
        transpose=False,
        scanning_continuity=False,
        no_ffn=False,
        c_dim=None,
    ):
        super().__init__()
        self.dim = dim
        self.dct_size = dct_size
        self.num_blocks = dct_size
        self.block_size = dim // self.num_blocks
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.reverse = reverse
        self.transpose = transpose
        self.scanning_continuity = scanning_continuity
        self.no_ffn = no_ffn
        c_dim = dim if c_dim is None else c_dim

        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(c_dim, 6 * dim if not self.no_ffn else 3 * dim, bias=True)
        )
        if not self.no_ffn:
            self.norm_2 = norm_cls(dim)

            mlp_hidden_dim = int(dim * 4)
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = GatedMLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        reserve_kernel = dct_size  # (dct_size + 1) // 2 # redundancy reduction by a half
        self.reserve_kernel = reserve_kernel

        self.dct_conv = init_dct_kernel(dim, dct_size, reserve_kernel)
        self.idct_conv = nn.Sequential(init_idct_kernel(dim, dct_size, reserve_kernel), nn.PixelShuffle(dct_size))

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        c: Optional[Tensor] = None,
        inference_params=None,
    ):
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = (
                self.norm(residual)
                if isinstance(self.norm, nn.Identity)
                else self.norm(residual.to(dtype=self.norm.weight.dtype))
            )
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )

        hidden_size = int(math.sqrt(hidden_states.size(1)))
        ### 2D DCT
        h = rearrange(hidden_states, "b (h w) d -> b d h w", h=hidden_size)
        h = self.dct_conv(h)
        h = rearrange(
            h, "b (c p1 p2) h w -> b (h p1 w p2) c", c=hidden_states.size(-1), p1=self.reserve_kernel
        ).contiguous()

        if self.transpose:
            h = rearrange(h, "n (h w) c -> n (w h) c", h=hidden_size)

        if self.scanning_continuity:
            h = rearrange(h.clone(), "n (w h) c -> n c w h", h=hidden_size)
            h[:, :, 1::2] = h[:, :, 1::2].flip(-1)
            h = rearrange(h, "n c w h -> n (w h) c", h=hidden_size)

        if self.reverse:
            h = h.flip(1)

        if not self.no_ffn:
            shift_ssm, scale_ssm, gate_ssm, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            h = h + gate_ssm.unsqueeze(1) * self.mixer(
                modulate(h, shift_ssm, scale_ssm), c, inference_params=inference_params
            )
            h = h + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm_2(h), shift_mlp, scale_mlp))
        else:
            shift_ssm, scale_ssm, gate_ssm = self.adaLN_modulation(c).chunk(3, dim=1)
            h = h + gate_ssm.unsqueeze(1) * self.mixer(
                modulate(h, shift_ssm, scale_ssm), c, inference_params=inference_params
            )

        # transform back
        if self.reverse:
            h = h.flip(1)

        if self.scanning_continuity:
            h = rearrange(h.clone(), "n (w h) c -> n c w h", h=hidden_size)
            h[:, :, 1::2] = h[:, :, 1::2].flip(-1)
            h = rearrange(h, "n c w h -> n (w h) c", h=hidden_size)

        if self.transpose:
            h = rearrange(h, "n (h w) c -> n (w h) c", h=hidden_size)

        h = rearrange(
            h,
            "b (h p1 w p2) c -> b (c p1 p2) h w",
            c=hidden_states.size(-1),
            p1=self.reserve_kernel,
            p2=self.reserve_kernel,
            h=hidden_size // self.reserve_kernel,
        ).contiguous()
        out = self.idct_conv(h)
        out = rearrange(out, "b c h w -> b (h w) c")

        return out, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class MoEBlock(nn.Module):
    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False):

        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None):
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class DiMBlockCombined(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
        drop_path=0.0,
        reverse=False,
        transpose=False,
        scanning_continuity=False,
        use_gated_mlp=True,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.reverse = reverse
        self.transpose = transpose
        self.scanning_continuity = scanning_continuity

        self.norm = norm_cls(dim)
        self.spatial_mamba = DiMBlockRaw(
            dim // 2,
            mixer_cls,
            norm_cls=nn.Identity,
            drop_path=0.0,
            fused_add_norm=False,
            residual_in_fp32=residual_in_fp32,
            reverse=reverse,
            transpose=transpose,
            scanning_continuity=scanning_continuity,
            c_dim=dim,
        )

        self.freq_mamba = WaveDiMBlock(
            dim // 2,
            mixer_cls,
            norm_cls=nn.Identity,
            drop_path=0.0,
            fused_add_norm=False,
            residual_in_fp32=residual_in_fp32,
            reverse=False,
            transpose=reverse,  # transpose, # disable if only left2right scanning is used
            scanning_continuity=scanning_continuity,
            no_ffn=True,
            c_dim=dim,
            num_wavelet_lv=2,
        )

        self.proj = CrossAttentionFusion(dim, num_heads=8, qkv_bias=True, swap_k=False)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

        self.norm_2 = norm_cls(dim)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim, bias=True))
        mlp_hidden_dim = int(dim * 4)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_gated_mlp:
            self.mlp = GatedMLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        c: Optional[Tensor] = None,
        inference_params=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))

        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )

        x1, x2 = hidden_states.chunk(2, dim=2)
        x1, _ = self.spatial_mamba(x1, None, c, inference_params)
        x2, _ = self.freq_mamba(x2, None, c, inference_params)
        if isinstance(self.proj, CrossAttentionFusion):
            x = self.proj(x1, x2)
        else:
            x = torch.cat((x1, x2), dim=2)
            x = self.proj(x)

        # FFN
        hidden_states = hidden_states + x
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(3, dim=1)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm_2(hidden_states), shift_mlp, scale_mlp)
        )

        return hidden_states, residual


class DiMBlockCombinedFourier(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        mixer_cls_2,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
        drop_path=0.0,
        reverse=False,
        transpose=False,
        scanning_continuity=False,
        use_gated_mlp=True,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.reverse = reverse
        self.transpose = transpose
        self.scanning_continuity = scanning_continuity

        self.norm = norm_cls(dim)
        self.spatial_mamba = DiMBlockRaw(
            dim // 2,
            mixer_cls,
            norm_cls=nn.Identity,
            drop_path=0.0,
            fused_add_norm=False,
            residual_in_fp32=residual_in_fp32,
            reverse=reverse,
            transpose=transpose,
            scanning_continuity=scanning_continuity,
            c_dim=dim,
        )

        self.freq_mamba = DCTBlock(
            dim // 2,
            mixer_cls_2,
            norm_cls=nn.Identity,
            drop_path=0.0,
            fused_add_norm=False,
            residual_in_fp32=residual_in_fp32,
            reverse=False,
            transpose=False,
            scanning_continuity=scanning_continuity,
            no_ffn=True,
            c_dim=dim,
            dct_size=4,
        )

        self.proj = CrossAttentionFusion(dim, num_heads=8, qkv_bias=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

        self.norm_2 = norm_cls(dim)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim, bias=True))
        mlp_hidden_dim = int(dim * 4)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_gated_mlp:
            self.mlp = GatedMLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        c: Optional[Tensor] = None,
        inference_params=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))

        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )

        x1, x2 = hidden_states.chunk(2, dim=2)
        x1, _ = self.spatial_mamba(x1, None, c, inference_params)
        x2, _ = self.freq_mamba(x2, None, c, inference_params)
        if isinstance(self.proj, CrossAttentionFusion):
            x = self.proj(x1, x2)
        else:
            x = torch.cat((x1, x2), dim=2)
            x = self.proj(x)

        # FFN
        hidden_states = hidden_states + x
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(3, dim=1)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm_2(hidden_states), shift_mlp, scale_mlp)
        )

        return hidden_states, residual


class DiMBlockCombinedEinFFT(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
        drop_path=0.0,
        reverse=False,
        transpose=False,
        scanning_continuity=False,
        use_gated_mlp=True,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.reverse = reverse
        self.transpose = transpose
        self.scanning_continuity = scanning_continuity

        self.norm = norm_cls(dim)
        self.spatial_mamba = DiMBlockRaw(
            dim // 2,
            mixer_cls,
            norm_cls=nn.Identity,
            drop_path=0.0,
            fused_add_norm=False,
            residual_in_fp32=residual_in_fp32,
            reverse=reverse,
            transpose=transpose,
            scanning_continuity=scanning_continuity,
            c_dim=dim,
        )

        self.freq_mamba = EinFFT(
            dim // 2,
        )

        self.proj = CrossAttentionFusion(dim, num_heads=8, qkv_bias=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

        self.norm_2 = norm_cls(dim)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim, bias=True))
        mlp_hidden_dim = int(dim * 4)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_gated_mlp:
            self.mlp = GatedMLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        c: Optional[Tensor] = None,
        inference_params=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))

        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )

        x1, x2 = hidden_states.chunk(2, dim=2)
        x1, _ = self.spatial_mamba(x1, None, c, inference_params)
        x2 = self.freq_mamba(x2)
        if isinstance(self.proj, CrossAttentionFusion):
            x = self.proj(x1, x2)
        else:
            x = torch.cat((x1, x2), dim=2)
            x = self.proj(x)

        # FFN
        hidden_states = hidden_states + x
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(3, dim=1)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm_2(hidden_states), shift_mlp, scale_mlp)
        )

        return hidden_states, residual


class DiMBlockRaw(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
        drop_path=0.0,
        reverse=False,
        transpose=False,
        scanning_continuity=False,
        c_dim=None,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.reverse = reverse
        self.transpose = transpose
        self.scanning_continuity = scanning_continuity
        c_dim = dim if c_dim is None else c_dim

        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(c_dim, 3 * dim, bias=True))

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        c: Optional[Tensor] = None,
        inference_params=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = (
                self.norm(residual)
                if isinstance(self.norm, nn.Identity)
                else self.norm(residual.to(dtype=self.norm.weight.dtype))
            )
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )

        l = hidden_states.shape[1]
        h = w = int(np.sqrt(l))
        if self.transpose:
            hidden_states = rearrange(hidden_states, "n (h w) c -> n (w h) c", h=h, w=w)

        if self.scanning_continuity:
            hidden_states = rearrange(hidden_states.clone(), "n (w h) c -> n c w h", h=h, w=w)
            hidden_states[:, :, 1::2] = hidden_states[:, :, 1::2].flip(-1)
            hidden_states = rearrange(hidden_states, "n c w h -> n (w h) c", h=h, w=w)

        if self.reverse:
            hidden_states = hidden_states.flip(1)

        shift_ssm, scale_ssm, gate_ssm = self.adaLN_modulation(c).chunk(3, dim=1)
        hidden_states = hidden_states + gate_ssm.unsqueeze(1) * self.mixer(
            modulate(hidden_states, shift_ssm, scale_ssm), c, inference_params=inference_params
        )

        # transform back
        if self.reverse:
            hidden_states = hidden_states.flip(1)

        if self.scanning_continuity:
            hidden_states = rearrange(hidden_states.clone(), "n (w h) c -> n c w h", h=h, w=w)
            hidden_states[:, :, 1::2] = hidden_states[:, :, 1::2].flip(-1)
            hidden_states = rearrange(hidden_states, "n c w h -> n (w h) c", h=h, w=w)

        if self.transpose:
            hidden_states = rearrange(hidden_states, "n (h w) c -> n (w h) c", h=h, w=w)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_gated_mlp=True, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_gated_mlp:
            self.mlp = GatedMLP(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        else:
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c=None, **kwargs):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiM(nn.Module,PyTorchModelHubMixin):
    def __init__(
        self,
        img_resolution=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1024,
        depth=16,
        label_dropout=0.1,
        num_classes=1000,
        learn_sigma=False,
        ssm_cfg=None,
        rms_norm=False,
        residual_in_fp32=True,
        fused_add_norm=False,
        scan_type="none",
        initializer_cfg=None,
        num_moe_experts=8,
        mamba_moe_layers=None,
        add_bias_linear=False,
        gated_linear_unit=True,
        routing_mode="top1",
        is_moe=False,
        pe_type="ape",
        block_type="linear",
        cond_mamba=False,
        scanning_continuity=False,
        enable_fourier_layers=False,
        learnable_pe=False,
        skip=False,
        drop_path=0.0,
        use_final_norm=False,
        use_attn_every_k_layers=-1,
        use_gated_mlp=True,
        use_independent_attn=False,
    ):
        super().__init__()
        if block_type == "raw":
            self.depth = int(depth * 3) if not enable_fourier_layers else depth  # int(depth*1.5)
        else:
            self.depth = depth
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.initializer_cfg = initializer_cfg
        self.enable_fourier_layers = enable_fourier_layers
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32
        self.add_before = False
        self.use_attn_every_k_layers = use_attn_every_k_layers

        self.use_independent_attn = use_independent_attn
        if self.use_independent_attn:
            num_transformer_blocks = self.depth // use_attn_every_k_layers - 1
            self.depth = self.depth - self.depth // use_attn_every_k_layers

        # using rotary embedding
        self.pe_type = pe_type
        # block type
        self.block_type = block_type

        self.x_embedder = PatchEmbed(img_resolution, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, label_dropout)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=learnable_pe)

        if self.pe_type == "rope":
            self.emb_sin, self.emb_cos = get_2d_sincos_rotary_embed(hidden_size, int(num_patches**0.5))
            self.emb_sin = torch.from_numpy(self.emb_sin).to(dtype=torch.float32)
            self.emb_cos = torch.from_numpy(self.emb_cos).to(dtype=torch.float32)
        elif self.pe_type == "cpe":
            self.pos_cnn = AdaInPosCNN(hidden_size, hidden_size)

        dpr = [x.item() for x in torch.linspace(0, drop_path, self.depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        grid_size = int(math.sqrt(num_patches))

        def gen_paths(N, scan_type):
            path_type = scan_type.split("_")[0]
            num_paths = int(scan_type.split("_")[1])
            path_gen_fn = SCAN_ZOO[path_type]
            zz_paths = path_gen_fn(N)[:num_paths]

            zz_paths_rev = [reverse_permut_np(x) for x in zz_paths]
            zz_paths = [torch.from_numpy(x)[None,] for x in zz_paths]
            zz_paths_rev = [torch.from_numpy(x)[None,] for x in zz_paths_rev]
            zz_paths = torch.cat(zz_paths * self.depth, dim=0)
            zz_paths_rev = torch.cat(zz_paths_rev * self.depth, dim=0)

            assert len(zz_paths) == len(zz_paths_rev), f"{len(zz_paths)} != {len(zz_paths_rev)}"

            block_kwargs = {}
            block_kwargs["zigzag_paths"] = zz_paths
            block_kwargs["zigzag_paths_reverse"] = zz_paths_rev
            block_kwargs["scan_type"] = scan_type
            return block_kwargs

        if scan_type.startswith("zigma") or scan_type.startswith("sweep") or scan_type.startswith("jpeg"):
            block_kwargs = gen_paths(grid_size, scan_type)
        else:
            block_kwargs = {}
        block_kwargs2 = gen_paths(grid_size, "jpeg_2")  # NOTE: fixed

        self.blocks = nn.ModuleList(
            [
                create_block(
                    hidden_size,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=1e-5,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    scan_type=scan_type,
                    drop_path=inter_dpr[i],
                    num_moe_experts=num_moe_experts,
                    mamba_moe_layers=mamba_moe_layers,
                    add_bias_linear=add_bias_linear,
                    gated_linear_unit=gated_linear_unit,
                    routing_mode=routing_mode,
                    is_moe=is_moe,
                    block_type=block_type,
                    # alternate orders (each ssm handle one order)
                    reverse=(scan_type == "none") and (i % 2 > 0),
                    transpose=(scan_type == "none")
                    and (i % 4 >= 2),
                    cond_mamba=cond_mamba,
                    scanning_continuity=scanning_continuity,
                    use_gated_mlp=use_gated_mlp,
                    block_kwargs=block_kwargs,
                    block_kwargs2=block_kwargs2,
                )
                for i in range(self.depth)
            ]
        )

        if enable_fourier_layers:
            path_type = "jpeg"
            num_paths = 8
            zz_paths, zz_paths_rev = gen_paths(grid_size, path_type, num_paths)
            fourier_block_kwargs = {}
            fourier_block_kwargs["zigzag_paths"] = zz_paths
            fourier_block_kwargs["zigzag_paths_reverse"] = zz_paths_rev
            self.fourier_blocks = nn.ModuleList(
                [
                    WaveDiMBlock(
                        hidden_size,
                        mixer_cls=partial(
                            CondMamba, layer_idx=i, scan_type="none", d_cond=hidden_size, **fourier_block_kwargs
                        ),
                        residual_in_fp32=residual_in_fp32,
                        fused_add_norm=fused_add_norm,
                        drop_path=dpr[i],
                        norm_cls=partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=1e-5),
                        reverse=(i % 2 > 0),
                        transpose=False,
                        scanning_continuity=scanning_continuity,
                    )
                    for i in range(self.depth)
                ]
            )

        if self.use_attn_every_k_layers > 0:
            if self.use_independent_attn:
                self.attn_block = nn.ModuleList(
                    [DiTBlock(hidden_size, 16, use_gated_mlp=use_gated_mlp) for i in range(num_transformer_blocks)]
                )
            else:
                self.attn_block = DiTBlock(hidden_size, 16, use_gated_mlp=use_gated_mlp)

        self.norm_f = (
            (nn.LayerNorm if not rms_norm else RMSNorm)(
                hidden_size,
                eps=1e-5,
            )
            if use_final_norm
            else None
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=self.depth,
                **(self.initializer_cfg if self.initializer_cfg is not None else {}),
            )
        )

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y=None, inference_params=None, **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        if t is None:
            # for compute Gflops
            t = torch.randint(0, 1000, (x.shape[0],), device=x.device)
        if y is None:
            y = torch.ones(x.size(0), dtype=torch.long, device=x.device) * (self.y_embedder.get_in_channels() - 1)
        t = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        c = t + y  # (N, D)

        # add rope !
        if self.pe_type == "ape":
            x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        elif self.pe_type == "rope":
            self.emb_cos = self.emb_cos.to(x.device)
            self.emb_sin = self.emb_sin.to(x.device)
            x = apply_rotary(self.x_embedder(x), self.emb_sin, self.emb_cos)
        elif self.pe_type == "cpe":
            x = self.x_embedder(x)
            h = w = int(self.x_embedder.num_patches**0.5)
            x = self.pos_cnn(x, c, H=h, W=w)
        else:
            raise ("Unsupport PE")

        residual = None
        freq_residual = None
        for idx, block in enumerate(self.blocks):
            if self.add_before and self.enable_fourier_layers:
                if isinstance(self.fourier_blocks[idx], DCTBlock):
                    x, freq_residual = self.fourier_blocks[idx](x, freq_residual)
                elif isinstance(self.fourier_blocks[idx], WaveDiMBlock):
                    x, freq_residual = self.fourier_blocks[idx](x, freq_residual, c)
                else:
                    x = self.fourier_blocks[idx](x)

            if self.pe_type == "ape":
                # ViM raw
                x, residual = block(x, residual, c, inference_params=inference_params)
            elif self.pe_type == "rope":
                # use RoPE
                x, residual = block(x, residual, c, inference_params=inference_params)
            elif self.pe_tpe == "cpe":
                # use CPE
                x, residual = block(x, residual, c, inference_params=inference_params)

            if not self.add_before and self.enable_fourier_layers:
                if isinstance(self.fourier_blocks[idx], DCTBlock):
                    x, freq_residual = self.fourier_blocks[idx](x, freq_residual)
                elif isinstance(self.fourier_blocks[idx], WaveDiMBlock):
                    x, freq_residual = self.fourier_blocks[idx](x, freq_residual, c)
                else:
                    x = self.fourier_blocks[idx](x)

            if self.use_attn_every_k_layers > 0 and (idx + 1) % self.use_attn_every_k_layers == 0:
                if self.use_independent_attn:
                    attn_idx = int((idx + 1) // self.use_attn_every_k_layers - 1)
                    x = self.attn_block[attn_idx](x, c)
                else:
                    x = self.attn_block(x, c)

        if self.norm_f is not None:
            if not self.fused_add_norm:
                if residual is None:
                    residual = x
                else:
                    residual = residual + self.drop_path(x)
                x = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
            else:
                # Set prenorm=False here since we don't need the residual
                fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
                x = fused_add_norm_fn(
                    self.drop_path(x),
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )

        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y=None, inference_params=None, cfg_scale=1.0, **kwargs):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y, inference_params)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def forward_with_adacfg(self, x, t, y=None, inference_params=None, cfg_scale=3.8, scale_pow=4.0, **kwargs):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        if cfg_scale is not None:
            half = x[: len(x) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.forward(combined, t, y, inference_params)
            eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            scale_step = (1 - torch.cos(((1 - t) ** scale_pow) * math.pi)) * 1 / 2  # power-cos scaling
            real_cfg_scale = (cfg_scale - 1) * scale_step + 1
            real_cfg_scale = real_cfg_scale[: len(x) // 2].view(-1, 1, 1, 1)

            half_eps = uncond_eps + real_cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)
        else:
            model_out = self.forward(x, t, y, inference_params)
            eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
            return torch.cat([eps, rest], dim=1)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: blk.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, blk in enumerate(self.blocks)
        }


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.0,
    rms_norm=False,
    residual_in_fp32=True,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    scan_type="none",
    add_bias_linear=False,
    gated_linear_unit=True,
    routing_mode: str = "sinkhorn",  # 'sinkhorn', 'top1', 'top2', 'sinkhorn_top2'
    num_moe_experts: int = 8,
    mamba_moe_layers: list = None,
    is_moe: bool = False,
    block_type="linear",
    reverse=False,
    transpose=False,
    cond_mamba=False,  # conditional mode
    scanning_continuity=False,
    skip=False,
    use_gated_mlp=True,
    block_kwargs={},
    block_kwargs2={},
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs)
    if layer_idx % 2 == 0 or not is_moe:
        if cond_mamba:
            mixer_cls = partial(
                CondMamba,
                layer_idx=layer_idx,
                scan_type=scan_type,
                d_cond=d_model,
                **ssm_cfg,
                **block_kwargs,
                **factory_kwargs,
            )
        else:
            mixer_cls = partial(Mamba, layer_idx=layer_idx, scan_type=scan_type, **ssm_cfg, **factory_kwargs)
        if block_type == "raw":
            block = DiMBlockRaw(
                d_model,
                mixer_cls,
                norm_cls=norm_cls,
                drop_path=drop_path,
                fused_add_norm=fused_add_norm,
                residual_in_fp32=residual_in_fp32,
                reverse=reverse,
                transpose=transpose,
                scanning_continuity=scanning_continuity,
            )
        elif block_type == "wave":
            block = WaveDiMBlock(
                d_model,
                mixer_cls,
                norm_cls=norm_cls,
                drop_path=drop_path,
                fused_add_norm=fused_add_norm,
                residual_in_fp32=residual_in_fp32,
                reverse=reverse,
                transpose=transpose,
                scanning_continuity=scanning_continuity,
                skip=skip,
                window_scan=False,
            )
        elif block_type == "window":
            block = DiMBlockWindow(
                d_model,
                mixer_cls,
                norm_cls=norm_cls,
                drop_path=drop_path,
                fused_add_norm=fused_add_norm,
                residual_in_fp32=residual_in_fp32,
                reverse=False,
                transpose=reverse,
                scanning_continuity=scanning_continuity,
                skip=skip,
                shift_window=False,
            )
        elif block_type == "combined":
            block = DiMBlockCombined(
                d_model,
                mixer_cls,
                norm_cls=norm_cls,
                drop_path=drop_path,
                fused_add_norm=fused_add_norm,
                residual_in_fp32=residual_in_fp32,
                reverse=reverse,
                transpose=transpose,
                scanning_continuity=scanning_continuity,
                use_gated_mlp=use_gated_mlp,
            )
        elif block_type == "combined_fourier":
            mixer_cls_2 = partial(
                CondMamba, layer_idx=layer_idx, d_cond=d_model, **ssm_cfg, **block_kwargs2, **factory_kwargs
            )
            block = DiMBlockCombinedFourier(
                d_model,
                mixer_cls,
                mixer_cls_2,
                norm_cls=norm_cls,
                drop_path=drop_path,
                fused_add_norm=fused_add_norm,
                residual_in_fp32=residual_in_fp32,
                reverse=reverse,
                transpose=transpose,
                scanning_continuity=scanning_continuity,
                use_gated_mlp=use_gated_mlp,
            )
        elif block_type == "combined_einfft":
            block = DiMBlockCombinedEinFFT(
                d_model,
                mixer_cls,
                norm_cls=norm_cls,
                drop_path=drop_path,
                fused_add_norm=fused_add_norm,
                residual_in_fp32=residual_in_fp32,
                reverse=reverse,
                transpose=transpose,
                scanning_continuity=scanning_continuity,
                use_gated_mlp=use_gated_mlp,
            )
        else:
            block = DiMBlock(
                d_model,
                mixer_cls,
                norm_cls=norm_cls,
                drop_path=drop_path,
                fused_add_norm=fused_add_norm,
                residual_in_fp32=residual_in_fp32,
                reverse=reverse,
                transpose=transpose,
                scanning_continuity=scanning_continuity,
                skip=skip,
            )
    else:
        mixer_cls = partial(
            SwitchMLP,
            layer_idx=layer_idx,
            add_bias_linear=add_bias_linear,
            gated_linear_unit=gated_linear_unit,
            routing_mode=routing_mode,
            num_moe_experts=num_moe_experts,
            mamba_moe_layers=mamba_moe_layers,
        )
        block = MoEBlock(
            d_model,
            mixer_cls=mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
    block.layer_idx = layer_idx
    return block


def DiM_XL_2(**kwargs):
    return DiM(
        depth=24,  # 28, double 28 if use DiMBlockRaw
        hidden_size=1152,
        patch_size=2,
        initializer_cfg=None,
        ssm_cfg=None,
        **kwargs,
    )


def DiM_L_2(**kwargs):
    return DiM(
        depth=16,  # 24, double 24 if use DiMBlockRaw
        hidden_size=1024,
        patch_size=2,
        initializer_cfg=None,
        ssm_cfg=None,
        **kwargs,
    )


def DiM_L_2_v1(**kwargs):
    return DiM(
        depth=20,  # 24, double 24 if use DiMBlockRaw
        hidden_size=1024,
        patch_size=2,
        initializer_cfg=None,
        ssm_cfg=None,
        **kwargs,
    )


def DiM_B_2(**kwargs):
    return DiM(
        depth=12,  # 12, double 12 if use DiMBlockRaw
        hidden_size=768,
        patch_size=2,
        initializer_cfg=None,
        ssm_cfg=None,
        **kwargs,
    )


def DiM_L_4(**kwargs):
    return DiM(
        depth=16,  # 24, double 24 if use DiMBlockRaw
        hidden_size=1024,
        patch_size=4,
        initializer_cfg=None,
        ssm_cfg=None,
        **kwargs,
    )


def DiM_L_4_v1(**kwargs):
    return DiM(
        depth=20,  # 24, double 24 if use DiMBlockRaw
        hidden_size=1024,
        patch_size=4,
        initializer_cfg=None,
        ssm_cfg=None,
        **kwargs,
    )


DiM_models = {
    "DiM-XL/2": DiM_XL_2,
    "DiM-L/2": DiM_L_2,
    "DiM-L/2-v1": DiM_L_2_v1,
    "DiM-B/2": DiM_B_2,
    "DiM-L/4": DiM_L_4,
    "DiM-L/4-v1": DiM_L_4_v1,
}
