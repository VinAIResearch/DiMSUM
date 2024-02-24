##########
# There is few notes for designing model mamba for diffusion.
# 1. How to integrate condition data and time data
# a) using Uvit. Concat directly with time+cond embedding (easy to do, low recommandation)
# b) using DiT. Use adaIn module (need to investigate more)
# c) Consider paper recently such as MDT and DiffiT
# d) rms or layernorm (most gen model use layer norm) ? ==> I think layer norm is better
# e) should we use fused norm or not ?


# 2. Choose the backbone vision mamba. Consider Umamba, Vim and VMamba (need to read more about these paper)
# Vim: Why both direction ? Why using causal conv1d ? Can we integrate other transformer design technique (SE network, SwinTrans) ?
#
#

import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
import math
import numpy as np


from timm.models.vision_transformer import VisionTransformer, _cfg, Mlp, PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath
from timm.models.vision_transformer import _load_weights


import einops
import torch.utils.checkpoint
from timm.models.layers import DropPath
from timm.models.vision_transformer import _load_weights

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
    
    
__all__ = [
    'vim_tiny_patch16_224', 'vim_small_patch16_224', 'vim_base_patch16_224',
    'vim_tiny_patch16_384', 'vim_small_patch16_384', 'vim_base_patch16_384',
]

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

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


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
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb



class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, cond_mode):
        super().__init__()
        self.cond_mode = cond_mode
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        if self.cond_mode == "AdaIn":
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )

    def forward(self, x, c=None):
        x = self.norm_final(x)
        if c is not None and self.cond_mode == "AdaIn":
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            x = modulate(x, shift, scale)
        x = self.linear(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 mixer_cls, 
                 cond_mode,
                 norm_cls=nn.LayerNorm, 
                 fused_add_norm=False, 
                 residual_in_fp32=False,
                 drop_path=0., 
                 skip=False,):
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
        
        My comment: They use the simple architecture here but we can adapt it
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.norm = norm_cls(dim)
        # this one have dropout while mamba_simple file does not have
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
            
        if self.cond_mode == "AdaIn":
            self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 3 * dim, bias=True)
        )

    def forward(self, 
                hidden_states: Tensor, 
                residual: Optional[Tensor] = None, 
                inference_params=None, 
                skip=None,
                c=None):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if c is not None and self.cond_mode == "AdaIn":
            shift_msa, scale_msa, gate_msa = self.adaLN_modulation(c).chunk(3, dim=1)
            
        if self.skip_linear is not None:
            hidden_states = self.skip_linear(torch.cat([hidden_states, skip], dim=-1))
        
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
        if c is not None and self.cond_mode == "AdaIn":
            hidden_states = modulate(hidden_states, shift_msa, scale_msa)      
            hidden_states = gate_msa.unsqueeze(1)*self.mixer(hidden_states, inference_params=inference_params)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    
    
class TransBlock(nn.Module):
    def __init__(self, 
                dim, 
                mixer_cls, 
                cond_mode,
                norm_cls=nn.LayerNorm, 
                fused_add_norm=False, 
                residual_in_fp32=False,
                drop_path=0., 
                skip=False,
                mlp_ratio=4,):
        
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.norm = norm_cls(dim)
        self.norm_linear = norm_cls(dim)
        self.cond_mode = cond_mode
        # this one have dropout while mamba_simple file does not have
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
            
            assert isinstance(
                self.norm_linear, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
            
        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        if self.cond_mode == "AdaIn":
            self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
            
    def forward(self, 
                hidden_states: Tensor,
                residual: Optional[Tensor] = None, 
                inference_params=None, 
                skip=None, 
                c=None,):
        
        if c is not None and self.cond_mode == "AdaIn":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        if self.skip_linear is not None:
            hidden_states = self.skip_linear(torch.cat([hidden_states, skip], dim=-1))
        
        
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
        if c is not None and self.cond_mode == "AdaIn":
            hidden_states = modulate(hidden_states, shift_msa, scale_msa)      
            hidden_states = gate_msa.unsqueeze(1)*self.mixer(hidden_states, inference_params=inference_params)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        
        if not self.fused_add_norm:
            residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_linear, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_linear.weight,
                self.norm_linear.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm_linear.eps,
            )
        if c is not None and self.cond_mode == "AdaIn":
            hidden_states = modulate(hidden_states, shift_mlp, scale_mlp)
            hidden_states = gate_mlp.unsqueeze(1)*self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    skip=False,
    bimamba_type="none",
    blk_type="simple",
    mlp_ratio=4,
    cond_mode="concat",
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba_type=bimamba_type, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if blk_type == "simple":
        block = Block(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            drop_path=drop_path,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            skip=skip,
            cond_mode=cond_mode
        )
    elif blk_type == "trans":
        block = TransBlock(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            drop_path=drop_path,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            skip=skip,
            mlp_ratio=mlp_ratio,
            cond_mode=cond_mode
        )
    block.layer_idx = layer_idx
    return block


class MambaDiffV1(nn.Module):
    ### Design based on Uvit
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3,
                 learn_sigma=False,
                 embed_dim=768, 
                 depth=12, 
                 mlp_time_embed=False, 
                 num_classes=-1,
                 skip=True,
                 conv=True,
                 ssm_cfg=None, 
                 class_drop_rate=0.,
                 drop_path_rate=0.0, # using DropPath (usal for classification)
                 norm_epsilon: float = 1e-6, 
                 rms_norm: bool = False, 
                 initializer_cfg=None,
                 fused_add_norm=False, # use fused norm from Tri Dao repo
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 bimamba_type="none", # bidirection mamba or not
                 cond_mode="AdaIn", # concat or adain (concat following UViT and AdaIn following DiT)
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs) 
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.out_chans = in_chans * 2 if learn_sigma else in_chans
        self.cond_mode = cond_mode
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        # Create Embedding Layer 
        self.x_embedder = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.x_embedder.num_patches
        self.t_embedder = TimestepEmbedder(embed_dim)
        if self.num_classes > 0:
            self.y_embedder = LabelEmbedder(self.num_classes, embed_dim, class_dropout_prob=class_drop_rate)

        if not self.cond_mode == "AdaIn":
            if self.num_classes > 0:
                self.extras = 2
            else:
                self.extras = 1
            self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        
        # Setup dropout for Mamba layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        
        # Create Mamba block layer
        self.in_blocks = nn.ModuleList([
                create_block(
                        embed_dim,
                        ssm_cfg=ssm_cfg,
                        norm_epsilon=norm_epsilon,
                        rms_norm=rms_norm,
                        residual_in_fp32=residual_in_fp32,
                        fused_add_norm=fused_add_norm,
                        layer_idx=i,
                        bimamba_type=bimamba_type,
                        drop_path=inter_dpr[i],
                        blk_type="trans",
                        cond_mode=cond_mode,
                        **factory_kwargs,
                ) for i in range(depth // 2)])

        self.mid_block = create_block(
                        embed_dim,
                        ssm_cfg=ssm_cfg,
                        norm_epsilon=norm_epsilon,
                        rms_norm=rms_norm,
                        residual_in_fp32=residual_in_fp32,
                        fused_add_norm=fused_add_norm,
                        layer_idx=(depth // 2),
                        bimamba_type=bimamba_type,
                        drop_path=inter_dpr[depth // 2],
                        blk_type="trans",
                        cond_mode=cond_mode,
                        **factory_kwargs,
                )

        self.out_blocks = nn.ModuleList([
                create_block(
                        embed_dim,
                        ssm_cfg=ssm_cfg,
                        norm_epsilon=norm_epsilon,
                        rms_norm=rms_norm,
                        residual_in_fp32=residual_in_fp32,
                        fused_add_norm=fused_add_norm,
                        layer_idx=i,
                        bimamba_type=bimamba_type,
                        drop_path=inter_dpr[i],
                        skip=skip,
                        blk_type="trans",
                        cond_mode=cond_mode,
                        **factory_kwargs,
                ) for i in range(depth // 2 + 1, depth + 1)])

        # Setup Final Layer
        self.final_layer = FinalLayer(embed_dim, patch_size, self.out_chans, cond_mode=cond_mode)
        self.final_conv = nn.Conv2d(self.in_chans, self.in_chans, 3, padding=1) if conv else nn.Identity()

        # Initialize weight for model (positional encoding is fixed while time and class is trainable)
        self.initialize_weights()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}
    
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        if self.cond_mode == "AdaIn":
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        else:
            trunc_normal_(self.pos_embed, std=.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        if self.num_classes > 0:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in Mamba blocks:
        if self.cond_mode == "AdaIn":
            for block in self.in_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            for block in self.out_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.mid_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.mid_block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        if self.cond_mode == "AdaIn":
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    
    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_chans
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    

    def forward(self, x, timesteps, y=None, inference_params=None):
        x = self.x_embedder(x)
        B, L, D = x.shape
        t = self.t_embedder(timesteps)
        c = None
        if self.cond_mode == "concat":
            t = t.unsqueeze(1)
            x = torch.cat((t, x), dim=1)
            if y is not None:
                y = self.y_embedder(y, self.training)
                y = y.unsqueeze(1)
                x = torch.cat((y, x), dim=1)
        else:
            if y is not None:
                y = self.y_embedder(y, self.training)
                c = t + y
            else:
                c = t
        x = x + self.pos_embed
        
        # Run the transformer block
        residual = None
        skips = []
        for blk in self.in_blocks:
            x, residual = blk(x, residual, inference_params=inference_params, c = c)
            skips.append(x)
        x, residual = self.mid_block(x, residual, inference_params=inference_params, c = c)
        for blk in self.out_blocks:
            x, residual = blk(x, residual, inference_params=inference_params, skip=skips.pop(), c = c)

        # Run the final block
        x = self.final_layer(x, c)
        if not self.cond_mode == "AdaIn":
            assert x.size(1) == self.extras + L
            x = x[:, self.extras:, :]
        x = self.unpatchify(x)
        x = self.final_conv(x)
        return x
    
    
# model = MambaDiffV1(img_size=32,
#                     patch_size=2, 
#                     in_chans=4, 
#                     embed_dim=768, 
#                     depth=12,
#                     mlp_time_embed=True, 
#                     num_classes=-1,
#                     skip=True,
#                     conv=True,
#                     ssm_cfg=None, 
#                     drop_rate=0.,
#                     drop_path_rate=0.0,
#                     norm_epsilon=1e-5, 
#                     rms_norm=True, 
#                     initializer_cfg=None,
#                     fused_add_norm=True, 
#                     residual_in_fp32=True, 
#                     bimamba_type="v2").to("cuda")

# x = torch.randn(2,4,32,32).to("cuda")
# t = torch.randint(0, 1000, (2, )).to("cuda")
# out = model(x, t)
# print(out.shape)

def MambaDiffV1_XL_2():
    # note: most of generative model using layer norm instead of rms norm
    return MambaDiffV1(img_size=32,
                        patch_size=2, 
                        in_chans=4, 
                        embed_dim=768, 
                        depth=12,
                        mlp_time_embed=True, 
                        num_classes=-1,
                        skip=False,
                        conv=True,
                        ssm_cfg=None, 
                        class_drop_rate=0.,
                        drop_path_rate=0.0,
                        norm_epsilon=1e-6, 
                        rms_norm=False, 
                        initializer_cfg=None,
                        fused_add_norm=True, 
                        residual_in_fp32=True, 
                        bimamba_type="v2",
                        cond_mode="concat").to("cuda")
    
    
mamba_models = {
    'MambaDiffV1_XL_2': MambaDiffV1_XL_2
}