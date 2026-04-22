"""
Custom ViT-B/32 vision encoder ported from bi_clip_pipeline_for_perturbations.

Matches the exact architecture used to train the perceptual-init yfcc3m
checkpoints so that weights can be loaded without key-mapping or activation
function mismatches. State-dict keys are identical to those in the Lightning
.ckpt files (after stripping the 'model.visual.' prefix).
"""
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        return super().forward(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if hasattr(F, "scaled_dot_product_attention"):
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            x = x.transpose(1, 2).reshape(B, N, C)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        ls_init_value: Optional[float] = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop,
        )
        if ls_init_value is not None:
            self.gamma_1 = nn.Parameter(ls_init_value * torch.ones(dim))
            self.gamma_2 = nn.Parameter(ls_init_value * torch.ones(dim))
        else:
            self.gamma_1 = None
            self.gamma_2 = None
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # Indices 0 and 3 match bi_clip: Linear, GELU, Dropout, Linear, Dropout.
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor):
        if self.gamma_1 is None:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        else:
            x = x + self.gamma_1 * self.attn(self.norm1(x))
            x = x + self.gamma_2 * self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer matching the bi_clip_pipeline_for_perturbations
    architecture exactly. State-dict keys are compatible with checkpoints
    saved from that pipeline (after stripping the 'model.visual.' prefix).
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 32,
        width: int = 768,
        layers: int = 12,
        heads: int = 12,
        mlp_ratio: float = 4.0,
        output_dim: int = 512,
        ls_init_value: Optional[float] = None,
        patch_dropout: float = 0.0,
        no_ln_pre: bool = False,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        pos_embed_type: str = "learnable",
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.width = width
        self.output_dim = output_dim
        grid_size = image_size // patch_size
        self.num_patches = grid_size * grid_size

        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.class_embedding = nn.Parameter(torch.randn(width))
        self.positional_embedding = nn.Parameter(
            torch.randn(self.num_patches + 1, width) * 0.01
        )
        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)
        self.patch_dropout = patch_dropout
        self.transformer = nn.ModuleList([
            TransformerBlock(
                dim=width, num_heads=heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, ls_init_value=ls_init_value,
                act_layer=act_layer, norm_layer=norm_layer,
            )
            for _ in range(layers)
        ])
        self.ln_post = norm_layer(width)
        self.proj = nn.Parameter(torch.randn(width, output_dim) * (1.0 / width ** 0.5))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls_tok = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_tok, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        for block in self.transformer:
            x = block(x)
        x = x[:, 0]
        x = self.ln_post(x)
        x = x @ self.proj
        return x


class PerceptualInitCLIPWrapper(nn.Module):
    """
    Wraps the perceptual-init ViT-B/32 visual encoder for use in the
    cvpr_perturbations inference pipeline.

    forward() returns L2-normalised 512-d image features, which are directly
    compatible with compute_model_rdm and the behavioral RSA pipeline.
    """

    def __init__(self, visual_encoder: VisionTransformer):
        super().__init__()
        self.visual = visual_encoder

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.visual(images)
        return F.normalize(features, dim=-1)
