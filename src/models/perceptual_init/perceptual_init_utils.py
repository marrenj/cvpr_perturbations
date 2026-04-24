"""
Utilities for loading perceptual-init yfcc3m ViT-B/32 checkpoints.

Checkpoints are PyTorch Lightning .ckpt files produced by
bi_clip_pipeline_for_perturbations. The visual encoder weights live under
the 'model.visual.' prefix inside the Lightning state dict.
"""
from pathlib import Path
from typing import Optional
import pickle
import re

import torch


class _TolerantUnpickler(pickle.Unpickler):
    """Return a stub class for any global we can't import.

    Lightning checkpoints pickle hyperparameter objects referencing the
    training project's modules (e.g. a top-level `config` package). Those
    modules aren't on sys.path at inference time, but we only consume the
    tensor state_dict, so stubbing unresolved classes lets the rest of the
    checkpoint deserialize without error.
    """

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return type(
                f"_Stub_{module}_{name}".replace(".", "_"),
                (),
                {
                    "__init__": lambda self, *a, **kw: None,
                    "__setstate__": lambda self, state: None,
                    "__reduce__": lambda self: (object, ()),
                },
            )


class _TolerantPickleModule:
    Unpickler = _TolerantUnpickler
    UnpicklingError = pickle.UnpicklingError
    load = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)

from src.models.perceptual_init.model import (
    VisionTransformer,
    PerceptualInitCLIPWrapper,
    LayerNorm,
)


# ViT-B/32 config taken verbatim from
# bi_clip_pipeline_for_perturbations/config/modelconfig/vit_b_32.json
_VITB32_CONFIG = dict(
    image_size=224,
    patch_size=32,
    width=768,
    layers=12,
    heads=12,        # width (768) // head_width (64)
    mlp_ratio=4.0,
    output_dim=512,  # embed_dim
    ls_init_value=None,
    patch_dropout=0.0,
    no_ln_pre=False,
    pos_embed_type="learnable",
)


def create_vitb32_visual_encoder() -> VisionTransformer:
    """Instantiate the ViT-B/32 visual encoder matching the bi_clip architecture."""
    return VisionTransformer(**_VITB32_CONFIG)


def load_visual_state_dict_from_lightning_checkpoint(
    checkpoint_path: str,
    map_location: str = "cpu",
    visual_prefix: str = "model.visual.",
) -> dict:
    """
    Extract the visual encoder state dict from a Lightning .ckpt file.

    Strips the 'model.visual.' prefix so the returned dict loads directly
    into VisionTransformer via load_state_dict(strict=True).
    """
    ckpt = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
        pickle_module=_TolerantPickleModule,
    )
    state_dict = ckpt.get("state_dict", ckpt)

    visual_sd = {
        k[len(visual_prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(visual_prefix)
    }

    if not visual_sd:
        top_keys = list(state_dict.keys())[:15]
        raise RuntimeError(
            f"No keys found with prefix '{visual_prefix}' in {checkpoint_path}. "
            f"Top-level keys (first 15): {top_keys}"
        )
    return visual_sd


def initialize_perceptual_init_model(
    checkpoint_path: str,
    device: torch.device,
    logger=None,
) -> PerceptualInitCLIPWrapper:
    """
    Build a PerceptualInitCLIPWrapper with weights loaded from a
    bi_clip_pipeline_for_perturbations Lightning checkpoint.
    """

    def _log(msg):
        if logger:
            logger.info(msg)

    _log("Creating ViT-B/32 visual encoder (bi_clip architecture).")
    visual_encoder = create_vitb32_visual_encoder()

    _log(f"Loading visual weights from: {checkpoint_path}")
    visual_sd = load_visual_state_dict_from_lightning_checkpoint(
        checkpoint_path, map_location=str(device)
    )
    missing, unexpected = visual_encoder.load_state_dict(visual_sd, strict=True)
    if missing:
        _log(f"  Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        _log(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    _log(f"  Loaded {len(visual_sd)} tensors into visual encoder.")

    model = PerceptualInitCLIPWrapper(visual_encoder).to(device)
    model.eval()
    return model


def parse_epoch_from_lightning_filename(filename: str) -> str:
    """
    Parse the epoch number from a Lightning checkpoint filename.

    Handles common patterns:
      epoch=5-step=600.ckpt  ->  '5'
      epoch_5.ckpt           ->  '5'
      last.ckpt              ->  'last'
    """
    m = re.search(r"epoch[=\-_]?(\d+)", filename, re.IGNORECASE)
    if m:
        return m.group(1)
    if "last" in filename.lower():
        return "last"
    return Path(filename).stem


def sorted_lightning_checkpoints(ckpt_dir: Path):
    """
    Return all .ckpt files in ckpt_dir sorted by epoch number.
    Files without a parsable epoch number (e.g. last.ckpt) are appended last.
    """
    files = list(ckpt_dir.glob("*.ckpt"))

    def _sort_key(p):
        m = re.search(r"epoch[=\-_]?(\d+)", p.name, re.IGNORECASE)
        return int(m.group(1)) if m else 10 ** 9

    return sorted(files, key=_sort_key)
