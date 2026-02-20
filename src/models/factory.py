# src/models/factory.py
import torch
from torch.nn import DataParallel
import timm
import wandb

from src.models.clip_hba.clip_hba_utils import (
    CLIPHBA,
    apply_dora_to_ViT,
    switch_dora_layers,
)
from src.data.spose_dimensions import classnames66


# Maps the user-facing architecture names to timm model identifiers
_TIMM_ARCH_MAP = {
    "ViT-B/16": "vit_base_patch16_224",
    "ViT-L/14": "vit_large_patch14_224",
    "RN50":     "resnet50",
    "RN101":    "resnet101",
}


def build_cliphba(
    backbone, vision_layers, transformer_layers, rank,
    cuda, device, wandb_watch_model, wandb_log_freq,
):
    """Build a CLIP-HBA model with DoRA adapters for fine-tuning."""
    pos_embedding = backbone != "RN50"
    print(f"pos_embedding is {pos_embedding}")

    model = CLIPHBA(
        classnames=classnames66,
        backbone_name=backbone,
        pos_embedding=pos_embedding,
    )

    apply_dora_to_ViT(
        model,
        n_vision_layers=vision_layers,
        n_transformer_layers=transformer_layers,
        r=rank,
        dora_dropout=0.1,
    )
    switch_dora_layers(model, freeze_all=True, dora_state=True)

    if cuda == -1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)

    model.to(device)

    if wandb_watch_model:
        wandb.watch(model, log="all", log_freq=wandb_log_freq)

    return model


def build_timm_model(
    architecture, pretrained, num_classes,
    cuda, device, wandb_watch_model, wandb_log_freq,
):
    """Build a ViT or ResNet model via timm for from-scratch training."""
    timm_name = _TIMM_ARCH_MAP.get(architecture)
    if timm_name is None:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Supported timm architectures: {list(_TIMM_ARCH_MAP)}"
        )

    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)

    if cuda == -1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)

    model.to(device)

    if wandb_watch_model:
        wandb.watch(model, log="all", log_freq=wandb_log_freq)

    return model


def build_model(
    architecture,
    pretrained,
    clip_hba_backbone,
    vision_layers,
    transformer_layers,
    rank,
    cuda,
    device,
    wandb_watch_model,
    wandb_log_freq,
    num_classes=1000,
):
    """
    Factory that returns a configured, device-ready model.

    Args:
        architecture:      ``'CLIP-HBA'`` for fine-tuning, or a timm architecture
                           name (``'ViT-B/16'``, ``'ViT-L/14'``, ``'RN50'``,
                           ``'RN101'``) for from-scratch training.
        pretrained:        Load pretrained weights from timm (scratch only).
        clip_hba_backbone: CLIP backbone name (fine-tuning only).
        vision_layers:     DoRA vision layers (fine-tuning only).
        transformer_layers: DoRA text layers (fine-tuning only).
        rank:              DoRA rank (fine-tuning only).
        cuda:              -1 for all GPUs, 0/1 for a specific GPU.
        device:            torch.device already configured.
        wandb_watch_model: Whether to call wandb.watch on the model.
        wandb_log_freq:    Frequency for wandb gradient logging.
        num_classes:       Output head size for timm models (default 1000 for
                           ImageNet; ignored for CLIP-HBA).

    Returns:
        torch.nn.Module: Configured model on the requested device.
    """
    if architecture == "CLIP-HBA":
        return build_cliphba(
            clip_hba_backbone,
            vision_layers,
            transformer_layers,
            rank,
            cuda,
            device,
            wandb_watch_model,
            wandb_log_freq,
        )

    return build_timm_model(
        architecture,
        pretrained,
        num_classes,
        cuda,
        device,
        wandb_watch_model,
        wandb_log_freq,
    )
