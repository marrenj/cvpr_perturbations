# src/models/factory.py

import torch
from torchvision.models import vit_b_16, vit_l_14, resnet50  # or vit_b_8, etc.

def build_model(arch_cfg):
    model_type = arch_cfg.get('type', 'CLIP')
    if model_type == 'ViT-B/16':
        # Create a Vision Transformer with specified parameters
        model = vit_b_16(pretrained=arch_cfg.get('pretrained', False))
        # Optionally resize or replace the head if num_classes differs
        # e.g., model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
        return model

    elif model_type == 'ViT-L/14':
        # Create a Vision Transformer with specified parameters
        model = vit_l_14(pretrained=arch_cfg.get('pretrained', False))
        # Optionally resize or replace the head if num_classes differs
        # e.g., model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
        return model

    elif model_type == 'RN50':
        # Create a ResNet with specified parameters
        model = resnet50(pretrained=arch_cfg.get('pretrained', False))
        # Optionally resize or replace the head if num_classes differs
        # e.g., model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
        return model

    elif model_type == 'CLIP-HBA':
        # Existing CLIP-HBA logic, e.g.:
        from src.models.clip_hba.clip_hba_utils import initialize_cliphba_model
        backbone = arch_cfg.get('backbone', 'RN50')
        model = initialize_cliphba_model(backbone)  # original code path
        return model

    else:
        raise ValueError(f"Unknown architecture type: {model_type}")