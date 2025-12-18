import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import numpy as np

from torch.nn import functional as F
from tqdm import tqdm

from torch.optim import AdamW
from torch.nn import DataParallel

import random
import math
from functions.spose_dimensions import *
import sys
sys.path.append('../')
from src.models.CLIPs.clip_hba import clip

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import logging
import sys
from datetime import datetime
import csv
import os
import scipy.io
from scipy.stats import spearmanr


def seed_everything(seed):
    # Set the seed for PyTorch's random number generators
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set the seed for Python's random number generator
    random.seed(seed)

    # Set the seed for NumPy's random number generator
    np.random.seed(seed)

    # Ensure that the CuDNN backend is deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_file_path):
    """
    Set up logger to write to both console and file.
    
    Args:
        log_file_path (str): Path to the log file
    
    Returns:
        logger: Configured logger object
    """
    # Create logger
    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                datefmt='%Y-%m-%d %H:%M:%S')
    
    # File handler
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def load_random_states(random_state_path, epoch, optimizer=None, dataloader_generator=None, logger=None):
    """
    Load all random states from a checkpoint for 100% reproducibility.
    
    Args:
        random_state_path: Directory where checkpoints are saved
        epoch: Epoch number to load from
        optimizer: Optional optimizer to restore state
        dataloader_generator: Optional generator to restore state
        logger: Optional logger for logging messages
    
    Returns:
        bool: True if successful, False otherwise
    """
    log = logger.info if logger else print
    
    checkpoint_file = os.path.join(random_state_path, f"epoch{epoch}_random_states.pth")
    
    if not os.path.exists(checkpoint_file):
        log(f"Warning: Random state checkpoint not found: {checkpoint_file}")
        return False
    
    checkpoint = torch.load(checkpoint_file)
    
    # Restore all random states
    torch.set_rng_state(checkpoint['torch_rng_state'])
    np.random.set_state(checkpoint['numpy_rng_state'])
    random.setstate(checkpoint['python_rng_state'])

    # Restore CUDA random states if available
    if torch.cuda.is_available() and 'cuda_rng_state' in checkpoint:
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        if 'cuda_rng_state_all' in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state_all'])
    
    # Restore optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        log(f"Restored optimizer state from epoch {epoch}")
    
    # Restore DataLoader generator state if provided
    if dataloader_generator is not None and 'dataloader_generator_state' in checkpoint:
        dataloader_generator.set_state(checkpoint['dataloader_generator_state'])
        log(f"Restored DataLoader generator state from epoch {epoch}")
    
    log(f"Random states loaded from: {checkpoint_file}")
    return True


def load_dataset_split_indices(split_indices_path, logger=None):
    """
    Load dataset split indices from a saved checkpoint.
    
    Args:
        split_indices_path: Path to the saved split indices file
        logger: Optional logger for logging messages
    
    Returns:
        dict: Dictionary containing train_indices, test_indices, and metadata
              or None if file doesn't exist
    """
    log = logger.info if logger else print
    
    if not os.path.exists(split_indices_path):
        log(f"Split indices file not found: {split_indices_path}")
        return None
    
    split_info = torch.load(split_indices_path)
    log(f"Loaded dataset split indices from: {split_indices_path}")
    log(f"  Train samples: {len(split_info['train_indices'])}")
    log(f"  Test samples: {len(split_info['test_indices'])}")
    log(f"  Random seed used: {split_info['random_seed']}")
    
    return split_info


class SubsetWithIndices(Dataset):
    """
    Subset of a dataset at specified indices, similar to torch.utils.data.Subset
    but allows explicit index specification.
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    
    def __len__(self):
        return len(self.indices)


class ThingsDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.img_dir = img_dir
        self.transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.52997664, 0.48070561, 0.41943838],
                                            std=[0.27608301, 0.26593025, 0.28238822])
                    ])

        # Load and filter annotations based on the 'set' column
        self.annotations = pd.read_csv(csv_file, index_col=0)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_name = self.annotations.iloc[index, 0]
        img_path = os.path.join(self.img_dir, image_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        targets = torch.tensor(self.annotations.iloc[index, 1:].values.astype('float32'))
        
        return image_name, image, targets


def replace_with_gaussian_noise(image, mean, std):
    """
    Replace an image with Gaussian noise.
    
    Args:
        image: Input image tensor of shape [C, H, W]
        mean: Mean of the Gaussian distribution
        std: Standard deviation of the Gaussian distribution
    
    Returns:
        Tensor of Gaussian noise with the same shape as input image
    """
    C, H, W = image.size()
    noise = torch.randn((C, H, W), device=image.device) * std + mean
    return noise


# create another dataset class for the inference data (48 Things images)
class ThingsInferenceDataset(Dataset):
    def __init__(self, inference_csv_file, img_dir, RDM48_triplet_dir):
        self.img_dir = img_dir
        self.RDM48_triplet_dir = RDM48_triplet_dir
        self.transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.52997664, 0.48070561, 0.41943838],
                                            std=[0.27608301, 0.26593025, 0.28238822])
                    ])

        # Load and filter annotations based on the 'set' column
        self.annotations = pd.read_csv(inference_csv_file, index_col=0)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_name = self.annotations.iloc[index, 0]
        img_path = os.path.join(self.img_dir, image_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image_name, image


def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, os.path.expanduser("~/.cache/clip"))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class CLIPHBA(nn.Module):
    def __init__(self, classnames, backbone_name='RN50', pos_embedding=False):
        super().__init__()

        self.num_clip = len(classnames)
        self.clip_model = load_clip_to_cpu(backbone_name)
        self.clip_model.float()
        self.pos_embedding = pos_embedding

        # Disable gradients for all parameters first
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Tokenize all prompts at once and store them as a tensor
        self.tokenized_prompts = torch.stack([clip.tokenize(classname) for classname in classnames])
        self._cached_tokenized_prompts = None
        self._cached_device = None


    def forward(self, image):
        if self.clip_model.training:
            self.clip_model.eval()

        # Cache tokenized prompts on the device after first use
        device = image.device
        if self._cached_tokenized_prompts is None or self._cached_device != device:
            self._cached_tokenized_prompts = self.tokenized_prompts.to(device)
            self._cached_device = device

        # Process all tokenized prompts in a single forward pass
        pred_score = self.clip_model(image, self._cached_tokenized_prompts, self.pos_embedding)

        pred_score = pred_score.float()  # Adjust the dimensions accordingly

        # print(f"pred_score: {pred_score}")

        return pred_score


class LoRALayer(nn.Module):
    def __init__(self, original_layer, r=8, lora_alpha=16, lora_dropout=0.1):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)

        self.lora_A = nn.Parameter(torch.randn(self.r, original_layer.out_features))
        self.lora_B = nn.Parameter(torch.zeros(original_layer.in_features, self.r))
        self.scaling = self.lora_alpha / self.r

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))

    def forward(self, x):
        lora_B = self.lora_B.to(dtype=x.dtype)
        lora_A = self.lora_A.to(dtype=x.dtype)
        return self.original_layer(x) + (self.lora_dropout(x) @ lora_B @ lora_A) * self.scaling

    @property
    def weight(self):
        return (self.original_layer.weight.to(self.lora_B.dtype) + (self.lora_B @ self.lora_A) * self.scaling).to(self.original_layer.weight.dtype)

    @property
    def bias(self):
        return self.original_layer.bias


def apply_lora_to_ViT(model, n_vision_layers=1, n_transformer_layers=1, r=8, lora_dropout=0.1):
    """
    Applies LoRA to the 'out_proj' of the 11th and the last (23rd) ResidualAttentionBlock in the
    VisionTransformer's transformer.

    :param model: The PyTorch model to modify.
    :param r: The rank of the LoRA approximation.
    :param lora_dropout: The dropout rate for LoRA layers.
    """
    if isinstance(model, torch.nn.DataParallel):
        model_module = model.module
    else:
        model_module = model

    # Specific blocks to modify
    block_indices = -n_vision_layers

    for idx in range(block_indices, 0):
        # Access the specific ResidualAttentionBlock
        target_block = model_module.clip_model.visual.transformer.resblocks[idx]
        # Access the 'out_proj' within the MultiheadAttention of the block
        target_layer = target_block.attn.out_proj
        # Replace the original layer with a LoRALayer
        lora_layer = LoRALayer(target_layer, r=r, lora_dropout=lora_dropout)
        target_block.attn.out_proj = lora_layer

    block_indices = -n_transformer_layers
    for idx in range(block_indices, 0):
        # Access the specific ResidualAttentionBlock
        target_block = model_module.clip_model.transformer.resblocks[idx]
        # Access the 'out_proj' within the MultiheadAttention of the block
        target_layer = target_block.attn.out_proj
        # Replace the original layer with a LoRALayer
        lora_layer = LoRALayer(target_layer, r=r, lora_dropout=lora_dropout)
        target_block.attn.out_proj = lora_layer


def unfreeze_lora_layers(model, freeze_all=True):
    """
    Freeze or unfreeze the model's parameters based on the presence of LoRA layers.
    If a LoRALayer is encountered, only its specific LoRA parameters are unfrozen.
    """
    for name, param in model.named_parameters():
        # Initially set requires_grad based on the freeze_all flag
        param.requires_grad = not freeze_all

    if freeze_all:
        # If freezing all parameters, selectively unfreeze LoRA parameters
        def recursive_unfreeze_lora(module):
            for child_name, child in module.named_children():
                if isinstance(child, LoRALayer):
                    # Unfreeze only LoRA-specific parameters within LoRALayer
                    child.lora_A.requires_grad = True
                    child.lora_B.requires_grad = True
                    # Keep the original layer's parameters frozen
                    child.original_layer.weight.requires_grad = False
                    if child.original_layer.bias is not None:
                        child.original_layer.bias.requires_grad = False
                else:
                    recursive_unfreeze_lora(child)

        # Apply selective unfreezing to the entire model
        if isinstance(model, torch.nn.DataParallel):
            recursive_unfreeze_lora(model.module)
        else:
            recursive_unfreeze_lora(model)


class DoRALayer(nn.Module):
    def __init__(self, original_layer, r=8, dora_alpha=16, dora_dropout=0.1):
        super(DoRALayer, self).__init__()
        self.original_layer = original_layer
        self.r = r  # Low-rank factor
        self.dora_alpha = dora_alpha  # Scaling parameter
        self.dora_dropout = nn.Dropout(p=dora_dropout)

        # Decompose original weights into magnitude and direction
        with torch.no_grad():
            W = original_layer.weight.data.clone()  # [out_features, in_features]
            W = W.T  # Transpose to [in_features, out_features]
            S = torch.norm(W, dim=0)  # Magnitudes (norms of columns), shape [out_features]
            D = W / S  # Direction matrix with unit-norm columns, shape [in_features, out_features]

        # Store S as a trainable parameter
        self.m = nn.Parameter(S)  # [out_features]
        # Store D as a buffer (since we don't want to update it directly)
        self.register_buffer('D', D)  # [in_features, out_features]

        # LoRA adaptation of D
        self.delta_D_A = nn.Parameter(torch.zeros(self.r, original_layer.out_features))
        self.delta_D_B = nn.Parameter(torch.zeros(original_layer.in_features, self.r))

        # Scaling
        self.scaling = self.dora_alpha / self.r

        # Initialize delta_D_A and delta_D_B
        self.reset_parameters()

        # Copy the bias from the original layer
        if self.original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone())
        else:
            self.bias = None

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.delta_D_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.delta_D_B, a=math.sqrt(5))

    @property
    def weight(self):
        # Compute adapted D
        delta_D = (self.delta_D_B @ self.delta_D_A) * self.scaling  # [in_features, out_features]

        D_new = self.D + delta_D  # [in_features, out_features]

        # Normalize columns of D_new
        D_norms = torch.norm(D_new, dim=0, keepdim=True) + 1e-8  # [1, out_features], add epsilon to avoid division by zero
        D_normalized = D_new / D_norms  # [in_features, out_features]

        # Reconstruct the adapted weight
        W = D_normalized * self.m  # [in_features, out_features], m is [out_features]

        W = W.T  # Transpose back to [out_features, in_features]

        return W

    def forward(self, x):
        # Compute adapted D
        delta_D = (self.delta_D_B @ self.delta_D_A) * self.scaling  # [in_features, out_features]
        delta_D = self.dora_dropout(delta_D)

        D_new = self.D + delta_D  # [in_features, out_features]

        # Normalize columns of D_new
        D_norms = torch.norm(D_new, dim=0, keepdim=True) + 1e-8  # [1, out_features]
        D_normalized = D_new / D_norms  # [in_features, out_features]

        # Reconstruct the adapted weight
        W = D_normalized * self.m  # [in_features, out_features]
        W = W.T  # [out_features, in_features]

        # Compute output
        return F.linear(x, W, self.bias)


def apply_dora_to_ViT(model, n_vision_layers=1, n_transformer_layers=1, r=8, dora_dropout=0.1, seed=123):

    if isinstance(model, torch.nn.DataParallel):
        model_module = model.module
    else:
        model_module = model

    # Specific blocks to modify in the visual transformer
    block_indices = range(-n_vision_layers, 0)  # Adjusted for proper indexing

    for idx in block_indices:
        # Access the specific ResidualAttentionBlock
        target_block = model_module.clip_model.visual.transformer.resblocks[idx]
        # Access the 'out_proj' within the MultiheadAttention of the block
        target_layer = target_block.attn.out_proj
        # Replace the original layer with a DoRALayer
        dora_layer = DoRALayer(target_layer, r=r, dora_dropout=dora_dropout)
        target_block.attn.out_proj = dora_layer

    # Specific blocks to modify in the main transformer
    block_indices = range(-n_transformer_layers, 0)

    for idx in block_indices:
        # Access the specific ResidualAttentionBlock
        target_block = model_module.clip_model.transformer.resblocks[idx]
        # Access the 'out_proj' within the MultiheadAttention of the block
        target_layer = target_block.attn.out_proj
        # Replace the original layer with a DoRALayer
        dora_layer = DoRALayer(target_layer, r=r, dora_dropout=dora_dropout)
        target_block.attn.out_proj = dora_layer


def switch_dora_layers(model, freeze_all=True, dora_state=True):
    """
    Freeze or unfreeze the model's parameters based on the presence of DoRA layers.
    If a DoRALayer is encountered, only its specific DoRA parameters are unfrozen.
    """
    for name, param in model.named_parameters():
        # Initially set requires_grad based on the freeze_all flag
        param.requires_grad = not freeze_all

    if freeze_all:
        # If freezing all parameters, selectively unfreeze DoRA parameters
        def recursive_unfreeze_dora(module):
            for child_name, child in module.named_children():
                if isinstance(child, DoRALayer):
                    # Unfreeze DoRA-specific parameters within DoRALayer
                    child.m.requires_grad = dora_state
                    child.delta_D_A.requires_grad = dora_state
                    child.delta_D_B.requires_grad = dora_state
                    # Keep the original layer's parameters frozen
                    if child.bias is not None:
                        child.bias.requires_grad = False
                else:
                    recursive_unfreeze_dora(child)

        # Apply selective unfreezing to the entire model
        if isinstance(model, torch.nn.DataParallel):
            recursive_unfreeze_dora(model.module)
        else:
            recursive_unfreeze_dora(model)


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # return sum(p.numel() for p in model.parameters())


def unfreeze_image_layers(model):
    # Handle DataParallel wrapper
    if isinstance(model, torch.nn.DataParallel):
        model_module = model.module
    else:
        model_module = model

    # Unfreezing the last layer of the image encoder
        
    for param in model_module.clip_model.visual.layer3.parameters():
        param.requires_grad = True

    for param in model_module.clip_model.visual.layer4.parameters():
        param.requires_grad = True

    for param in model_module.clip_model.visual.attnpool.parameters():
        param.requires_grad = True


def unfreeze_image_layers_all(model):
    # Handle DataParallel wrapper
    if isinstance(model, torch.nn.DataParallel):
        model_module = model.module
    else:
        model_module = model

    # Unfreezing the last layer of the image encoder
        
    for param in model_module.clip_model.visual.parameters():
        param.requires_grad = True


def evaluate_model(model, data_loader, device, criterion):
    model.eval()
    total_loss = 0.0

    # Wrap data_loader with tqdm for a progress bar
    with torch.no_grad(), tqdm(enumerate(data_loader), total=len(data_loader), desc="Evaluating", file=sys.stderr) as progress_bar:
        for batch_idx, (image_names, images, targets) in progress_bar:

            images = images.to(device)
            targets = targets.to(device)

            predictions = model(images)

            loss = criterion(predictions, targets)
            progress_bar.set_postfix({'loss': loss.item()})
            total_loss += loss.item() * images.size(0) 

    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss


def behavioral_RSA(model, inference_loader, device, logger=None):
    model.eval()
    image_names = []
    predictions = []
    
    # Use logger if provided, otherwise use print
    log = logger.info if logger else print

    with torch.no_grad():
        for batch_idx, (image_name, image) in enumerate(inference_loader):
            image = image.to(device)
            
            output = model(image)

            image_names.extend(image_name)

            predictions.extend(output.cpu().numpy())

        log(f"First 10 image names: {image_names[:5]}")

        predictions_emb = np.array(predictions) 

        log(f"Embedding matrix shape: {predictions_emb.shape}\n")

        model_rdm = 1 - np.corrcoef(predictions_emb)
        np.fill_diagonal(model_rdm, 0)
        # log(f"RDM shape: {model_rdm.shape}\n")
        # log("First 5x5 of the RDM:")
        # log(model_rdm[:5, :5])
        # log("\n")

        reference_rdm_dict = scipy.io.loadmat(inference_loader.dataset.RDM48_triplet_dir)
        reference_rdm = reference_rdm_dict['RDM48_triplet']
        # log("First 5x5 of the reference RDM:")
        # log(reference_rdm[:5, :5])
        # log("\n")
        
        # Extract upper triangular elements (excluding diagonal) for correlation
        # This avoids double-counting and diagonal elements
        upper_tri_indices = np.triu_indices_from(reference_rdm, k=1)
    
        reference_values = reference_rdm[upper_tri_indices]
        log(f"First 5 reference rdm values: {reference_values[:5]}\n")
        model_values = model_rdm[upper_tri_indices]
        log(f"First 5 model rdm values: {model_values[:5]}\n")
    
        # Compute Spearman correlation
        rho, p_value = spearmanr(reference_values, model_values)
    
        return rho, p_value, model_rdm


def save_dora_parameters(model, dora_parameters_path, epoch, logger=None):
    """
    Save DoRA parameters for specific modules in the model.
    Each module's parameters are saved to a separate file to avoid overwriting.
    """
    # Use logger if provided, otherwise use print
    log = logger.info if logger else print
    
    modules_to_save = [
        ("clip_model.visual.transformer.resblocks.22.attn.out_proj", "visual_resblock_22_attn"),
        ("clip_model.visual.transformer.resblocks.23.attn.out_proj", "visual_resblock_23_attn"),
        ("clip_model.transformer.resblocks.11.attn.out_proj", "transformer_resblock_11_attn"),
    ]

    dora_params = {}

    # save the parameters for each module
    for module_path, module_name in modules_to_save:
        # Traverse the model to get the module.
        module = model
        for attr in module_path.split("."):
            module = getattr(module, attr)

        # Extract DoRA parameters
        dora_params[f'{module_path}.m'] = module.m.detach().cpu()
        dora_params[f'{module_path}.delta_D_A'] = module.delta_D_A.detach().cpu()
        dora_params[f'{module_path}.delta_D_B'] = module.delta_D_B.detach().cpu()

        # # Log parameter shapes
        # log(f"\n{module_name} parameter shapes:")
        # log(f"  m: shape {dora_params[f'{module_path}.m'].shape}")
        # log(f"  delta_D_A: shape {dora_params[f'{module_path}.delta_D_A'].shape}")
        # log(f"  delta_D_B: shape {dora_params[f'{module_path}.delta_D_B'].shape}")
    
    # Save the parameters
    save_path = os.path.join(dora_parameters_path, f"epoch{epoch + 1}_dora_params.pth")
    torch.save(dora_params, save_path)


def save_random_states(optimizer, epoch, random_state_path, dataloader_generator, logger=None):
    """
    Save all random states and optimizer state for 100% reproducibility.
    
    Args:
        optimizer: The optimizer whose state to save
        epoch: Current epoch number
        random_state_path: Directory to save the checkpoint
        dataloader_generator: The generator used for DataLoader shuffling
        logger: Optional logger for logging messages
    """
    log = logger.info if logger else print
    
    # Create checkpoint with all random states
    checkpoint = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'torch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(),
        'dataloader_generator_state': dataloader_generator.get_state(),
    }
    
    # Save CUDA random states for all GPUs if available
    if torch.cuda.is_available():
        checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state()
        checkpoint['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
    
    # Save the checkpoint
    os.makedirs(random_state_path, exist_ok=True)
    checkpoint_file = os.path.join(random_state_path, f"epoch{epoch + 1}_random_states.pth")
    torch.save(checkpoint, checkpoint_file)
    log(f"Random states saved: {checkpoint_file}")


def shuffle_targets(targets, perturb_seed=None, generator=None):
    """
    Shuffle the targets tensor while preserving the same target values.
    This breaks the image-target correspondence by randomly reassigning targets to different images.
    
    Args:
        targets: Original targets tensor of shape [batch_size, num_targets]
        perturb_seed: Seed for random number generation (if None, uses current state).
                      Ignored if generator is provided.
        generator: Optional torch.Generator for reproducible shuffling. If provided,
                   uses this generator instead of perturb_seed.
    
    Returns:
        Shuffled targets tensor with the same values but different assignments
    """
    
    if generator is None and perturb_seed is not None:
        # Save current random states
        torch_state = torch.get_rng_state()
        np_state = np.random.get_state()
        python_state = random.getstate()
        
        # Set seeds for reproducibility
        torch.manual_seed(perturb_seed)
        np.random.seed(perturb_seed)
        random.seed(perturb_seed)
    
    # Create a copy of the targets to avoid modifying the original
    shuffled_targets = targets.clone()
    
    # Get batch size
    batch_size = targets.shape[0]
    
    # Generate random permutation indices for the batch dimension
    if generator is not None:
        perm_indices = torch.randperm(batch_size, device=targets.device, generator=generator)
    else:
        perm_indices = torch.randperm(batch_size, device=targets.device)
    
    # Shuffle the targets by reordering the batch dimension
    shuffled_targets = shuffled_targets[perm_indices]
    
    if generator is None and perturb_seed is not None:
        # Restore original random states
        torch.set_rng_state(torch_state)
        np.random.set_state(np_state)
        random.setstate(python_state)
    
    return shuffled_targets


def train_model(model, train_loader, test_loader, inference_loader, device, optimizer, criterion, epochs, training_res_path, training_run, perturb_length, perturb_seed, mean, std, perturb_distribution, perturb_type, logger=None, early_stopping_patience=5, checkpoint_path='clip_hba_model_cv.pth', dora_parameters_path='./dora_params', random_state_path='./random_states', dataloader_generator=None, resume_from_epoch=0, previous_training_res_path=None):
    model.train()
    best_test_loss = float('inf')
    epochs_no_improve = 0

    # Use logger if provided, otherwise use print
    log = logger.info if logger else print

    best_test_loss = 500000

    # Create folder to store DoRA parameters
    os.makedirs(dora_parameters_path, exist_ok=True)

    headers = ['epoch', 'train_loss', 'test_loss', 'behavioral_rsa_rho', 'behavioral_rsa_p_value', 'used_random_targets', 'used_shuffled_targets', 'used_uniform_images', 'used_image_noise']

    # Check if we're resuming from the same file (in-place resume)
    resuming_same_file = (previous_training_res_path == training_res_path and 
                          os.path.exists(training_res_path) and 
                          resume_from_epoch > 0)
    
    if resuming_same_file:
        # When resuming from the same file, we keep the existing CSV and just append new rows
        log("Resuming from existing CSV file - will append new epochs")
        # Verify the file has the expected structure
        try:
            with open(training_res_path, 'r') as check_file:
                reader = csv.reader(check_file)
                existing_headers = next(reader, None)
                if existing_headers != headers:
                    log(f"Warning: CSV headers don't match. Expected {headers}, found {existing_headers}")
        except Exception as e:
            if logger:
                logger.warning(f"Could not verify existing CSV file: {e}")
    else:
        # Initialize CSV: if resuming from a different file, pre-populate rows up to resume_from_epoch
        with open(training_res_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            if previous_training_res_path and resume_from_epoch > 0 and os.path.exists(previous_training_res_path):
                try:
                    with open(previous_training_res_path, 'r') as prev_file:
                        prev_reader = csv.reader(prev_file)
                        next(prev_reader, None)  # skip header
                        for row in prev_reader:
                            try:
                                epoch_val = int(row[0])
                            except Exception:
                                continue
                            if epoch_val <= resume_from_epoch:
                                writer.writerow(row)
                except Exception as e:
                    if logger:
                        logger.warning(f"Could not pre-populate training CSV from {previous_training_res_path}: {e}")

    for epoch in range(resume_from_epoch, epochs):
        total_loss = 0.0
        used_random_targets = False
        used_shuffled_targets = False
        used_uniform_images = False
        used_image_noise = False

        # Check if this epoch is within the perturbation window
        perturb_start_epoch = training_run - 1  # Convert to 0-indexed
        perturb_end_epoch = perturb_start_epoch + perturb_length - 1
        
        if perturb_start_epoch <= epoch <= perturb_end_epoch and perturb_type == 'random_target':
            logger.info("="*80)
            log(f"\n*** USING RANDOM TARGETS FOR EPOCH {epoch+1} (Perturbation window: epochs {perturb_start_epoch+1}-{perturb_end_epoch+1}) ***")
            logger.info("="*80)
            log(f"Random target seed: {perturb_seed}")
            used_random_targets = True
        elif perturb_start_epoch <= epoch <= perturb_end_epoch and perturb_type == 'image_noise':
            logger.info("="*80)
            log(f"\n*** USING IMAGE NOISE FOR EPOCH {epoch+1} (Perturbation window: epochs {perturb_start_epoch+1}-{perturb_end_epoch+1}) ***")
            logger.info("="*80)
            log(f"Image noise seed: {perturb_seed}")
            log(f"Image noise distribution - mean: {mean:.4f}, std: {std:.4f}")
            used_image_noise = True       
        elif perturb_start_epoch <= epoch <= perturb_end_epoch and perturb_type == 'label_shuffle':
            logger.info("="*80)
            log(f"\n*** USING SHUFFLED TARGETS FOR EPOCH {epoch+1} (Perturbation window: epochs {perturb_start_epoch+1}-{perturb_end_epoch+1}) ***")
            logger.info("="*80)
            log(f"Shuffle target seed: {perturb_seed}")
            used_shuffled_targets = True
        elif perturb_start_epoch <= epoch <= perturb_end_epoch and perturb_type == 'uniform_images':
            logger.info("="*80)
            log(f"\n*** USING UNIFORM GRAYSCALE IMAGES FOR EPOCH {epoch+1} (Perturbation window: epochs {perturb_start_epoch+1}-{perturb_end_epoch+1}) ***")
            logger.info("="*80)
            log(f"All images will be replaced with uniform grayscale (value=0.5)")
            used_uniform_images = True

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", file=sys.stderr)
        for batch_idx, (image_names, images, targets) in progress_bar:
    
            images = images.to(device)
            targets = targets.to(device)

            # Apply image noise if flag is set
            if used_image_noise:
                # Set seed for reproducibility across batches
                torch.manual_seed(perturb_seed + training_run * 1000 + batch_idx)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(perturb_seed + training_run * 1000 + batch_idx)
                
                # Debug output for first batch
                if batch_idx == 0:
                    log(f"\n{'='*80}")
                    log(f"IMAGE NOISE DEBUG - Batch {batch_idx}, Epoch {epoch+1}")
                    log(f"{'='*80}")
                    log(f"  Batch size: {images.shape[0]}")
                    log(f"  Image shape: {images.shape[1:]}")
                    log(f"  Original image stats: min={images[0].min().item():.3f}, max={images[0].max().item():.3f}, mean={images[0].mean().item():.3f}")
                
                # Apply noise to each image in the batch
                for i in range(len(images)):
                    images[i] = replace_with_gaussian_noise(images[i], mean, std)
                
                # Debug output for first batch
                if batch_idx == 0:
                    log(f"  After noise - image stats: min={images[0].min().item():.3f}, max={images[0].max().item():.3f}, mean={images[0].mean().item():.3f}")
                    log(f"{'='*80}\n")

            if used_uniform_images:
                # Replace all images with uniform grayscale images (value=0.5)
                images = torch.ones_like(images) * 0.5
                # Debug output for first batch
                if batch_idx == 0:
                    log(f"\n{'='*80}")
                    log(f"UNIFORM IMAGE DEBUG - Batch {batch_idx}, Epoch {epoch+1}")
                    log(f"{'='*80}")
                    log(f"  Batch size: {images.shape[0]}")
                    log(f"  Image shape: {images.shape[1:]}")
                    log(f"  All pixel values set to: 0.5")
                    log(f"  Sample image stats: min={images[0].min().item():.3f}, max={images[0].max().item():.3f}, mean={images[0].mean().item():.3f}")
                    log(f"{'='*80}\n")

            if used_random_targets:
                batch_generator = torch.Generator(device=device)
                batch_generator.manual_seed(perturb_seed + training_run * 1000 + batch_idx) # make this seed different for each batch
                
                # Generate random targets using the epoch generator (different for each batch)
                if perturb_distribution == 'normal':
                    random_targets = torch.randn(targets.shape, device=device, dtype=torch.float32, generator=batch_generator)
                elif perturb_distribution == 'target':
                    random_targets = torch.randn(targets.shape, device=device, dtype=torch.float32, generator=batch_generator) * std + mean
                targets = random_targets

                # DEBUG: Check if random targets are valid
                if batch_idx == 0:
                    log(f"Random targets stats: min={targets.min().item():.6f}, max={targets.max().item():.6f}, mean={targets.mean().item():.6f}")
                if torch.isnan(targets).any():
                    log(f"ERROR: NaN detected in random targets!")
                    log(f"Random targets sample: {targets[0][:5]}")
                    continue

            if used_shuffled_targets:
                batch_generator = torch.Generator(device=device)
                batch_generator.manual_seed(perturb_seed + training_run * 1000 + batch_idx) # make this seed different for each batch
                
                # Debug output: Show targets before and after shuffling (only for first batch to avoid clutter)
                if batch_idx == 0:
                    num_samples_to_show = min(5, targets.shape[0])
                    image_names_before = list(image_names)  # Store original image names
                    targets_before = targets.clone().cpu()  # Store original targets on CPU
                    
                    log(f"\n{'='*80}")
                    log(f"LABEL SHUFFLING DEBUG - Batch {batch_idx}, Epoch {epoch+1}")
                    log(f"{'='*80}")
                    log(f"BEFORE SHUFFLING (Image -> Target mapping):")
                    log(f"  Batch size: {targets.shape[0]}")
                    for i in range(num_samples_to_show):
                        target_sample = targets_before[i].numpy()[:5]  # First 5 dimensions
                        log(f"  [{i}] Image='{image_names_before[i]}' -> Target[0:5]={target_sample}")
                    if targets.shape[0] > num_samples_to_show:
                        log(f"  ... ({targets.shape[0] - num_samples_to_show} more samples)")
                
                # Perform shuffling
                targets = shuffle_targets(targets, generator=batch_generator)
                
                # Debug output: Show targets after shuffling
                if batch_idx == 0:
                    log(f"\nAFTER SHUFFLING (Image -> Target mapping):")
                    log(f"  Batch size: {targets.shape[0]}")
                    targets_after = targets.cpu()  # Convert to CPU for comparison
                    for i in range(num_samples_to_show):
                        target_sample = targets_after[i].numpy()[:5]  # First 5 dimensions
                        # Find which original position this target came from by comparing target values
                        original_idx = None
                        for j in range(targets_before.shape[0]):
                            if torch.allclose(targets_after[i], targets_before[j], atol=1e-5):
                                original_idx = j
                                break
                        if original_idx is not None and original_idx != i:
                            log(f"  [{i}] Image='{image_names[i]}' -> Target[0:5]={target_sample} | *** SHUFFLED from original [{original_idx}] ('{image_names_before[original_idx]}')")
                        elif original_idx == i:
                            log(f"  [{i}] Image='{image_names[i]}' -> Target[0:5]={target_sample} | (No change - same position)")
                        else:
                            log(f"  [{i}] Image='{image_names[i]}' -> Target[0:5]={target_sample} | (Could not find original match)")
                    if targets.shape[0] > num_samples_to_show:
                        log(f"  ... ({targets.shape[0] - num_samples_to_show} more samples)")
                    log(f"{'='*80}\n")


            optimizer.zero_grad()
            predictions = model(images)

            # DEBUG: Check if predictions are valid
            if torch.isnan(predictions).any():
                log(f"ERROR: NaN detected in model predictions!")
                log(f"Predictions sample: {predictions[0][:5]}")
                continue
            
            loss = criterion(predictions, targets)
            # DEBUG: Check if loss is valid
            if torch.isnan(loss) or torch.isinf(loss):
                log(f"ERROR: NaN/Inf detected in loss!")
                continue
            progress_bar.set_postfix({'loss': loss.item()})
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
        avg_train_loss = total_loss / len(train_loader.dataset)

        # Evaluate after every epoch
        avg_test_loss = evaluate_model(model, test_loader, device, criterion)
        log(f"Epoch {epoch+1}: Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_test_loss:.4f}")
        
        # Conduct behavioral RSA at every epoch
        rho, p_value, model_rdm = behavioral_RSA(model, inference_loader, device, logger=logger)
        log(f"Behavioral RSA Correlation & p-value: {rho:.4f}, {p_value:.4f}")
        model.train() # put the model back in training mode
        
        # Log if perturbations were used
        if used_random_targets:
            log(f"*** RANDOM TARGETS WERE USED IN THIS EPOCH ***")
        if used_shuffled_targets:
            log(f"*** SHUFFLED TARGETS WERE USED IN THIS EPOCH ***")
        if used_uniform_images:
            log(f"*** UNIFORM GRAYSCALE IMAGES WERE USED IN THIS EPOCH ***")
        if used_image_noise:
            log(f"*** GAUSSIAN NOISE WAS APPLIED TO IMAGES IN THIS EPOCH ***")

        # Prepare the data row with the epoch number and loss values
        data_row = [epoch + 1, avg_train_loss, avg_test_loss, rho, p_value, used_random_targets, used_shuffled_targets, used_uniform_images, used_image_noise]

        # Append the data row to the CSV file
        with open(training_res_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data_row)

        # Save the DoRA parameters
        save_dora_parameters(model, dora_parameters_path, epoch, logger=logger)
        log("\n\n*********************************")
        log(f"DoRA parameters saved for epoch {epoch+1}")
        log("\n\n*********************************")

        # Save random states and optimizer after every epoch for full reproducibility
        if dataloader_generator is not None:
            save_random_states(optimizer, epoch, random_state_path, dataloader_generator, logger=logger)

        # Calculate perturbation window 
        perturb_start_epoch = training_run - 1  # Convert to 0-indexed
        perturb_end_epoch = perturb_start_epoch + perturb_length - 1
        in_perturbation_window = perturb_start_epoch <= epoch <= perturb_end_epoch

        # Check for early stopping and saving checkpoint
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            epochs_no_improve = 0
        else:
            # only increment the counter if not in the perturbation window
            if not in_perturbation_window:
                epochs_no_improve += 1
            # if in the perturbation window, don't increment the counter
        
        # Trigger early stopping if the number of epochs without improvement reaches the early stopping patience
        if epochs_no_improve == early_stopping_patience:
            log("\n\n*********************************")
            log(f"Early stopping triggered at epoch {epoch+1}")
            log("*********************************\n\n")
            break


def run_behavioral_training(config):
    """
    Run behavioral training with the given configuration.
    
    Args:
        config (dict): Configuration dictionary containing training parameters
    """
    torch.manual_seed(config['random_seed'])
    torch.cuda.manual_seed_all(config['random_seed'])
    np.random.seed(config['random_seed'])
    random.seed(config['random_seed'])
    
    # Clear CUDA cache between runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    seed_everything(config['random_seed'])

    # Set up logger
    log_dir = os.path.dirname(config['checkpoint_path'])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_log_{timestamp}.txt')
    logger = setup_logger(log_file)
    
    logger.info("="*80)
    logger.info("Starting Training Run")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)
    
    # Initialize dataset
    dataset = ThingsDataset(csv_file=config['csv_file'], img_dir=config['img_dir'])

    embeddings = dataset.annotations.iloc[:, 1:].values.astype('float32')

    if config['perturb_distribution'] == 'normal':
        mean = 0
        std = 1
    elif config['perturb_distribution'] == 'target':
        mean = np.mean(embeddings)
        std = np.std(embeddings)
    
    # Split dataset using baseline split
    baseline_split_path = config.get('baseline_split_indices_path')
    # Load the split indices from baseline training
    split_info = load_dataset_split_indices(baseline_split_path, logger=logger)
    train_dataset = SubsetWithIndices(dataset, split_info['train_indices'])
    test_dataset = SubsetWithIndices(dataset, split_info['test_indices'])
    logger.info("Using baseline dataset split")

    # Initialize inference dataset
    inference_dataset = ThingsInferenceDataset(inference_csv_file=config['inference_csv_file'], img_dir=config['img_dir'], RDM48_triplet_dir=config['RDM48_triplet_dir'])
    
    # Create a generator for reproducible DataLoader shuffling
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(config['random_seed'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, generator=dataloader_generator)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    inference_loader = DataLoader(inference_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Determine pos_embedding based on backbone
    pos_embedding = False if config['backbone'] == 'RN50' else True
    logger.info(f"pos_embedding is {pos_embedding}")
    
    # Initialize model
    model = CLIPHBA(classnames=classnames66, backbone_name=config['backbone'], 
                    pos_embedding=pos_embedding)
    
    # Set device
    if config['cuda'] == -1:
        device = torch.device("cuda")
    elif config['cuda'] == 0:
        device = torch.device("cuda:0")
    elif config['cuda'] == 1:
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")
    
    # Apply DoRA
    apply_dora_to_ViT(model, 
                      n_vision_layers=config['vision_layers'],
                      n_transformer_layers=config['transformer_layers'],
                      r=config['rank'],
                      dora_dropout=0.1)
    switch_dora_layers(model, freeze_all=True, dora_state=True)

    training_run = config['training_run']
    
    # Determine which DoRA checkpoint to load: resume source (if provided) or baseline at training_run-1
    dora_params_path = None
    if config.get('resume_from_epoch', 0) > 0 and config.get('resume_dora_parameters_path'):
        dora_checkpoint = config['resume_from_epoch']
        dora_params_path = os.path.join(config['resume_dora_parameters_path'], f"epoch{dora_checkpoint}_dora_params.pth")
    else:
        dora_checkpoint = training_run - 1
        dora_params_path = os.path.join(config['baseline_dora_directory'], f"epoch{dora_checkpoint}_dora_params.pth")

    # Load DoRA parameters if available/appropriate
    if dora_params_path and os.path.exists(dora_params_path) and (config['training_run'] >= 1):
        dora_params_state_dict = torch.load(dora_params_path)
        model.load_state_dict(dora_params_state_dict, strict=False)
        logger.info(f"Loaded DoRA parameters from {dora_params_path}")
    else:
        logger.info(f"Using original DoRA parameters from model initialization")
    
    # Use DataParallel if using all GPUs
    if config['cuda'] == -1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)
    
    model.to(device)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=config['lr'])

    # If resuming from a specific epoch, load the random states from prior run if provided, else baseline
    resume_from_epoch = config.get('resume_from_epoch', 0)
    if resume_from_epoch > 0:
        prior_random_state_path = config.get('resume_random_state_path') or config.get('baseline_random_state_path')
        if prior_random_state_path:
            logger.info(f"Resuming from epoch {resume_from_epoch}")
            success = load_random_states(
                prior_random_state_path, 
                resume_from_epoch, 
                optimizer=optimizer,
                dataloader_generator=dataloader_generator,
                logger=logger
            )
            if success:
                logger.info(f"Successfully restored all random states from epoch {resume_from_epoch}")
            else:
                logger.warning(f"Could not load random states - starting with fresh random state")
        else:
            logger.warning("baseline_random_state_path not provided in config, cannot restore random states")
    
    # Print training information
    logger.info("\nModel Configuration:")
    logger.info("-------------------")
    for key, value in config.items():
        logger.info(f"{key}: {value}")
    logger.info("\nUpdating layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(name)
    logger.info(f"\nNumber of trainable parameters: {count_trainable_parameters(model)}\n")
    
    # Train model
    train_model(model, train_loader, test_loader, inference_loader, device, optimizer, 
                criterion=config['criterion'], epochs=config['epochs'], 
                training_res_path=config['training_res_path'], training_run=training_run, perturb_length=config['perturb_length'], perturb_seed=config['perturb_seed'],
                mean=mean, std=std, perturb_distribution=config['perturb_distribution'], perturb_type=config['perturb_type'],
                logger=logger,
                early_stopping_patience=config['early_stopping_patience'],
                checkpoint_path=config['checkpoint_path'],
                dora_parameters_path=config['dora_parameters_path'],
                random_state_path=config.get('random_state_path', './random_states'),
                dataloader_generator=dataloader_generator,
                resume_from_epoch=config.get('resume_from_epoch', 0),
                previous_training_res_path=config.get('previous_training_res_path')
                )