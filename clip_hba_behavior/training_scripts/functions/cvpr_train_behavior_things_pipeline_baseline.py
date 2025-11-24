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
from src.models.clip_hba_utils import save_dora_parameters

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import logging
from datetime import datetime
import csv
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


    def forward(self, image):
        if self.clip_model.training:
            self.clip_model.eval()

        # Move tokenized prompts to the same device as the input image
        tokenized_prompts = self.tokenized_prompts.to(image.device)

        # Process all tokenized prompts in a single forward pass
        pred_score = self.clip_model(image, tokenized_prompts, self.pos_embedding)

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
    with torch.no_grad(), tqdm(enumerate(data_loader), total=len(data_loader), desc="Evaluating") as progress_bar:
        for batch_idx, (_, images, targets) in progress_bar:
            images = images.to(device)
            targets = targets.to(device)

            predictions = model(images)

            loss = criterion(predictions, targets)
            progress_bar.set_postfix({'loss': loss.item()})
            total_loss += loss.item() * images.size(0) 

    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss


def behavioral_RSA(model, inference_loader, device):
    model.eval()
    image_names = []
    predictions = []

    with torch.no_grad():
        for batch_idx, (image_name, image) in enumerate(inference_loader):
            image = image.to(device)
            
            output = model(image)

            image_names.extend(image_name)

            predictions.extend(output.cpu().numpy())

        print(f"First 10 image names: {image_names[:5]}")

        predictions_emb = np.array(predictions) 

        print(f"Embedding matrix shape: {predictions_emb.shape}\n")

        model_rdm = 1 - np.corrcoef(predictions_emb)
        np.fill_diagonal(model_rdm, 0)
        print(f"RDM shape: {model_rdm.shape}\n")
        print("First 5x5 of the RDM:")
        print(model_rdm[:5, :5])
        print("\n")

        reference_rdm_dict = scipy.io.loadmat(inference_loader.dataset.RDM48_triplet_dir)
        reference_rdm = reference_rdm_dict['RDM48_triplet']
        print("First 5x5 of the reference RDM:")
        print(reference_rdm[:5, :5])
        print("\n")
        
        # Extract upper triangular elements (excluding diagonal) for correlation
        # This avoids double-counting and diagonal elements
        upper_tri_indices = np.triu_indices_from(reference_rdm, k=1)
    
        reference_values = reference_rdm[upper_tri_indices]
        print(f"First 5 reference rdm values: {reference_values[:5]}\n")
        model_values = model_rdm[upper_tri_indices]
        print(f"First 5 model rdm values: {model_values[:5]}\n")
    
        # Compute Spearman correlation
        rho, p_value = spearmanr(reference_values, model_values)
    
        return rho, p_value, model_rdm


# def save_dora_parameters(model, dora_parameters_path, epoch):
#     """
#     Save DoRA parameters for specific modules in the model.
#     Each module's parameters are saved to a separate file to avoid overwriting.
#     """
#     modules_to_save = [
#         ("clip_model.visual.transformer.resblocks.22.attn.out_proj", "visual_resblock_22_attn"),
#         ("clip_model.visual.transformer.resblocks.23.attn.out_proj", "visual_resblock_23_attn"),
#         ("clip_model.transformer.resblocks.11.attn.out_proj", "transformer_resblock_11_attn"),
#     ]

#     dora_params = {}

#     # save the parameters for each module
#     for module_path, module_name in modules_to_save:
#         # Traverse the model to get the module.
#         # This works by splitting the module_path string (e.g., "clip_model.visual.transformer.resblocks.22.attn.out_proj")
#         # into its components, and then repeatedly calling getattr to descend into the model's attribute tree.
#         # For example, getattr(model, "clip_model") -> getattr(model.clip_model, "visual") -> ... etc.
#         module = model
#         for attr in module_path.split("."):
#             module = getattr(module, attr)

#         # Extract DoRA parameters
#         dora_params[f'{module_path}.m'] = module.m.detach().cpu()
#         dora_params[f'{module_path}.delta_D_A'] = module.delta_D_A.detach().cpu()
#         dora_params[f'{module_path}.delta_D_B'] = module.delta_D_B.detach().cpu()

#         # Print parameter shapes
#         print(f"\n{module_name} parameter shapes:")
#         print(f"  m: shape {dora_params[f'{module_path}.m'].shape}")
#         print(f"  delta_D_A: shape {dora_params[f'{module_path}.delta_D_A'].shape}")
#         print(f"  delta_D_B: shape {dora_params[f'{module_path}.delta_D_B'].shape}")
    
#     # Save the parameters
#     save_path = os.path.join(dora_parameters_path, f"epoch{epoch + 1}_dora_params.pth")
#     torch.save(dora_params, save_path)


def save_random_states(optimizer, epoch, random_state_path, dataloader_generator, logger=None):
    """
    Save all random states and optimizer state for 100% reproducibility.
    
    Args:
        optimizer: The optimizer whose state to save
        epoch: Current epoch number
        random_state_path: Directory to save the checkpoint
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


def train_model(model, train_loader, test_loader, inference_loader, device, optimizer, criterion, epochs, training_res_path, logger=None, early_stopping_patience=5, checkpoint_path='clip_hba_model_cv.pth', dora_parameters_path='./dora_params', random_state_path='./random_states', dataloader_generator=None, vision_layers=1, transformer_layers=1):
    model.train()
    best_test_loss = float('inf')
    epochs_no_improve = 0

    # Use logger if provided, otherwise use print
    log = logger.info if logger else print

    # initial evaluation
    log("*********************************")
    log("Evaluating initial model")
    best_test_loss = evaluate_model(model, test_loader, device, criterion)
    initial_behavioral_rsa_rho, initial_behavioral_rsa_p_value, initial_model_rdm = behavioral_RSA(model, inference_loader, device)
    log(f"Initial Validation Loss: {best_test_loss:.4f}")
    log(f"Initial Behavioral RSA Correlation & p-value: {initial_behavioral_rsa_rho:.4f}, {initial_behavioral_rsa_p_value:.4f}")
    log("*********************************\n")

    # Create folder to store DoRA parameters
    os.makedirs(dora_parameters_path, exist_ok=True)

    # Create directory for training results CSV if it doesn't exist
    os.makedirs(os.path.dirname(training_res_path), exist_ok=True)

    headers = ['epoch', 'train_loss', 'test_loss', 'behavioral_rsa_rho', 'behavioral_rsa_p_value']

    with open(training_res_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

    for epoch in range(epochs):
        total_loss = 0.0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (_, images, targets) in progress_bar:

            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(images)
            
            loss = criterion(predictions, targets)
            progress_bar.set_postfix({'loss': loss.item()})
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
        avg_train_loss = total_loss / len(train_loader.dataset)

        # Evaluate after every epoch
        avg_test_loss = evaluate_model(model, test_loader, device, criterion)
        print(f"Epoch {epoch+1}: Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_test_loss:.4f}")
        log(f"Epoch {epoch+1}: Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_test_loss:.4f}")

        # Conduct behavioral RSA at every epoch
        rho, p_value, model_rdm = behavioral_RSA(model, inference_loader, device)
        log(f"Behavioral RSA Correlation & p-value: {rho:.4f}, {p_value:.4f}")
        model.train() # put the model back in training mode

        # Prepare the data row with the epoch number and loss values
        data_row = [epoch + 1, avg_train_loss, avg_test_loss, rho, p_value]

        # Append the data row to the CSV file
        with open(training_res_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data_row)

        # Save random states and optimizer after every epoch for full reproducibility
        save_random_states(optimizer, epoch, random_state_path, dataloader_generator, logger=logger)

        # Save the DoRA parameters
        save_dora_parameters(
            model,
            dora_parameters_path,
            epoch,
            vision_layers,
            transformer_layers,
            log_fn=log,
        )
        log(f"DoRA parameters saved for epoch {epoch+1}")

        # Check for early stopping and saving checkpoint
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

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
    
    # Split dataset
    train_size = int(config['train_portion'] * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    split_info = {
        'train_indices': train_dataset.indices.copy() if hasattr(train_dataset, 'indices') else list(train_dataset.indices),
        'test_indices': test_dataset.indices.copy() if hasattr(test_dataset, 'indices') else list(test_dataset.indices),
        'random_seed': config['random_seed'],
        'train_portion': config['train_portion']
    }
    split_file = os.path.join(config['random_state_path'], 'dataset_split_indices.pth')
    os.makedirs(config['random_state_path'], exist_ok=True)
    torch.save(split_info, split_file)
    logger.info(f"Dataset split indices saved: {split_file}")

    # Initialize inference dataset
    inference_dataset = ThingsInferenceDataset(inference_csv_file=config['inference_csv_file'], img_dir=config['img_dir'], RDM48_triplet_dir=config['RDM48_triplet_dir'])

    # Create a generator for reproducible DataLoader shuffling
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(config['random_seed'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, generator=dataloader_generator)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    inference_loader = DataLoader(inference_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Determine pos_embedding based on backbone
    pos_embedding = False if config['backbone'] == 'RN50' else True
    print(f"pos_embedding is {pos_embedding}")
    
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
    
    # Use DataParallel if using all GPUs
    if config['cuda'] == -1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)
    
    model.to(device)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=config['lr'])
    
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
    train_model(
        model,
        train_loader,
        test_loader,
        inference_loader,
        device,
        optimizer,
        config['criterion'],
        config['epochs'],
        config['training_res_path'],
        logger=logger,
        early_stopping_patience=config['early_stopping_patience'],
        checkpoint_path=config['checkpoint_path'],
        dora_parameters_path=config['dora_parameters_path'],
        random_state_path=config['random_state_path'],
        dataloader_generator=dataloader_generator,
        vision_layers=config['vision_layers'],
        transformer_layers=config['transformer_layers'],
    )
