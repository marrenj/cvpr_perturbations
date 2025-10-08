import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split, Subset
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

from scipy.stats import spearmanr
import scipy.io

import csv

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


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


class ThingsDataset(Dataset):
    def __init__(self, csv_file, img_dir, filter_indices=None):
        self.img_dir = img_dir
        self.transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.52997664, 0.48070561, 0.41943838],
                                            std=[0.27608301, 0.26593025, 0.28238822])
                    ])

        # Load the full CSV
        full_annotations = pd.read_csv(csv_file, index_col=0)

        if filter_indices is not None:
            self.annotations = full_annotations.loc[filter_indices]
        else:
            self.annotations = full_annotations

        # Store the original indices from the CSV
        self.original_indices = self.annotations.index.tolist()
        print(f"Dataset created with {len(self.annotations)} samples")
        print(f"Original indices: {self.original_indices[:10]}...")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_name = self.annotations.iloc[index, 0]
        img_path = os.path.join(self.img_dir, image_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        targets = torch.tensor(self.annotations.iloc[index, 1:].values.astype('float32'))
        
        # Return the original image index, not the local subset index
        original_index = self.original_indices[index]

        return image_name, image, targets, original_index


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
        for batch_idx, (_, images, targets, _) in progress_bar:
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


def save_dora_parameters(model, dora_parameters_path, epoch):
    """
    Save DoRA parameters for specific modules in the model.
    Each module's parameters are saved to a separate file to avoid overwriting.
    """
    modules_to_save = [
        ("clip_model.visual.transformer.resblocks.22.attn.out_proj", "visual_resblock_22_attn"),
        ("clip_model.visual.transformer.resblocks.23.attn.out_proj", "visual_resblock_23_attn"),
        ("clip_model.transformer.resblocks.11.attn.out_proj", "transformer_resblock_11_attn"),
    ]

    dora_params = {}

    # save the parameters for each module
    for module_path, module_name in modules_to_save:
        # Traverse the model to get the module.
        # This works by splitting the module_path string (e.g., "clip_model.visual.transformer.resblocks.22.attn.out_proj")
        # into its components, and then repeatedly calling getattr to descend into the model's attribute tree.
        # For example, getattr(model, "clip_model") -> getattr(model.clip_model, "visual") -> ... etc.
        module = model
        for attr in module_path.split("."):
            module = getattr(module, attr)

        # Extract DoRA parameters
        dora_params[f'{module_path}.m'] = module.m.detach().cpu()
        dora_params[f'{module_path}.delta_D_A'] = module.delta_D_A.detach().cpu()
        dora_params[f'{module_path}.delta_D_B'] = module.delta_D_B.detach().cpu()

        # Save the parameters
        save_path = os.path.join(dora_parameters_path, f"epoch{epoch + 1}_dora_params.pth")
        torch.save(dora_params, save_path)

        # Print parameter shapes
        print(f"\n{module_name} parameter shapes:")
        print(f"  m: shape {dora_params[f'{module_path}.m'].shape}")
        print(f"  delta_D_A: shape {dora_params[f'{module_path}.delta_D_A'].shape}")
        print(f"  delta_D_B: shape {dora_params[f'{module_path}.delta_D_B'].shape}")


def generate_random_targets(targets_shape, device, random_seed=None):
    """
    Generate completely random targets with the same shape as the original targets.
    The random targets are seeded for reproducibility.
    
    Args:
        targets_shape: Shape of the original targets tensor
        device: Device to place the random targets on
        random_seed: Seed for random number generation (if None, uses current state)
    
    Returns:
        Random targets tensor with the same shape as original targets
    """
    
    if random_seed is not None:
        # Save current random states
        torch_state = torch.get_rng_state()
        np_state = np.random.get_state()
        python_state = random.getstate()
        
        # Set seeds for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
    
    # Generate random targets with the same shape
    # Using normal distribution with mean 0 and std 1 (similar to typical target ranges)
    random_targets = torch.randn(targets_shape, device=device, dtype=torch.float32)
    
    if random_seed is not None:
        # Restore original random states
        torch.set_rng_state(torch_state)
        np.random.set_state(np_state)
        random.setstate(python_state)
    
    return random_targets


def load_checkpoint(model, checkpoint_path, device):
    """
    Load model checkpoint from the specified path.
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    print("Checkpoint loaded successfully!")
    
    return model


def train_model(model, train_loader, test_loader, inference_loader, device, optimizer, criterion, epochs, training_res_path, batch_size, csv_file, img_dir, early_stopping_patience=5, checkpoint_path='clip_hba_model_cv.pth', dora_parameters_path='./dora_params', model_embedding_path='./model_embeddings', model_rdm_path='./model_rdms', random_target_epoch=None, random_target_seed=42, resume_from_epoch=0):
    model.train()
    best_test_loss = float('inf')
    epochs_no_improve = 0

    # Skip initial evaluation if resuming from a checkpoint
    if resume_from_epoch > 0:
        print(f"*********************************")
        print(f"Resuming training from epoch {resume_from_epoch + 1}")
        print("*********************************\n")
        
        # Load the best test loss from the existing CSV file
        try:
            existing_df = pd.read_csv(training_res_path)
            if len(existing_df) > 0:
                best_test_loss = existing_df['test_loss'].min()
                print(f"Best test loss from previous training: {best_test_loss:.4f}")
        except Exception as e:
            print(f"Could not load existing training results: {e}")
            print("Starting with default best test loss")
    else:
        # initial evaluation
        print("*********************************")
        print("Evaluating initial model")
        best_test_loss = evaluate_model(model, test_loader, device, criterion)
        print(f"Initial Validation Loss: {best_test_loss:.4f}")
        print("*********************************\n")

    # Create folder to store DoRA parameters
    os.makedirs(dora_parameters_path, exist_ok=True)

    # Only write headers if starting from epoch 0
    if resume_from_epoch == 0:
        headers = ['epoch', 'train_loss', 'test_loss', 'behavioral_rsa_rho', 'behavioral_rsa_p_value', 'used_random_targets']
        with open(training_res_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

    for epoch in range(resume_from_epoch, epochs):
        total_loss = 0.0
        used_random_targets = False

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (_, images, targets, _) in progress_bar:
    
            images = images.to(device)
            targets = targets.to(device)

            # Check if this is the epoch where we should use random targets
            if random_target_epoch is not None and epoch == random_target_epoch - 1:  # -1 because epochs are 0-indexed
                print(f"\n*** USING RANDOM TARGETS FOR EPOCH {epoch+1} ***")
                print(f"Random target seed: {random_target_seed}")
                targets = generate_random_targets(targets.shape, device, random_target_seed)
                used_random_targets = True

            optimizer.zero_grad()
            predictions = model(images)
            
            loss = criterion(predictions, targets)
            progress_bar.set_postfix({'loss': loss.item(), 'random_targets': used_random_targets})
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
        avg_train_loss = total_loss / len(train_loader.dataset)

        # Evaluate after every epoch
        avg_test_loss = evaluate_model(model, test_loader, device, criterion)
        print(f"Epoch {epoch+1}: Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_test_loss:.4f}")
        if used_random_targets:
            print(f"*** RANDOM TARGETS WERE USED IN THIS EPOCH ***")

        # Conduct behavioral RSA at every epoch
        rho, p_value, model_rdm = behavioral_RSA(model, inference_loader, device)
        print(f"Behavioral RSA Correlation & p-value: {rho:.4f}, {p_value:.4f}")
        
        # Prepare the data row with the epoch number and loss values
        data_row = [epoch + 1, avg_train_loss, avg_test_loss, rho, p_value, used_random_targets]

        # Append the data row to the CSV file
        with open(training_res_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data_row)

        # Save the model RDM 
        os.makedirs(model_rdm_path, exist_ok=True)
        np.save(f'{model_rdm_path}/model_rdm_epoch{epoch+1}.npy', model_rdm)

        # Save the DoRA parameters
        save_dora_parameters(model, dora_parameters_path, epoch)
        
        # Check for early stopping and saving checkpoint
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            epochs_no_improve = 0
            # Save the model checkpoint
            torch.save(model.state_dict(),checkpoint_path)
            print("\n\n-----------------------------------")
            print(f"Checkpoint saved for epoch {epoch+1}")
            print("-----------------------------------\n\n")
        else:
            epochs_no_improve += 1

        if epochs_no_improve == early_stopping_patience:
            print("\n\n*********************************")
            print(f"Early stopping triggered at epoch {epoch+1}")
            print("*********************************\n\n")
            break


def run_behavioral_traning(config):
    """
    Run behavioral training with the given configuration.
    
    Args:
        config (dict): Configuration dictionary containing training parameters
    """
    seed_everything(config['random_seed'])
    
    # Initialize trainingdataset
    dataset = ThingsDataset(csv_file=config['csv_file'], img_dir=config['img_dir'])
    
    # Split dataset
    train_size = int(config['train_portion'] * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Print first 5 samples of the train dataset
    for i in range(5):
        sample = train_dataset[i]
        print(f"Train dataset sample: {sample}")
    
    # Initialize inference dataset
    inference_dataset = ThingsInferenceDataset(inference_csv_file=config['inference_csv_file'], img_dir=config['img_dir'], RDM48_triplet_dir=config['RDM48_triplet_dir'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    inference_loader = DataLoader(inference_dataset, batch_size=config['batch_size'], shuffle=False)

    # Print first 5 samples of the train loader
    for i in range(5):
        sample = train_loader.dataset[i]
        print(f"Train data loader sample: {sample}")
    
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
    
    # Load checkpoint if resuming
    if config.get('resume_from_epoch', 0) > 0:
        model = load_checkpoint(model, config['resume_checkpoint_path'], device)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=config['lr'])
    
    # Print training information
    print("\nModel Configuration:")
    print("-------------------")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("\nUpdating layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    print(f"\nNumber of trainable parameters: {count_trainable_parameters(model)}\n")

    # Print random target information if specified
    if config.get('random_target_epoch') is not None:
        print(f"*** RANDOM TARGET CONFIGURATION ***")
        print(f"Random targets will be used in epoch: {config['random_target_epoch']}")
        print(f"Random target seed: {config.get('random_target_seed', 42)}")
        print("*** END RANDOM TARGET CONFIGURATION ***\n")

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
        config['batch_size'],
        config['csv_file'],
        config['img_dir'],
        config['early_stopping_patience'],
        config['checkpoint_path'],
        config['dora_parameters_path'],
        config['model_embedding_path'],
        config['model_rdm_path'],
        config.get('random_target_epoch'),
        config.get('random_target_seed', 42),
        config.get('resume_from_epoch', 0)
    )
