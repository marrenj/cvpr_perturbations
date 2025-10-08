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

        return image


# class ThingsCurriculumDataset(Dataset):
#     def __init__(self, array_of_image_names, csv_file, img_dir):
#         self.img_dir = img_dir
#         self.transform = transforms.Compose([
#                         transforms.Resize((224, 224)),
#                         transforms.ToTensor(),
#                         transforms.Normalize(mean=[0.52997664, 0.48070561, 0.41943838],
#                                             std=[0.27608301, 0.26593025, 0.28238822])
#                     ])

#         # Load and filter annotations based on the 'set' column
#         self.annotations = array_of_image_names
#         self.targets = pd.read_csv(csv_file, index_col=0)

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, index):
#         image_name = self.annotations[index]
#         img_path = os.path.join(self.img_dir, image_name)
#         image = Image.open(img_path).convert("RGB")
#         image = self.transform(image)


#         targets = torch.tensor(self.annotations[index, 1:].values.astype('float32'))
        
#         return image_name, image, targets


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
    predictions = []

    with torch.no_grad():
        for batch_idx, (image) in enumerate(inference_loader):
            image = image.to(device)
            
            output = model(image)

            predictions.extend(output.cpu().numpy())

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
    # define the three modules to save
    modules_to_save = [
        ("clip_model.visual.transformer.resblocks.22.attn.out_proj", "visual_resblock_22_attn"),
        ("clip_model.visual.transformer.resblocks.23.attn.out_proj", "visual_resblock_23_attn"),
        ("clip_model.transformer.resblocks.11.attn.out_proj", "transformer_resblock_11_attn"),
    ]

    # save the parameters for each module
    for module_path, module_name in modules_to_save:
        # Get the module
        module = model
        for attr in module_path.split("."):
            module = getattr(module, attr)

        # Extract DoRA parameters
        dora_params = {
            'm':module.m.detach().cpu(),
            'delta_D_A':module.delta_D_A.detach().cpu(),
            'delta_D_B':module.delta_D_B.detach().cpu(),
        }

        # Save the parameters
        save_path = os.path.join(dora_parameters_path, f"epoch{epoch + 1}_dora_params.pth")
        torch.save(dora_params, save_path)

        # Print parameter shapes
        print(f"\n{module_name} parameter shapes:")
        print(f"  m: shape {dora_params['m'].shape}")
        print(f"  delta_D_A: shape {dora_params['delta_D_A'].shape}")
        print(f"  delta_D_B: shape {dora_params['delta_D_B'].shape}")


# def pre_calculate_difficulties(model, train_loader, device, difficulty_criterion, batch_size, img_dir):
#         """
#         Pre-calculate difficulties for all images before training starts.
#         Uses difficulty_criterion.
#         """
#         print("Pre-calculating difficulties for all images...")
#         model.eval() # put the model in evaluation mode
#         all_losses = [] # initialize the list to store the losses
        
#         with torch.no_grad(), tqdm(enumerate(train_loader), total=len(train_loader), desc="Pre-calculating difficulties") as progress_bar:
#             for batch_idx, (_, images, targets, original_index) in progress_bar:
#                 images = images.to(device) # move the images to the device
#                 targets = targets.to(device) # move the targets to the device

#                 predictions = model(images) # get the predictions from the model

#                 loss = difficulty_criterion(predictions, targets) # calculate the loss between targets and predictions based on the difficulty criterion
#                 per_exemplar_loss = torch.mean(loss, dim=1) # calculate the mean loss for each exemplar
#                 all_losses.append({'original_index': original_index, 'exemplar_loss': per_exemplar_loss}) # append the image name and the loss for each exemplar
        
#         # Print the length of the all_losses list
#         print(f"All losses length: {len(all_losses)}")

#         # Print the first 5 elements of the all_losses list
#         print(f"All losses first 5 elements: {all_losses[:5]}")

#         # Consolidate all losses and indices from all batches
#         all_indices_tensor = torch.cat([d['original_index'] for d in all_losses])
#         all_losses_tensor = torch.cat([d['exemplar_loss'] for d in all_losses])

#         # Create a DataFrame for sorting and handling
#         loss_df = pd.DataFrame({
#         'original_index': all_indices_tensor.cpu().numpy(),
#         'loss_value': all_losses_tensor.cpu().numpy()
#         })

#         # Print the first 5 elements of the loss df
#         print(f"Loss df first 5 elements: {loss_df[:5]}")

#         print(f"Loss df shape: {loss_df.shape}")

#         # Sort the DataFrame from smallest to largest loss
#         sorted_loss_df = loss_df.sort_values(by='loss_value', ascending=True)

#         print("\nExample of sorted image indices (from lowest loss to highest loss):")
#         print(sorted_loss_df[:20]) # Display the first 20 original image indices and their losses

        # Get the original dataset and the train subset indices
        # The train_loader is a DataLoader whose .dataset is a Subset of the original ThingsDataset.
        # train_loader.dataset: Subset object (the training subset)
        # train_loader.dataset.dataset: the original ThingsDataset
        # train_loader.dataset.dataset['original_index']: indices of the original ThingsDataset used in this subset
        #original_dataset = train_loader.dataset.dataset  # This is the full ThingsDataset
        #train_subset_indices = train_loader.dataset.dataset['original_index']  # These are the indices used for the train subset

        # Print the first element of the original dataset
        #print(f"Original dataset first element: {original_dataset[0]}")

        # Print the first 5 elements of the train subset indices
        #print(f"Train subset indices first 5 elements: {train_subset_indices[:5]}")

        #curriculum_dataset = ThingsDataset(csv_file=config['csv_file'], img_dir=config['img_dir'], filter_indices=sorted_loss_df['original_index'].values)

        #for i in range(5):
        #    sample = curriculum_dataset[i]
        #    print(f"Curriculum dataset sample: {sample}")

        # Create a new curriculum dataset by reordering the train subset
        # sorted_loss_df['original_index'] contains the original dataset indices in difficulty order
        # We need to map these back to the train subset indices
        #train_indices_set = set(train_subset_indices)
        #sorted_train_indices = []

        #for idx in sorted_loss_df['original_index'].values:
        #    if idx in train_indices_set:
                # Find the position of this index in the train subset
        #        train_pos = [i for i, x in enumerate(train_subset_indices) if x == idx]
        #        if len(train_pos) > 0:
        #            sorted_train_indices.append(train_pos[0])

        #curriculum_dataset = Subset(original_dataset, [train_subset_indices[i] for i in sorted_train_indices])

        # Show first few samples 
        #print(f"\nFirst 3 samples from curriculum dataset:")
        #for i in range(min(3, len(curriculum_dataset))):
        #    sample = curriculum_dataset[i]
        #    print(f"Sample {i}: {sample[0:]}")  

        #curriculum_data_loader = DataLoader(curriculum_dataset, batch_size, shuffle=False)

        #print(f"\nCreated a new data loader with {len(curriculum_dataset)} samples.")

        #return sorted_loss_df


def update_difficulties(model, train_loader, device, difficulty_criterion):
        """Update difficulties for all images.
        Use difficulty_criterion.
        """
        print(f"Re-calculating difficulties for all images...")
        model.eval() # put the model in evaluation mode
        image_names = []
        original_indices = []
        predictions = []
        mse_losses = [] # initialize the list to store the losses
        
        with torch.no_grad(), tqdm(enumerate(train_loader), total=len(train_loader), desc="Calculating difficulties") as progress_bar:
            for batch_idx, (batch_image_names, images, targets, batch_original_indices) in progress_bar:
                images = images.to(device) # move the images to the device
                targets = targets.to(device) # move the targets to the device

                batch_outputs = model(images) # get the predictions from the model

                loss = difficulty_criterion(batch_outputs, targets) # calculate the loss between targets and predictions based on the difficulty criterion
                per_exemplar_loss = torch.mean(loss, dim=1) # calculate the mean loss for each exemplar

                predictions.extend(batch_outputs.cpu().numpy())
                image_names.extend(batch_image_names)
                original_indices.extend(batch_original_indices.cpu().numpy())
                mse_losses.extend(per_exemplar_loss.cpu().numpy()) # append the image name and the loss for each exemplar

        predictions = np.array(predictions)
        mse_losses = np.array(mse_losses)

        model_embeddings = pd.DataFrame({
            'image_name': image_names,
            'original_index': original_indices,
            'loss_value': mse_losses,
            'predictions': list(predictions)
        })
        
        # model_embeddings = pd.DataFrame(predictions)
        # model_embeddings['image_name'] = image_names
        # model_embeddings['original_index'] = original_indices
        # model_embeddings['loss_value'] = mse_losses

        # Sort the embeddings DataFrame by loss value
        sorted_model_embeddings = model_embeddings.sort_values(by='loss_value', ascending=False)

        #print("\nExample of sorted image indices (from lowest loss to highest loss):")
        #print(sorted_loss_df[:20]) # Display the first 20 original image indices and their losses

        # Get the original dataset and the train subset indices
        # The train_loader is a DataLoader whose .dataset is a Subset of the original ThingsDataset.
        # train_loader.dataset: Subset object (the training subset)
        # train_loader.dataset.dataset: the original ThingsDataset
        # train_loader.dataset.dataset['original_index']: indices of the original ThingsDataset used in this subset
        #original_dataset = train_loader.dataset.dataset  # This is the full ThingsDataset
        #train_subset_indices = train_loader.dataset.dataset['original_index']  # These are the indices used for the train subset

        # Print the first element of the original dataset
        #print(f"Original dataset first element: {original_dataset[0]}")

        # Print the first 5 elements of the train subset indices
        #print(f"Train subset indices first 5 elements: {train_subset_indices[:5]}")

        #curriculum_dataset = ThingsDataset(csv_file=config['csv_file'], img_dir=config['img_dir'], filter_indices=sorted_loss_df['original_index'].values)

        #for i in range(5):
        #    sample = curriculum_dataset[i]
        #    print(f"Curriculum dataset sample: {sample}")

        # Create a new curriculum dataset by reordering the train subset
        # sorted_loss_df['original_index'] contains the original dataset indices in difficulty order
        # We need to map these back to the train subset indices
        #train_indices_set = set(train_subset_indices)
        #sorted_train_indices = []

        #for idx in sorted_loss_df['original_index'].values:
        #    if idx in train_indices_set:
                # Find the position of this index in the train subset
        #        train_pos = [i for i, x in enumerate(train_subset_indices) if x == idx]
        #        if len(train_pos) > 0:
        #            sorted_train_indices.append(train_pos[0])

        #curriculum_dataset = Subset(original_dataset, [train_subset_indices[i] for i in sorted_train_indices])

        # Show first few samples 
        #print(f"\nFirst 3 samples from curriculum dataset:")
        #for i in range(min(3, len(curriculum_dataset))):
        #    sample = curriculum_dataset[i]
        #    print(f"Sample {i}: {sample[0:]}")  

        #curriculum_data_loader = DataLoader(curriculum_dataset, batch_size, shuffle=False)

        #print(f"\nCreated a new data loader with {len(curriculum_dataset)} samples.")

        return sorted_model_embeddings


def train_model(model, train_loader, test_loader, inference_loader, device, optimizer, criterion, epochs, training_res_path, difficulty_criterion, batch_size, csv_file, img_dir, early_stopping_patience=5, start_rate=.3, grow_rate=.05, grow_interval=5, checkpoint_path='clip_hba_model_cv.pth', dora_parameters_path='./dora_params', model_embedding_path='./model_embeddings', curriculum_subset_path='./curriculum_subsets', model_rdm_path='./model_rdms'):
    model.train()
    best_test_loss = float('inf')
    epochs_no_improve = 0

    # initial evaluation
    print("*********************************")
    print("Evaluating initial model")
    best_test_loss = evaluate_model(model, test_loader, device, criterion)
    print(f"Initial Validation Loss: {best_test_loss:.4f}")
    print("*********************************\n")

    # Calculate when we'll reach 100%
    updates_to_100 = int(np.ceil((1.0 - start_rate) / grow_rate))
    epochs_to_100 = updates_to_100 * grow_interval
    print(f"  Expected to reach 100% in {epochs_to_100} epochs ({updates_to_100} updates)")

    curriculum_update_epochs = [i * grow_interval for i in range(1, updates_to_100 + 1)]

    print(f"Curriculum updates will occur at epochs: {curriculum_update_epochs}")

    # Create folder to store DoRA parameters
    os.makedirs(dora_parameters_path, exist_ok=True)

    headers = ['epoch', 'train_loss', 'test_loss', 'behavioral_rsa_rho', 'behavioral_rsa_p_value']

    with open(training_res_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

    for epoch in range(epochs):
        total_loss = 0.0

        # If the epoch is in the curriculum update epochs, then re-calculate the difficulties and create a new curriculum data loader for this epoch
        # if epoch in curriculum_update_epochs:
        #     print(f"Re-calculating difficulties at epoch {epoch + 1}")
        #     sorted_loss_df = update_difficulties(model, train_loader, device, difficulty_criterion)
        #     curriculum_dataset = ThingsDataset(csv_file, img_dir, filter_indices=sorted_loss_df['original_index'].values)
            
        #     for i in range(5):
        #         sample = curriculum_dataset[i]
        #         print(f"Curriculum dataset sample: {sample}")

        #     curriculum_data_loader = DataLoader(curriculum_dataset, batch_size, shuffle=False)
        
        sorted_model_embeddings = update_difficulties(model, train_loader, device, difficulty_criterion)
        curriculum_dataset = ThingsDataset(csv_file, img_dir, filter_indices=sorted_model_embeddings['original_index'].values)
        curriculum_data_loader = DataLoader(curriculum_dataset, batch_size, shuffle=False)

        # Calculate the number of samples to use based on the current epoch
        total_curriculum_samples = len(curriculum_data_loader.dataset)
        current_portion = start_rate + grow_rate * (epoch // grow_interval)
        current_portion = min(1.0, current_portion)
        num_samples_to_use = int(total_curriculum_samples * current_portion)

        # Dynamically adjust the portion of curriculum data used for training as epochs progress.
        # Use a Subset of the curriculum dataset that grows every 'grow_interval' epochs.
       
        # Create a Subset of the curriculum dataset for this epoch
        # (Assumes curriculum_data_loader was created with shuffle=False)
        # from torch.utils.data import Subset, DataLoader

        # This line selects the first num_samples_to_use samples from the curriculum_data_loader.dataset in order (not randomly).
        # curriculum_subset is a torch.utils.data.Subset object that contains the first num_samples_to_use samples
        # from the curriculum_data_loader.dataset (which is a ThingsDataset instance sorted by difficulty).
        # Each item in curriculum_subset is a tuple: (image_name, image, targets, original_index).
        curriculum_subset = Subset(curriculum_data_loader.dataset, range(num_samples_to_use))
        print(f"Curriculum subset length: {len(curriculum_subset)}")

        # Curriculum subset save epochs
        curriculum_save_epochs = list(range(0, epochs, 10))

        if epoch == 0 or epoch in curriculum_save_epochs:
            # Save the curriculum subset to a csv file
            print(f"Saving curriculum subset to csv file at epoch {epoch+1}")
            # Extract the data from curriculum_subset to create a proper DataFrame
            curriculum_data = []
            for i in range(len(curriculum_subset)):
                image_name, image, _, original_index = curriculum_subset[i]
                curriculum_data.append({
                'image_name': image_name,
                'original_index': original_index.item() if hasattr(original_index, 'item') else original_index
                })
            # curriculum_data.append({
            #     'image_name': curriculum_subset.image_name,
            #     'original_index': curriculum_subset.original_index,
            #     'targets': curriculum_subset.targets,
            # })
            curriculum_subset_df = pd.DataFrame(curriculum_data)
            os.makedirs(curriculum_subset_path, exist_ok=True)
            curriculum_subset_df.to_csv(f'{curriculum_subset_path}/curriculum_subset_epoch{epoch+1}.csv', index=False)

            # Save the sorted model embeddings df to a csv file
            print(f"Saving sorted model embeddings df to csv file at epoch {epoch+1}")
            os.makedirs(model_embedding_path, exist_ok=True)
            sorted_model_embeddings.to_csv(f'{model_embedding_path}/sorted_model_embeddings_epoch{epoch+1}.csv', index=False)

        #print(f"Curriculum subset first 5 elements: {curriculum_subset[:5]}")
        curriculum_loader_for_epoch = DataLoader(curriculum_subset, batch_size=batch_size, shuffle=True)

        # Now, use curriculum_loader_for_epoch instead of curriculum_data_loader in your training loop:
        progress_bar = tqdm(enumerate(curriculum_loader_for_epoch), total=len(curriculum_loader_for_epoch), desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (_, images, targets, _) in progress_bar:

            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(images)
            
            loss = criterion(predictions, targets)
            progress_bar.set_postfix({'loss': loss.item()})
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
        avg_train_loss = total_loss / len(curriculum_loader_for_epoch.dataset)

        updates_completed = epoch // grow_interval
        print(f"Updates completed: {updates_completed}")
        print(f"Current data portion: {current_portion}")

        # Evaluate after every epoch
        avg_test_loss = evaluate_model(model, test_loader, device, criterion)
        print(f"Epoch {epoch+1}: Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_test_loss:.4f}")

        # Conduct behavioral RSA at every epoch
        rho, p_value, model_rdm = behavioral_RSA(model, inference_loader, device)
        print(f"Behavioral RSA Correlation & p-value: {rho:.4f}, {p_value:.4f}")
        
        # Prepare the data row with the epoch number and loss values
        data_row = [epoch + 1, avg_train_loss, avg_test_loss, rho, p_value]

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
        if current_portion != 1.0: # if the data portion is not 100%, then early stopping should not be enabled
            epochs_no_improve = 0 
            # Save the model checkpoint
            torch.save(model.state_dict(),checkpoint_path)
            print("\n\n-----------------------------------")
            print(f"Checkpoint saved for epoch {epoch+1}")
            print("-----------------------------------\n\n")
        else: # early stopping can be enabled as soon as the model has seen the 100% of thefull dataset
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
            
            # Save the curriculum subset to a csv file
            print(f"Training complete. Saving curriculum subset to csv file at finalepoch ({epoch+1})")
            # Extract the data from curriculum_subset to create a proper DataFrame
            curriculum_data = []
            for i in range(len(curriculum_subset)):
                image_name, image, _, original_index = curriculum_subset[i]
                curriculum_data.append({
                    'image_name': image_name,
                    'original_index': original_index.item() if hasattr(original_index, 'item') else original_index
                })
            curriculum_subset_df = pd.DataFrame(curriculum_data)
            os.makedirs(curriculum_subset_path, exist_ok=True)
            curriculum_subset_df.to_csv(f'{curriculum_subset_path}/curriculum_subset_epoch{epoch+1}.csv', index=False)
            # Save the sorted model embeddings df to a csv file
            print(f"Training complete. Saving sorted model embeddings df to csv file at final epoch ({epoch+1})")
            os.makedirs(model_embedding_path, exist_ok=True)
            sorted_model_embeddings.to_csv(f'{model_embedding_path}/sorted_model_embeddings_epoch{epoch+1}.csv', index=False)
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
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
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
    
    # Pre-calculate difficulties for baby step curriculum learning
    # sorted_loss_df = pre_calculate_difficulties(model, train_loader, device, difficulty_criterion=config['difficulty_criterion'], batch_size=config['batch_size'], img_dir=config['img_dir'])

    # curriculum_dataset = ThingsDataset(csv_file=config['csv_file'], img_dir=config['img_dir'], filter_indices=sorted_loss_df['original_index'].values)

    # for i in range(5):
    #     sample = curriculum_dataset[i]
    #     print(f"Curriculum dataset sample: {sample}")

    # curriculum_data_loader = DataLoader(curriculum_dataset, batch_size=config['batch_size'], shuffle=False)

    # Train model
    train_model(model, train_loader, test_loader, inference_loader, device, optimizer, 
                config['criterion'], 
                config['epochs'], 
                config['training_res_path'],
                config['difficulty_criterion'],
                config['batch_size'],
                config['csv_file'],
                config['img_dir'],
                config['early_stopping_patience'],
                config['start_rate'], 
                config['grow_rate'], 
                config['grow_interval'],
                config['checkpoint_path'],
                config['dora_parameters_path'],
                config['model_embedding_path'],
                config['curriculum_subset_path'],
                config['model_rdm_path'])

    # Print all model components (layers/modules) that are unfrozen and require grad
    # print("\nUnfrozen model components (modules with parameters that require grad):")
    # for name, module in model.named_modules():
    #     # Check if any parameter in the module requires grad
    #     params = list(module.parameters(recurse=False))
    #     if params and any(p.requires_grad for p in params):
    #         print(f"Module: {name} ({module.__class__.__name__})")
    #         for pname, param in module.named_parameters(recurse=False):
    #             if param.requires_grad:
    #                 print(f"    Parameter: {pname} | shape: {tuple(param.shape)}")


    # attention weights