import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import numpy as np
from torch.nn import functional as F
import copy
from tqdm import tqdm
import random
import math
from functions.spose_dimensions import *
import sys
sys.path.append('../')
from src.models.CLIPs.clip_hba import clip
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
import shutil


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


class ImageDataset(Dataset):
    def __init__(self, stimuli_file, concept_index_file, img_dir):
        self.img_dir = img_dir
        self.transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.52997664, 0.48070561, 0.41943838],
                                            std=[0.27608301, 0.26593025, 0.28238822])
                    ])

        # Load the full CSV
        self.stimuli_metadata = pd.read_csv(stimuli_file, index_col=0)
        self.concept_index = np.load(concept_index_file)

        print(self.stimuli_metadata)
        print(self.concept_index)

    def __len__(self):
        return len(self.stimuli_metadata)

    def __getitem__(self, index):
        # get the row at the specified index
        row = self.stimuli_metadata.iloc[index]

        # extract the concept and image name from the row
        concept = row['concept']
        image_name = row['stimulus']

        # Build the image path
        image_path = os.path.join(self.img_dir, concept, image_name)

        # Load and transform the image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image_name, image, concept
    

def run_image(model, data_loader, device=torch.device("cuda:0")):
    model.eval()
    model.to(device)
    image_names = []
    concepts = []
    predictions = []
    
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Processing images")
    
    with torch.no_grad():
        for batch_idx, (batch_image_names, batch_images, batch_concepts) in progress_bar:
            batch_images = batch_images.to(device)
            batch_outputs = model(batch_images)
            
            predictions.extend(batch_outputs.cpu().numpy())
            image_names.extend(batch_image_names)
            concepts.extend(batch_concepts)

        # min max to 0-1
        predictions = np.array(predictions)
        # predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())

        hba_embedding = pd.DataFrame(predictions)
        hba_embedding['image'] = image_names
        hba_embedding['concept'] = concepts
        hba_embedding = hba_embedding[['image', 'concept'] + [col for col in hba_embedding if col != 'image' and col != 'concept']]
        #emb_save_path = f"{save_folder}/static_embedding.csv"
        #hba_embedding.to_csv(emb_save_path, index=False)
        #print(f"Embedding saved to {emb_save_path}")

        #rdm generation
        # rdm = 1 - np.corrcoef(np.array(predictions))
        # np.fill_diagonal(rdm, 0)
        # rdm_save_path = f"{save_folder}/static_rdm.npy"
        # np.save(rdm_save_path, rdm)
        # print(f"RDM saved to {rdm_save_path}")
        # print(f"-----------------------------------------------\n")
    return hba_embedding



def run_behavior_inference(config):
    """
    Run inference using the provided configuration.
    
    Args:
        config (dict): Configuration dictionary containing:
            - img_dir (str): Directory containing input images
            - load_hba (bool): Whether to load HBA weights
            - backbone (str): CLIP backbone model name
            - model_path (str): Path to the trained model
            - save_folder (str): Output directory path
            - batch_size (int): Batch size for inference
            - cuda (str): Device specification
    """
    # Create the directory if it doesn't exist
    print(f"\nEmbedding will be saved to folder: {config['save_folder']}\n")
    if os.path.exists(config['save_folder']):
        shutil.rmtree(config['save_folder'])
    os.makedirs(config['save_folder'])

    classnames = classnames66
    
    # Determine pos_embedding based on backbone
    pos_embedding = False if config['backbone'] == 'RN50' else True
    print(f"pos_embedding is {pos_embedding}")

    # Initialize model
    model = CLIPHBA(classnames=classnames, 
                    backbone_name=config['backbone'], 
                    pos_embedding=pos_embedding)

    # Load the dataset
    dataset = ImageDataset(stimuli_file=config['stimuli_file'], concept_index_file=config['concept_index_file'], img_dir=config['img_dir'])
    data_loader = DataLoader(dataset, 
                           batch_size=config['batch_size'], 
                           shuffle=False,
                           num_workers=3,7
                           pin_memory=True) 

    # Load HBA weights if specified
    if config['load_hba']:
        apply_dora_to_ViT(model, 
                         n_vision_layers=2, 
                         n_transformer_layers=1, 
                         r=32, 
                         dora_dropout=0.1)
        # Load all the parameters in the final trained model
        model_state_dict = torch.load(config['model_path'])
        # List all the files in the dora_params_path
        dora_params_files = os.listdir(config['dora_params_path'])
        dora_params_files = sorted(dora_params_files, key=lambda f: int(f.split('_')[0].replace('epoch', '')))
        
        # Filter to start from epoch 8
        dora_params_files = [f for f in dora_params_files if int(f.split('_')[0].replace('epoch', '')) >= 8]

        for file in dora_params_files:
            # This line creates a new dictionary called dora_params_state_dict.
            # It iterates over every key-value pair in model_state_dict (which is the state dict loaded from the model_path).
            # Replace the keys and values in model_state_dict only for the dora layers listed in the dora_params_path
            epoch = file.split('_')[0]
            dora_params_state_dict = torch.load(config['dora_params_path'] + '/' + file)
            for key, value in dora_params_state_dict.items():
                model_state_dict[key] = value

            # For each key, it removes the prefix "module." if it exists, by replacing "module." with an empty string.
            # This is necessary because sometimes models are saved with a "module." prefix (e.g., when using DataParallel),
            # but the current model's state_dict does not expect this prefix.
            # The value is kept unchanged.
            # The result is a state_dict with keys matching the current model, ready to be loaded.
            adjusted_state_dict = {key.replace("module.", ""): value 
                                 for key, value in model_state_dict.items()}

            model.load_state_dict(adjusted_state_dict)
            
            device = torch.device(config['cuda'])
    
            # Run the model and save output embeddings
            hba_embedding = run_image(model, data_loader, device=device)
            
            # Create the 720_concepts subdirectory if it doesn't exist
            output_dir = f"{config['save_folder']}_720_concepts"
            os.makedirs(output_dir, exist_ok=True)
            
            embedding_save_path = f"{output_dir}/720_concept_embeddings_{epoch}.csv"
            hba_embedding.to_csv(embedding_save_path, index=False)
            print(f"Embedding saved to {embedding_save_path}")
            print(f"-----------------------------------------------\n")        
    else:
        print(f"Using Original CLIP {config['backbone']}")