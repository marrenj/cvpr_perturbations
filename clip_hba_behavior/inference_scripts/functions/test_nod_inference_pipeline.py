import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
from pathlib import Path
import numpy as np
from torch.nn import functional as F
import copy
from tqdm import tqdm
import random
import math
import sys
sys.path.append('../')
from src.models.CLIPs.clip_hba import clip
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
import shutil
from functions.spose_dimensions import classnames66


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
    cache_dir = Path.home() / ".cache" / "clip"
    model_path = clip._download(url, str(cache_dir))

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
        self._tokenized_prompts_device = None  # Cache for device


    def forward(self, image):
        if self.clip_model.training:
            self.clip_model.eval()

        # Process all tokenized prompts in a single forward pass
        if self._tokenized_prompts_device != image.device:
            self.tokenized_prompts = self.tokenized_prompts.to(image.device)
            self._tokenized_prompts_device = image.device

        pred_score = self.clip_model(image, self.tokenized_prompts, self.pos_embedding)

        return pred_score.float()
    

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
    def __init__(self, category_index_file, img_dir, max_images_per_category=2):
        # Convert paths to Path objects for OS-agnostic handling
        self.img_dir = Path(img_dir)
        self.category_index_file = Path(category_index_file)
        
        self.transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.52997664, 0.48070561, 0.41943838],
                                            std=[0.27608301, 0.26593025, 0.28238822])
                    ])

        # Load the full CSV
        self.category_index = pd.read_csv(self.category_index_file)

        # Pre-compute all image paths for all categories
        # Sample max_images_per_category images from each category
        self.image_paths = []
        self.image_names = []
        self.categories = []

        for _, row in self.category_index.iterrows():
            category = row['category']
            category_path = self.img_dir / category
            
            if category_path.is_dir():
                # Get all images in this category
                all_images = [f.name for f in category_path.iterdir() 
                             if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
                
                # Sample max_images_per_category images randomly
                random.seed(42)  # For reproducibility
                sampled_images = random.sample(all_images, min(max_images_per_category, len(all_images)))
                
                for image_file in sampled_images:
                    # Store relative path for compatibility
                    image_path = Path(category) / image_file
                    self.image_paths.append(str(image_path))
                    self.image_names.append(image_file)
                    self.categories.append(category)
        
        print(f"Dataset loaded with {len(self.image_paths)} images from {len(self.category_index)} categories ({max_images_per_category} per category)")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Get the image at the specified index
        image_path = self.img_dir / self.image_paths[index]
        image_name = self.image_names[index]
        category = self.categories[index]
        
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        return image_name, image, category


def cache_dataloader_to_pinned_memory(data_loader):
    cached_batches = []
    caching_bar = tqdm(enumerate(data_loader),
                       total=len(data_loader),
                       desc="Caching images")

    with torch.no_grad():
        for _, (batch_image_names, batch_images, batch_categories) in caching_bar:
            cached_batches.append((
                list(batch_image_names),
                batch_images.pin_memory(),
                list(batch_categories)
            ))

    return cached_batches


def run_image(model, cached_batches, device=torch.device("cuda:0")):
    model.eval()
    image_names = []
    categories = []
    all_predictions = []
    
    progress_bar = tqdm(enumerate(cached_batches),
                        total=len(cached_batches),
                        desc=f"Processing images")
    
    with torch.no_grad():
        for batch_idx, (batch_image_names, pinned_batch_images, batch_categories) in progress_bar:
            batch_images = pinned_batch_images.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                batch_outputs = model(batch_images)

            
            all_predictions.append(batch_outputs.cpu())
            image_names.extend(batch_image_names)
            categories.extend(batch_categories)

        # min max to 0-1
        predictions = torch.cat(all_predictions, dim=0).numpy()
        # predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())

        hba_embedding = pd.DataFrame(predictions)
        hba_embedding['image'] = image_names
        hba_embedding['category'] = categories
        hba_embedding = hba_embedding[['image', 'category'] + [col for col in hba_embedding if col != 'image' and col != 'category']]

    return hba_embedding



def run_behavior_inference(config):
    """
    Run inference using the provided configuration.
    
    Args:
        config (dict): Configuration dictionary containing:
            - img_dir (str or Path): Directory containing input images
            - load_hba (bool): Whether to load HBA weights
            - backbone (str): CLIP backbone model name
            - model_path (str or Path): Path to the trained model
            - save_folder (str or Path): Output directory path
            - batch_size (int): Batch size for inference
            - cuda (str): Device specification
            - dora_params_path (str or Path): Path to DoRA parameters directory
            - training_res_path (str or Path, optional): Path to training results CSV
            - min_epoch_to_process (int, optional): Minimum epoch to process
    """
    # Convert paths to Path objects for OS-agnostic handling
    save_folder = Path(config['save_folder'])
    output_dir = save_folder
    
    # Create the directory if it doesn't exist
    print(f"\nEmbedding will be saved to folder: {save_folder}\n")
    save_folder.mkdir(parents=True, exist_ok=True)

    classnames = classnames66
    
    # Determine pos_embedding based on backbone
    pos_embedding = False if config['backbone'] == 'RN50' else True
    print(f"pos_embedding is {pos_embedding}")

    # Initialize model
    model = CLIPHBA(classnames=classnames, 
                    backbone_name=config['backbone'], 
                    pos_embedding=pos_embedding)

    # Load the dataset
    dataset = ImageDataset(category_index_file=config['category_index_file'], img_dir=config['img_dir'])
    data_loader = DataLoader(dataset, 
                           batch_size=config['batch_size'], 
                           shuffle=False,
                           num_workers=4,
                           pin_memory=True,
                           persistent_workers=True,
                           prefetch_factor=2) 

    cached_batches = cache_dataloader_to_pinned_memory(data_loader)
    del data_loader

    # Load HBA weights if specified
    device = torch.device(config['cuda'])
    
    if config['load_hba']:
        apply_dora_to_ViT(model, 
                         n_vision_layers=2, 
                         n_transformer_layers=1, 
                         r=32, 
                         dora_dropout=0.1)
    model.to(device)

    if config['load_hba']:
        
        # Convert dora_params_path to Path object
        dora_params_path = Path(config['dora_params_path'])
        
        # List all the files in the dora_params_path
        dora_params_files = [f.name for f in dora_params_path.iterdir() if f.is_file()]
        dora_params_files = sorted(dora_params_files, key=lambda f: int(f.split('_')[0].replace('epoch', '')))

        # Filter epochs based on minimum test loss if training results CSV is provided
        if 'training_res_path' in config and config['training_res_path']:
            training_res_path = Path(config['training_res_path'])
            if training_res_path.exists():
                training_df = pd.read_csv(training_res_path)
                
                # Find the epoch with minimum test loss
                min_test_loss_idx = training_df['test_loss'].idxmin()
                min_test_loss_epoch = int(training_df.loc[min_test_loss_idx, 'epoch'])
                
                print(f"Minimum test loss occurred at epoch {min_test_loss_epoch}")
                print(f"Original epoch files: {len(dora_params_files)}")
                
                # Filter DoRA parameter files to only include epochs up to minimum test loss
                filtered_dora_files = []
                for file in dora_params_files:
                    epoch_num = int(file.split('_')[0].replace('epoch', ''))
                    if epoch_num <= min_test_loss_epoch:
                        filtered_dora_files.append(file)
                
                dora_params_files = filtered_dora_files
                print(f"Filtered epoch files: {len(dora_params_files)}")
                total_files = len([f for f in dora_params_path.iterdir() if f.is_file()])
                print(f"Skipping {total_files - len(dora_params_files)} epochs after minimum test loss\n")

        # Final cross-run skip based on min_epoch_to_process
        if 'min_epoch_to_process' in config and config['min_epoch_to_process'] is not None:
            min_epoch = int(config['min_epoch_to_process'])
            before = len(dora_params_files)
            dora_params_files = [
                f for f in dora_params_files
                if int(f.split('_')[0].replace('epoch', '')) >= min_epoch
            ]
            print(f"Applying min_epoch_to_process={min_epoch}: filtered {before - len(dora_params_files)} epoch files; "
                  f"{len(dora_params_files)} remain.")

        for file in dora_params_files:
            # Extract epoch number from filename
            epoch = file.split('_')[0].replace('epoch', '')

            embedding_save_path = output_dir / f"nod_embeddings_epoch{epoch}.csv"

            # Skip if output already exists
            if embedding_save_path.exists():
                print(f"Epoch {epoch} already processed, skipping...")
                continue
            
            # Load DoRA parameters for this epoch
            dora_file_path = dora_params_path / file
            dora_params_state_dict = torch.load(dora_file_path, weights_only=True)
            model.load_state_dict(dora_params_state_dict, strict=False)
    
            # Run the model and save output embeddings
            hba_embedding = run_image(model, cached_batches, device=device)
            
            hba_embedding.to_csv(embedding_save_path, index=False)
            print(f"Embedding saved to {embedding_save_path}")
            print(f"-----------------------------------------------\n")        
    else:
        print(f"Using Original CLIP {config['backbone']}")