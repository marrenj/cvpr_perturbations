import torch
import torch.nn as nn
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import scipy.io
import scipy.stats
import os
import csv
from scipy.stats import spearmanr
from tqdm import tqdm
import torchvision.models as Models
import numpy as np
import torchvision.transforms as T
import torchvision.datasets as Datasets
from torch.utils.data import DataLoader


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


seed_everything(1)

# Initialize training dataset
csv_file = '../Data/spose_embedding66d_rescaled_1806train.csv'
img_dir = '../Data/Things1854'
dataset = ThingsDataset(csv_file, img_dir)

# Split training dataset into training and validation
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Initialize inference dataset
inference_csv_file = '../Data/spose_embedding66d_rescaled_48val_reordered.csv'
RDM48_triplet_dir = '../Data/RDM48_triplet.mat'
inference_dataset = ThingsInferenceDataset(inference_csv_file, img_dir, RDM48_triplet_dir)
   
# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
inference_loader = DataLoader(inference_dataset, batch_size, shuffle=False)


# Load the AlexNet model
model = Models.alexnet(weights=None)
# This line replaces the last fully connected layer of AlexNet's classifier with a new linear layer that outputs 100 features instead of the original number of classes.
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 100)


# Load the alexnet checkpoints
alexnet_folder = '../../../../teba/seeds_checkpoints/alexnet/seed_1'
alexnet_checkpoints = {}
for file in os.listdir(alexnet_folder):
    if file.endswith('.pth'):
        alexnet_checkpoints[file.split('_')[1].split('.')[0]] = torch.load(os.path.join(alexnet_folder, file))

device = torch.device('cuda:1')
criterion = nn.MSELoss()

headers = ['epoch', 'test_loss', 'behavioral_rsa_rho', 'behavioral_rsa_p_value']
alexnet_things_inference_res_path = '../alexnet_things_inference_res/alexnet_things_inference_res.csv'

with open(alexnet_things_inference_res_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)

# For each key in the alexnet_checkpoints dictionary, load the weights from the epoch into the model
for epoch, state in alexnet_checkpoints.items():
    model.load_state_dict(state, strict=True)
    model.eval()
    model = model.to(device)

    # Evaluate the model's performance on the training test set
    avg_test_loss = evaluate_model(model, test_loader, device, criterion)

    # Conduct behavioral RSA a this epoch
    rho, p_value, model_rdm = behavioral_RSA(model, inference_loader, device)

    # Prepare the data row with the epoch number and loss values
    data_row = [epoch, avg_test_loss, rho, p_value]

    # Append the data row to the CSV file
    with open(alexnet_things_inference_res_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data_row)




# # Load the weights from the epoch into the model
# state = torch.load(ckpt_path, map_location=device)
# model.load_state_dict(state, strict=True)
# model.eval()
# model = model.to(device)


