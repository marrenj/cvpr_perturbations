import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import pandas as pd
from PIL import Image


# TODO: Add image_category as an object to return
class ThingsDataset(Dataset):
    def __init__(self, img_annotations_file, img_dir):
        self.img_dir = img_dir
        self.transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.52997664, 0.48070561, 0.41943838],
                                            std=[0.27608301, 0.26593025, 0.28238822])
                    ])

        # Load and filter annotations based on the 'set' column
        self.annotations = pd.read_csv(img_annotations_file, index_col=0)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_name = self.annotations.iloc[index, 0]
        img_path = os.path.join(self.img_dir, image_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)


        targets = torch.tensor(self.annotations.iloc[index, 1:].values.astype('float32'))
        
        return image_name, image, targets