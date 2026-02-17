from torchvision import transforms
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from PIL import Image


class ImagenetDataset(Dataset):
    def __init__(self, category_index_file, img_dir):
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

        for category in self.category_index['category'].unique():
            category_path = self.img_dir / category
            
            if category_path.is_dir():
                # # Get all images in this category
                # all_images = [f.name for f in category_path.iterdir() 
                #              if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
                
                # # Sample max_images_per_category images randomly
                # random.seed(42)  # For reproducibility
                # sampled_images = random.sample(all_images, min(max_images_per_category, len(all_images)))

                # sample the image_name from the category_index
                sampled_images = self.category_index[self.category_index['category'] == category]['image'].sample(min(max_images_per_category, len(self.category_index[self.category_index['category'] == category])))
                
                for image_file in sampled_images:
                    # Store relative path for compatibility
                    image_path = Path(category) / image_file
                    self.image_paths.append(str(image_path))
                    self.image_names.append(image_file)
                    self.categories.append(category)
        
        print(f"Dataset loaded with {len(self.image_paths)} images from {self.category_index['category'].nunique()} categories ({max_images_per_category} per category)")

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