import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm


class NightsTripletDataset(Dataset):
    """Dataset for NIGHTS triplet evaluation."""
    
    def __init__(self, nights_dir, split='test'):
        """
        Args:
            nights_dir: Path to NIGHTS dataset directory
            split: 'train', 'val', or 'test'
            transform: Image transformations
        """
        self.nights_dir = Path(nights_dir)
        self.split = split
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.52997664, 0.48070561, 0.41943838],
                                 std=[0.27608301, 0.26593025, 0.28238822])
    ])
        
        # Load triplet metadata
        csv_path = self.nights_dir / 'data.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"NIGHTS data.csv not found at {csv_path}")
        
        # Load all triplets and filter by split
        all_triplets = pd.read_csv(csv_path)

        # Filter to only include rows where split column matches requested split
        if 'split' not in all_triplets.columns:
            raise ValueError(f"Column 'split' not found in data.csv. Available columns: {all_triplets.columns.tolist()}")

        self.triplets = all_triplets[all_triplets['split'] == split].reset_index(drop=True)

        if len(self.triplets) == 0:
            raise ValueError(f"No triplets found for split '{split}' in data.csv. Available splits: {all_triplets['split'].unique().tolist()}")

        print(f"Loaded {len(self.triplets)} triplets from NIGHTS {split} split (filtered from {len(all_triplets)} total triplets)")
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        row = self.triplets.iloc[idx]
        
        # Load images
        ref_img = Image.open(self.nights_dir / row['ref_path']).convert('RGB')
        img0 = Image.open(self.nights_dir / row['left_path']).convert('RGB')
        img1 = Image.open(self.nights_dir / row['right_path']).convert('RGB')
        
        # Apply transforms
        ref_img = self.transform(ref_img)
        img0 = self.transform(img0)
        img1 = self.transform(img1)
        
        # Determine winner: 0 if left won (left_vote=1), 1 if right won (right_vote=1)
        left_vote = int(row['left_vote'])
        right_vote = int(row['right_vote'])
    
        if left_vote == 1 and right_vote == 0:
            winner = 0  # Left image is the winner
        elif left_vote == 0 and right_vote == 1:
            winner = 1  # Right image is the winner
        else:
            raise ValueError(f"Invalid vote combination: left_vote={left_vote}, right_vote={right_vote}. Exactly one should be 1.")
        
        return ref_img, img0, img1, winner, idx