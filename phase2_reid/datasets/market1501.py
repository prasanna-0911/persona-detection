"""
Market-1501 Dataset Loader for Person Re-Identification

This module provides dataset classes for loading and processing
the Market-1501 dataset for training Re-ID models with triplet loss.

Dataset structure:
    Market-1501-v15.09.15/
    ├── bounding_box_train/     # Training images
    ├── bounding_box_test/      # Testing images (gallery)
    └── query/                  # Query images for evaluation
"""

import os
import re
import random
from collections import defaultdict

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class Market1501Dataset(Dataset):
    """
    Market-1501 dataset for triplet training.
    
    Returns triplets of (anchor, positive, negative) images where:
    - anchor: Reference image of a person
    - positive: Different image of the SAME person
    - negative: Image of a DIFFERENT person
    
    Args:
        root_dir: Path to Market-1501-v15.09.15 folder
        mode: 'train' or 'test'
        transform: Optional image transformations
    """
    
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform or self._default_transform()
        
        # Select appropriate folder
        if mode == 'train':
            self.data_dir = os.path.join(root_dir, 'bounding_box_train')
        else:
            self.data_dir = os.path.join(root_dir, 'bounding_box_test')
        
        # Organize images by person ID
        self.images_by_id = defaultdict(list)
        self._load_dataset()
        
        # List of valid person IDs
        self.person_ids = list(self.images_by_id.keys())
        
        # Flat list for iteration
        self.all_images = []
        for pid, imgs in self.images_by_id.items():
            for img_path in imgs:
                self.all_images.append((img_path, pid))
        
        print(f"Loaded {len(self.all_images)} images of {len(self.person_ids)} identities")
    
    def _default_transform(self):
        """Standard preprocessing for Re-ID"""
        return transforms.Compose([
            transforms.Resize((256, 128)),  # Standard Re-ID size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_dataset(self):
        """Parse filenames and organize by person ID"""
        # Filename pattern: 0001_c1s1_001051_00.jpg
        # Format: personID_cameraID_sequenceID_frameID.jpg
        pattern = re.compile(r'(\d+)_c(\d)s(\d)_(\d+)_(\d+)\.jpg')
        
        for filename in os.listdir(self.data_dir):
            if not filename.endswith('.jpg'):
                continue
            
            match = pattern.match(filename)
            if match:
                person_id = int(match.group(1))
                
                # Skip junk images (ID <= 0)
                if person_id <= 0:
                    continue
                
                img_path = os.path.join(self.data_dir, filename)
                self.images_by_id[person_id].append(img_path)
    
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        """
        Get a triplet of images.
        
        Returns:
            anchor: Tensor of anchor image
            positive: Tensor of positive image (same person)
            negative: Tensor of negative image (different person)
            anchor_id: Person ID of anchor
        """
        anchor_path, anchor_id = self.all_images[idx]
        
        # Get positive: same person, different image
        positive_candidates = [p for p in self.images_by_id[anchor_id] if p != anchor_path]
        if len(positive_candidates) == 0:
            positive_path = anchor_path  # Fallback
        else:
            positive_path = random.choice(positive_candidates)
        
        # Get negative: different person
        negative_id = random.choice([pid for pid in self.person_ids if pid != anchor_id])
        negative_path = random.choice(self.images_by_id[negative_id])
        
        # Load and transform images
        anchor = self._load_image(anchor_path)
        positive = self._load_image(positive_path)
        negative = self._load_image(negative_path)
        
        return anchor, positive, negative, anchor_id
    
    def _load_image(self, path):
        """Load and transform a single image"""
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


class Market1501Query(Dataset):
    """
    Query dataset for Re-ID evaluation.
    
    Used to evaluate model by finding matching persons in gallery.
    """
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.query_dir = os.path.join(root_dir, 'query')
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.images = []
        pattern = re.compile(r'(\d+)_c(\d)s(\d)_(\d+)_(\d+)\.jpg')
        
        for filename in os.listdir(self.query_dir):
            if filename.endswith('.jpg'):
                match = pattern.match(filename)
                if match:
                    person_id = int(match.group(1))
                    camera_id = int(match.group(2))
                    self.images.append({
                        'path': os.path.join(self.query_dir, filename),
                        'person_id': person_id,
                        'camera_id': camera_id
                    })
        
        print(f"Loaded {len(self.images)} query images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        item = self.images[idx]
        img = Image.open(item['path']).convert('RGB')
        img = self.transform(img)
        return img, item['person_id'], item['camera_id']


class Market1501Gallery(Dataset):
    """
    Gallery dataset for Re-ID evaluation.
    
    Contains images to search through when matching query images.
    """
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.gallery_dir = os.path.join(root_dir, 'bounding_box_test')
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.images = []
        pattern = re.compile(r'(-?\d+)_c(\d)s(\d)_(\d+)_(\d+)\.jpg')
        
        for filename in os.listdir(self.gallery_dir):
            if filename.endswith('.jpg'):
                match = pattern.match(filename)
                if match:
                    person_id = int(match.group(1))
                    camera_id = int(match.group(2))
                    # Skip junk images
                    if person_id < 0:
                        continue
                    self.images.append({
                        'path': os.path.join(self.gallery_dir, filename),
                        'person_id': person_id,
                        'camera_id': camera_id
                    })
        
        print(f"Loaded {len(self.images)} gallery images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        item = self.images[idx]
        img = Image.open(item['path']).convert('RGB')
        img = self.transform(img)
        return img, item['person_id'], item['camera_id']


# Test the dataset
if __name__ == '__main__':
    # Update path as needed
    dataset = Market1501Dataset(
        root_dir='path/to/Market-1501-v15.09.15',
        mode='train'
    )
    
    anchor, positive, negative, pid = dataset[0]
    print(f"Anchor shape: {anchor.shape}")
    print(f"Person ID: {pid}")
    print(f"Total samples: {len(dataset)}")
