
"""
Market-1501 Dataset Loader for Person Re-ID
"""

import os
import re
from PIL import Image
from collections import defaultdict
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class Market1501Dataset(Dataset):
    """
    Market-1501 dataset for triplet training
    Returns: (anchor, positive, negative) triplets
    """
    
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform or self._default_transform()
        
        if mode == 'train':
            self.data_dir = os.path.join(root_dir, 'bounding_box_train')
        else:
            self.data_dir = os.path.join(root_dir, 'bounding_box_test')
        
        self.images_by_id = defaultdict(list)
        self._load_dataset()
        
        self.person_ids = list(self.images_by_id.keys())
        
        self.all_images = []
        for pid, imgs in self.images_by_id.items():
            for img_path in imgs:
                self.all_images.append((img_path, pid))
        
        print(f"Loaded {len(self.all_images)} images of {len(self.person_ids)} identities")
    
    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_dataset(self):
        pattern = re.compile(r'(\d+)_c(\d)s(\d)_(\d+)_(\d+)\.jpg')
        
        for filename in os.listdir(self.data_dir):
            if not filename.endswith('.jpg'):
                continue
            
            match = pattern.match(filename)
            if match:
                person_id = int(match.group(1))
                
                if person_id <= 0:
                    continue
                
                img_path = os.path.join(self.data_dir, filename)
                self.images_by_id[person_id].append(img_path)
    
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        anchor_path, anchor_id = self.all_images[idx]
        
        positive_candidates = [p for p in self.images_by_id[anchor_id] if p != anchor_path]
        if len(positive_candidates) == 0:
            positive_path = anchor_path
        else:
            positive_path = random.choice(positive_candidates)
        
        negative_id = random.choice([pid for pid in self.person_ids if pid != anchor_id])
        negative_path = random.choice(self.images_by_id[negative_id])
        
        anchor = self._load_image(anchor_path)
        positive = self._load_image(positive_path)
        negative = self._load_image(negative_path)
        
        return anchor, positive, negative, anchor_id
    
    def _load_image(self, path):
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


class Market1501Query(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.query_dir = os.path.join(root_dir, 'query')
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
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
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        item = self.images[idx]
        img = Image.open(item['path']).convert('RGB')
        img = self.transform(img)
        return img, item['person_id'], item['camera_id']
