
"""
Re-ID Network Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ReIDNetwork(nn.Module):
    def __init__(self, embedding_dim=128, pretrained=True):
        super(ReIDNetwork, self).__init__()
        
        if pretrained:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet50(weights=None)
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        self.embedding = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_dim)
        )
        
        self.embedding_dim = embedding_dim
        
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        embedding = self.embedding(features)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding
    
    def get_embedding(self, x):
        return self.forward(x)
