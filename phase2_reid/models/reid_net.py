"""
Person Re-Identification Network

This module implements a Re-ID network using ResNet50 backbone
with a custom embedding head for person re-identification.

Architecture:
    Input Image (256x128) → ResNet50 Backbone → Global Avg Pool → 
    FC Layers → 128-dim L2-normalized embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ReIDNetwork(nn.Module):
    """
    Person Re-Identification Network.
    
    Uses ResNet50 pretrained on ImageNet as backbone,
    followed by embedding layers to produce a 128-dimensional
    feature vector for each person image.
    
    Args:
        embedding_dim: Dimension of output embedding (default: 128)
        pretrained: Whether to use ImageNet pretrained weights
    """
    
    def __init__(self, embedding_dim=128, pretrained=True):
        super(ReIDNetwork, self).__init__()
        
        # Load ResNet50 backbone
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            resnet = models.resnet50(weights=weights)
        else:
            resnet = models.resnet50(weights=None)
        
        # Remove final classification layer, keep up to avgpool
        # ResNet50 output: 2048 features after global average pooling
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Embedding head: 2048 → 512 → embedding_dim
        self.embedding = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_dim)
        )
        
        self.embedding_dim = embedding_dim
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, 3, 256, 128]
            
        Returns:
            L2-normalized embedding of shape [batch_size, embedding_dim]
        """
        # Extract features from backbone
        features = self.backbone(x)  # [B, 2048, 1, 1]
        features = features.view(features.size(0), -1)  # [B, 2048]
        
        # Project to embedding space
        embedding = self.embedding(features)  # [B, embedding_dim]
        
        # L2 normalize for cosine similarity
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding
    
    def get_embedding(self, x):
        """Alias for forward pass, used during inference"""
        return self.forward(x)
    
    def extract_features(self, x):
        """Extract features before embedding layer (for analysis)"""
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        return features


class ReIDNetworkWithClassifier(nn.Module):
    """
    Re-ID Network with additional classification head.
    
    Useful for training with both Triplet Loss and Cross-Entropy Loss.
    The classification head can improve feature learning.
    
    Args:
        num_classes: Number of person identities in training set
        embedding_dim: Dimension of embedding
        pretrained: Whether to use pretrained weights
    """
    
    def __init__(self, num_classes, embedding_dim=128, pretrained=True):
        super(ReIDNetworkWithClassifier, self).__init__()
        
        # Load backbone
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            resnet = models.resnet50(weights=weights)
        else:
            resnet = models.resnet50(weights=None)
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Embedding head
        self.embedding = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_dim)
        )
        
        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
    def forward(self, x, return_features=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor
            return_features: If True, return only embeddings (for inference)
            
        Returns:
            If return_features: normalized embeddings
            Else: (logits, embeddings) tuple
        """
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        embedding = self.embedding(features)
        embedding_norm = F.normalize(embedding, p=2, dim=1)
        
        if return_features:
            return embedding_norm
        
        logits = self.classifier(embedding)
        return logits, embedding_norm


# Test the model
if __name__ == '__main__':
    # Create model
    model = ReIDNetwork(embedding_dim=128, pretrained=False)
    
    # Test input
    dummy_input = torch.randn(4, 3, 256, 128)
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output L2 norm: {torch.norm(output, dim=1)}")  # Should be ~1.0
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
