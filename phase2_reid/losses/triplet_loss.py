"""
Triplet Loss for Person Re-Identification

Triplet loss ensures that:
- Images of the same person are close in embedding space
- Images of different persons are far apart

Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)

Where:
- d(a, p): Distance between anchor and positive (same person)
- d(a, n): Distance between anchor and negative (different person)
- margin: Minimum separation between positive and negative pairs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Standard Triplet Loss.
    
    Computes loss for triplets of (anchor, positive, negative) embeddings.
    
    Args:
        margin: Minimum distance margin between positive and negative pairs
    """
    
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positive: Positive embeddings [batch_size, embedding_dim]
            negative: Negative embeddings [batch_size, embedding_dim]
            
        Returns:
            loss: Scalar triplet loss
        """
        # Squared Euclidean distance
        dist_pos = torch.sum((anchor - positive) ** 2, dim=1)
        dist_neg = torch.sum((anchor - negative) ** 2, dim=1)
        
        # Triplet loss with margin
        losses = F.relu(dist_pos - dist_neg + self.margin)
        
        return losses.mean()


class TripletLossWithHardMining(nn.Module):
    """
    Triplet Loss with Batch Hard Mining.
    
    Instead of using random triplets, this finds:
    - Hardest positive: Same person, but furthest away
    - Hardest negative: Different person, but closest
    
    This makes training more efficient and effective.
    
    Args:
        margin: Minimum distance margin
    """
    
    def __init__(self, margin=0.3):
        super(TripletLossWithHardMining, self).__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels):
        """
        Compute triplet loss with hard mining.
        
        Args:
            embeddings: All embeddings in batch [batch_size, embedding_dim]
            labels: Person IDs for each embedding [batch_size]
            
        Returns:
            loss: Scalar triplet loss
        """
        # Compute pairwise distance matrix
        dist_matrix = self._pairwise_distances(embeddings)
        
        # Create masks for positive and negative pairs
        labels = labels.unsqueeze(0)
        positive_mask = (labels == labels.T).float()
        negative_mask = (labels != labels.T).float()
        
        # Remove self-comparisons from positive mask
        positive_mask.fill_diagonal_(0)
        
        # Find hardest positive (max distance among same-person pairs)
        hard_positives = (dist_matrix * positive_mask).max(dim=1)[0]
        
        # Find hardest negative (min distance among different-person pairs)
        # Add large value to same-person pairs so they're not selected
        dist_neg = dist_matrix + positive_mask * 1e6
        hard_negatives = dist_neg.min(dim=1)[0]
        
        # Compute triplet loss
        losses = F.relu(hard_positives - hard_negatives + self.margin)
        
        # Only count valid triplets
        valid_triplets = (hard_positives > 0) & (hard_negatives < 1e6)
        
        if valid_triplets.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        return losses[valid_triplets].mean()
    
    def _pairwise_distances(self, embeddings):
        """
        Compute pairwise Euclidean distances.
        
        Args:
            embeddings: [batch_size, embedding_dim]
            
        Returns:
            distances: [batch_size, batch_size] distance matrix
        """
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        dot_product = torch.mm(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances = F.relu(distances)  # Numerical stability
        
        return torch.sqrt(distances + 1e-8)


class CombinedLoss(nn.Module):
    """
    Combined Triplet Loss + Cross-Entropy Loss.
    
    Cross-Entropy helps with classification accuracy.
    Triplet Loss helps with metric learning.
    
    Args:
        num_classes: Number of person identities
        margin: Triplet loss margin
        triplet_weight: Weight for triplet loss
        ce_weight: Weight for cross-entropy loss
    """
    
    def __init__(self, num_classes, margin=0.3, triplet_weight=1.0, ce_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.triplet_loss = TripletLoss(margin=margin)
        self.ce_loss = nn.CrossEntropyLoss()
        self.triplet_weight = triplet_weight
        self.ce_weight = ce_weight
    
    def forward(self, anchor_emb, positive_emb, negative_emb, 
                logits=None, labels=None):
        """
        Compute combined loss.
        
        Args:
            anchor_emb, positive_emb, negative_emb: Triplet embeddings
            logits: Classification logits (optional)
            labels: Ground truth labels (optional)
            
        Returns:
            total_loss: Combined loss
        """
        # Triplet loss
        triplet = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
        
        total_loss = self.triplet_weight * triplet
        
        # Add cross-entropy if logits provided
        if logits is not None and labels is not None:
            ce = self.ce_loss(logits, labels)
            total_loss += self.ce_weight * ce
        
        return total_loss


# Test losses
if __name__ == '__main__':
    # Test basic triplet loss
    triplet_loss = TripletLoss(margin=0.3)
    
    anchor = F.normalize(torch.randn(8, 128), dim=1)
    positive = F.normalize(torch.randn(8, 128), dim=1)
    negative = F.normalize(torch.randn(8, 128), dim=1)
    
    loss = triplet_loss(anchor, positive, negative)
    print(f"Triplet Loss: {loss.item():.4f}")
    
    # Test hard mining loss
    hard_loss = TripletLossWithHardMining(margin=0.3)
    
    embeddings = F.normalize(torch.randn(16, 128), dim=1)
    labels = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8])
    
    loss = hard_loss(embeddings, labels)
    print(f"Hard Mining Loss: {loss.item():.4f}")
