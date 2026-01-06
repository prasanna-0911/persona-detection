
"""
Triplet Loss for Person Re-Identification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        dist_pos = torch.sum((anchor - positive) ** 2, dim=1)
        dist_neg = torch.sum((anchor - negative) ** 2, dim=1)
        losses = F.relu(dist_pos - dist_neg + self.margin)
        return losses.mean()
