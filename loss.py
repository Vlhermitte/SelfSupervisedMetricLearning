import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the loss function (Triplet loss)
class TripletLoss(nn.Module):
    """
    Triplet loss
    """
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        d_pos = F.pairwise_distance(anchor, positive, p=2)
        d_neg = F.pairwise_distance(anchor, negative, p=2)
        loss = F.relu(d_pos - d_neg + self.margin)
        return loss.mean()