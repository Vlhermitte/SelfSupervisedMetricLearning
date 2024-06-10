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


# Define the loss function (Contrastive loss)
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss (SimCLR)
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, z_i, z_j, temperature=0.5):
        N = len(z_i)
        z = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_i_j = torch.diag(sim, N)
        sim_j_i = torch.diag(sim, -N)
        positive = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(2*N, 1)
        negative = sim[torch.ones(2*N, dtype=torch.bool)].reshape(2*N, -1)
        logits = torch.cat((positive, negative), dim=1)
        logits /= temperature
        labels = torch.zeros(2*N, dtype=torch.long, device=z.device)
        loss = F.cross_entropy(logits, labels)
        return loss
