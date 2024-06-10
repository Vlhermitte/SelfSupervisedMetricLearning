import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights

# Define the model (CNN)
class BloodNet(nn.Module):
    """
    A network that maps an image to an embedding (descriptor) with global pooling
    """
    def __init__(self, use_max=True):
        super(BloodNet, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.pool = nn.AdaptiveMaxPool2d(1) if use_max else nn.AdaptiveAvgPool2d(1)

    def forward(self, x, eps=1e-6):
        x = self.resnet(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.normalize(x, p=2, dim=1, eps=eps)
        return x

    def __str__(self):
        return "BloodNet"


class BloodSimNet(nn.Module):
    """
    A network that maps an image to an embedding (descriptor) with global pooling and a projection layer
    """
    def __init__(self, projection_dim=64, use_max=True):
        super(BloodSimNet, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.pool = nn.AdaptiveMaxPool2d(1) if use_max else nn.AdaptiveAvgPool2d(1)
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim),
        )

    def forward(self, x, eps=1e-6):
        z = self.resnet(x)
        z = self.pool(z)
        z = z.view(z.size(0), -1)
        z = self.projector(z)
        z = F.normalize(z, p=2, dim=1, eps=eps)
        return z

    def __str__(self):
        return "BloodSimNet"
