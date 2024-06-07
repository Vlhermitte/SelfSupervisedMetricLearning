import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the model (CNN)
class BloodNet(nn.Module):
    """
    A network that maps an image to an embedding (descriptor) with global pooling
    """
    def __init__(self, use_max=True):
        super(BloodNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # Use either max or average pooling
        self.pool = nn.AdaptiveMaxPool2d((1, 1)) if use_max else nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, eps=1e-6):
        x = self.conv(x)
        x = self.pool(x)
        x = F.normalize(x, p=2, dim=1, eps=eps)
        x = x.view(x.size(0), -1)
        return x


class SiameseNetwork(nn.Module):
    """
    Siamese network for self-supervised learning
    """

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(512 * 6 * 6, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x