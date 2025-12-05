import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=13, pretrained=True, dropout=0.4):
        super().__init__()

        self.resnet = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)
