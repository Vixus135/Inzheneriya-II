import torch
import torch.nn as nn
from torchvision import models


class LandscapeClassifier(nn.Module):
    """Классификатор ландшафтов на базе ResNet"""
    
    def __init__(self, num_classes: int = 6, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Заморозить веса backbone (для первого этапа обучения)"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Разморозить только последний слой
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_backbone(self):
        """Разморозить все веса для fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True