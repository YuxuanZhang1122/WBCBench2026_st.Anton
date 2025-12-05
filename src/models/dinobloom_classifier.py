import torch
import torch.nn as nn
from typing import Literal

from src.models.dinobloom import DinoBloomFeatureExtractor
from src.models.classifiers import get_classifier


class DinoBloomClassifier(nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int,
        classifier_head: str = "linear",
        mlp_hidden_dims: list = [256],
        dropout: float = 0.3,
        feature_type: Literal["cls", "avg_patch", "concat"] = "cls",
        resolution_strategy: Literal["resize", "interpolate"] = "resize",
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.feature_type = feature_type

        self.feature_extractor = DinoBloomFeatureExtractor(
            checkpoint_path=checkpoint_path,
            resolution_strategy=resolution_strategy,
            freeze=freeze_backbone
        )

        if feature_type == "cls":
            input_dim = 768
        elif feature_type == "avg_patch":
            input_dim = 768
        elif feature_type == "concat":
            input_dim = 1536
        else:
            raise ValueError(f"Invalid feature_type: {feature_type}")

        self.classifier = get_classifier(
            classifier_head,
            input_dim,
            num_classes,
            mlp_hidden_dims,
            dropout
        )

    def forward(self, x):
        with torch.no_grad() if self.feature_extractor.backbone.training == False else torch.enable_grad():
            cls_token, avg_patch, concat_features = self.feature_extractor(x)

        if self.feature_type == "cls":
            features = cls_token
        elif self.feature_type == "avg_patch":
            features = avg_patch
        elif self.feature_type == "concat":
            features = concat_features

        logits = self.classifier(features)
        return logits

    def freeze_backbone(self):
        for param in self.feature_extractor.backbone.parameters():
            param.requires_grad = False
        self.feature_extractor.backbone.eval()

    def unfreeze_backbone(self):
        for param in self.feature_extractor.backbone.parameters():
            param.requires_grad = True
        self.feature_extractor.backbone.train()
