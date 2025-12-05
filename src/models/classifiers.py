import torch
import torch.nn as nn
from typing import List


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class MLP2LayerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)


class MLP3LayerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.3
    ):
        super().__init__()
        layers = []

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


def get_classifier(
    classifier_type: str,
    input_dim: int,
    num_classes: int,
    hidden_dims: List[int] = [256],
    dropout: float = 0.3
) -> nn.Module:
    if classifier_type == "linear":
        return LinearClassifier(input_dim, num_classes)
    elif classifier_type == "mlp_2layer":
        return MLP2LayerClassifier(input_dim, num_classes, hidden_dims[0], dropout)
    elif classifier_type == "mlp_3layer":
        return MLP3LayerClassifier(input_dim, num_classes, hidden_dims, dropout)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
