"""Statistical models."""
import torch
from torch import Tensor, nn


class Siamese(nn.Module):
    """Siamese architecture."""

    def __init__(self, n_feature: int, n_hidden: int):
        super(Siamese, self).__init__()
        self.net_fingerprints = nn.Sequential(
            nn.Linear(n_feature, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )
        self.net_descriptors_3d = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
        )

    def forward(self, tensor_1: Tensor, tensor_2: Tensor):
        """Run the model."""
        x = self.net_fingerprints(tensor_1)
        y = self.net_descriptors_3d(tensor_2)
        return torch.abs(x - y)


class FullyConnected(nn.Module):
    """Fully connected network for non-linear regression."""

    def __init__(self, n_feature: int, n_hidden: int):
        super(FullyConnected, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(n_feature, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, tensor: Tensor) -> Tensor:
        """Run the model."""
        return self.seq(tensor)


def select_model(opts: dict) -> nn.Module:
    """Select a model using the input provided by the user."""
    return FullyConnected(opts.model.input_cells, opts.model.hidden_cells)
