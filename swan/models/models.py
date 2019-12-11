"""Statistical models."""
from torch import Tensor, nn


class FullyConnected(nn.Module):
    """Fully connected network for non-linear regression."""

    def __init__(self, n_feature: int, n_hidden: int):
        """Create a deep feed foward network."""
        super().__init__()
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


class ChemiNet(nn.Module):
    """Create a molecular graph convolutional network.

    Use the architecture reported at: https://arxiv.org/abs/1803.06236
    """

    def __init__(self):
        """Create the network architecture."""
        pass

    def forward(self, atoms_tensor, bonds_tensor) -> Tensor:
        """Rum model."""
        pass


def select_model(opts: dict) -> nn.Module:
    """Select a model using the input provided by the user."""
    if 'fingerprint' in opts.featurizer:
        return FullyConnected(opts.model.input_cells, opts.model.hidden_cells)
    else:
        raise NotImplementedError
