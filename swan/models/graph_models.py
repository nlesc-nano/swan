"""Statistical models."""
import torch.nn.functional as F
import torch_geometric as tg
from flamingo.features.featurizer import (NUMBER_ATOMIC_GRAPH_FEATURES,
                                          NUMBER_BOND_GRAPH_FEATURES)
from torch import Tensor, nn
from torch.nn import BatchNorm1d
from torch_geometric.nn import NNConv

__all__ = ["ChemiNet"]


class ChemiNet(nn.Module):
    """Create a molecular graph convolutional network.

    Use the convolution reported at: https://arxiv.org/abs/1704.01212
    """

    def __init__(self, output_channels: int = 50):
        """Create the network architecture."""
        super().__init__()
        self.output_channels = output_channels
        self.lin1 = nn.Sequential(
            nn.Linear(NUMBER_BOND_GRAPH_FEATURES, NUMBER_ATOMIC_GRAPH_FEATURES * output_channels),
            nn.ReLU(),
        )
        self.lin2 = nn.Sequential(
            nn.Linear(NUMBER_BOND_GRAPH_FEATURES, output_channels * output_channels // 2),
            nn.ReLU(),
        )
        self.conv1 = NNConv(NUMBER_ATOMIC_GRAPH_FEATURES, output_channels, self.lin1)
        self.conv2 = NNConv(output_channels, output_channels // 2, self.lin2)
        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(output_channels // 2, output_channels // 2),
            nn.ReLU(),
            nn.Linear(output_channels // 2, 1)
        )
        self.bn1 = BatchNorm1d(output_channels)
        self.bn2 = BatchNorm1d(output_channels // 2)

    def forward(self, data: tg.data.Dataset) -> Tensor:
        """Run model."""
        x = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, data.edge_index, data.edge_attr))
        x = self.bn2(x)
        x = tg.nn.global_add_pool(x, data.batch)
        return self.output_layer(x)
