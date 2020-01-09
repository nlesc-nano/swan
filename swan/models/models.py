"""Statistical models."""
from torch import Tensor, nn
from torch_geometric.nn import NNConv
from torch.nn import BatchNorm1d
import torch_geometric as tg
import torch.nn.functional as F
from ..features.featurizer import NUMBER_ATOMIC_GRAPH_FEATURES, NUMBER_BOND_GRAPH_FEATURES


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

    def forward(self, data) -> Tensor:
        """Run model."""
        x = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, data.edge_index, data.edge_attr))
        x = self.bn2(x)
        x = tg.nn.global_add_pool(x, data.batch)
        return self.output_layer(x)


def select_model(opts: dict) -> nn.Module:
    """Select a model using the input provided by the user."""
    if 'fingerprint' in opts.featurizer:
        return FullyConnected(opts.model.input_cells, opts.model.hidden_cells)
    elif 'graph' in opts.featurizer:
        return ChemiNet()
    else:
        raise NotImplementedError
