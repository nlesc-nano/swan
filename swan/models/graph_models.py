"""Statistical models."""
import torch
import torch.nn.functional as F
from flamingo.features.featurizer import NUMBER_ATOMIC_GRAPH_FEATURES, NUMBER_BOND_GRAPH_FEATURES

from torch.nn import GRU, Linear, ReLU, Sequential
from torch_geometric.nn import NNConv, Set2Set

__all__ = ["MPNN"]


class MPNN(torch.nn.Module):
    """Create a molecular graph convolutional network.

    Use the convolution reported at: https://arxiv.org/abs/1704.01212
    This network was taking from: https://github.com/rusty1s/pytorch_geometric/blob/master/examples/qm9_nn_conv.py
    """
    def __init__(self, dim=64, batch_size=128):
        super(MPNN, self).__init__()
        self.lin0 = torch.nn.Linear(NUMBER_ATOMIC_GRAPH_FEATURES, dim)

        nn = Sequential(Linear(NUMBER_BOND_GRAPH_FEATURES, batch_size), ReLU(), Linear(batch_size, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)
