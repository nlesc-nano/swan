"""Statistical models."""
import torch
import torch.nn.functional as F
from swan.dataset.features.featurizer import NUMBER_ATOMIC_GRAPH_FEATURES, NUMBER_BOND_GRAPH_FEATURES

from torch.nn import GRU, Linear, ReLU, Sequential
from torch_geometric.nn import NNConv, Set2Set

__all__ = ["MPNN"]


class MPNN(torch.nn.Module):
    """Create a molecular graph convolutional network.

    Use the convolution NN reported at: https://arxiv.org/abs/1704.01212
    This network was taking from: https://github.com/rusty1s/pytorch_geometric/blob/master/examples/qm9_nn_conv.py
    """
    def __init__(self, num_labels: int = 1, num_iterations: int = 3, output_channels: int = 10, batch_size: int = 128):
        """NN initialization.

        Parameters
        ----------
        num_labels
            How many labels to predict
        num_iterations
            Time steps (T)
        output_channels
            Number of output message channels
        batch_size
            Batch size used for training

        """

        super(MPNN, self).__init__()
        # Number of iterations to propagate the message
        self.iterations = num_iterations
        # Input layer
        self.lin0 = torch.nn.Linear(NUMBER_ATOMIC_GRAPH_FEATURES, output_channels)

        # NN that transform the states into message using the edge features
        nn = Sequential(Linear(NUMBER_BOND_GRAPH_FEATURES, batch_size), ReLU(), Linear(batch_size, output_channels * output_channels))
        self.conv = NNConv(output_channels, output_channels, nn, aggr='mean')
        # Combine the old state with the new one using a Gated Recurrent Unit
        self.gru = GRU(output_channels, output_channels)
        # Pooling function
        self.set2set = Set2Set(output_channels, processing_steps=self.iterations)
        # Fully connected output layers
        self.lin1 = torch.nn.Linear(2 * output_channels, output_channels)
        self.lin2 = torch.nn.Linear(output_channels, num_labels)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        # propagation in "time" of the messages
        for i in range(self.iterations):
            # Collect the message from the neighbors
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            # update the state
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        # Pool the state vectors
        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        return self.lin2(out)
