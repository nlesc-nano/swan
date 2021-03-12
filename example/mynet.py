from torch import Tensor, nn


class FingerprintFullyConnected(nn.Module):
    """Fully connected network for non-linear regression."""

    def __init__(self, input_features: int = 2048, hidden_cells: int = 100, num_labels: int = 1):
        """Create a deep feed foward network."""
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_features, hidden_cells),
            nn.ReLU(),
            nn.Linear(hidden_cells, hidden_cells),
            nn.ReLU(),
            nn.Linear(hidden_cells, num_labels),
        )

    def forward(self, tensor: Tensor) -> Tensor:
        """Run the model."""
        return self.seq(tensor)
