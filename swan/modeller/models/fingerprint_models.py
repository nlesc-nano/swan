"""Statistical models."""

from torch import Tensor, nn

__all__ = ["FingerprintFullyConnected"]


class FingerprintFullyConnected(nn.Module):
    """Fully connected network for non-linear regression."""

    def __init__(self, input_units: int = 2048, hidden_units: int = 100, output_units: int = 1,
                activation: nn.Module = nn.ReLU):
        """Create a deep feed foward network."""
        super().__init__()
        self.activation = activation

        self.net = nn.Sequential(
            nn.Linear(input_units, hidden_units),
            self.activation(),
            nn.Linear(hidden_units, hidden_units),
            self.activation(),
            nn.Linear(hidden_units, output_units),
        )

    def forward(self, tensor: Tensor) -> Tensor:
        """Run the model."""
        return self.net(tensor)
