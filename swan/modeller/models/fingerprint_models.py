"""Statistical models."""

from torch import Tensor, nn
from itertools import chain
from typing import Tuple

__all__ = ["FingerprintFullyConnected"]


class FingerprintFullyConnected(nn.Module):
    """Fully connected network for non-linear regression."""

    def __init__(self, input_units: int = 2048, hidden_units: Tuple[int, ...] = (100, 100), output_units: int = 1,
                activation: nn.Module = nn.ReLU):
        """Create a deep feed foward network."""
        super().__init__()
        self.activation = activation
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.output_units = output_units

        self.layers = self._construct_layers()
        self.net = nn.Sequential(*self.layers)

    def forward(self, tensor: Tensor) -> Tensor:
        """Run the model."""
        return self.net(tensor)

    def _construct_layers(self):
        in_sizes = [self.input_units] + list(self.hidden_units)
        out_sizes = list(self.hidden_units) + [self.output_units]
        
        linear_layers = [nn.Linear(n_in, n_out) for n_in, n_out in zip(in_sizes, out_sizes)]
        
        activations = [self.activation() for _ in linear_layers]
        
        all_layers = list(chain(*zip(linear_layers, activations)))[:-1]  # no activation after last layer
        
        return all_layers
    
    def get_config(self):
        return {
            'input_units': self.input_units,
            'hidden_units': self.hidden_units,
            'output_units': self.output_units,
            'activation': self.activation
        }
