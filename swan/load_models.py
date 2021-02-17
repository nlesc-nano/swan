"""Select the model to use."""
from flamingo.utils import Options
from torch import nn

from .models import FingerprintFullyConnected, ChemiNet


def select_model(opts: Options) -> nn.Module:
    """Select a model using the input provided by the user."""
    if 'fingerprint' in opts.featurizer:
        return FingerprintFullyConnected(opts.model.input_cells, opts.model.hidden_cells)
    elif 'graph' in opts.featurizer:
        return ChemiNet()
    else:
        raise NotImplementedError
