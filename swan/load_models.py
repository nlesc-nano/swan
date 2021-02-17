"""Select the model to use."""
from flamingo.utils import Options
from torch import nn

from .models import ChemiNet, FingerprintFullyConnected

DEFAULT_MODELS = {
    "fingerprintfullyconnected": FingerprintFullyConnected,
    "cheminet": ChemiNet,
}


def select_model(opts: Options) -> nn.Module:
    """Select a model using the input provided by the user."""
    name = opts.name.lower()
    model = DEFAULT_MODELS.get(name, None)
    if model is None:
        raise NotImplementedError

    return model(**opts.parameters)
