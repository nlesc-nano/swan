"""Select the model to use."""
import importlib

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
        module = importlib.import_module(opts.path)
        model = getattr(module, opts.name)

    return model(**opts.parameters)
