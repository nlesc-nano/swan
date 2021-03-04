"""Select the model to use."""
import importlib

from flamingo.utils import Options
from torch import nn

from .models import MPNN, FingerprintFullyConnected

DEFAULT_MODELS = {
    "fingerprintfullyconnected": FingerprintFullyConnected,
    "mpnn": MPNN,
}


def select_model(opts: Options) -> nn.Module:
    """Select a model using the input provided by the user."""
    name = opts.name.lower()
    model = DEFAULT_MODELS.get(name, None)
    if model is None:
        raise RuntimeError(f"Model {name} is not None")
        # if opts["model_path"] is None:
        #     msg = f"Model {name} is not known by Swan. Provide a model_path to the Network definition"
        #     raise RuntimeError(msg)
        # else:
        #     module = importlib.import_module(opts.model_path)
        #     model = getattr(module, opts.name)
        #     print("model: ", model)

    return model(**opts.parameters)
