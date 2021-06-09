import torch

from swan.dataset import FingerprintsData
from swan.modeller import GPModeller
from swan.modeller.models import GaussianProcess

from .utils_test import PATH_TEST, remove_files

torch.set_default_dtype(torch.float32)


def test_gaussian_process():
    data = FingerprintsData(PATH_TEST / "thousand.csv", properties=["gammas"])
    net = GaussianProcess()
    modeller = GPModeller(net, data, use_cuda=False, replace_state=False)
