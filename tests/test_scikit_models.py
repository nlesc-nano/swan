import numpy as np
from scipy import stats
from sklearn.gaussian_process.kernels import ConstantKernel

from swan.dataset import FingerprintsData
from swan.modeller import SKModeller

from .utils_test import PATH_TEST

DATA = FingerprintsData(PATH_TEST / "thousand.csv", properties=["Hardness (eta)"], sanitize=False)
DATA.scale_labels()


def run_test(model: str, **kwargs):
    """Run the training and validation step for the given model."""
    modeller = SKModeller(model, DATA)
    modeller.train_model()
    predicted, expected = modeller.validate_model()
    reg = stats.linregress(predicted.flatten(), expected.flatten())
    assert not np.isnan(reg.rvalue)


def test_decision_tree():
    """Check the interface to the Decisiontree class."""
    run_test("decision_tree")


def test_svm():
    """Check the interface to the support vector machine."""
    run_test("svm")


def test_gaussian_process():
    """Check the interface to the support vector machine."""
    kernel = ConstantKernel(constant_value=10)
    run_test("gaussian_process", kernel=kernel)
