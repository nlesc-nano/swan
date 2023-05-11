import numpy as np
from scipy import stats
from sklearn.gaussian_process.kernels import ConstantKernel

from swan.dataset import FingerprintsData
from swan.modeller import SKModeller

from tests.utils_test import PATH_TEST


def run_test(model: str, **kwargs):
    """Run the training and validation step for the given model."""
    data = FingerprintsData(PATH_TEST / "thousand.csv", properties=["Hardness (eta)"], sanitize=False)
    data.scale_labels()
    modeller = SKModeller(model, data)
    modeller.train_model()
    predicted, expected = modeller.validate_model()
    reg = stats.linregress(predicted.flatten(), expected.flatten())
    assert not np.isnan(reg.rvalue)


def run_prediction(model: str):
    """Check the prediction functionality."""
    data = FingerprintsData(PATH_TEST / "smiles.csv", sanitize=False)
    modeller = SKModeller(model, data)
    modeller.load_model("swan_skmodeller.pkl")
    modeller.data.load_scale()
    predicted = modeller.predict(data.fingerprints)
    assert not np.isnan(predicted).all()


def test_decision_tree():
    """Check the interface to the Decisiontree class."""
    model = "decision_tree"
    run_test(model)
    run_prediction(model)


def test_svm():
    """Check the interface to the support vector machine."""
    model = "svm"
    run_test(model)
    run_prediction(model)


def test_gaussian_process():
    """Check the interface to the support vector machine."""
    kernel = ConstantKernel(constant_value=10)
    model = "gaussian_process"
    run_test(model, kernel=kernel)
    run_prediction(model)
