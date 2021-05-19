from scipy import stats
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

from swan.dataset import FingerprintsData
from swan.modeller import SKModeller

from .utils_test import PATH_TEST

DATA = FingerprintsData(PATH_TEST / "thousand.csv", properties=["gammas"], sanitize=False)
DATA.scale_labels()


def run_test(model: str, **kwargs):
    """Run the training and validation step for the given model."""
    modeller = SKModeller(DATA, model)
    modeller.train_model()
    predicted, expected = modeller.valid_model()
    reg = stats.linregress(predicted.flatten(), expected.flatten())
    assert reg.rvalue > 0


def test_decision_tree():
    """Check the interface to the Decisiontree class."""
    run_test("decisiontree")


def test_svm():
    """Check the interface to the support vector machine."""
    run_test("svm")


def test_gaussian_process():
    """Check the interface to the support vector machine."""
    kernel = DotProduct() + WhiteKernel()
    run_test("gaussianprocess", kernel=kernel)
