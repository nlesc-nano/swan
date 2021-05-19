from swan.dataset import FingerprintsData
from swan.modeller import SKModeller
from scipy import stats

from .utils_test import PATH_TEST


def test_decision_tree():
    """Check the interface to the Decisiontree class."""
    data = FingerprintsData(PATH_TEST / "thousand.csv", properties=["gammas"], sanitize=False)

    data.scale_labels()
    modeller = SKModeller(data, "decisiontree")
    modeller.train_model()
    predicted, expected = modeller.valid_model()
    reg = stats.linregress(predicted.flatten(), expected.flatten())
    assert reg.rvalue > 0
