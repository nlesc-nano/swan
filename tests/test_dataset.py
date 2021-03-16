import unittest

from swan.dataset import FingerprintsDataset, MolGraphDataset
from swan.dataset import FingerprintsData, MolGraphData
from .utils_test import PATH_TEST


class TestDataSet(unittest.TestCase):
    """Test case for the data set we use."""
    def setUp(self):
        self.data = PATH_TEST / "thousand.csv"

    def test_fingerprint(self):
        dataset = FingerprintsDataset(self.data, properties=["gammas"])
        print(dataset.data)

    def test_graph(self):
        dataset = MolGraphDataset(self.data, properties=["gammas"])
        print(dataset.data)

    def test_fingerprint_loader(self):
        data = FingerprintsData(self.data, properties=["gammas"])
        data.create_data_loader()

    def test_grapj_loader(self):
        data = MolGraphData(self.data, properties=["gammas"])
        data.create_data_loader()
