import unittest
from swan.dataset import FingerprintsData, GraphData
from .utils_test import PATH_TEST


class TestDataSet(unittest.TestCase):
    """Test case for the data set we use."""
    def setUp(self):
        self.data = PATH_TEST / "thousand.csv"

    def test_fingerprint(self):
        data = FingerprintsData(self.data, properties=["gammas"])
        data.create_data_loader()

    def test_graph(self):
        data = GraphData(self.data, properties=["gammas"])
        data.create_data_loader()
