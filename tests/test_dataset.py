from pathlib import Path
import unittest
import numpy as np

from pytest_mock import MockFixture

from swan.dataset import FingerprintsDataset, MolGraphDataset

from .utils_test import PATH_TEST


class TestDataSet(unittest.TestCase):
    """Test case for the data set we use."""
    def setUp(self):
        self.data = PATH_TEST / "thousand.csv"

    def test_fingerprint(self):
        dataset = FingerprintsDataset(self.data, properties=["gammas"])

    def test_graph(self):
        dataset = MolGraphDataset(self.data, properties=["gammas"])


if __name__ == "__main__":
    unittest.main()