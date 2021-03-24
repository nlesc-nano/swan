
from swan.dataset import FingerprintsData, GraphData, DGLGraphData
from .utils_test import PATH_TEST

PATH_CSV = PATH_TEST / "thousand.csv"


def test_fingerprint_dataset():
    """Check that the fingerprint dataset is loaded correctly."""
    data = FingerprintsData(PATH_CSV, properties=["gammas"])
    data.create_data_loader()


def test_torch_geometric_dataset():
    """Check that the torch_geometric dataset is loaded correctly."""
    data = GraphData(PATH_CSV, properties=["gammas"])
    data.create_data_loader()


def test_torch_geometric_dataset_with_optimization():
    """Check that the torch_geometric dataset generates a guess for the molecular coordinates."""
    data = GraphData(PATH_CSV, properties=["gammas"], optimize_molecule=True)
    data.create_data_loader()


def test_dgl_dataset():
    """Check that the DGL dataset is loaded correctly."""
    data = DGLGraphData(PATH_CSV, properties=["gammas"])
    data.create_data_loader()
