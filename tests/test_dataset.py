
from swan.dataset import FingerprintsData, TorchGeometricGraphData, DGLGraphData
from tests.utils_test import PATH_TEST


PATH_CSV = PATH_TEST / "thousand.csv"


def test_fingerprint_dataset():
    """Check that the fingerprint dataset is loaded correctly."""
    data = FingerprintsData(PATH_CSV, properties=["Hardness (eta)"])
    data.create_data_loader()


def test_torch_geometric_dataset():
    """Check that the torch_geometric dataset is loaded correctly."""
    data = TorchGeometricGraphData(PATH_CSV, properties=["Hardness (eta)"])
    data.create_data_loader()


def test_torch_geometric_dataset_with_optimization():
    """Check that the torch_geometric dataset generates a guess for the molecular coordinate."""
    data = TorchGeometricGraphData(PATH_CSV, properties=["Hardness (eta)"], optimize_molecule=True)
    data.create_data_loader()


def test_dgl_dataset():
    """Check that the DGL dataset is loaded correctly."""
    data = DGLGraphData(PATH_CSV, properties=["Hardness (eta)"])
    data.create_data_loader()


def test_dataset_with_geometries():
    """Provide a Path with molecular geometries."""
    path_csv = PATH_TEST / "cdft_properties.csv"
    path_geometries = PATH_TEST / "cdft_geometries.json"

    data = TorchGeometricGraphData(
        path_csv, properties=["Electrophilicity index (w=omega)"], file_geometries=path_geometries, sanitize=False)
    data.create_data_loader()
