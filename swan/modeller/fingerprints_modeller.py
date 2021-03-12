import numpy as np
from torch.utils.data import DataLoader
from .modeller_base import ModellerBase
from ..dataset import FingerprintsDataset


class FingerprintModeller(ModellerBase):
    """Object to create models using fingerprints."""
    def create_data_loader(self, indices: np.ndarray) -> DataLoader:
        """Create a DataLoader instance for the data."""
        dataset = FingerprintsDataset(self.data.loc[indices],
                                      self.opts.properties,
                                      self.opts.featurizer.fingerprint,
                                      self.opts.featurizer.nbits)

        return DataLoader(dataset=dataset,
                          batch_size=self.opts.torch_config.batch_size)
