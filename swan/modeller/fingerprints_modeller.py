import numpy as np
from torch.utils.data import DataLoader
from .modeller_base import ModellerBase


class FingerprintModeller(ModellerBase):
    """Object to create models using fingerprints."""
    def create_data_loader(self, indices: np.ndarray) -> DataLoader:
        return DataLoader(dataset=self.dataset,
                          batch_size=self.opts.torch_config.batch_size)
