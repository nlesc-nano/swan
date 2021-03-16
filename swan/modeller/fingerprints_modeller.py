import numpy as np
from torch.utils.data import DataLoader, random_split
from .modeller_base import ModellerBase


class FingerprintModeller(ModellerBase):
    """Object to create models using fingerprints."""
    def create_data_loader(self,
                           frac=[0.8, 0.2],
                           batch_size: int = 64) -> DataLoader:

        ntotal = self.dataset.__len__()
        ntrain = int(frac[0] * ntotal)
        nvalid = ntotal - ntrain

        self.train_dataset, self.valid_dataset = random_split(
            self.dataset, [ntrain, nvalid])

        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=batch_size)

        self.valid_loader = DataLoader(dataset=self.valid_dataset,
                                       batch_size=batch_size)
