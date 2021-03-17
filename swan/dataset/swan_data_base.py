from torch.utils.data import random_split


class SwanDataBase:
    """Base class for the data loaders."""
    def __init__(self):

        self.dataframe = None
        self.dataset = None
        self.train_dataset = None
        self.valid_dataset = None

        self.train_loader = None
        self.valid_loader = None
        self.data_loader_fun = None

    def create_data_loader(self,
                           frac=[0.8, 0.2],
                           batch_size: int = 64) -> None:
        """create the train/valid data loaders

        Parameters
        ----------
        frac : list, optional
            fraction to divide the dataset, by default [0.8, 0.2]
        batch_size : int, optional
            batchsize, by default 64
        """

        ntotal = self.dataset.__len__()
        ntrain = int(frac[0] * ntotal)
        nvalid = ntotal - ntrain

        self.train_dataset, self.valid_dataset = random_split(
            self.dataset, [ntrain, nvalid])

        self.train_loader = self.data_loader_fun(dataset=self.train_dataset,
                                                 batch_size=batch_size)

        self.valid_loader = self.data_loader_fun(dataset=self.valid_dataset,
                                                 batch_size=batch_size)

    @staticmethod
    def get_item(batch_data):
        """get the data/ground truth of a minibatch

        Parameters
        ----------
        batch_data : [type]
            data of the mini batch

        Raises
        ------
        NotImplementedError
            Not implemented in the base class
        """
        raise NotImplementedError("get item not implemented")
