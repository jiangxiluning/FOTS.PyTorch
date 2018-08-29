import torch.utils.data as torchdata
import numpy as np
from torchvision import datasets, transforms
from base import BaseDataLoader
from .dataset import SynthTextDataset


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, config):
        super(MnistDataLoader, self).__init__(config)
        self.data_dir = config['data_loader']['data_dir']
        self.data_loader = torchdata.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])), batch_size=256, shuffle=False)
        self.x = []
        self.y = []
        for data, target in self.data_loader:
            self.x += [i for i in data.numpy()]
            self.y += [i for i in target.numpy()]
        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def __next__(self):
        batch = super(MnistDataLoader, self).__next__()
        batch = [np.array(sample) for sample in batch]
        return batch

    def _pack_data(self):
        packed = list(zip(self.x, self.y))
        return packed

    def _unpack_data(self, packed):
        unpacked = list(zip(*packed))
        unpacked = [list(item) for item in unpacked]
        return unpacked

    def _update_data(self, unpacked):
        self.x, self.y = unpacked

    def _n_samples(self):
        return len(self.x)


class SynthTextDataLoader(BaseDataLoader):

    def __init__(self, config):
        super(SynthTextDataLoader, self).__init__()
        self.config = config
        dataRoot = self.config['data_dir']
        batchSize = self.config['batch_size']
        shuffle = self.config['shuffle']
        ds = SynthTextDataset(dataRoot)
        self.data_loader = torchdata.DataLoader(ds, batch_size = batchSize, shuffle = shuffle,
                                                transforms = transforms.Compose([]))

    def __next__(self):
        pass


