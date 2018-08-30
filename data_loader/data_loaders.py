import numpy as np
import torch
import torch.utils.data as torchdata
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
        self.length = len(ds)
        self.data_loader = torchdata.DataLoader(ds, batch_size = batchSize, shuffle = shuffle,
                                                collate_fn = self.__collate_fn)
        self.__images = []
        self.__scroeMap = []
        self.__geoMap = []
        self.__transcripts = []

    def __next__(self):
        pass

    def _n_samples(self):
        return self.length

    def __collate_fn(self, batch):
        img, score_map, geo_map, training_mask, transcript = zip(*batch)
        bs = len(score_map)
        images = []
        score_maps = []
        geo_maps = []
        training_masks = []
        transcripts = []
        for i in range(bs):
            if img[i] is not None:
                a = torch.from_numpy(img[i])
                a = a.permute(2, 0, 1)
                images.append(a)
                b = torch.from_numpy(score_map[i])
                b = b.permute(2, 0, 1)
                score_maps.append(b)
                c = torch.from_numpy(geo_map[i])
                c = c.permute(2, 0, 1)
                geo_maps.append(c)
                d = torch.from_numpy(training_mask[i])
                d = d.permute(2, 0, 1)
                training_masks.append(d)

        images = torch.stack(images, 0)
        score_maps = torch.stack(score_maps, 0)
        geo_maps = torch.stack(geo_maps, 0)
        training_masks = torch.stack(training_masks, 0)
        # TODO: need to implement the transformation for transcript as we need to compute ctc loss

        return images, score_maps, geo_maps, training_masks, transcripts


