import torch.utils.data as torchdata

from ..base import BaseDataLoader
from .dataset import SynthTextDataset
from .datautils import collate_fn


class SynthTextDataLoaderFactory(BaseDataLoader):

    def __init__(self, config):
        super(SynthTextDataLoaderFactory, self).__init__(config)
        dataRoot = self.config['data_loader']['data_dir']
        self.workers = self.config['data_loader']['workers']
        ds = SynthTextDataset(dataRoot)

        self.__trainDataset, self.__valDataset = self.__train_val_split(ds)

    def train(self):
        trainLoader = torchdata.DataLoader(self.__trainDataset, num_workers = self.num_workers, batch_size = self.batch_size,
                                           shuffle = self.shuffle, collate_fn = collate_fn)
        return trainLoader

    def val(self):
        shuffle = self.config['validation']['shuffle']
        valLoader = torchdata.DataLoader(self.__valDataset, num_workers = self.num_workers, batch_size = self.batch_size,
                                         shuffle = shuffle, collate_fn = collate_fn)
        return valLoader

    def __train_val_split(self, ds):
        '''

        :param ds: dataset
        :return:
        '''
        split = self.config['validation']['validation_split']

        try:
            split = float(split)
        except:
            raise RuntimeError('Train and val splitting ratio is invalid.')

        val_len = int(split * len(ds))
        train_len = len(ds) - val_len
        train, val = torchdata.random_split(ds, [train_len, val_len])
        return train, val

    def split_validation(self):
        raise NotImplementedError


class OCRDataLoaderFactory(BaseDataLoader):

    def __init__(self, config, ds):
        super(OCRDataLoaderFactory, self).__init__(config)
        self.workers = self.config['data_loader']['workers']
        self.__trainDataset, self.__valDataset = self.__train_val_split(ds)

    def train(self):
        trainLoader = torchdata.DataLoader(self.__trainDataset, num_workers = self.num_workers, batch_size = self.batch_size,
                                           shuffle = self.shuffle, collate_fn = collate_fn)
        return trainLoader

    def val(self):
        shuffle = self.config['validation']['shuffle']
        valLoader = torchdata.DataLoader(self.__valDataset, num_workers = self.num_workers, batch_size = self.batch_size,
                                         shuffle = shuffle, collate_fn = collate_fn)
        return valLoader

    def __train_val_split(self, ds):
        '''

        :param ds: dataset
        :return:
        '''
        split = self.config['validation']['validation_split']

        try:
            split = float(split)
        except:
            raise RuntimeError('Train and val splitting ratio is invalid.')

        val_len = int(split * len(ds))
        train_len = len(ds) - val_len
        train, val = torchdata.random_split(ds, [train_len, val_len])
        return train, val

    def split_validation(self):
        raise NotImplementedError