import argparse
import json
import logging
import math
import os
import pathlib

from torch.utils.data import random_split

from data_loader import SynthTextDataLoaderFactory
from logger import Logger
from model.loss import *
from model.model import *
from model.metric import *
from trainer import Trainer

logging.basicConfig(level=logging.INFO, format='')


ICDAR2015_DATA_ROOT = pathlib.Path('/Users/luning/Dev/data/icdar2015/train')


def train_val_split(dataset, ratio: str='8:2'):
    '''

    :param ratio: train v.s. val etc. 8:2
    :param dataset:
    :return:
    '''

    try:
        train_part, val_part = ratio.split(':')
        train_part, val_part = int(train_part), int(val_part)
    except:
        print('ratio is illegal.')
        train_part, val_part = 8, 2

    train_len =  math.floor(len(dataset) * (train_part / (train_part + val_part)))
    val_len = len(dataset) - train_len

    train, val = random_split(dataset, [train_len, val_len])
    return train, val


def main(config, resume):
    train_logger = Logger()

    # Synth800K
    data_loader = SynthTextDataLoaderFactory(config)
    train = data_loader.train()
    val = data_loader.val()

    # icdar 2015
    # custom_dataset = MyDataset(DATA_ROOT / 'ch4_training_images',
    #                            DATA_ROOT / 'ch4_training_localization_transcription_gt')
    #
    # train_dataset, val_dataset = train_val_split(custom_dataset)
    # data_loader = DataLoader(train_dataset, collate_fn = collate_fn, batch_size = 32, shuffle = True)
    # valid_data_loader = DataLoader(val_dataset, collate_fn = collate_fn, batch_size = 32, shuffle = True)

    model = eval(config['arch'])(config['model'])
    model.summary()

    loss = eval(config['loss'])(config['model'])
    metrics = [eval(metric) for metric in config['metrics']]

    trainer = Trainer(model, loss, metrics,
                      resume=resume,
                      config=config,
                      data_loader=train,
                      valid_data_loader=val,
                      train_logger=train_logger)

    trainer.train()


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')

    args = parser.parse_args()

    config = None
    if args.resume is not None:
        if args.config is not None:
            logger.warning('Warning: --config overridden by --resume')
        config = torch.load(args.resume)['config']
    elif args.config is not None:
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
        assert not os.path.exists(path), "Path {} already exists!".format(path)
    assert config is not None

    main(config, args.resume)
