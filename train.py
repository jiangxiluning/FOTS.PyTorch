import argparse
import json
from loguru import logger
import os
import pathlib

from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from easydict import EasyDict

from FOTS.model.model import FOTSModel
from FOTS.model.loss import *
from FOTS.data_loader.data_module import SynthTextDataModule, ICDARDataModule


def main(config, resume: bool):

    model = FOTSModel(config)
    if resume:
        assert pathlib.Path(config.pretrain).exists()
        resume_ckpt = config.pretrain
        logger.info('Resume training from: {}'.format(config.pretrain))
    else:
        if config.pretrain:
            assert pathlib.Path(config.pretrain).exists()
            logger.info('Finetune with: {}'.format(config.pretrain))
            model = model.load_from_checkpoint(config.pretrain, config=config, map_location='cpu')
            resume_ckpt = None
        else:
            resume_ckpt = None

    if config.data_loader.dataset == 'synth800k':
        data_module = SynthTextDataModule(config)
    else:
        data_module = ICDARDataModule(config)
    data_module.setup()

    root_dir = str(pathlib.Path(config.trainer.save_dir).absolute() / config.name)
    
    
    every_n_train_steps = config.trainer.get('every_n_train_steps', None)
    every_n_epochs = config.trainer.get('every_n_epochs', None)
    
    if (every_n_train_steps is None) and (every_n_epochs is None):
        logger.warning('None of checkpoint stratedge is set. Use 1 for every_n_epochs.')
        every_n_epochs = 1
        
    save_top_k = config.trainer.get('save_top_k', 1)
    
    
    checkpoint_callback = ModelCheckpoint(dirpath=root_dir + '/checkpoints', 
                                          monitor=config.trainer.monitor,
                                          mode=config.trainer.monitor_mode,
                                          save_top_k=save_top_k,
                                          every_n_train_steps=every_n_train_steps,
                                          every_n_val_epochs=every_n_epochs
                                          )
    wandb_dir = pathlib.Path(root_dir) / 'wandb'
    if not wandb_dir.exists():
        wandb_dir.mkdir(parents=True, exist_ok=True)
    wandb_logger = WandbLogger(name=config.name,
                               project='FOTS',
                               config=config,
                               save_dir=root_dir)
    if not config.cuda:
        gpus = 0
    else:
        gpus = config.gpus
        
    if config.model.mode == 'detection':
        find_unused_parameters = True
    else:
        find_unused_parameters = False

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        max_epochs=config.trainer.epochs,
        default_root_dir=root_dir,
        gpus=gpus,
        accelerator='ddp',
        benchmark=True,
        sync_batchnorm=True,
        precision=config.precision,
        log_gpu_memory=config.trainer.log_gpu_memory,
        log_every_n_steps=config.trainer.log_every_n_steps,
        overfit_batches=config.trainer.overfit_batches,
        weights_summary='full',
        terminate_on_nan=config.trainer.terminate_on_nan,
        fast_dev_run=config.trainer.fast_dev_run,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
        resume_from_checkpoint=resume_ckpt,
        plugins=DDPPlugin(find_unused_parameters=find_unused_parameters))
    trainer.fit(model=model, datamodule=data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', action='store_true',
                        help='path to latest checkpoint (default: None)')

    args = parser.parse_args()

    config = None
    if args.config is not None:
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
        # assert not os.path.exists(path), "Path {} already exists!".format(path)
    else:
        if args.resume is not None:
            logger.warning('Warning: --config overridden by --resume')
            config = torch.load(args.resume, map_location='cpu')['config']

    assert config is not None
    config = EasyDict(config)
    main(config, args.resume)
