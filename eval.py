import argparse

import json
import torch
import logging
import pathlib
import traceback

from pytorch_lightning import Trainer

from FOTS.model.model import FOTSModel
from FOTS.utils.bbox import Toolbox

import easydict
from FOTS.data_loader.data_module import ICDARDataModule


logging.basicConfig(level=logging.DEBUG, format='')


def load_model(model_path, with_gpu):
    model = FOTSModel(config)

    if config.data_loader.dataset == 'synth800k':
        data_module = SynthTextDataModule(config)
    else:
        data_module = ICDARDataModule(config)

    root_dir = str(pathlib.Path(config.trainer.save_dir).absolute() / config.name)
    checkpoint_callback = ModelCheckpoint(dirpath=root_dir + '/checkpoints', period=1)
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
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch)
    trainer.fit(model=model, datamodule=data_module)


def main(args:argparse.Namespace):
    model_path = args.model
    input_dir = args.input_dir
    output_dir = args.output_dir
    with_image = True if output_dir else False
    with_gpu = True if torch.cuda.is_available() else False

    config = json.load(open(args.config))
    #with_gpu = False


    config = easydict.EasyDict(config)
    model = FOTSModel.load_from_checkpoint(checkpoint_path=model_path,
                                           map_location='cpu', config=config)
    model = model.to('cuda:0')
    model.eval()
    for image_fn in input_dir.glob('*.jpg'):
        try:
            with torch.no_grad():
                ploy, im = Toolbox.predict(image_fn, model, with_image, output_dir, with_gpu=True)
                print(len(ploy))
        except Exception as e:
            traceback.print_exc()


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='Model eval')
    parser.add_argument('-m', '--model', default=None, type=pathlib.Path, required=True,
                        help='path to model')
    parser.add_argument('-o', '--output_dir', default=None, type=pathlib.Path,
                        help='output dir for drawn images')
    parser.add_argument('-i', '--input_dir', default=None, type=pathlib.Path, required=False,
                        help='dir for input images')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    args = parser.parse_args()
    main(args)









