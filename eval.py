import argparse
import pathlib
import logging
from dataclasses import dataclass
from typing import List

import json
import torch
import easydict
import cv2
import numpy as np
import tqdm
import subprocess
import zipfile
import sys

from FOTS.data_loader.data_module import ICDARDataModule
from FOTS.model.model import FOTSModel

logging.basicConfig(level=logging.DEBUG, format='')

DET_CMD = '{} scripts/detection/script.py -g=scripts/detection/gt.zip -s={}'
E2E_CMD = '{} scripts/e2e/script.py -g=scripts/e2e/gt.zip -s={}'

@dataclass
class Result:
    image_path: pathlib.Path
    boxes: List[List[int]]
    transcripts: List[str]
    pred_size: tuple # (h, w)


def calculate_metric(output_dir: pathlib.Path, mode='detection'):

    results_zip = output_dir / 'results.zip'
    results_dir = output_dir / 'results'

    with zipfile.ZipFile(results_zip, mode='w') as zf:
        for i in results_dir.glob('*.txt'):
            zf.write(i, arcname=i.name)

    if mode == 'detection':
        subprocess.run(DET_CMD.format(sys.executable, results_zip.as_posix()), shell=True, text=True)
    elif mode == 'e2e':
        subprocess.run(E2E_CMD.format(sys.executable, results_zip.as_posix()), shell=True, text=True)
    else:
        raise ValueError('Mode {} is not supported.'.format(mode))



def main(args: argparse.Namespace):
    model_path = args.model
    output_dir = pathlib.Path(args.output_dir)
    output_image_dir = output_dir / 'images'
    output_results_dir = output_dir / 'results'
    output_image_dir.mkdir(exist_ok=True, parents=True)
    output_results_dir.mkdir(exist_ok=True, parents=True)

    with_gpu = True if torch.cuda.is_available() else False
    with_gpu = with_gpu & args.cuda

    if with_gpu:
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')


    config = json.load(open(args.config))
    config = easydict.EasyDict(config)
    #with_gpu = False

    config.data_loader.batch_size = 4
    config.data_loader.workers = 8
    data_module = ICDARDataModule(config)
    data_module.setup()

    model = FOTSModel.load_from_checkpoint(checkpoint_path=model_path,
                                           map_location='cpu', config=config)
    model = model.to(device)
    model.eval()

    image_dict = dict()

    for batch in tqdm.tqdm(data_module.val_dataloader()):
        output = model(images=batch['images'].to(device),
                       s=batch['score_maps'],g=batch['geo_maps'])

        mapping = output['mapping'].cpu().numpy().astype(np.int)
        image_paths = batch['image_names']
        boxes = output['bboxes'].cpu().numpy()
        #boxes = batch['bboxes'].cpu().numpy()
        transrcipts = output['transcripts']
        geo_maps = output['geo_maps']

        if transrcipts[0]:
            transrcipts = output['transcripts'][0].cpu().numpy(), output['transcripts'][1].cpu().numpy()
            assert len(boxes) == len(transrcipts[0])

        for i, image_index in enumerate(mapping):
            stem_key = pathlib.Path(image_paths[image_index]).stem
            if stem_key not in image_dict:
                result = Result(image_paths[image_index], [], [], pred_size=(geo_maps.shape[2]/config.data_loader.scale,
                                                                             geo_maps.shape[3]/config.data_loader.scale))
                image_dict[stem_key] = result
            else:
                result = image_dict[stem_key]

            box = boxes[i]
            pts = box.astype(np.int)
            result.boxes.append(pts)
            if transrcipts[0]:
                result.transcripts.append(transrcipts[0][i])
            else:
                result.transcripts.append(None)

    for k, v in image_dict.items():
        output_image_path = (output_image_dir / k).with_suffix('.jpg')
        output_result_path = output_results_dir / 'res_{}.txt'.format(k)
        f = output_result_path.open(mode='w')
        image = cv2.imread(v.image_path, cv2.IMREAD_COLOR)
        h, w, _ = image.shape

        for box, transcript in zip(v.boxes, v.transcripts):
            pts = box[:8].reshape(4, 2)
            pts[:, 0] = pts[:, 0] * w / v.pred_size[1]
            pts[:, 1] = pts[:, 1] * h / v.pred_size[0]


            image = cv2.polylines(image, [pts], True, [0, 0, 255], thickness=2)
            colors = [(255, 0, 0),
                      (0, 255, 0),
                      (0, 0, 255),
                      (0, 0, 0)]
            for i, p in enumerate(pts):
                cv2.circle(image, tuple(p), radius=5, color=colors[i])


            box_list = [str(p) for p in pts.flatten().tolist()]
            if transcript is not None:
                origin = box[0]
                font = cv2.FONT_HERSHEY_PLAIN
                img = cv2.putText(image, transcript, (origin[0], origin[1] - 10), font, 0.5, (255, 255, 255))
            else:
                line = ','.join(box_list)
            f.write(line + '\n')

        f.close()
        cv2.imwrite(output_image_path.as_posix(), image)

    calculate_metric(output_dir, mode=config.model.mode)



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
    parser.add_argument('--cuda', help='with cuda or not', dest='cuda', action='store_true')
    parser.add_argument('--gpu', default=0, type=int, help='gpu device id')
    args = parser.parse_args()
    main(args)









