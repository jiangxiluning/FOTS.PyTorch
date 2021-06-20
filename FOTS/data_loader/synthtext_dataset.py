#   ______                                           __                 
#  /      \                                         /  |                
# /$$$$$$  | __   __   __   ______   _______        $$ |       __    __ 
# $$ |  $$ |/  | /  | /  | /      \ /       \       $$ |      /  |  /  |
# $$ |  $$ |$$ | $$ | $$ |/$$$$$$  |$$$$$$$  |      $$ |      $$ |  $$ |
# $$ |  $$ |$$ | $$ | $$ |$$    $$ |$$ |  $$ |      $$ |      $$ |  $$ |
# $$ \__$$ |$$ \_$$ \_$$ |$$$$$$$$/ $$ |  $$ |      $$ |_____ $$ \__$$ |
# $$    $$/ $$   $$   $$/ $$       |$$ |  $$ |      $$       |$$    $$/ 
#  $$$$$$/   $$$$$/$$$$/   $$$$$$$/ $$/   $$/       $$$$$$$$/  $$$$$$/ 
#
# File: synthtext_dataset.py
# Author: Owen Lu
# Date: 2021/4/11
# Email: jiangxiluning@gmail.com
# Description:
import typing
import pathlib

import loguru
import scipy.io as sio
from torch.utils.data import Dataset
import imgaug.augmentables.polys as ia_polys
import imgaug.augmentables.segmaps as ia_segmaps

from .transforms import Transform
from .datautils import *
from ..utils.util import str_label_converter

from . import utils as data_utils


class SynthTextDataset(Dataset):

    def __init__(self, data_root,
                 scale: float = 0.25,
                 size: int = 640,
                 transform=None,
                 vis=False):
        self.dataRoot = pathlib.Path(data_root)
        if not self.dataRoot.exists():
            raise FileNotFoundError('Dataset folder is not exist.')

        self.targetFilePath = self.dataRoot / 'gt.mat'
        if not self.targetFilePath.exists():
            raise FileExistsError('Target file is not exist.')
        targets = {}
        sio.loadmat(self.targetFilePath, targets, squeeze_me=True, struct_as_record=False,
                    variable_names=['imnames', 'wordBB', 'txt'])

        self.imageNames = targets['imnames']
        self.wordBBoxes = targets['wordBB']
        self.transcripts = targets['txt']
        self.transform: Transform = transform

        self.vis = vis
        self.scale = scale
        self.size = size

    def visualize(self,
                  image_name: str,
                  image: np.ndarray,
                  polygons: typing.List[typing.List[float]],
                  score_map: np.ndarray,
                  training_mask: np.ndarray):

        polygon_list = []
        for polygon in polygons:
            polygon_list.append(ia_polys.Polygon(
                np.array(polygon).reshape(4, 2)
            ))

        polygons_on_image = ia_polys.PolygonsOnImage(polygons=polygon_list, shape=image.shape)
        new_image = polygons_on_image.draw_on_image(image)
        cv2.imwrite(image_name + '.jpg', new_image)

        score_map = ia_segmaps.SegmentationMapsOnImage(score_map.astype(dtype=np.uint8), shape=image.shape)
        new_image = score_map.draw_on_image(image.astype(dtype=np.uint8))
        cv2.imwrite(image_name + '_score.jpg', new_image[0])

        training_mask = ia_segmaps.SegmentationMapsOnImage(training_mask.astype(dtype=np.uint8), shape=image.shape)
        new_image = training_mask.draw_on_image(image.astype(dtype=np.uint8))
        cv2.imwrite(image_name + '_mask.jpg', new_image[0])

    def __getitem__(self, index):
        """

        :param index:
        :return:
            imageName: path of image
            wordBBox: bounding boxes of words in the image
            transcript: corresponding transcripts of bounded words
        """
        try:
            image_path = self.imageNames[index]
            word_b_boxes = self.wordBBoxes[index] # 2 * 4 * num_words
            transcripts = self.transcripts[index]

            im = cv2.imread((self.dataRoot / 'imgs' / image_path).as_posix())
            image_path = pathlib.Path(image_path)
            word_b_boxes = np.expand_dims(word_b_boxes, axis=2) if (word_b_boxes.ndim == 2) else word_b_boxes
            _, _, num_of_words = word_b_boxes.shape
            text_polys = word_b_boxes.transpose((2, 1, 0))
            if isinstance(transcripts, str):
                transcripts = transcripts.split()
            transcripts = [word for line in transcripts for word in line.split()]

            if num_of_words == len(transcripts):
                h, w, _ = im.shape
                text_polys = check_and_validate_polys(text_polys, (h, w))

                max_tries = 5
                if self.transform:
                    while True and (max_tries != 0):
                        transformed_im, transformed_text_polys = self.transform(im, text_polys)
                        valid_text_polys = [polygon for polygon in transformed_text_polys if polygon.is_fully_within_image(image=im)]
                        if len(valid_text_polys) > 0:
                            text_polys = valid_text_polys
                            transcripts = [transcripts[i] for i, polygon in enumerate(text_polys) if polygon.is_fully_within_image(image=im)]
                            im = transformed_im
                            break
                        max_tries -= 1

                    if max_tries == 0:
                        return self.__getitem__(np.random.randint(0, len(self)))

                polys = np.stack([poly.coords for poly in text_polys])
                score_map, geo_map, training_mask, rectangles, rois = data_utils.get_score_geo(im, polys,
                                                                                               np.ones(polys.shape[0]),
                                                                                               self.scale, self.size)

                # predict 出来的feature map 是 128 * 128， 所以 gt 需要取 /4 步长
                image = im[:, :, ::-1].astype(np.float32)  # bgr -> rgb
                assert len(transcripts) == len(rectangles)

                if len(transcripts) == 0:
                    raise RuntimeError('No text found.')

                if self.vis:
                    self.visualize(image=image,
                                   polygons=rectangles,
                                   score_map=score_map,
                                   training_mask=training_mask,
                                   image_name=image_path.stem)

                transcripts = str_label_converter.encode(transcripts)

                image = normalize_iamge(image)

                return image_path.as_posix(), image, score_map, geo_map, training_mask, transcripts, rectangles, rois
            else:
                return self.__getitem__(np.random.randint(0, len(self)))

        except Exception as e:
            # loguru.logger.warning('Something wrong with data processing. Resample.')
            # return self.__getitem__(np.random.randint(0, len(self)))
            raise e

    def __len__(self):
        return len(self.imageNames)
