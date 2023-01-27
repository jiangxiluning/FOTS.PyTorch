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
from sklearn.model_selection import train_test_split

from .transforms import Transform
from .datautils import *
from ..utils.util import str_label_converter, dump_results
from . import utils as data_utils


class SynthTextDatasetFactory:

    def __init__(self, data_root, random_state=1024, val_size=0.2, test_size=0.0):
        self.dataRoot = pathlib.Path(data_root)
        if not self.dataRoot.exists():
            raise FileNotFoundError('Dataset folder is not exist.')

        self.targetFilePath = self.dataRoot / 'gt.mat'
        if not self.targetFilePath.exists():
            raise FileExistsError('Target file is not exist.')
        self.targets = {}
        sio.loadmat(self.targetFilePath.as_posix(), self.targets, squeeze_me=True, struct_as_record=False,
                    variable_names=['imnames', 'wordBB', 'txt'])

        self.image_folder = self.dataRoot / 'imgs'

        assert val_size + test_size < 1
        assert val_size >= 0.0
        assert test_size >= 0.0
        samples = list(range(len(self.targets['imnames'])))

        if test_size:
            trainval, self.test = train_test_split(samples, test_size=test_size, random_state=random_state, shuffle=True)
        else:
            trainval = samples
            self.test = None

        if val_size:
            self.train, self.val = train_test_split(trainval, test_size=val_size, random_state=random_state, shuffle=True)
        else:
            self.train = samples
            self.val = None

    def train_ds(self, scale, size, transform, vis):
        return SynthTextDataset(self.targets, self.image_folder,
                                indices=self.train, scale=scale, size=size, transform=transform, vis=vis)

    def val_ds(self, scale, size, transform, vis):
        if not self.val:
            return None
        return SynthTextDataset(self.targets,
                                self.image_folder,
                                indices=self.val, scale=scale, size=size, transform=transform, vis=vis)

    def test_ds(self, scale, size, transform, vis):
        if not self.test:
            return None
        return SynthTextDataset(self.targets,
                                self.image_folder,
                                indices=self.test, scale=scale, size=size, transform=transform, vis=vis)



class SynthTextDataset(Dataset):

    def __init__(self, targets,
                 image_folder,
                 indices,
                 scale: float = 0.25,
                 size: int = 640,
                 transform=None,
                 vis=False):

        self.targets = targets
        self.indices = indices
        self.imageNames = targets['imnames']
        self.wordBBoxes = targets['wordBB']
        self.transcripts = targets['txt']
        self.transform: Transform = transform
        self.image_folder = image_folder

        self.vis = vis
        self.scale = scale
        self.size = size

    def visualize(self,
                  image_name: str,
                  image: np.ndarray,
                  polygons: typing.List[typing.List[float]],
                  score_map: np.ndarray,
                  training_mask: np.ndarray,
                  rois: np.ndarray,
                  transcripts: typing.List[str] = None):

        polygon_list = []
        for polygon in polygons:
            polygon_list.append(ia_polys.Polygon(
                np.array(polygon).reshape(4, 2)
            ))

        polygons_on_image = ia_polys.PolygonsOnImage(polygons=polygon_list, shape=image.shape)
        new_image = polygons_on_image.draw_on_image(image)

        colors = [(255, 0, 0),
                  (0, 255, 0),
                  (0, 0, 255),
                  (0, 0, 0)]

        for polygon in polygons:
            for i, p in enumerate(polygon.reshape(4, 2)):
                cv2.circle(new_image, tuple(p), radius=5, color=colors[i])

        cv2.imwrite(image_name + '_bbox.jpg', new_image[:, :, ::-1])

        rois_list = []

        new_image = image.copy()
        for roi, transcript in zip(rois, transcripts):
            center = (roi[0], roi[1])
            wh = (roi[3], roi[2])
            angle = -roi[4]
            box = cv2.boxPoints((center, wh, angle))
            rois_list.append(ia_polys.Polygon(
                box
            ))
            origin = box[0]
            font = cv2.FONT_HERSHEY_PLAIN
            new_image = cv2.putText(new_image, transcript, (int(origin[0]), int(origin[1] + 10)), font, 1, (0, 255, 0), 1)

        rois_on_image = ia_polys.PolygonsOnImage(polygons=rois_list, shape=image.shape)
        new_image = rois_on_image.draw_on_image(new_image)
        cv2.imwrite(image_name + '_rois.jpg', new_image[:, :, ::-1])

        score_map = ia_segmaps.SegmentationMapsOnImage(score_map.astype(dtype=np.uint8), shape=image.shape)
        new_image = score_map.draw_on_image(image.astype(dtype=np.uint8))
        cv2.imwrite(image_name + '_score.jpg', new_image[0][:, :, ::-1])


        training_mask = cv2.resize(training_mask, dsize=(image.shape[1], image.shape[0]))
        new_image = image * (1 - training_mask[:,:, None])
        cv2.imwrite(image_name + '_mask.jpg', new_image[:, :, ::-1])

    def __getitem__(self, index):
        """

        :param index:
        :return:
            imageName: path of image
            wordBBox: bounding boxes of words in the image
            transcript: corresponding transcripts of bounded words
        """
        try:
            sample = self.indices[index]
            image_path = self.imageNames[sample]
            word_b_boxes = self.wordBBoxes[sample] # 2 * 4 * num_words
            transcripts = self.transcripts[sample]
            image_path = (self.image_folder / image_path).absolute()
            im = cv2.imread(image_path.as_posix())

            word_b_boxes = np.expand_dims(word_b_boxes, axis=2) if (word_b_boxes.ndim == 2) else word_b_boxes
            _, _, num_of_words = word_b_boxes.shape
            text_polys = word_b_boxes.transpose((2, 1, 0))
            if isinstance(transcripts, str):
                transcripts = transcripts.split()
            transcripts = [word for line in transcripts for word in line.split()]

            if num_of_words == len(transcripts):
                h, w, _ = im.shape
                text_polys = check_and_validate_polys(text_polys, (h, w))

                max_tries = 10
                if self.transform:
                    while max_tries > 0:
                        transformed_im, transformed_text_polys = self.transform(im, text_polys)

                        valid_polygons = []
                        valid_transcripts = []
                        for i, polygon in enumerate(transformed_text_polys):
                            if polygon.is_fully_within_image(image=transformed_im):
                                valid_polygons.append(polygon)
                                valid_transcripts.append(transcripts[i])

                        if len(valid_polygons) > 0:
                            text_polys = valid_polygons
                            transcripts = valid_transcripts
                            im = transformed_im
                            break

                        max_tries -= 1

                    if max_tries == 0:
                        # loguru.logger.debug('Max tries has reached.')
                        return self.__getitem__(np.random.randint(0, len(self)))

                polys = np.stack([poly.coords for poly in text_polys])
                labels = np.ones(polys.shape[0], dtype=np.int32)
                score_map, geo_map, training_mask, rectangles, rois = data_utils.get_score_geo(im, polys,
                                                                                               labels,
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
                                   image_name=image_path.stem,
                                   rois=rois,
                                   transcripts=transcripts)

                transcripts = str_label_converter.encode(transcripts)

                image = normalize_iamge(image)

                return image_path.as_posix(), image, score_map, geo_map, training_mask, transcripts, rectangles, rois, labels
            else:
                return self.__getitem__(np.random.randint(0, len(self)))

        except Exception as e:
            loguru.logger.warning('Something wrong with data processing. Resample.')
            return self.__getitem__(np.random.randint(0, len(self)))

    def __len__(self):
        return len(self.indices)
