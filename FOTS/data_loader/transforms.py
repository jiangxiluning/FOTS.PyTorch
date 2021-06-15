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
# File: transforms.py
# Author: Owen Lu
# Date: 2021/3/25
# Email: jiangxiluning@gmail.com
# Description:
import typing

import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage


class Transform:
    def __init__(self,
                 long_sizes: typing.Iterable = (640, 800, 1333, 2560),
                 angles: typing.Iterable = (-10, 10),
                 height_ratios: typing.Iterable = (0.8, 1.2),
                 cropped_size: tuple = (640, 640),
                 blur_prob: float = 0.3,
                 color_jitter_prob: float = 0.3,
                 output_size: tuple = (640, 640),
                 is_training=False):

        self.long_sizes = long_sizes
        self.angles = angles
        self.height_ratios = height_ratios
        self.cropped_size = cropped_size
        self.blur_prob = blur_prob
        self.color_jitter_prob = color_jitter_prob
        self.output_size = output_size
        self.is_training = is_training

    def __call__(self, *args, **kwargs) -> typing.Tuple[np.ndarray, typing.List[Polygon]]:

        if self.is_training:
            resize = iaa.Resize(size=dict(longer_side=self.long_sizes,
                                          width='keep-aspect-ratio'))
            rotate = iaa.Rotate(rotate=self.angles, fit_output=True)
            resize_height = iaa.Resize(size=dict(height=self.height_ratios,
                                                 width='keep'))
            crop = iaa.CropToFixedSize(width=self.cropped_size[0], height=self.cropped_size[1])
            fix_resize = iaa.Resize(size=self.output_size)
            blur = iaa.GaussianBlur()
            blur = iaa.Sometimes(p=self.blur_prob,
                                 then_list=blur)
            color_jitter = iaa.MultiplyBrightness()
            color_jitter = iaa.Sometimes(p=self.color_jitter_prob,
                                         then_list=color_jitter)

            augs = [resize,
                    rotate,
                    resize_height,
                    crop,
                    fix_resize,
                    blur,
                    color_jitter]

            ia = iaa.Sequential(augs)
        else:
            fix_resize = iaa.Resize(size=self.output_size)
            ia = iaa.Sequential([fix_resize])

        image = args[0]
        polygons = args[1]

        polygon_list = []
        for i in range(polygons.shape[0]):
            polygon_list.append(Polygon(polygons[i].tolist()))

        polygons_on_image = PolygonsOnImage(polygon_list, shape=image.shape)

        image_aug, polygons_aug = ia(image=image, polygons=polygons_on_image)

        return image_aug, polygons_aug.polygons




