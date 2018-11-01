from base import BaseModel
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from .modules import shared_conv
from .modules.roi_rotate import ROIRotate
import pretrainedmodels as pm


class FOTSModel(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        self.mode = config['mode']

        bbNet =  pm.__dict__['resnet50'](pretrained='imagenet') # resnet50 in paper
        self.sharedConv = shared_conv.SharedConv(bbNet)
        self.recognizer = Recognizer()
        self.detector = Detector()
        self.roirotate = ROIRotate()

    def forward(self, *input):
        '''

        :param input:
        :return:
        '''
        image, boxes = input
        score_map, geo_map, recog_map = None, None, None
        feature_map = self.sharedConv.forward(image)
        if self.mode == 'detection':
            score_map, geo_map = self.detector(feature_map)

        if self.mode == 'recognition':
            recog_map = self.recognizer(feature_map)

        if self.mode == 'united':
            score_map, geo_map = self.detector(feature_map)
            crops, padded_width = self.roirotate(feature_map, boxes)
            recog_map = self.recognizer(crops)

        return score_map, geo_map, recog_map


class Recognizer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, *input):
        return None


class Detector(nn.Module):

    def __init__(self):
        super().__init__()
        self.scoreMap = nn.Conv2d(32, 1, kernel_size = 1)
        self.geoMap = nn.Conv2d(32, 4, kernel_size = 1)
        self.angleMap = nn.Conv2d(32, 1, kernel_size = 1)

    def forward(self, *input):
        final,  = input

        score = self.scoreMap(final)
        score = torch.sigmoid(score)

        geoMap = self.geoMap(final)
        # 出来的是 normalise 到 0 -1 的值是到上下左右的距离，但是图像他都缩放到  512 * 512 了，但是 gt 里是算的绝对数值来的
        geoMap = torch.sigmoid(geoMap) * 512

        angleMap = self.angleMap(final)
        angleMap = (torch.sigmoid(angleMap) - 0.5) * math.pi / 2

        geometry = torch.cat([geoMap, angleMap], dim=1)

        return score, geometry
