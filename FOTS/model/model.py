from ..base import BaseModel
import torch.nn as nn
import torch
import math
from .modules import shared_conv
from .modules.roi_rotate import ROIRotate
from .modules.crnn import CRNN
from .keys import keys
import pretrainedmodels as pm
from ..utils.bbox import Toolbox


class FOTSModel(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        self.mode = config['mode']

        bbNet =  pm.__dict__['resnet50'](pretrained='imagenet') # resnet50 in paper
        self.sharedConv = shared_conv.SharedConv(bbNet)

        nclass = len(keys) + 1
        self.recognizer = Recognizer(nclass).double()
        self.detector = Detector()
        self.roirotate = ROIRotate()

    def forward(self, *input):
        '''

        :param input:
        :return:
        '''
        image, boxes = input
        if image.is_cuda:
            device = image.get_device()
        else:
            device = torch.device('cpu')

        score_map, geo_map, preds, actual_length, pred_boxes, indices = None, None, None, None, None, None
        feature_map = self.sharedConv.forward(image)
        if self.mode == 'detection':
            score_map, geo_map = self.detector(feature_map)

        if self.mode == 'recognition':
            preds, actual_length, indices = self.recognizer(feature_map)

        if self.mode == 'united':
            score_map, geo_map = self.detector(feature_map)


            score = score_map.permute(0, 2, 3, 1)
            geometry = geo_map.permute(0, 2, 3, 1)
            score = score.detach().cpu().numpy()
            geometry = geometry.detach().cpu().numpy()

            timer = {'net': 0, 'restore': 0, 'nms': 0}
            pred_boxes, timer = Toolbox.detect(score_map=score, geo_map=geometry, timer=timer)

            if self.training:
                rois, lengths, indices = self.roirotate(feature_map, boxes)
            else:
                if not pred_boxes:
                    return score_map, geo_map, (preds, actual_length), pred_boxes, indices

                rois, lengths, indices = self.roirotate(feature_map, pred_boxes)

            rois = torch.tensor(rois).to(device)
            rois = rois.permute(0, 3, 1, 2)
            lengths = torch.tensor(lengths).to(device)
            preds, actual_length = self.recognizer(rois, lengths)

        return score_map, geo_map, (preds, actual_length), pred_boxes, indices


class Recognizer(nn.Module):

    def __init__(self, nclass):
        super().__init__()
        self.crnn = CRNN(8, 32, nclass, 256)

    def forward(self, rois, lengths):
        return self.crnn(rois, lengths)


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
