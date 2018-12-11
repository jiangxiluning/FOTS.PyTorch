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
import numpy as np


class FOTSModel(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        self.mode = config['model']['mode']

        bbNet =  pm.__dict__['resnet50'](pretrained='imagenet') # resnet50 in paper
        self.sharedConv = shared_conv.SharedConv(bbNet)

        nclass = len(keys) + 1
        self.recognizer = Recognizer(nclass)
        self.detector = Detector()
        self.roirotate = ROIRotate()

    def forward(self, *input):
        '''

        :param input:
        :return:
        '''
        image, boxes, mapping = input

        if image.is_cuda:
            device = image.get_device()
        else:
            device = torch.device('cpu')


        feature_map = self.sharedConv.forward(image)

        score_map, geo_map = self.detector(feature_map)

        if self.training:
            rois, lengths, indices = self.roirotate(feature_map, boxes[:, :8], mapping)
            pred_mapping = mapping
            pred_boxes = boxes
        else:
            score = score_map.permute(0, 2, 3, 1)
            geometry = geo_map.permute(0, 2, 3, 1)
            score = score.detach().cpu().numpy()
            geometry = geometry.detach().cpu().numpy()

            timer = {'net': 0, 'restore': 0, 'nms': 0}

            pred_boxes = []
            pred_mapping = []
            for i in range(score.shape[0]):
                s = score[i, :, :, 0]
                g = geometry[i, :, :, ]
                bb, _ = Toolbox.detect(score_map=s, geo_map=g, timer=timer)
                bb_size = bb.shape[0]

                if len(bb) > 0:
                    pred_mapping.append(np.array([i] * bb_size))
                    pred_boxes.append(bb)

            if len(pred_mapping) > 0:
                pred_boxes = np.concatenate(pred_boxes)
                pred_mapping = np.concatenate(pred_mapping)
                rois, lengths, indices = self.roirotate(feature_map, pred_boxes[:, :8], pred_mapping)
            else:
                return score_map, geo_map, (None, None), pred_boxes, pred_mapping, None

        lengths = torch.tensor(lengths).to(device)
        preds = self.recognizer(rois, lengths)
        preds = preds.permute(1, 0, 2) # B, T, C -> T, B, C

        return score_map, geo_map, (preds, lengths), pred_boxes, pred_mapping, indices


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
