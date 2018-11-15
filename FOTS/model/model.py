from ..base import BaseModel
import torch.nn as nn
import torch
import math
from .modules import shared_conv
from .modules.roi_rotate import ROIRotate
from .modules.crnn import CRNN
from .keys import keys
import pretrainedmodels as pm
import torch.optim as optim
from ..utils.bbox import Toolbox


class FOTSModel:

    def __init__(self, config):

        self.mode = config['mode']

        bbNet =  pm.__dict__['resnet50'](pretrained='imagenet') # resnet50 in paper
        self.sharedConv = shared_conv.SharedConv(bbNet, config)

        nclass = len(keys) + 1
        self.recognizer = Recognizer(nclass, config).double()
        self.detector = Detector(config)
        self.roirotate = ROIRotate()

    def parallelize(self):
        self.sharedConv = torch.nn.DataParallel(self.sharedConv)
        self.recognizer = torch.nn.DataParallel(self.recognizer)
        self.detector = torch.nn.DataParallel(self.detector)

    def to(self, device):
        self.sharedConv = self.sharedConv.to(device)
        self.detector = self.detector.to(device)
        self.recognizer = self.recognizer.to(device)

    def summary(self):
        self.sharedConv.summary()
        self.detector.summary()
        self.recognizer.summary()

    def optimize(self, optimizer_type, params):
        optimizer = getattr(optim, optimizer_type)(
            [{
                'params': self.sharedConv.parameters(),
                'params': self.detector.parameters(),
                'params': self.recognizer.parameters()
            }],
            **params
        )
        return optimizer

    def train(self):
        self.sharedConv.train()
        self.detector.train()
        self.recognizer.train()

    def eval(self):
        self.sharedConv.eval()
        self.detector.eval()
        self.recognizer.eval()

    def state_dict(self):
        sd = {
            '0': self.sharedConv.state_dict(),
            '1': self.detector.state_dict(),
            '2': self.recognizer.state_dict()
        }
        return sd

    def load_state_dict(self, sd):
        self.sharedConv.load_state_dict(sd['0'])
        self.detector.load_state_dict(sd['1'])
        self.recognizer.load_state_dict(sd['2'])

    @property
    def training(self):
        return self.sharedConv.training and self.detector.training and self.recognizer.training

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


class Recognizer(BaseModel):

    def __init__(self, nclass, config):
        super().__init__(config)
        self.crnn = CRNN(8, 32, nclass, 256)

    def forward(self, rois, lengths):
        return self.crnn(rois, lengths)


class Detector(BaseModel):

    def __init__(self, config):
        super().__init__(config)
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
