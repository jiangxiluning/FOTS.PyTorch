import math
from typing import List, Any

import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning.core import LightningModule

from ..base import BaseModel
from .modules import shared_conv
from .modules.roi_rotate import ROIRotate
from .modules.crnn import CRNN
from ..utils.util import keys
from .loss import FOTSLoss
from ..utils.bbox import Toolbox
from ..utils.post_processor import PostProcessor
from ..utils.util import visualize

class FOTSModel(LightningModule):

    def __init__(self, config):
        super(FOTSModel, self).__init__()
        self.config = config

        self.mode = config['model']['mode']

        bbNet = torch.hub.load(self.config.backbone_weights, 'resnet50', pretrained=True, source='local')
        self.sharedConv = shared_conv.SharedConv(bbNet, config)

        nclass = len(keys) + 1
        self.recognizer = Recognizer(nclass, config)
        self.detector = Detector(config)
        self.roirotate = ROIRotate()

        self.max_transcripts_pre_batch = self.config.data_loader.max_transcripts_pre_batch

        self.loss = FOTSLoss(config=config)

        self.postprocessor = PostProcessor()

    def configure_optimizers(self):
        optimizer = getattr(optim, self.config.optimizer_type)(
            self.parameters(),
            **self.config.optimizer
        )

        if not self.config.lr_scheduler.name:
            return optimizer
        else:
            if self.config.lr_scheduler.name == 'StepLR':
                lr_scheduler = StepLR(optimizer, **self.config.lr_scheduler.args)
            else:
                raise NotImplementedError()
            return dict(optimizer=optimizer, lr_scheduler=lr_scheduler)

    def forward(self, images, boxes=None, mapping=None):
        feature_map = self.sharedConv.forward(images)

        score_map, geo_map = self.detector(feature_map)

        if self.training:
            boxes = boxes.cpu().numpy()
            rois, lengths, indices = self.roirotate(feature_map, boxes[:, :8], mapping)
            pred_mapping = mapping
            pred_boxes = boxes

            preds = self.recognizer(rois, lengths)
            preds = preds.permute(1, 0, 2) # B, T, C -> T, B, C

            data = dict(score_maps=score_map,
                        geo_maps=geo_map,
                        transcripts=(preds, lengths),
                        bboxes=pred_boxes,
                        mapping=pred_mapping,
                        indices=indices)
            return data

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

                preds = self.recognizer(rois, lengths)
                preds = preds.permute(1, 0, 2) # B, T, C -> T, B, C

                data = dict(score_maps=score_map,
                            geo_maps=geo_map,
                            transcripts=(preds, lengths),
                            bboxes=pred_boxes,
                            mapping=pred_mapping,
                            indices=indices)
                return data
            else:
                data = dict(score_maps=score_map,
                            geo_maps=geo_map,
                            transcripts=(None, None),
                            bboxes=None,
                            mapping=None,
                            indices=None)

                return data

    def training_step(self, *args, **kwargs):
        input_data = args[0]
        bboxes = input_data['bboxes']
        mapping = input_data['mapping']
        sampling_indices = torch.randperm(bboxes.size(0))[:self.max_transcripts_pre_batch]
        bboxes = bboxes[sampling_indices]
        mapping = mapping[sampling_indices]

        output = self.forward(images=input_data['images'],
                              boxes=bboxes,
                              mapping=mapping)

        sorted_indices = output['indices']
        y_true_recog = (input_data['transcripts'][0][sampling_indices][sorted_indices],
                        input_data['transcripts'][1][sampling_indices][sorted_indices])

        loss_dict = self.loss(y_true_cls=input_data['score_maps'],
                              y_pred_cls=output['score_maps'],
                              y_true_geo=input_data['geo_maps'],
                              y_pred_geo=output['geo_maps'],
                              y_true_recog=y_true_recog,
                              y_pred_recog=output['transcripts'],
                              training_mask=input_data['training_masks'])

        loss = loss_dict['reg_loss'] + loss_dict['cls_loss'] + loss_dict['recog_loss']
        self.log('loss', loss, logger=True)
        self.log('reg_loss', loss_dict['reg_loss'], logger=True, prog_bar=True)
        self.log('cls_loss', loss_dict['cls_loss'], logger=True, prog_bar=True)
        self.log('recog_loss', loss_dict['recog_loss'], logger=True, prog_bar=True)

        return loss

    def validation_step(self, *args, **kwargs):
        input_data = args[0]
        output = self.forward(images=input_data['images'])
        output['images_names'] = input_data['image_names']
        return output

    def validation_step_end(self, *args, **kwargs):
        output: dict = args[0]

        boxes_list = []
        transcripts_list = []

        pred_boxes = output['bboxes']
        pred_mapping = output['mapping']
        image_names = output['images_names']

        if pred_boxes is None:
            return dict(image_names=image_names, boxes_list=boxes_list, transcripts_list=transcripts_list)

        print(pred_boxes)
        return
        # pred_transcripts, pred_lengths = output['transcripts']
        # indices = output['indices']
        # # restore order
        # # pred_transcripts = pred_transcripts[:, ::-1, :][indices[::-1]]
        # # pred_lengths = pred_lengths[:, ::-1, :][indices[::-1]]
        #
        # for index in range(len(image_names)):
        #     selected_indices = np.argwhere(pred_mapping == index)
        #     boxes = pred_boxes[selected_indices]
        #     transcripts = pred_transcripts[:, selected_indices, ]
        #     lengths = pred_lengths[selected_indices]
        #     boxes, transcripts = self.postprocessor(boxes=boxes, transcripts=(transcripts, lengths))
        #     boxes_list.append(boxes)
        #     transcripts_list.append(transcripts)
        #     image_path = '/media/mydisk/ocr/det/icdar2015/detection/test/imgs/' + image_names[index]
        #     visualize(image_path, boxes, transcripts)
        #
        #     # visualize_box
        #
        # return dict(image_names=image_names, boxes_list=boxes_list, transcripts_list=transcripts_list)


class Recognizer(BaseModel):

    def __init__(self, nclass, config):
        super().__init__(config)
        self.crnn = CRNN(8, 32, nclass, 256)

    def forward(self, rois, lengths):
        return self.crnn(rois, lengths)


class Detector(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        self.scoreMap = nn.Conv2d(32, 1, kernel_size=1)
        self.geoMap = nn.Conv2d(32, 4, kernel_size=1)
        self.angleMap = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, *input):
        final,  = input

        score = self.scoreMap(final)
        score = torch.sigmoid(score)

        geoMap = self.geoMap(final)
        # 出来的是 normalise 到 0 -1 的值是到上下左右的距离，但是图像他都缩放到  640 * 640 了，但是 gt 里是算的绝对数值来的
        geoMap = torch.sigmoid(geoMap) * 640  # TODO: 640 is the image size

        angleMap = self.angleMap(final)
        angleMap = (torch.sigmoid(angleMap) - 0.5) * math.pi / 2

        geometry = torch.cat([geoMap, angleMap], dim=1)

        return score, geometry
