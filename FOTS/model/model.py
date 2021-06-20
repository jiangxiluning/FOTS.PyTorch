import math
from typing import List, Any

import cv2
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
from ..rroi_align.functions.rroi_align import RRoiAlignFunction
from ..utils.detect import get_boxes

class FOTSModel(LightningModule):

    def __init__(self, config):
        super(FOTSModel, self).__init__()
        self.config = config

        self.mode = config['model']['mode']

        bbNet = torch.hub.load(self.config.backbone_weights, 'resnet50', pretrained=True, source='local')
        self.sharedConv = shared_conv.SharedConv(bbNet, config)

        # 'blank' and '<other>'
        nclass = len(keys) + 2
        self.recognizer = Recognizer(nclass, config)
        self.detector = Detector(config)
        #self.roirotate = ROIRotate()
        self.roirotate = RRoiAlignFunction()
        self.pooled_height = 8
        self.spatial_scale = 1.0

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

    def forward(self, images, boxes=None, rois=None):
        feature_map = self.sharedConv.forward(images)

        score_map, geo_map = self.detector(feature_map)

        if self.training:
            if self.mode == 'detection':
                data = dict(score_maps=score_map,
                            geo_maps=geo_map,
                            transcripts=(None, None),
                            bboxes=boxes,
                            mapping=None,
                            indices=None)
                return data

            # there are some hard samples, ###

            # for memory concern, we fix the number of transcript to train
            sampled_indices = torch.randperm(rois.size(0))[:self.max_transcripts_pre_batch]
            rois = rois[sampled_indices]

            ratios = rois[:, 4] / rois[:, 3]
            maxratio = ratios.max().item()
            pooled_width = np.ceil(self.pooled_height * maxratio).astype(int)

            roi_features = self.roirotate.apply(feature_map, rois, self.pooled_height, pooled_width, self.spatial_scale)
            lengths = torch.ceil(self.pooled_height * ratios)

            pred_mapping = rois[:, 0]
            pred_boxes = boxes

            preds = self.recognizer(roi_features, lengths.cpu())
            preds = preds.permute(1, 0, 2) # B, T, C -> T, B, C

            data = dict(score_maps=score_map,
                        geo_maps=geo_map,
                        transcripts=(preds, lengths),
                        bboxes=pred_boxes,
                        mapping=pred_mapping,
                        indices=sampled_indices)
            return data

        else:
            score = score_map.cpu().numpy()
            geometry = geo_map.cpu().numpy()

            pred_boxes = []
            rois = []
            for i in range(score.shape[0]):
                s = score[i]
                g = geometry[i]
                bb = get_boxes(s, g, score_thresh=0.9)
                if bb is not None:
                    roi = []
                    for _, gt in enumerate(bb[:, :8].reshape(-1, 4, 2)):
                        rr = cv2.minAreaRect(gt)
                        center = rr[0]
                        (w, h) = rr[1]
                        min_rect_angle = rr[2]

                        if h > w:
                            min_rect_angle = min_rect_angle + 180
                            roi.append([i, center[0], center[1], w, h, -min_rect_angle])
                        else:
                            roi.append([i, center[0], center[1], h, w, -min_rect_angle])

                    pred_boxes.append(bb)
                    rois.append(np.stack(roi))

            if self.mode == 'detection':

                if len(pred_boxes) > 0:
                    pred_boxes = torch.as_tensor(np.concatenate(pred_boxes), dtype=feature_map.dtype, device=feature_map.device)
                    rois = torch.as_tensor(np.concatenate(rois), dtype=feature_map.dtype, device=feature_map.device)
                    pred_mapping = rois[:, 0]
                else:
                    pred_boxes = None
                    pred_mapping = None

                data = dict(score_maps=score_map,
                            geo_maps=geo_map,
                            transcripts=(None, None),
                            bboxes=pred_boxes,
                            mapping=pred_mapping,
                            indices=None)
                return data

            if len(rois) > 0:
                pred_boxes = torch.as_tensor(np.concatenate(pred_boxes), dtype=feature_map.dtype, device=feature_map.device)
                rois = torch.as_tensor(np.concatenate(rois), dtype=feature_map.dtype, device=feature_map.device)
                pred_mapping = rois[:, 0]

                ratios = rois[:, 4] / rois[:, 3]
                maxratio = ratios.max().item()
                pooled_width = np.ceil(self.pooled_height * maxratio).astype(int)
                roi_features = self.roirotate.apply(feature_map, rois, self.pooled_height, pooled_width, self.spatial_scale)

                lengths = torch.ceil(self.pooled_height * ratios)

                preds = self.recognizer(roi_features, lengths.cpu())
                preds = preds.permute(1, 0, 2)  # B, T, C -> T, B, C

                data = dict(score_maps=score_map,
                            geo_maps=geo_map,
                            transcripts=(preds, lengths),
                            bboxes=pred_boxes,
                            mapping=pred_mapping,
                            indices=None)
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
        rois = input_data['rois']

        output = self.forward(images=input_data['images'],
                              boxes=bboxes,
                              rois=rois)

        sampled_indices = output['indices']
        y_true_recog = (input_data['transcripts'][0][sampled_indices],
                        input_data['transcripts'][1][sampled_indices])

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

        self.log('pred_boxes', pred_boxes.shape[0] if pred_boxes is not None else 0, prog_bar=True)

        if pred_boxes is None:
            return dict(image_names=image_names, boxes_list=boxes_list, transcripts_list=transcripts_list)

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
        self.size = config.data_loader.size

    def forward(self, *input):
        final,  = input

        score = self.scoreMap(final)
        score = torch.sigmoid(score)

        geoMap = self.geoMap(final)
        # 出来的是 normalise 到 0 -1 的值是到上下左右的距离，但是图像他都缩放到  640 * 640 了，但是 gt 里是算的绝对数值来的
        geoMap = torch.sigmoid(geoMap) * self.size  # TODO: 640 is the image size

        angleMap = self.angleMap(final)
        angleMap = (torch.sigmoid(angleMap) - 0.5) * math.pi

        geometry = torch.cat([geoMap, angleMap], dim=1)

        return score, geometry
