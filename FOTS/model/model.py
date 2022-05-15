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
from .modules.crnn import CRNN
from ..utils.util import keys
from .loss import FOTSLoss
from ..utils.post_processor import PostProcessor
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
        self.spatial_scale = config.data_loader.scale

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

    def forward(self, images, boxes=None, rois=None, s=None, g=None):
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
            rois = rois[: self.max_transcripts_pre_batch]

            ratios = rois[:, 4] / rois[:, 3]
            maxratio = ratios.max().item()
            pooled_width = np.ceil(self.pooled_height * maxratio).astype(int)

            roi_features = self.roirotate.apply(feature_map, rois, self.pooled_height, pooled_width, self.spatial_scale)
            # self.debug_rois(images, rois, self.pooled_height, pooled_width, 1.0)

            lengths = torch.ceil(self.pooled_height * ratios)

            pred_mapping = rois[:, 0]
            pred_boxes = boxes

            preds = self.recognizer(roi_features, lengths.cpu())
            preds = preds.permute(1, 0, 2) # B, T, C -> T, B, C

            data = dict(score_maps=score_map,
                        geo_maps=geo_map,
                        transcripts=(preds, lengths),
                        bboxes=pred_boxes,
                        mapping=pred_mapping)
            return data

        else:
            # if s is not None and g is not None:
            #     score_map = s
            #     geo_map = g

            score = score_map.detach().cpu().numpy()
            geometry = geo_map.detach().cpu().numpy()

            pred_boxes = []
            rois = []
            for i in range(score.shape[0]):
                s = score[i]
                g = geometry[i]
                bb = get_boxes(s, g, score_thresh=0.9)
                if bb is not None:
                    roi = []
                    for _, gt in enumerate(bb[:, :8].reshape(-1, 4, 2)):
                        center = (gt[0, :] + gt[1, :] + gt[2, :] + gt[3, :]) / 4
                        dw = gt[1, :] - gt[0, :]
                        dh = gt[0, :] - gt[3, :]
                        poww = pow(dw, 2)
                        powh = pow(dh, 2)

                        w = np.sqrt(poww[0] + poww[1])
                        #h = np.sqrt(powh[0] + powh[1]) + random.randint(-2, 2)
                        h = np.sqrt(powh[0] + powh[1])
                        angle_gt = (np.arctan2((gt[1, 1] - gt[0, 1]), gt[1, 0] - gt[0, 0]) + np.arctan2(
                            (gt[2, 1] - gt[3, 1]), gt[2, 0] - gt[3, 0])) / 2  # 求角度
                        angle_gt = -angle_gt / 3.1415926535 * 180
                        roi.append([i, center[0], center[1], h, w, angle_gt])

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
                            mapping=pred_mapping)
                return data

            if len(rois) > 0:
                pred_boxes = torch.as_tensor(np.concatenate(pred_boxes), dtype=feature_map.dtype, device=feature_map.device)
                rois = torch.as_tensor(np.concatenate(rois), dtype=feature_map.dtype, device=feature_map.device)
                pred_mapping = rois[:, 0]

                ratios = rois[:, 4] / rois[:, 3]
                maxratio = ratios.max().item()
                pooled_width = np.ceil(self.pooled_height * maxratio).astype(int)
                roi_features = self.roirotate.apply(feature_map, rois, self.pooled_height, pooled_width, self.spatial_scale)
                # self.debug_rois(images, rois, self.pooled_height, pooled_width, 1.0)

                lengths = torch.ceil(self.pooled_height * ratios)

                preds = self.recognizer(roi_features, lengths.cpu())
                preds = preds.permute(1, 0, 2)  # B, T, C -> T, B, C

                data = dict(score_maps=score_map,
                            geo_maps=geo_map,
                            transcripts=(preds, lengths),
                            bboxes=pred_boxes,
                            mapping=pred_mapping)
                return data
            else:
                data = dict(score_maps=score_map,
                            geo_maps=geo_map,
                            transcripts=(None, None),
                            bboxes=None,
                            mapping=None)

                return data

    def debug_rois(self,
                   images:torch.Tensor,
                   rois: torch.Tensor,
                   pooled_height: int,
                   pooled_width: int,
                   scale_factor: float=1.0):

        resized_image = nn.functional.interpolate(images, scale_factor=scale_factor, mode='bilinear')
        resized_image = resized_image.contiguous()
        rois = rois.contiguous()

        roi_features = self.roirotate.apply(resized_image, rois, pooled_height, pooled_width, scale_factor)

        resized_image = resized_image.permute(0, 2, 3, 1).contiguous().detach().cpu().numpy()
        resized_image = resized_image * np.array([0.229, 0.224, 0.225])
        resized_image = resized_image + np.array([0.485, 0.456, 0.406])
        resized_image = resized_image * 255

        rois = rois.detach().cpu().numpy()
        roi_feature = roi_features.permute(0, 2, 3, 1).contiguous().detach().cpu().numpy()
        padding_mask = roi_feature != 0

        roi_feature *= np.array([0.229, 0.224, 0.225])
        roi_feature +=  np.array([0.485, 0.456, 0.406])
        roi_feature *= 255
        roi_feature = roi_feature * padding_mask

        image_indices = rois[:, 0].astype(int)

        for i in range(image_indices.shape[0]):
            index = image_indices[i]
            image = resized_image[index]
            roi = rois[i][1:]
            feature = roi_feature[i]
            center = (roi[0], roi[1])
            wh = (roi[3], roi[2])
            angle = -roi[4]
            box = cv2.boxPoints((center, wh, angle))
            new_image = cv2.polylines(image[:, :, ::-1].astype(np.uint8), [box.astype(np.int32)], color=(0, 255, 0), isClosed=True, thickness=1)
            cv2.imwrite('./roi_test/image_{}_roi_{}.jpg'.format(index, i), new_image)
            cv2.imwrite('./roi_test/image_{}_roi_{}_cropped.jpg'.format(index, i), feature[:, :, ::-1].astype(np.uint8))

    def training_step(self, *args, **kwargs):
        input_data = args[0]
        bboxes = input_data['bboxes']
        rois = input_data['rois']
        labels = input_data['labels']

        valid_transcripts = input_data['transcripts'][0][labels]
        valid_length = input_data['transcripts'][1][labels]
        valid_rois = rois[labels]

        sampled_indices = torch.randperm(valid_transcripts.size(0))[:self.max_transcripts_pre_batch]

        valid_transcripts = valid_transcripts[sampled_indices]
        valid_length = valid_length[sampled_indices]
        valid_rois = valid_rois[sampled_indices]

        output = self.forward(images=input_data['images'],
                              boxes=bboxes,
                              rois=valid_rois)

        y_true_recog = (valid_transcripts, valid_length)

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
        angleMap = (torch.sigmoid(angleMap) - 0.5) * math.pi / 2 # -pi/2  pi/2

        geometry = torch.cat([geoMap, angleMap], dim=1)

        return score, geometry
