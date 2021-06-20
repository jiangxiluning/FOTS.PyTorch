import torch
import torch.nn as nn
from torch.nn import CTCLoss
import einops

from .ghm import GHMC



import torch
import torch.nn as nn


def get_dice_loss(gt_score, pred_score):
    inter = torch.sum(gt_score * pred_score)
    union = torch.sum(gt_score) + torch.sum(pred_score) + 1e-5
    return 1. - (2 * inter / union)


def get_geo_loss(gt_geo, pred_geo):
    d1_gt, d2_gt, d3_gt, d4_gt, angle_gt = torch.split(gt_geo, 1, 1)
    d1_pred, d2_pred, d3_pred, d4_pred, angle_pred = torch.split(pred_geo, 1, 1)
    area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
    area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)
    w_union = torch.min(d3_gt, d3_pred) + torch.min(d4_gt, d4_pred)
    h_union = torch.min(d1_gt, d1_pred) + torch.min(d2_gt, d2_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    iou_loss_map = -torch.log((area_intersect + 1.0)/(area_union + 1.0))
    angle_loss_map = 1 - torch.cos(angle_pred - angle_gt)
    return iou_loss_map, angle_loss_map


class DetectionLoss(nn.Module):
    def __init__(self, weight_angle=10):
        super(DetectionLoss, self).__init__()
        self.weight_angle = weight_angle
        self.ghmc = GHMC()

    def forward(self, gt_score, pred_score, gt_geo, pred_geo, ignored_map):
        if torch.sum(gt_score) < 1:
            return torch.sum(pred_score + pred_geo) * 0

        # classify_loss = self.ghmc(einops.rearrange(pred_score, 'b c h w -> (b h w) c'),
        #                           einops.rearrange(gt_score, 'b c h w -> (b h w) c'),
        #                           einops.rearrange(ignored_map, 'b c h w -> (b h w) c'))

        classify_loss = get_dice_loss(gt_score, pred_score*(1-ignored_map.byte()))
        iou_loss_map, angle_loss_map = get_geo_loss(gt_geo, pred_geo)

        angle_loss = torch.sum(angle_loss_map*gt_score) / torch.sum(gt_score)
        iou_loss = torch.sum(iou_loss_map*gt_score) / torch.sum(gt_score)
        geo_loss = self.weight_angle * angle_loss + iou_loss
        # print('classify loss is {:.8f}, angle loss is {:.8f}, iou loss is {:.8f}'.format(classify_loss, angle_loss, iou_loss))
        return geo_loss, classify_loss

# class DetectionLoss(nn.Module):
#     def __init__(self):
#         super(DetectionLoss, self).__init__()
#         self.ghmc = GHMC()
#
#
#     def forward(self, y_true_cls, y_pred_cls,
#                 y_true_geo, y_pred_geo,
#                 training_mask):
#
#         classification_loss = self.__dice_coefficient(y_true_cls, y_pred_cls, training_mask)
#
#         # classification_loss = self.ghmc(einops.rearrange(y_pred_cls, 'b c h w -> (b h w) c'),
#         #                                 einops.rearrange(y_true_cls, 'b c h w -> (b h w) c'),
#         #                                 einops.rearrange(training_mask, 'b c h w -> (b h w) c'))
#
#
#         # d1 -> top, d2->right, d3->bottom, d4->left
#         #     d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
#         d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = torch.split(y_true_geo, 1, 1)
#         #     d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
#         d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = torch.split(y_pred_geo, 1, 1)
#         area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
#         area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
#         w_union = torch.min(d2_gt, d2_pred) + torch.min(d4_gt, d4_pred)
#         h_union = torch.min(d1_gt, d1_pred) + torch.min(d3_gt, d3_pred)
#         area_intersect = w_union * h_union
#         area_union = area_gt + area_pred - area_intersect
#         L_AABB = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
#         L_theta = 1 - torch.cos(theta_pred - theta_gt)
#         L_g = L_AABB + 10 * L_theta
#
#         return torch.mean(L_g * y_true_cls * training_mask), classification_loss
#
#     def __dice_coefficient(self, y_true_cls, y_pred_cls, training_mask):
#         '''
#         dice loss
#         :param y_true_cls:
#         :param y_pred_cls:
#         :param training_mask:
#         :return:
#         '''
#         eps = 1e-5
#         intersection = torch.sum(y_true_cls * y_pred_cls * training_mask)
#         union = torch.sum(y_true_cls * training_mask) + torch.sum(y_pred_cls * training_mask) + eps
#         loss = 1. - (2 * intersection / union)
#
#         return loss
#
#     def __cross_entroy(self, y_true_cls, y_pred_cls, training_mask):
#         #import ipdb; ipdb.set_trace()
#         return torch.nn.functional.binary_cross_entropy(y_pred_cls*training_mask, (y_true_cls*training_mask))


class RecognitionLoss(nn.Module):

    def __init__(self):
        super(RecognitionLoss, self).__init__()
        self.ctc_loss = CTCLoss(zero_infinity=True) # pred, pred_len, labels, labels_len

    def forward(self, *input):
        gt, pred = input[0], input[1]
        loss = self.ctc_loss( torch.log_softmax(pred[0], dim=-1), gt[0].cpu(), pred[1].int(), gt[1].cpu())
        if torch.isnan(loss):
            raise RuntimeError()
        return loss


class FOTSLoss(nn.Module):

    def __init__(self, config):
        super(FOTSLoss, self).__init__()
        self.mode = config['model']['mode']
        self.detectionLoss = DetectionLoss()
        self.recogitionLoss = RecognitionLoss()

    def forward(self, y_true_cls, y_pred_cls,
                y_true_geo, y_pred_geo,
                y_true_recog, y_pred_recog,
                training_mask):

        if self.mode == 'recognition':
            recognition_loss = self.recogitionLoss(y_true_recog, y_pred_recog)
            reg_loss = torch.tensor([0.], device=recognition_loss.device)
            cls_loss = torch.tensor([0.], device=recognition_loss.device)
        elif self.mode == 'detection':
            reg_loss, cls_loss = self.detectionLoss(y_true_cls, y_pred_cls,
                                                y_true_geo, y_pred_geo, training_mask)
            recognition_loss = torch.tensor([0.], device=reg_loss.device)
        elif self.mode == 'united':
            reg_loss, cls_loss = self.detectionLoss(y_true_cls, y_pred_cls, y_true_geo, y_pred_geo, training_mask)
            if y_true_recog:
                recognition_loss = self.recogitionLoss(y_true_recog, y_pred_recog)
                if recognition_loss <0 :
                    import ipdb; ipdb.set_trace()

        #recognition_loss = recognition_loss.to(detection_loss.device)
        return dict(reg_loss=reg_loss, cls_loss=cls_loss, recog_loss=recognition_loss)
