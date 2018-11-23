import numpy as np
import torch
from ..base import BaseTrainer
from ..utils.bbox import Toolbox
from ..model.keys import keys
from ..utils.util import strLabelConverter
from ..utils.util import show_box

class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """
    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, toolbox: Toolbox, valid_data_loader=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, metrics, resume, config, train_logger)
        self.config = config
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False
        self.log_step = int(np.sqrt(self.batch_size))
        self.toolbox = toolbox
        self.labelConverter = strLabelConverter(keys)

    def _to_tensor(self, *tensors):
        t = []
        for __tensors in tensors:
            t.append(__tensors.to(self.device))
        return t

    def _eval_metrics(self, pred, gt):
        precious, recall, hmean = self.metrics[0](pred, gt)
        return np.array([precious, recall, hmean])

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(3) # precious, recall, hmean
        for batch_idx, gt in enumerate(self.data_loader):
            try:
                imagePaths, img, score_map, geo_map, training_mask, transcripts, boxes, mapping= gt
                img, score_map, geo_map, training_mask = self._to_tensor(img, score_map, geo_map, training_mask)

                # import cv2
                # for i in range(img.shape[0]):
                #     image = img[i]
                #     for tt, bb in zip(transcripts[i], boxes[i]):
                #         show_box(image.permute(1, 2, 0).detach().cpu().numpy()[:,:, ::-1].astype(np.uint8).copy(), bb, tt)

                self.optimizer.zero_grad()
                pred_score_map, pred_geo_map, pred_recog, pred_boxes, pred_mapping, indices = self.model.forward(img, boxes, mapping)

                transcripts = transcripts[indices]
                pred_boxes = pred_boxes[indices]
                pred_mapping = pred_mapping[indices]
                labels, label_lengths = self.labelConverter.encode(transcripts.tolist())
                recog = (labels, label_lengths)

                det_loss, reg_loss = self.loss(score_map, pred_score_map, geo_map, pred_geo_map, recog, pred_recog, training_mask)
                loss = det_loss + reg_loss
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                pred_transcripts = []
                if len(pred_mapping) > 0:
                    pred_mapping = pred_mapping[indices]
                    pred_boxes = pred_boxes[indices]
                    pred_fns = [imagePaths[i] for i in pred_mapping]

                    pred, lengths = pred_recog
                    _, pred = pred.max(2)
                    for i in range(lengths.numel()):
                        l = lengths[i]
                        p = pred[:l, i]
                        t = self.labelConverter.decode(p, l)
                        pred_transcripts.append(t)
                    pred_transcripts = np.array(pred_transcripts)

                gt_fns = [imagePaths[i] for i in mapping]
                total_metrics += self._eval_metrics((pred_boxes, pred_transcripts, pred_fns),
                                                        (boxes, transcripts, gt_fns))

                if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                    self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} Detection Loss: {:.6f} Recognition Loss:{:.6f}'.format(
                        epoch,
                        batch_idx * self.data_loader.batch_size,
                        len(self.data_loader) * self.data_loader.batch_size,
                        100.0 * batch_idx / len(self.data_loader),
                        loss.item(), det_loss.item(), reg_loss.item()))
            except:
                print(imagePaths)
                raise

        log = {
            'loss': total_loss / len(self.data_loader),
            'precious': total_metrics[0] / len(self.data_loader),
            'recall': total_metrics[1] / len(self.data_loader),
            'hmean': total_metrics[2] / len(self.data_loader)
        }

        if self.valid:
            val_log = self._valid_epoch()
            log = {**log, **val_log}

        return log

    def _valid_epoch(self):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_metrics = np.zeros(3)
        with torch.no_grad():
            for batch_idx, gt in enumerate(self.valid_data_loader):
                try:
                    imagePaths, img, score_map, geo_map, training_mask, transcripts, boxes, mapping = gt
                    img, score_map, geo_map, training_mask = self._to_tensor(img, score_map, geo_map, training_mask)

                    pred_score_map, pred_geo_map, pred_recog, pred_boxes, pred_mapping, indices = self.model.forward(img, boxes, mapping)
                    pred_transcripts = []
                    pred_fns = []
                    if len(pred_mapping) > 0:
                        pred_mapping = pred_mapping[indices]
                        pred_boxes = pred_boxes[indices]
                        pred_fns = [imagePaths[i] for i in pred_mapping]

                        pred, lengths = pred_recog
                        _, pred = pred.max(2)
                        for i in range(lengths.numel()):
                            l = lengths[i]
                            p = pred[:l, i]
                            t = self.labelConverter.decode(p, l)
                            pred_transcripts.append(t)
                        pred_transcripts = np.array(pred_transcripts)

                    gt_fns = [imagePaths[i] for i in mapping]
                    total_val_metrics += self._eval_metrics((pred_boxes, pred_transcripts, pred_fns),
                                                            (boxes, transcripts, gt_fns))
                except:
                    print(imagePaths)
                    raise

        return {
            'val_precious': total_val_metrics[0] / len(self.valid_data_loader),
            'val_recall': total_val_metrics[1] / len(self.valid_data_loader),
            'val_hmean': total_val_metrics[2] / len(self.valid_data_loader)
        }
