from FOTS.model.modules.roi_rotate import ROIRotate
from FOTS.utils.util import show_box
import numpy as np
import cv2
import torch

def test_roi():
    img_fn = '/Users/luning/Dev/data/icdar/icdar2015/4.4/training/ch4_training_images/img_1.jpg'
    img_gt = '/Users/luning/Dev/data/icdar/icdar2015/4.4/training/ch4_training_localization_transcription_gt/gt_img_1.txt'
    img = cv2.imread(img_fn)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = []
    transcripts = []
    with open(img_gt, mode = 'r', encoding = 'utf-8-sig') as f:
        for line in f:
            text = line.strip().split(',')
            box = [int(x) for x in text[:8]]
            transcripts.append(text[8])
            boxes.append(box)

    roi_rotate = ROIRotate()

    boxes = np.array(boxes).astype(np.float32) * 4
    img = img[np.newaxis]
    img = torch.tensor(img)
    img = img.permute(0, 3, 1, 2)

    rois, l, indices = roi_rotate(img, boxes, [0]*boxes.shape[0])
    for i in range(rois.shape[0]):
        roi = rois[i].astype(np.uint8)
        cv2.imshow('img', cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
        cv2.waitKey()