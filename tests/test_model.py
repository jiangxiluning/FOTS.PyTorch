from FOTS.model.modules.roi_rotate import ROIRotate
from FOTS.utils.util import show_box
import numpy as np
import cv2
import torch
import pytest
import torchvision

#@pytest.mark.skip
def test_roi():
    img_fn = '/Users/luning/Dev/data/icdar/icdar2015/4.4/training/ch4_training_images/img_1.jpg'
    img_gt = '/Users/luning/Dev/data/icdar/icdar2015/4.4/training/ch4_training_localization_transcription_gt/gt_img_1.txt'
    img = cv2.imread(img_fn)
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


    to_tensor = torchvision.transforms.ToTensor()
    img = to_tensor(img)
    img = img[np.newaxis]
    img.requires_grad = True
    #img = img.permute(0, 3, 1, 2)

    rois, l, indices = roi_rotate(img, boxes, [0]*boxes.shape[0])
    #rois.backward(torch.randn((7,3,8,55), dtype=torch.double))
    rois = rois.permute(0, 2, 3, 1)
    for i in range(rois.shape[0]):
        roi = rois[i].detach().cpu().numpy()
        cv2.imshow('img', (roi*255).astype(np.uint8))
        cv2.waitKey()



@pytest.mark.skip
def test_affine():
    img_fn = '/Users/luning/Dev/data/icdar/icdar2015/4.4/training/ch4_training_images/img_100.jpg'
    img_gt = '/Users/luning/Dev/data/icdar/icdar2015/4.4/training/ch4_training_localization_transcription_gt/gt_img_100.txt'
    img = cv2.imread(img_fn)
    cv2.imshow('img', img)
    cv2.waitKey()

    r = np.radians(-45)
    matrix = np.float64([[np.cos(r), np.sin(r), 0], [-np.sin(r), np.cos(r), 0]])
    #matrix = np.float64([[1, 0, 0], [0, 1, 0]])
    img_cv = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))

    cv2.imshow('img', img_cv)
    cv2.waitKey()

    #matrix = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype = torch.float64)

    matrix = ROIRotate.param2theta(matrix, img.shape[1], img.shape[0])

    matrix *= 1e5
    matrix = torch.Tensor(matrix)
    matrix /= 1e5

    matrix = matrix[np.newaxis]

    to_tensor = torchvision.transforms.ToTensor()

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = to_tensor(img)
    img = img[np.newaxis]

    grid = torch.nn.functional.affine_grid(matrix, img.size())
    x = torch.nn.functional.grid_sample(img, grid)

    x = x[0].permute(1, 2, 0).detach().cpu().numpy()
    x = (x*255).astype(np.uint8)
    cv2.imshow('img', x)
    cv2.waitKey()

