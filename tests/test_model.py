from FOTS.model.modules.roi_rotate import ROIRotate
from FOTS.utils.util import show_box
import numpy as np
import cv2
import torch
import pytest
import torchvision
import scipy.io as sio

@pytest.mark.skip
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
def test_rect_icdar():
    img_fn = '/Users/luning/Dev/data/icdar/icdar2015/4.4/training/ch4_training_images/img_283.jpg'
    img_gt = '/Users/luning/Dev/data/icdar/icdar2015/4.4/training/ch4_training_localization_transcription_gt/gt_img_283.txt'
    img = cv2.imread(img_fn)
    boxes = []
    transcripts = []
    with open(img_gt, mode = 'r', encoding = 'utf-8-sig') as f:
        for line in f:
            text = line.strip().split(',')
            box = [int(x) for x in text[:8]]
            transcripts.append(text[8])
            boxes.append(box)

    boxes = np.array(boxes).astype(np.float32)

    for i in range(len(boxes)):
        x1, y1, x2, y2, x3, y3, x4, y4 = boxes[i]
        print(boxes[i])
        rotated_rect = cv2.minAreaRect(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
        print(rotated_rect)
        show_box(img, boxes[i].reshape( 4, 2), transcripts[i])


def test_rect_synth800k():
    data_root = '/Users/luning/Dev/data/SynthText/SynthText/'
    img_fn = 'desert_78_83'
    gt_fn = data_root + 'gt.mat'
    targets = {}
    targets = sio.loadmat(gt_fn, targets, squeeze_me = True, struct_as_record = False,
                variable_names = ['imnames', 'wordBB', 'txt'])

    imageNames = targets['imnames']
    wordBBoxes = targets['wordBB']
    transcripts = targets['txt']

    mask = [True if img_fn in i else False for i in imageNames]
    index = np.where(mask)[0][0]
    print(index)
    img_fn = imageNames[index]

    img_fn = data_root + img_fn

    img = cv2.imread(img_fn)
    boxes = wordBBoxes[index]
    transcripts = transcripts[index]
    transcripts = [word for line in transcripts for word in line.split()]


    boxes = np.expand_dims(boxes, axis = 2) if (boxes.ndim == 2) else boxes
    _, _, numOfWords = boxes.shape
    boxes = boxes.reshape([8, numOfWords], order = 'F').T  # num_words * 8

    for i in range(len(boxes)):
        x1, y1, x2, y2, x3, y3, x4, y4 = boxes[i]
        print(boxes[i])
        rotated_rect = cv2.minAreaRect(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
        print(rotated_rect)
        box_w, box_h = rotated_rect[0][0], rotated_rect[0][1]

        if box_w < box_h:
            box_w, box_h = box_h, box_w

        width = 8 * box_w / box_h
        print(width)

        show_box(img, boxes[i].reshape( 4, 2), transcripts[i])



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

