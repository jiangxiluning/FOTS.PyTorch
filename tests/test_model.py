import math
import random

import numpy as np
import cv2
import torch
import pytest
import torchvision
import scipy.io as sio
from imgaug.augmentables.polys import PolygonsOnImage, Polygon
from torch.autograd import gradcheck
from torch.utils.data import DataLoader

from FOTS.data_loader.transforms import Transform
from FOTS.model.modules.roi_rotate import ROIRotate
from FOTS.utils.util import show_box
from FOTS.data_loader.icdar_dataset import ICDARDataset
from FOTS.data_loader.synthtext_dataset import SynthTextDataset
from FOTS.data_loader.datautils import collate_fn
from FOTS.rroi_align.modules.rroi_align import _RRoiAlign



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
    data_root = '/home/luning/dev/data/SynthText800k/detection/'
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

    img_fn = data_root + '/imgs/' + img_fn

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

def test_transform():
    data_root = '/home/luning/dev/data/SynthText800k/detection/'
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

    img_fn = data_root + '/imgs/' + img_fn

    img = cv2.imread(img_fn)
    boxes = wordBBoxes[index]
    transcripts = transcripts[index]
    transcripts = [word for line in transcripts for word in line.split()]


    boxes = np.expand_dims(boxes, axis = 2) if (boxes.ndim == 2) else boxes
    _, _, numOfWords = boxes.shape
    # boxes = boxes.reshape([8, numOfWords]).T  # num_words * 8
    # boxes = boxes.reshape([numOfWords, 4, 2])
    boxes = boxes.transpose((2, 1, 0))

    polys = []
    for i in range(numOfWords):
        # box = boxes[:, :, i].T
        box = boxes[i]
        polys.append(Polygon(box.tolist()))
    polys_on_image = PolygonsOnImage(polygons=polys, shape=img.shape)

    image_before_aug = polys_on_image.draw_on_image(img)
    cv2.imwrite('before_aug.jpg', image_before_aug)

    transform = Transform()

    image_aug, polygons_aug = transform(img, boxes)

    polygons_on_image = PolygonsOnImage(polygons=polygons_aug, shape=image_aug.shape)
    image_aug = polygons_on_image.draw_on_image(image_aug)

    cv2.imwrite('aug.jpg', image_aug)


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


def test_synthtext_dataset():

    transform = Transform(is_training=True)
    ds = SynthTextDataset(data_root='/home/luning/dev/data/SynthText800k/detection/',
                          transform=transform,
                          vis=True)

    print(ds[0])


def test_icdar_dataset():

    transform = Transform(is_training=True)
    ds = ICDARDataset(data_root='/data/ocr/det/icdar2015/detection/train',
                      transform=transform,
                      vis=True)
    dataloader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
    for batch in dataloader:
        print(batch)


def test_rroi():
    path = 'FOTS/rroi_align/data/timg.jpeg'
    # path = './data/grad.jpg'
    im_data = cv2.imread(path)
    img = im_data.copy()
    im_data = torch.from_numpy(im_data).unsqueeze(0).permute(0,3,1,2)
    im_data = im_data
    im_data = im_data.to(torch.float).cuda()
    im_data = torch.tensor(im_data, requires_grad=True)

    # plt.imshow(img)
    # plt.show()

    # 参数设置
    debug = True
    # 居民身份证的坐标位置
    gt3 = np.asarray([[200,218],[198,207],[232,201],[238,210]])      # 签发机关
    gt1 = np.asarray([[205,150],[202,126],[365,93],[372,111]])     # 居民身份证
    # # gt2 = np.asarray([[205,150],[202,126],[365,93],[372,111]])     # 居民身份证
    gt2 = np.asarray([[206,111],[199,95],[349,60],[355,80]])       # 中华人民共和国
    gt4 = np.asarray([[312,127],[304,105],[367,88],[374,114]])       # 份证
    gt5 = np.asarray([[133,168],[118,112],[175,100],[185,154]])      # 国徽
    # gts = [gt1, gt2, gt3, gt4, gt5]
    gts = [gt2, gt4, gt5]

    # cv2.polylines(img, [gt2.astype(int)], isClosed=True, color=(0,0,255))
    # cv2.polylines(img, [gt4.astype(int)], isClosed=True, color=(0,0,255))
    # cv2.polylines(img, [gt5.astype(int)], isClosed=True, color=(0,0,255))

    # cv2.imshow('show', img)
    # cv2.waitKey()



    roi = []
    for i,gt in enumerate(gts):
        center = (gt[0, :] + gt[1, :] + gt[2, :] + gt[3, :]) / 4        # 求中心点

        dw = gt[2, :] - gt[1, :]
        dh =  gt[1, :] - gt[0, :]
        w = math.sqrt(dw[0] * dw[0] + dw[1] * dw[1])                    # 宽度和高度
        h = math.sqrt(dh[0] * dh[0] + dh[1] * dh[1])  + random.randint(-2, 2)

        angle_gt = (math.atan2((gt[2][1] - gt[1][1]), gt[2][0] - gt[1][0]) + math.atan2((gt[3][1] - gt[0][1]), gt[3][0] - gt[0][0]) ) / 2
        angle_gt = -angle_gt / 3.1415926535 * 180                       # 需要加个负号

        rr = cv2.minAreaRect(gt)
        center = rr[0]
        w, h = rr[1]
        angle = rr[2]
        if w < h:
            angle = angle + 180


        # roi.append([0, center[0], center[1], h, w, angle_gt])           # roi的参数
        roi.append([0, center[0], center[1], h, w, -angle])           # roi的参数

    rois = torch.tensor(roi)
    rois = rois.to(torch.float).cuda()

    pooled_height = 44
    maxratio = rois[:,4] / rois[:,3]
    maxratio = maxratio.max().item()
    pooled_width = math.ceil(pooled_height * maxratio)

    roipool = _RRoiAlign(pooled_height, pooled_width, 1.0)
    # 执行rroi_align操作
    pooled_feat = roipool(im_data, rois.view(-1, 6))

    # test = gradcheck(roipool, (im_data, rois.view(-1, 6)))
    # print(test)

    res = pooled_feat.pow(2).sum()
    # res = pooled_feat.sum()
    res.backward()


    if debug:
        for i in range(pooled_feat.shape[0]):
            x_d = pooled_feat.data.cpu().numpy()[i]
            x_data_draw = x_d.swapaxes(0, 2)
            x_data_draw = x_data_draw.swapaxes(0, 1)

            x_data_draw = np.asarray(x_data_draw, dtype=np.uint8)
            #cv2.imshow('im_data_gt %d' % i, x_data_draw)
            cv2.imwrite('./res%d.jpg' % i, x_data_draw)

        # cv2.imshow('img', img)

        # 显示梯度
        im_grad = im_data.grad.data.cpu().numpy()[0]
        im_grad = im_grad.swapaxes(0, 2)
        im_grad = im_grad.swapaxes(0, 1)

        im_grad = np.asarray(im_grad, dtype=np.uint8)
        cv2.imwrite('./grad.jpg',im_grad)

        #
        grad_img = img + im_grad
        cv2.imwrite('./grad_img.jpg', grad_img)
        print(pooled_feat.shape)