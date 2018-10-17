from roi_align.crop_and_resize import CropAndResizeFunction
from torch import nn

class ROIRotate(nn.Module):

    def __init__(self, height=8):
        '''

        :param height: heigth of feature map after affine transformation
        '''

        self.height = height

    def forward(self, images, feature_map, boxes_batch):
        '''

        :param im: N * 512 * 512 * 3
        :param feature_map:  N * 128 * 128 * 32
        :param boxes: list of box array
        :return:
        '''

        max_width = 0
        boxes_list = []
        boxes_indexes = []
        boxes_width = []
        for img_index, image, boxes in enumerate(zip(images, boxes_batch)):

            for box_index, box in enumerate(boxes):
                x1, y1, x2, y2, angle = box
                # 旋转 box 至水平
                x1_, y1_, x2_, y2_ = x1, y1, x2, y2

                box_h = y2_ - y1_
                box_w = x2_ - x1_
                width = self.height * box_w / box_h
                max_width = width if width > max_width else max_width

                boxes_list.append(box)
                boxes_indexes.append(img_index)
                boxes_width.append(width)

        boxes_padded_width = [max_width - w for w in boxes_width]

        crops = CropAndResizeFunction(self.height, max_width)(images, boxes_list, boxes_indexes)

        return crops, boxes_padded_width







