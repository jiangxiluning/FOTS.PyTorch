from roi_align.crop_and_resize import CropAndResizeFunction
from torch import nn
import cv2

class ROIRotate(nn.Module):

    def __init__(self, height=8):
        '''

        :param height: heigth of feature map after affine transformation
        '''

        self.height = height

    def forward(self, feature_map, predicted, boxes_batch):
        '''

        :param feature_map:  N * 128 * 128 * 32
        :param boxes_batch: list of box array , box on image with 512 x 512
        :param predicted: list of box array , box on image with 512 x 512
        :return:
        '''

        if self.training:

            max_width = 0
            boxes_list = []
            boxes_indexes = []
            boxes_width = []
            for img_index, boxes in enumerate(zip(boxes_batch)):

                for box_index, box in enumerate(boxes):
                    x1, y1, x2, y2, _, _, x4, y4 = box[:8] / 4 # 521 -> 128
                    angle = box[-1]

                    mapped_x1, mapped_y1 = (0, 0)
                    mapped_x4, mapped_y4 = (0, self.height)

                    box_h = y4 - y1
                    box_w = x2 - x1
                    width = self.height * box_w / box_h
                    max_width = width if width > max_width else max_width

                    mapped_x2, mapped_y2 = (width, 0)

                    affine_matrix = cv2.getAffineTransform([(x1, y1), (x2, y2), (x4, y4)], [
                        (mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (mapped_x4, mapped_y4)
                    ])

                    feature_map_rotated = cv2.transform(feature_map, affine_matrix)

                    boxes_list.append(box)
                    boxes_indexes.append(img_index)
                    boxes_width.append(width)

            boxes_padded_width = [max_width - w for w in boxes_width]

            crops = CropAndResizeFunction(self.height, max_width)(feature_map, boxes_list, boxes_indexes)

            return crops, boxes_padded_width

        else:
            # eval
            pass







