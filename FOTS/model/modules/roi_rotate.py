from torch import nn
import cv2
import numpy as np
import math
import torch
from ...utils.util import show_box

class ROIRotate(nn.Module):

    def __init__(self, height=8):
        '''

        :param height: heigth of feature map after affine transformation
        '''

        super().__init__()
        self.height = height

    def forward(self, feature_map, boxes, mapping):
        '''

        :param feature_map:  N * 128 * 128 * 32
        :param boxes: M * 8
        :param mapping: mapping for image
        :return: N * H * W * C
        '''

        max_width = 0
        boxes_width = []
        cropped_images = []
        feature_map = feature_map.permute(0, 2, 3, 1).detach().cpu().numpy()
        _, _, _ , channels = feature_map.shape

        for img_index, box in zip(mapping, boxes):
            feature = feature_map[img_index]  # B * H * W * C

            x1, y1, x2, y2, x3, y3, x4, y4 = box / 4  # 521 -> 128

            # show_box(feature, box / 4, 'ffffff', isFeaturemap=True)

            rotated_rect = cv2.minAreaRect(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
            box_w, box_h = rotated_rect[1][0], rotated_rect[1][1]

            if box_w <= box_h:
                box_w, box_h = box_h, box_w

            mapped_x1, mapped_y1 = (0, 0)
            mapped_x4, mapped_y4 = (0, self.height)

            width = math.ceil(self.height * box_w / box_h)
            max_width = width if width > max_width else max_width

            mapped_x2, mapped_y2 = (width, 0)

            affine_matrix = cv2.getAffineTransform(np.float32([(x1, y1), (x2, y2), (x4, y4)]), np.float32([
                (mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (mapped_x4, mapped_y4)
            ]))

            h, w, c = feature.shape

            all_channels = []

            for i in range(0, c, 3):  # transform feature map every 3 channels
                f_split = feature[:, :, i:i + 3]
                f_split_rotated = cv2.warpAffine(f_split, affine_matrix, dsize=(h, w))
                all_channels.append(f_split_rotated)
            all_channels = np.concatenate(all_channels, axis=-1)  # hstack all channels horizontally

            cropped_image = all_channels[0: self.height, 0:width, :]
            cropped_images.append(cropped_image)
            boxes_width.append(width)

        #boxes_padded_width = [max_width - w for w in boxes_width]
        cropped_images_padded = np.zeros((len(cropped_images), self.height, max_width, channels))

        for i in range(len(cropped_images)):
            _, w, _ = cropped_images[i].shape
            padded_part = np.zeros((self.height, max_width - w, channels))
            cropped_images_padded[i] = np.concatenate([cropped_images[i], padded_part], axis=1)

        lengths = np.array(boxes_width)
        indices = np.argsort(lengths) # sort images by its width cause pack padded tensor needs it
        indices = indices[::-1].copy() # descending order
        lengths = lengths[indices]
        cropped_images_padded = cropped_images_padded[indices]

        return cropped_images_padded, lengths, indices

    def warpAffine(self, featureMap, matrix, dsize):
        """

        :param feature: feature map
        :param matrix: affin matrix 2*3
        :param dsize: target size tuple
        :return:
        """
        m0 = matrix[0, 0]
        m1 = matrix[0, 1]
        m2 = matrix[0, 2]
        m3 = matrix[1, 0]
        m4 = matrix[1, 1]
        m5 = matrix[1, 2]

        detImg = torch.zeros((featureMap.shape[0], featureMap[1]), dtype=featureMap.dtype, device=featureMap.device)
        matrix = matrix.astype(np.double)

        D = m0 * m4 - m1 * m3
        D = 1.0 / D if D != 0 else 0.
        A11 = m4 * D
        A22 = m0 * D
        m0 = A11
        m1 *= -D
        m3 = m1
        m4 = A22
        b1 = -m0 * m2 - m1 * m5
        b2 = -m3 * m2 - m4 * m5
        m2 = b1
        m5 = b2






    def saturate_cast_short(self, n):
        return np.clip(n, np.iinfo(np.short).min, np.iinfo(np.short).max)


