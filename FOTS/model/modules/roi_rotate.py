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
        matrixes = []
        images = []

        for img_index, box in zip(mapping, boxes):
            feature = feature_map[img_index]  # B * H * W * C
            images.append(feature)

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
            matrixes.append(torch.tensor(affine_matrix, dtype=feature.dtype, device=feature.device))
            boxes_width.append(width)

        matrixes = torch.tensor(matrixes)
        feature = torch.tensor(feature)
        grid = nn.functional.affine_grid(matrixes, feature.size())
        feature_rotated = nn.functional.grid_sample(feature, grid)

        channels = feature_rotated.shape[3]
        cropped_images_padded = np.zeros((len(feature_rotated), self.height, max_width, channels))

        for i in range(feature_rotated.shape[0]):
            _, w, _ = boxes_width[i]
            padded_part = torch.zeros((self.height, max_width - w, channels))
            cropped_images_padded[i] = torch.cat([feature_rotated[i, 0:self.height, 0:w, :], padded_part], axis=1)

        lengths = np.array(boxes_width)
        indices = np.argsort(lengths) # sort images by its width cause pack padded tensor needs it
        indices = indices[::-1].copy() # descending order
        lengths = lengths[indices]
        cropped_images_padded = cropped_images_padded[indices]

        return cropped_images_padded, lengths, indices

    def warpAffine(self, featureMap, matrix, dsize):
        """

        :param feature: feature map B * H * W * C
        :param matrix: affin matrix 2*3
        :param dsize: target size tuple
        :return:
        """
        channels = featureMap.shape[2]
        detH, detW = dsize
        matrix = torch.tensor(matrix, device=featureMap.device, dtype=torch.double)
        detImg = torch.zeros((detH, detW, channels), dtype = featureMap.dtype, device = featureMap.device)



        m0 = matrix[0, 0]
        m1 = matrix[0, 1]
        m2 = matrix[0, 2]
        m3 = matrix[1, 0]
        m4 = matrix[1, 1]
        m5 = matrix[1, 2]

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

        for y in range(detH):
            for x in range(detW):
                fx = m0 * x + m1 * y + m2
                fy = m3 * x + m4 * y + m5

                sy = torch.floor(fy)
                fy -= sy
                sy = sy.int()

                if sy < 0 or sy >= detH:
                    continue

                cbufy = torch.zeros((2,), dtype=torch.uint8)
                cbufy[0] = self.saturate_cast_short((1.0 - fy) * 2048)
                cbufy[1] = torch.tensor(2048, dtype=torch.uint8, device=featureMap.device) - cbufy[0]

                sx = torch.floor(fx)
                fx -= sx
                sx = sx.int()

                if sx < 0 or sx >= detW:
                    continue

                cbufx = torch.zeros((2, ), dtype=torch.uint8)
                cbufx[0] = self.saturate_cast_short((1.0 - fx) * 2048)
                cbufx[1] = torch.tensor(2048, dtype=torch.uint8, device=featureMap.device) - cbufx[0]
                for k in range(channels):
                    if sy == (detH - 1) or sx == (detW - 1):
                        continue
                    else:
                        detImg[y, x, k] = (featureMap[sy, sx, k] * cbufx[0] * cbufy[0] +
                        featureMap[sy+1, sx, k] * cbufx[0] * cbufy[1] +
                        featureMap[sy, sx+1, k] * cbufx[1] * cbufy[0] +
                        featureMap[sy+1, sx+1, k] * cbufx[1] * cbufy[1]) >> 22

        return detImg




    def saturate_cast_short(self, n):
        return torch.clamp(n, np.iinfo(np.short).min, np.iinfo(np.short).max)


