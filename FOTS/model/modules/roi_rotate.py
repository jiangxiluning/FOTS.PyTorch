from torch import nn
import cv2
import numpy as np
import math
import torch
import skimage.transform as trans
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
        matrixes = []
        images = []

        for img_index, box in zip(mapping, boxes):
            feature = feature_map[img_index]  # B * H * W * C
            images.append(feature)

            x1, y1, x2, y2, x3, y3, x4, y4 = box / 4  # 521 -> 128


            # show_box(feature, box / 4, 'ffffff', isFeaturemap=True)

            rotated_rect = cv2.minAreaRect(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
            box_w, box_h = rotated_rect[1][0], rotated_rect[1][1]

            width = feature.shape[2]
            height = feature.shape[1]

            if box_w <= box_h:
                box_w, box_h = box_h, box_w

            mapped_x1, mapped_y1 = (0, 0)
            mapped_x4, mapped_y4 = (0, self.height)

            width_box = math.ceil(self.height * box_w / box_h)
            width_box = min(width_box, width) # not to exceed feature map's width
            max_width = width_box if width_box > max_width else max_width

            mapped_x2, mapped_y2 = (width_box, 0)

            src_pts = np.float32([(x1, y1), (x2, y2),(x4, y4)])
            dst_pts = np.float32([
                (mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (mapped_x4, mapped_y4)
            ])

            affine_matrix = cv2.getAffineTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))

            # iii = cv2.warpAffine((feature*255).permute(1,2,0).detach().cpu().numpy().astype(np.uint8),
            #                      affine_matrix, (width, height))
            #
            # cv2.imshow('img', iii)
            # cv2.waitKey()

            affine_matrix = ROIRotate.param2theta(affine_matrix, width, height)

            affine_matrix *= 1e20 # cancel the error when type conversion
            affine_matrix = torch.tensor(affine_matrix, device=feature.device, dtype=torch.float)
            affine_matrix /= 1e20

            # grid = torch.nn.functional.affine_grid(affine_matrix[np.newaxis], feature[np.newaxis].size())
            # x = torch.nn.functional.grid_sample(feature[np.newaxis], grid)
            #
            # x = x[0].permute(1, 2, 0).detach().cpu().numpy()
            # x = (x*255).astype(np.uint8)
            #
            # cv2.imshow('img', x)
            # cv2.waitKey()

            matrixes.append(affine_matrix)
            boxes_width.append(width_box)

        matrixes = torch.stack(matrixes)
        images = torch.stack(images)
        grid = nn.functional.affine_grid(matrixes, images.size())
        feature_rotated = nn.functional.grid_sample(images, grid)

        channels = feature_rotated.shape[1]
        cropped_images_padded = torch.zeros((len(feature_rotated), channels, self.height, max_width),
                                            dtype=feature_rotated.dtype,
                                            device=feature_rotated.device)

        for i in range(feature_rotated.shape[0]):
            w = boxes_width[i]
            if max_width == w:
                cropped_images_padded[i] = feature_rotated[i, :, 0:self.height, 0:w]
            else:
                padded_part = torch.zeros((channels, self.height, max_width - w),
                                          dtype=feature_rotated.dtype,
                                          device=feature_rotated.device)
                cropped_images_padded[i] = torch.cat([feature_rotated[i, :, 0:self.height, 0: w], padded_part], dim=-1)

        lengths = np.array(boxes_width)
        indices = np.argsort(lengths) # sort images by its width cause pack padded tensor needs it
        indices = indices[::-1].copy() # descending order
        lengths = lengths[indices]
        cropped_images_padded = cropped_images_padded[indices]

        return cropped_images_padded, lengths, indices

    @staticmethod
    def param2theta(param, w, h):
        param = np.vstack([param, [0, 0, 1]])
        param = np.linalg.inv(param)

        theta = np.zeros([2, 3])
        theta[0, 0] = param[0, 0]
        theta[0, 1] = param[0, 1] * h / w
        theta[0, 2] = param[0, 2] * 2 / w + theta[0, 0] + theta[0, 1] - 1
        theta[1, 0] = param[1, 0] * w / h
        theta[1, 1] = param[1, 1]
        theta[1, 2] = param[1, 2] * 2 / h + theta[1, 0] + theta[1, 1] - 1
        return theta


