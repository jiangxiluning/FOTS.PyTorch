from torch import nn
import cv2
import numpy as np

class ROIRotate(nn.Module):

    def __init__(self, height=8):
        '''

        :param height: heigth of feature map after affine transformation
        '''

        super().__init__()
        self.height = height

    def forward(self, feature_map, boxes):
        '''

        :param feature_map:  N * 128 * 128 * 32
        :param boxes: list of box array , box on image with 512 x 512
        :return: N * H * W * C
        '''

        max_width = 0
        boxes_width = []
        cropped_images = []
        feature_map = feature_map.permute(0, 2, 3, 1).detach().cpu().numpy()
        _, _, _ , channels = feature_map.shape
        for img_index in range(len(feature_map)):
            feature = feature_map[img_index]  # B * H * W * C

            for box in boxes[img_index]:
                x1, y1, x2, y2, _, _, x4, y4 = box[:8] / 4 # 521 -> 128

                mapped_x1, mapped_y1 = (0, 0)
                mapped_x4, mapped_y4 = (0, self.height)

                box_h = y4 - y1
                box_w = x2 - x1
                width = int(self.height * box_w / box_h)
                max_width = width if width > max_width else max_width

                mapped_x2, mapped_y2 = (width, 0)

                affine_matrix = cv2.getAffineTransform(np.float32([(x1, y1), (x2, y2), (x4, y4)]), np.float32([
                    (mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (mapped_x4, mapped_y4)
                ]))

                h, w, c = feature.shape

                all_channels = []

                for i in range(0, c, 3):  # transform feature map every 3 channels
                    f_split = feature[ :, :, i:i + 3]
                    f_split_rotated = cv2.warpAffine(f_split, affine_matrix, dsize=(h, w))
                    all_channels.append(f_split_rotated)
                all_channels = np.concatenate(all_channels, axis=-1)  # hstack all channels horizontally

                cropped_image = all_channels[0: self.height, 0:width, :]
                cropped_images.append(cropped_image)
                boxes_width.append(width)

        boxes_padded_width = [max_width - w for w in boxes_width]
        cropped_images_padded = np.zeros((len(boxes_padded_width), self.height, max_width, channels))

        for i in range(len(boxes_padded_width)):
            padded_width = boxes_padded_width[i]
            padded_part = np.zeros((self.height, padded_width, channels))
            cropped_images_padded[i] = np.concatenate([cropped_images[i], padded_part], axis=1)

        lengths = np.array([max_width] * len(boxes_padded_width)) - np.array(boxes_padded_width)
        indices = np.argsort(lengths) # sort images by its width cause pack padded tensor needs it
        indices = indices[::-1].copy() # descending order
        lengths = lengths[indices]
        cropped_images_padded = cropped_images_padded[indices]

        return cropped_images_padded, lengths, indices



