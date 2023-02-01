import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from ...base.base_model import BaseModel

SPEEDUP_SCALE = 512


class SharedConv(BaseModel):
    '''
    sharded convolutional layers
    '''

    def __init__(self, bbNet: nn.Module, config):
        super(SharedConv, self).__init__(config)
        self.backbone = bbNet
        # backbone as feature extractor
        """ for param in self.backbone.parameters():
            param.requires_grad = False """

        # Feature-merging branch
        # self.toplayer = nn.Conv2d(2048, 256, kernel_size = 1, stride = 1, padding = 0)  # Reduce channels

        self.mergeLayers1 = HLayer(2048, 1024)
        self.mergeLayers2 = HLayer(1024, 512)
        self.mergeLayers3 = HLayer(512, 256)

        # self.conv = nn.Conv2d(32, 32, kernel_size = 3, padding = 1)
        # self.bn5 = nn.BatchNorm2d(32)

    def forward(self, input):
        # bottom up

        f = self.__foward_backbone(input)
        output = self.mergeLayers1(f['conv5'], f['conv4'])
        output = self.mergeLayers2(output, f['conv3'])
        final = self.mergeLayers3(output, f['conv2'])

        # score = self.scoreMap(final)
        # score = torch.sigmoid(score)
        #
        # geoMap = self.geoMap(final)
        # # 出来的是 normalise 到 0 -1 的值是到上下左右的距离，但是图像他都缩放到  512 * 512 了，但是 gt 里是算的绝对数值来的
        # geoMap = torch.sigmoid(geoMap) * 512
        #
        # angleMap = self.angleMap(final)
        # angleMap = (torch.sigmoid(angleMap) - 0.5) * math.pi / 2
        #
        # geometry = torch.cat([geoMap, angleMap], dim = 1)
        #
        # return score, geometry

        return final

    def __foward_backbone(self, input):
        conv2 = None
        conv3 = None
        conv4 = None
        output = None # n * 7 * 7 * 2048

        for name, layer in self.backbone.named_children():
            input = layer(input)
            if name == 'layer1':
                conv2 = input
            elif name == 'layer2':
                conv3 = input
            elif name == 'layer3':
                conv4 = input
            elif name == 'layer4':
                output = input
                break
            
        return {'conv5': output,
                'conv4': conv4,
                'conv3': conv3,
                'conv2': conv2}

    def __unpool(self, input):
        _, _, H, W = input.shape
        return F.interpolate(input, mode = 'bilinear', scale_factor = 2, align_corners = True)

    def __mean_image_subtraction(self, images, means = [123.68, 116.78, 103.94]):
        '''
        image normalization
        :param images: bs * w * h * channel
        :param means:
        :return:
        '''
        num_channels = images.data.shape[1]
        if len(means) != num_channels:
            raise ValueError('len(means) must match the number of channels')
        for i in range(num_channels):
            images.data[:, i, :, :] -= means[i]

        return images


class DummyLayer(nn.Module):

    def forward(self, input_f):
        return input_f


class HLayer(nn.Module):

    def __init__(self, inputChannels, outputChannels):
        """

        :param inputChannels: channels of g+f
        :param outputChannels:
        """
        super(HLayer, self).__init__()
        self.conv2dOne = nn.Conv2d(inputChannels, outputChannels, kernel_size = 1)
        

    def forward(self, inputPrevG, inputF):
        inputPrevG = self.conv2dOne(inputPrevG)
        inputPrevG = F.interpolate(inputPrevG, mode = 'bilinear', scale_factor = 2, align_corners = True)
        return inputPrevG + inputF