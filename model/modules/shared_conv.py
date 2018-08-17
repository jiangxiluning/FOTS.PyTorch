import torch.nn as nn
import torch.nn.functional as F
import torch

class SharedConv(nn.Module):
    '''
    sharded convolutional layers
    '''

    def __init__(self, bbNet):
        super(SharedConv, self).__init__()
        # self.backbone = pm.__dict__['resnet50'](num_class=1000, pretrained='imagenet') # resnet50 in paper
        # self.backbone.eval()
        # self.preprocessTF = utils.TransformImage(self.backbone) # load transformation from model
        self.backbone = bbNet
        self.conv2Output = None
        self.conv3Output = None
        self.conv4Output = None
        self.__register_hooks()

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size = 1, stride = 1, padding = 0)  # Reduce channels

        self.mergeLayers0 = DummyLayer()

        self.mergeLayers1 = HLayer(2048 + 1024, 128)
        self.mergeLayers2 = HLayer(128 + 512, 64)
        self.mergeLayers3 = HLayer(64 + 32, 32)

        self.mergeLayers4 = nn.Conv2d(32, 32, kernel_size = 3)
        self.bn5 = nn.BatchNorm2d(32)





    def forward(self, input):

        # bottom up
        outputFeatures = self.backbone.features(input)  # n * 7 * 7 * 2048
        f1 = self.toplayer(outputFeatures)
        f = [f1, self.conv4Output, self.conv3Output, self.conv2Output]

        g = [None] * 4
        h = [None] * 4

        # i = 1
        h[0] = self.mergeLayers0(f[0])
        g[0] = self.__unpool(h[0])

        # i = 2
        h[1] = self.mergeLayers1(g[0], f[1])
        g[1] = self.__unpool(h[1])

        # i = 3
        h[2] = self.mergeLayers2(g[1], f[2])
        g[2] = self.__unpool(h[2])

        # i = 4
        h[3] = self.mergeLayers3(g[2], f[3])
        g[3] = self.__unpool(h[3])

        # final stage
        g[4] = self.mergeLayers4(h[3])
        g[4] = self.bn5(g[4])
        g[4] = F.relu(g[4])

        return g[4]

    def __unpool(self, input):
        _, _, H, W = input.size()
        return F.upsample_bilinear(input, H * 2, W * 2)

    def __register_hooks(self):

        def forward_hook_conv2(module, input, output):
            self.conv2Output = output

        def forward_hook_conv3(module, input, output):
            self.conv3Output = output

        def forward_hook_conv4(module, input, output):
            self.conv4Output = output

        # get intermediate output of pretrained model
        self.backbone.layer1[2].relu.register_forward_hook(forward_hook_conv2)
        self.backbone.layer2[3].relu.register_forward_hook(forward_hook_conv3)
        self.backbone.layer3[5].relu.register_forward_hook(forward_hook_conv4)



class DummyLayer(nn.Module):

    def forward(self, input_f):
        return input

class HLayer(nn.Module):

    def __init__(self, inputChannels, outputChannels):
        """

        :param inputChannels: channels of g+f
        :param outputChannels:
        """
        super(HLayer, self).__init__()

        self.conv2dOne = nn.Conv2d(inputChannels, outputChannels, kernel_size = 1)
        self.bnOne = nn.BatchNorm2d(outputChannels)

        self.conv2dTwo = nn.Conv2d(outputChannels, outputChannels, kernel_size = 3)
        self.bnTwo = nn.BatchNorm2d(outputChannels)


    def forward(self, inputPrevG, inputF):
        input = torch.cat([inputPrevG, inputF], dim = 1)
        output = self.conv2dOne(input)
        output = self.bnOne(input)
        output = F.relu(output)

        output = self.conv2dTwo(output)
        output = self.bnTwo(output)
        output = F.relu(output)

        return output