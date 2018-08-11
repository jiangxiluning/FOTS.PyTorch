import torch.nn as nn
import torch.nn.functional as F


class ShardedConv(nn.Module):
    '''
    sharded convolutional layers
    '''

    def __init__(self, bbNet):

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

        self.smooth = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size = 1, stride = 1, padding = 0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size = 1, stride = 1, padding = 0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, input):

        # bottom up
        outputFeatures = self.backbone.features(input)  # n * 7 * 7 * 2048
        p5 = self.toplayer(outputFeatures)
        p4 = self.__upsample_add(p5, self.latlayer1(self.conv4Output))
        p3 = self.__upsample_add(p4, self.latlayer2(self.conv3Output))
        p2 = self.__upsample_add(p3, self.latlayer3(self.conv2Output))
        p2 = self.smooth(p2)

        return p2

    def __upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def __register_hooks(self):

        def forward_hook_conv2(module, input, output):
            self.conv2Output = output

        def forward_hook_conv3(module, input, output):
            self.conv3Output = output

        def forward_hook_conv4(module, input, output):
            self.conv4Output = output

        # get intermediate output of pretrained model
        self.backbone.layer1[2].relu.register_forward_hook(forward_hook_conv2)
        self.backbone.layer1[2].relu.register_forward_hook(forward_hook_conv3)
        self.backbone.layer1[2].relu.register_forward_hook(forward_hook_conv4)

