from base import BaseModel
import torch.nn as nn
import torch.nn.functional as F
from .modules import shared_conv
import pretrainedmodels as pm


class FOTSModel(BaseModel):

    def __init__(self, config):
        super().__init__(config)

        bbNet =  pm.__dict__['resnet50'](pretrained='imagenet') # resnet50 in paper
        self.sharedConv = shared_conv.SharedConv(bbNet)

    def forward(self, input):
        return self.sharedConv.forward(input)