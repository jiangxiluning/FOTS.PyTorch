from base import BaseModel
import torch.nn as nn
import torch.nn.functional as F
from .modules import shared_conv
import pretrainedmodels as pm


class FOTSModel(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        self.mode = config['mode']

        bbNet =  pm.__dict__['resnet50'](pretrained='imagenet') # resnet50 in paper
        self.sharedConv = shared_conv.SharedConv(bbNet)
        self.recognizer = Recognizer()

    def forward(self, input):
        '''

        :param input:
        :return:
        '''

        score_map, geo_map = self.sharedConv.forward(input)

        if self.mode == 'detection':
            return score_map, geo_map, None
        elif self.mode == 'recognition':
            recog_map = self.recognizer(score_map, geo_map)
            return score_map, geo_map, recog_map


class Recognizer(nn.Module):

    def __init__(self):
        super(Recognizer, self).__init__()


    def forward(self, *input):
        return None