#   ______                                           __                 
#  /      \                                         /  |                
# /$$$$$$  | __   __   __   ______   _______        $$ |       __    __ 
# $$ |  $$ |/  | /  | /  | /      \ /       \       $$ |      /  |  /  |
# $$ |  $$ |$$ | $$ | $$ |/$$$$$$  |$$$$$$$  |      $$ |      $$ |  $$ |
# $$ |  $$ |$$ | $$ | $$ |$$    $$ |$$ |  $$ |      $$ |      $$ |  $$ |
# $$ \__$$ |$$ \_$$ \_$$ |$$$$$$$$/ $$ |  $$ |      $$ |_____ $$ \__$$ |
# $$    $$/ $$   $$   $$/ $$       |$$ |  $$ |      $$       |$$    $$/ 
#  $$$$$$/   $$$$$/$$$$/   $$$$$$$/ $$/   $$/       $$$$$$$$/  $$$$$$/ 
#
# File: post_processor.py
# Author: Owen Lu
# Date: 2021/4/11
# Email: jiangxiluning@gmail.com
# Description:
import typing

import numpy as np

from .util import StringLabelConverter


class PostProcessor:

    def __init__(self, use_beam_search=False):
        self.use_beam_sarch = use_beam_search
        pass

    def __call__(self, *args, **kwargs) -> typing.Tuple[typing.List, typing.List]:
        pred_boxes = kwargs['boxes']
        pred_transcripts, pred_lengths = kwargs['transcripts']

        boxes = []
        transcripts = []

        for i in range(pred_boxes.size(0)):
            boxes.append(pred_boxes[i].to_list())

            if not self.use_beam_sarch:
                transcript = np.argmax(pred_transcripts[i], axis=-1)
            else:
                raise NotImplementedError()

            transcript = StringLabelConverter.decode(transcript, pred_lengths[i])
            transcripts.append(transcript)

        return boxes, transcripts
