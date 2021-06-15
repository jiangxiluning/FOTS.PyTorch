import os
import pathlib
import typing

import numpy
import torch
import collections
import cv2
import numpy as np
from sklearn.decomposition import PCA
import string

keys = string.printable


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def show_box(image_name, image, box, transcirpt, isFeaturemap=False):
    pts = box.astype(np.int)

    if isFeaturemap: # dimension reduction
        h, w, c = image.shape
        pca = PCA(n_components=3)
        ii = image.reshape(h*w, c)
        ii = pca.fit_transform(ii)

        for c in range(3):
            max = np.max(ii[:, c])
            min = np.min(ii[:, c])
            x_std = (ii[:, c] - min) / (max - min)
            ii[:, c] = x_std * 255
        image = ii.reshape(h, w, -1).astype(np.uint8)

    img = cv2.polylines(image, [pts], True, [150, 200, 200])

    origin = pts[0]
    font = cv2.FONT_HERSHEY_PLAIN
    img = cv2.putText(img, transcirpt, (origin[0], origin[1] - 10), font, 0.5, (255, 255, 255))

    cv2.imwrite(image_name, img)
    # cv2.waitKey()

def visualize(image_path: str,
              boxes: numpy.ndarray,
              transrcipts: typing.List[str]):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    saved_path = pathlib.Path('/home/luning/dev/projects/FOTS.PyTorch/res_images')
    saved_path.mkdir(exist_ok=True)
    for box, transcript in zip(boxes, transrcipts):
        if box.shape[1] == 2:
            pts = box.T.astype(np.int)
        else:
            pts = box.astype(np.int)
        image = cv2.polylines(image, [pts], True, [150, 200, 200])
        origin = pts[0]
        font = cv2.FONT_HERSHEY_PLAIN
        img = cv2.putText(img, transcript, (origin[0], origin[1] - 10), font, 0.5, (255, 255, 255))

        image_name = pathlib.Path(image_path).stem

        cv2.imwrite(str((saved_path / 'res_'+image_name).with_suffix('.jpg')), img)

class StringLabelConverter(object):
    """Convert between str and label.
    NOTE:
        Insert `blank` to the alphabet for CTC.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False, max_length=50, raise_over_length=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet

        self.dict = {}
        for i, char in enumerate(iter(self.alphabet)):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

        self.dict['<other>'] = len(self.dict)
        self.max_length = max_length
        self.raise_over_length = raise_over_length

    def encode(self, text):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            text = [
                self.dict.get(char.lower() if self._ignore_case else char, self.dict['<other>'])
                for char in text
            ]
            length = [len(text)]
            return text, length
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            texts = []
            for s in text:
                text = self.encode(s)[0]
                if len(text) > self.max_length:
                    if self.raise_over_length:
                        raise ValueError('{} is over length {}'.format(text, self.max_length))
                    else:
                        text = text[:self.max_length]
                else:
                    text = text + [len(self.dict) + 1] * (self.max_length - len(text))
                texts.append(text)

            text = torch.tensor(texts, dtype=torch.long)

            return text, length

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length.item()
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


str_label_converter = StringLabelConverter(alphabet=keys, ignore_case=False)

if __name__ == '__main__':
    image = cv2.imread('/Users/luning/Dev/data/icdar/icdar2015/4.4/training/ch4_training_images/img_1.jpg')
    import pandas as pd
    gts = pd.read_csv('/Users/luning/Dev/data/icdar/icdar2015/4.4/training/ch4_training_localization_transcription_gt/gt_img_1.txt', header=None)
    for index, gt in gts.iterrows():
        x1, y1, x2, y2, x3, y3, x4, y4 = gt[:8]
        transcript = gt[8]
        show_box(image, np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], dtype=np.int), transcript)
