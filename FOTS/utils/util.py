import os
import torch
import collections
import cv2
import numpy as np
from sklearn.decomposition import PCA

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def show_box(image, box, transcirpt, isFeaturemap=False):
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

    cv2.imshow('text', img)
    cv2.waitKey()


class strLabelConverter(object):
    """Convert between str and label.
    NOTE:
        Insert `blank` to the alphabet for CTC.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet

        self.dict = {}
        for i, char in enumerate(iter(self.alphabet)):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

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
                self.dict.get(char.lower() if self._ignore_case else char, 0)
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.tensor(text), torch.tensor(length))

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


if __name__ == '__main__':
    image = cv2.imread('/Users/luning/Dev/data/icdar/icdar2015/4.4/training/ch4_training_images/img_1.jpg')
    import pandas as pd
    gts = pd.read_csv('/Users/luning/Dev/data/icdar/icdar2015/4.4/training/ch4_training_localization_transcription_gt/gt_img_1.txt', header=None)
    for index, gt in gts.iterrows():
        x1, y1, x2, y2, x3, y3, x4, y4 = gt[:8]
        transcript = gt[8]
        show_box(image, np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], dtype=np.int), transcript)
