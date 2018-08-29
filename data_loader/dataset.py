from torch.utils.data import Dataset
from .datautils import *
from scipy.io import sio

class MyDataset(Dataset):

    def __init__(self, img_root, txt_root):
        self.image_list, self.img_name = get_images(img_root)
        self.txt_root = txt_root

    def __getitem__(self, index):
        img, score_map, geo_map, training_mask = image_label(self.txt_root,
                                                             self.image_list, self.img_name, index,
                                                             input_size = 512,
                                                             random_scale = np.array([0.5, 1, 2.0, 3.0]),
                                                             background_ratio = 3. / 8)
        return img, score_map, geo_map, training_mask

    def __len__(self):
        return len(self.image_list)



class SynthTextDataset(Dataset):

    def __init__(self, data_root):
        self.dataRoot = pathlib.Path(data_root)
        if not self.dataRoot.exists():
            raise FileNotFoundError('Dataset folder is not exist.')

        self.targetFilePath = self.dataRoot / 'gt.mat'
        if not self.targetFilePath.exists():
            raise FileExistsError('Target file is not exist.')
        targets = {}
        sio.loadmat(self.targetFilePath, targets, squeeze_me=True, struct_as_record=False,
                    variable_name=['imnames', 'wordBB', 'txt'])

        self.imageNames = targets['imnames']
        self.wordBBoxes = targets['wordBB']
        self.transcripts = targets['txt']

    def __getitem__(self, index):
        """

        :param index:
        :return:
            imageName: path of image
            wordBBox: bounding boxes of words in the image
            transcript: corresponding transcripts of bounded words
        """
        imageName = self.imageNames[index]
        wordBBoxes = self.wordBBoxes[index] # 2 * 4 * num_words
        transcripts = self.transcripts[index]

        return imageName, wordBBoxes, transcripts

    def __len__(self):
        return len(self.imageNames)