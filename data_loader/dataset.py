from torch.utils.data import Dataset
from .datautils import *


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