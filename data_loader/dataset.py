from torch.utils.data import Dataset
from .datautils import *
import scipy.io as sio

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
                    variable_names=['imnames', 'wordBB', 'txt'])

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

        data = self.__transform((imageName, wordBBoxes, transcripts))

        try:
            if data is None:
                return self.__getitem__(np.random.randint(0, len(self)))
            else:
                return data
        except:
            return self.__getitem__(np.random.randint(0, len(self)))

    def __len__(self):
        return len(self.imageNames)

    def __transform(self, gt, input_size = 512, random_scale = np.array([0.5, 1, 2.0, 3.0]),
                        background_ratio = 3. / 8):
        '''

        :param gt: iamge path (str), wordBBoxes (2 * 4 * num_words), transcripts (multiline)

        :return:
        '''

        imagePath, wordBBoxes, transcripts = gt
        im = cv2.imread((self.dataRoot / imagePath).as_posix())
        wordBBoxes = np.expand_dims(wordBBoxes, axis = 2) if (wordBBoxes.ndim == 2) else wordBBoxes
        _, _, numOfWords = wordBBoxes.shape
        text_polys = wordBBoxes.reshape([8, numOfWords], order = 'F').T  # num_words * 8
        text_polys = text_polys.reshape(numOfWords, 4, 2)  # num_of_words * 4 * 2
        transcripts = [word for line in transcripts for word in line.split()]
        text_tags = np.zeros(numOfWords)

        if numOfWords == len(transcripts):
            h, w, _ = im.shape
            text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))

            rd_scale = np.random.choice(random_scale)
            im = cv2.resize(im, dsize = None, fx = rd_scale, fy = rd_scale)
            text_polys *= rd_scale

            # print rd_scale
            # random crop a area from image
            if np.random.rand() < background_ratio:
                # crop background
                im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background = True)
                # if text_polys.shape[0] > 0:
                #     # cannot find background
                #     pass
                # pad and resize image
                new_h, new_w, _ = im.shape
                max_h_w_i = np.max([new_h, new_w, input_size])
                im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype = np.uint8)
                im_padded[:new_h, :new_w, :] = im.copy()
                im = cv2.resize(im_padded, dsize = (input_size, input_size))
                score_map = np.zeros((input_size, input_size), dtype = np.uint8)
                geo_map_channels = 5
                #                     geo_map_channels = 5 if FLAGS.geometry == 'RBOX' else 8
                geo_map = np.zeros((input_size, input_size, geo_map_channels), dtype = np.float32)
                training_mask = np.ones((input_size, input_size), dtype = np.uint8)
            else:
                im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background = False)
                # if text_polys.shape[0] == 0:
                #     pass
                h, w, _ = im.shape

                # pad the image to the training input size or the longer side of image
                new_h, new_w, _ = im.shape
                max_h_w_i = np.max([new_h, new_w, input_size])
                im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype = np.uint8)
                im_padded[:new_h, :new_w, :] = im.copy()
                im = im_padded
                # resize the image to input size
                new_h, new_w, _ = im.shape
                resize_h = input_size
                resize_w = input_size
                im = cv2.resize(im, dsize = (resize_w, resize_h))
                resize_ratio_3_x = resize_w / float(new_w)
                resize_ratio_3_y = resize_h / float(new_h)
                text_polys[:, :, 0] *= resize_ratio_3_x
                text_polys[:, :, 1] *= resize_ratio_3_y
                new_h, new_w, _ = im.shape
                score_map, geo_map, training_mask = generate_rbox((new_h, new_w), text_polys, text_tags)

            # predict 出来的feature map 是 128 * 128， 所以 gt 需要取 /4 步长
            images = im[:, :, ::-1].astype(np.float32)  # bgr -> rgb
            score_maps = score_map[::4, ::4, np.newaxis].astype(np.float32)
            geo_maps = geo_map[::4, ::4, :].astype(np.float32)
            training_masks = training_mask[::4, ::4, np.newaxis].astype(np.float32)

            return images, score_maps, geo_maps, training_masks, transcripts

            #return images, score_maps, geo_maps, training_masks
