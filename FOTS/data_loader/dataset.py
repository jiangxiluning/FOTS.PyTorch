from torch.utils.data import Dataset
from .datautils import *
import scipy.io as sio
import logging
import numpy as np
from itertools import compress
import pathlib

logger = logging.getLogger(__name__)

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


class ICDAR(Dataset):

    structure = {
        '2015': {
            'training': {
                'images': 'ch4_training_images',
                'gt':'ch4_training_localization_transcription_gt',
                'voc_per_image': 'ch4_training_vocabularies_per_image',
                'voc_all': 'ch4_training_vocabulary.txt'
            },
            'test': {
                'images': 'ch4_test_images',
                'gt': 'Challenge4_Test_Task4_GT',
                'voc_per_image': 'ch4_test_vocabularies_per_image',
                'voc_all': 'ch4_test_vocabulary.txt'
            },
            'voc_generic': 'GenericVocabulary.txt'
        },
        '2013': {
            'training': {
                'images': 'ch2_training_images',
                'gt': 'ch2_training_localization_transcription_gt',
                'voc_per_image': 'ch2_training_vocabularies_per_image',
                'voc_all': 'ch2_training_vocabulary.txt'
            },
            'test': {
                'images': 'Challenge2_Test_Task12_Images',
                'voc_per_image': 'ch2_test_vocabularies_per_image',
                'voc_all': 'ch4_test_vocabulary.txt'
            },
            'voc_generic': 'GenericVocabulary.txt'
        },

    }

    def __init__(self, data_root, year='2013', type='training'):
        data_root = pathlib.Path(data_root)

        if year == '2013' and type == 'test':
            logger.warning('ICDAR 2013 does not contain test ground truth. Fall back to training instead.')

        self.structure = ICDAR.structure[year]
        self.imagesRoot = data_root / self.structure[type]['images']
        self.gtRoot = data_root / self.structure[type]['gt']
        self.images, self.bboxs, self.transcripts = self.__loadGT()

    def __loadGT(self):
        all_bboxs = []
        all_texts = []
        all_images = []
        for image in self.imagesRoot.glob('*.jpg'):
            all_images.append(image)
            gt = self.gtRoot / image.with_name('gt_{}'.format(image.stem)).with_suffix('.txt').name
            with gt.open(mode='r') as f:
                bboxes = []
                texts = []
                for line in f:
                    text = line.strip('\ufeff').strip('\xef\xbb\xbf').strip().split(',')
                    x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, text[:8]))
                    bbox = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    transcript = text[8]
                    bboxes.append(bbox)
                    texts.append(transcript)
                bboxes = np.array(bboxes)
                all_bboxs.append(bboxes)
                all_texts.append(texts)
        return all_images, all_bboxs, all_texts

    def __getitem__(self, index):
        imageName = self.images[index]
        bboxes = self.bboxs[index] # num_words * 8
        transcripts = self.transcripts[index]

        try:
            return self.__transform((imageName, bboxes, transcripts))
        except Exception as e:
            return self.__getitem__(torch.tensor(np.random.randint(0, len(self))))

    def __len__(self):
        return len(self.images)

    def __transform(self, gt, input_size = 512, random_scale = np.array([0.5, 1, 2.0, 3.0]),
                        background_ratio = 3. / 8):
        '''

        :param gt: iamge path (str), wordBBoxes (2 * 4 * num_words), transcripts (multiline)

        :return:
        '''

        imagePath, wordBBoxes, transcripts = gt
        im = cv2.imread(imagePath.as_posix())
        #wordBBoxes = np.expand_dims(wordBBoxes, axis = 2) if (wordBBoxes.ndim == 2) else wordBBoxes
        #_, _, numOfWords = wordBBoxes.shape
        numOfWords = len(wordBBoxes)
        text_polys = wordBBoxes  # num_words * 4 * 2
        transcripts = [word for line in transcripts for word in line.split()]
        text_tags = [True if(tag == '*' or tag == '###') else False for tag in transcripts] # ignore '###'

        if numOfWords == len(transcripts):
            h, w, _ = im.shape
            text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))

            rd_scale = np.random.choice(random_scale)
            im = cv2.resize(im, dsize = None, fx = rd_scale, fy = rd_scale)
            text_polys *= rd_scale

            rectangles = []

            # print rd_scale
            # random crop a area from image
            if np.random.rand() < background_ratio:
                # crop background
                im, text_polys, text_tags, selected_poly = crop_area(im, text_polys, text_tags, crop_background = True)
                if text_polys.shape[0] > 0:
                    # cannot find background
                    raise RuntimeError('cannot find background')
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
                im, text_polys, text_tags, selected_poly = crop_area(im, text_polys, text_tags, crop_background = False)
                if text_polys.shape[0] == 0:
                    raise RuntimeError('cannot find background')
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
                score_map, geo_map, training_mask, rectangles = generate_rbox((new_h, new_w), text_polys, text_tags)

            # predict 出来的feature map 是 128 * 128， 所以 gt 需要取 /4 步长
            images = im[:, :, ::-1].astype(np.float32)  # bgr -> rgb
            score_maps = score_map[::4, ::4, np.newaxis].astype(np.float32)
            geo_maps = geo_map[::4, ::4, :].astype(np.float32)
            training_masks = training_mask[::4, ::4, np.newaxis].astype(np.float32)

            transcripts = [transcripts[i] for i in selected_poly]
            mask = [not (word == '*' or word == '###') for word in transcripts]
            transcripts = list(compress(transcripts, mask))
            rectangles = list(compress(rectangles, mask)) # [ [pt1, pt2, pt3, pt3],  ]

            assert len(transcripts) == len(rectangles) # make sure length of transcipts equal to length of boxes
            if len(transcripts) == 0:
                raise RuntimeError('No text found.')

            return imagePath, images, score_maps, geo_maps, training_masks, transcripts, rectangles
        else:
            raise TypeError('Number of bboxes is inconsist with number of transcripts ')

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

        try:
            return self.__transform((imageName, wordBBoxes, transcripts))
        except:
            return self.__getitem__(torch.tensor(np.random.randint(0, len(self))))

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
        imagePath = pathlib.Path(imagePath)
        wordBBoxes = np.expand_dims(wordBBoxes, axis = 2) if (wordBBoxes.ndim == 2) else wordBBoxes
        _, _, numOfWords = wordBBoxes.shape
        text_polys = wordBBoxes.reshape([8, numOfWords], order = 'F').T  # num_words * 8
        text_polys = text_polys.reshape(numOfWords, 4, 2)  # num_of_words * 4 * 2
        transcripts = [word for line in transcripts for word in line.split()]
        text_tags = np.zeros(numOfWords) # 1 to ignore, 0 to hold

        if numOfWords == len(transcripts):
            h, w, _ = im.shape
            text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))

            rd_scale = np.random.choice(random_scale)
            im = cv2.resize(im, dsize = None, fx = rd_scale, fy = rd_scale)
            text_polys *= rd_scale

            rectangles = []

            # print rd_scale
            # random crop a area from image
            if np.random.rand() < background_ratio:
                # crop background
                im, text_polys, text_tags, selected_poly = crop_area(im, text_polys, text_tags, crop_background = True)
                if text_polys.shape[0] > 0:
                    # cannot find background
                    raise RuntimeError('Cannot find background.')
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
                im, text_polys, text_tags, selected_poly = crop_area(im, text_polys, text_tags, crop_background = False)
                if text_polys.shape[0] == 0:
                    raise RuntimeError('Cannot find background.')
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
                score_map, geo_map, training_mask, rectangles = generate_rbox((new_h, new_w), text_polys, text_tags)

            # predict 出来的feature map 是 128 * 128， 所以 gt 需要取 /4 步长
            images = im[:, :, ::-1].astype(np.float32)  # bgr -> rgb
            score_maps = score_map[::4, ::4, np.newaxis].astype(np.float32)
            geo_maps = geo_map[::4, ::4, :].astype(np.float32)
            training_masks = training_mask[::4, ::4, np.newaxis].astype(np.float32)

            transcripts = [transcripts[i] for i in selected_poly]
            rectangles = [rectangles[i] for i in selected_poly]

            assert len(transcripts) == len(rectangles)
            if len(transcripts) == 0:
                raise RuntimeError('No text found.')

            return imagePath, images, score_maps, geo_maps, training_masks, transcripts, rectangles
        else:
            raise TypeError('Number of bboxes is inconsist with number of transcripts ')
