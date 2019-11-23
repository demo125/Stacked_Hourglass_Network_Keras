import os
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
import scipy.misc
import json
import random
from PIL import Image


class MPIIDataGen(object):

    def __init__(self, jsonfile, imgpath, inres, outres, is_train):
        self.jsonfile = jsonfile
        self.imgpath = imgpath
        self.inres = inres
        self.outres = outres
        self.is_train = is_train
        self.nparts = 4
        self.anno = self._load_image_annotation()

    def _load_image_annotation(self):
        # load train or val annotation
        with open(self.jsonfile) as anno_file:
            anno = json.loads(json.load(anno_file))
        val_anno, train_anno = [], []
        for idx, val in enumerate(anno):
            if val['isValidation'] == True:
                val_anno.append(anno[idx])
            else:
                train_anno.append(anno[idx])

        if self.is_train:
            return train_anno
        else:
            return val_anno

    def get_dataset_size(self):
        return len(self.anno)

    def get_color_mean(self):
#         mean = np.array([0.4404, 0.4440, 0.4327], dtype=np.float)
        mean = np.array([0.20129337, 0.20129337, 0.20129337], dtype=np.float)
        return mean

    def get_annotations(self):
        return self.anno

    def generator(self, batch_size, num_hgstack, sigma=1, with_meta=False, is_shuffle=False,
                  rot_flag=False, scale_flag=False, flip_flag=False):
        '''
        Input:  batch_size * inres  * Channel (3)
        Output: batch_size * oures  * nparts
        '''
        train_input = np.zeros(shape=(batch_size, self.inres[0], self.inres[1], 3), dtype=np.float)
        gt_heatmap = np.zeros(shape=(batch_size, self.outres[0], self.outres[1], self.nparts), dtype=np.float)
        meta_info = list()

        if not self.is_train:
            assert (is_shuffle == False), 'shuffle must be off in val model'
            assert (rot_flag == False), 'rot_flag must be off in val model'

        while True:
            if is_shuffle:
                shuffle(self.anno)

            for i, kpanno in enumerate(self.anno):

                _imageaug, _gthtmap, _meta = self.process_image(i, kpanno, sigma, rot_flag, scale_flag, flip_flag)
                _index = i % batch_size
                
                train_input[_index, :, :, :] = _imageaug
                gt_heatmap[_index, :, :, :] = _gthtmap
                meta_info.append(_meta)
                
                if i % batch_size == (batch_size - 1):
                    out_hmaps = []
                    for m in range(num_hgstack):
                        out_hmaps.append(gt_heatmap)

                    if with_meta:
                        yield np.array(train_input), np.array(out_hmaps), np.array(meta_info)
                        meta_info = []
                    else:
                        yield np.array(train_input), np.array(out_hmaps)

    def process_image(self, sample_index, kpanno, sigma, rot_flag, scale_flag, flip_flag):
        imagefile = kpanno['img_paths']
        #todo zmenit na gray
        print(imagefile)
        image = np.array(Image.open(os.path.join(self.imgpath, imagefile)).convert('RGB'))
        # get center
        center = np.array(kpanno['objpos'])
        objects = np.array(kpanno['obj_locations'])
        scale = kpanno['scale_provided']

        # Adjust center/scale slightly to avoid cropping limbs
        if center[0] != -1:
            center[1] = center[1] + 15 * scale
            scale = scale * 1.25

        # filp
        if flip_flag and random.choice([0, 1]):
            image, objects, center = self.flip(image, objects, center)

        # scale
        if scale_flag:
            scale = scale * np.random.uniform(0.8, 1.2)

        # rotate image
        if rot_flag and random.choice([0, 1]):
            rot = np.random.randint(-1 * 30, 30)
        else:
            rot = 0
        
#         rot = 0

        cropimg = crop(image, center, scale, self.inres, rot)
        cropimg = normalize(cropimg, self.get_color_mean())

        # transform keypoints
        transformedKps = transform_kp(objects, center, scale, self.outres, rot)
        gtmap = generate_gtmap(transformedKps, sigma, self.outres)
        # meta info
        metainfo = {'sample_index': sample_index, 'center': center, 'scale': scale,
                    'pts': objects, 'tpts': transformedKps, 'name': imagefile}
        return cropimg, gtmap, metainfo

    @classmethod
    def get_kp_keys(cls):
        keys = ['left_kidney', 'right_kidney']
        return keys

    def flip(self, image, objects, center):

        import cv2

        objects = np.copy(objects)

        matchedParts = (
            [0, 1],  # left_kidney
            [2, 3],  # right_kidney
        )

        org_height, org_width, channels = image.shape

        # flip image
        flipimage = cv2.flip(image, flipCode=1)

        # flip each object
        objects[:, 0] = org_width - objects[:, 0]

        for i, j in matchedParts:
            temp = np.copy(objects[i, :])
            objects[i, :] = objects[j, :]
            objects[j, :] = temp

        # center
        flip_center = center
        flip_center[0] = org_width - center[0]

        return flipimage, objects, flip_center