import os
import numpy as np
from random import shuffle
import data_process
import scipy.misc
import json
import cv2
import random
from PIL import Image

class MPIIDataGen(object):

    def __init__(self, jsonfile, imgpath, inres, outres, is_train):
        self.jsonfile = jsonfile
        self.imgpath = imgpath
        self.inres = inres
        
        self.outres = outres
        self.is_train = is_train
        self.nparts = 2
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
        print('train/val size:', len(train_anno), len(val_anno))
        if self.is_train:
            return train_anno
        else:
            return val_anno

    def get_dataset_size(self):
        return len(self.anno)

    def get_color_mean(self):
        mean = np.array([0.2013, 0.2013, 0.2013])
        return mean

    def get_annotations(self):
        return self.anno

    def generator(self, batch_size, num_hgstack, sigma_scale=0.05, with_meta=False, is_shuffle=False,
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
                _imageaug, _gthtmap, _meta = self.process_image(i, kpanno, sigma_scale, rot_flag, scale_flag, flip_flag)
                _index = i % batch_size

                train_input[_index, :, :, :] = _imageaug
                gt_heatmap[_index, :, :, :] = _gthtmap
                meta_info.append(_meta)
                if i % batch_size == (batch_size - 1):
                    out_hmaps = []
                    for m in range(num_hgstack):
                        out_hmaps.append(gt_heatmap)

                    if with_meta:
                        yield train_input, out_hmaps, meta_info
                        meta_info = []
                    else:
                        yield train_input, out_hmaps

    def process_image(self, sample_index, kpanno, sigma_scale, rot_flag, scale_flag, flip_flag):
        imagefile = kpanno['img_paths']
        # image = scipy.misc.imread(os.path.join(self.imgpath, imagefile))
        image = np.array(Image.open(os.path.join(self.imgpath, imagefile.replace("\\", "/"))).convert('RGB'))
        
        # get center
        center = np.array(kpanno['objpos'])
        joints = np.array(kpanno['obj_locations'])
        
        joins_old = joints.copy()

        scale = kpanno['scale_provided']
        # Adjust center/scale slightly to avoid cropping limbs


        # filp
        if flip_flag and random.choice([0, 1]):
            image, joints, center = self.flip(image, joints, center)

        joints_c = [
                  [
                    joints[0][0] + (joints[1][0] - joints[0][0])/2,
                    joints[0][1] + (joints[1][1] - joints[0][1])/2,
                    joints[0][2]
                  ],
                  [
                    joints[2][0] + (joints[3][0] - joints[2][0])/2,
                    joints[2][1] + (joints[3][1] - joints[2][1])/2,
                    joints[2][2]
                  ]
        ]
        joints = np.array(joints_c)

        # scale
        if scale_flag:
            scale = scale * np.random.uniform(0.95, 1.05)

        # rotate image
        if rot_flag:
            rot = np.random.randint(-35, 35)
        else:
            rot = 0

        cropimg = data_process.crop(image, center, scale, self.inres, rot)
        cropimg = data_process.normalize(cropimg, self.get_color_mean())
        # transform keypoints
        transformedKps = data_process.transform_kp(joints, center, scale, self.outres, rot)

        # sigmas = [
        #   np.sqrt((joins_old[0][0] -  joins_old[1][0])**2  +  (joins_old[0][1] -  joins_old[1][1])**2),
        #    np.sqrt((joins_old[2][0] -  joins_old[3][0])**2 +  (joins_old[2][1] -  joins_old[3][1])**2),
        # ]

        # sigmas = np.array(sigmas)
        # c = (sigma_scale * (1/scale))
        # sigmas *= c
        # sigmas = sigmas ** 2.5
        
        sigmas = np.array([5.5, 4.5])
        
        gtmap = data_process.generate_gtmap(transformedKps, sigmas, self.outres)
        # meta info
        metainfo = {'sample_index': sample_index, 'center': center, 'scale': scale,
                    'pts': joints, 'tpts': transformedKps, 'name': imagefile}
        return cropimg, gtmap, metainfo

    @classmethod
    def get_kp_keys(cls):
        keys = ['left_kidney', 'right_kidney']
        return keys

    def flip(self, image, joints, center):


        joints = np.copy(joints)

        matchedParts = (
            [0, 1],  # left_kidney
            # [2, 3]  # right_kidney
        )

        org_height, org_width, channels = image.shape

        # flip image
        flipimage = cv2.flip(image, flipCode=1)

        # flip each joints
        joints[:, 0] = org_width - joints[:, 0]

        for i, j in matchedParts:
            temp = np.copy(joints[i, :])
            joints[i, :] = joints[j, :]
            joints[j, :] = temp

            #TODO flip x coords when flipepd to keep upper left and lower right corners
            # joints[i][0],joints[j][0] = joints[j][0], joints[i][0]



        # center
        flip_center = center
        flip_center[0] = org_width - center[0]

        return flipimage, joints, flip_center