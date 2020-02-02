import sys

sys.path.insert(0, "../data_gen/")
sys.path.insert(0, "../eval/")

import os
from hg_blocks import create_hourglass_network, euclidean_loss, bottleneck_block, bottleneck_mobile
from mpii_datagen import MPIIDataGen
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.models import load_model, model_from_json
from keras.optimizers import Adam, RMSprop
from keras.losses import mean_squared_error
import datetime
import scipy.misc
from data_process import normalize
import numpy as np
from eval_callback import EvalCallBack
from keras.callbacks import ReduceLROnPlateau
import keras.backend as K
from keras.callbacks import TensorBoard
from PIL import Image


class HourglassNet(object):

    def __init__(self, num_classes, num_stacks, num_channels, inres, outres):
        self.num_classes = num_classes
        self.num_stacks = num_stacks
        self.num_channels = num_channels
        self.inres = inres
        self.outres = outres

    def build_model(self, mobile=False, show=False):
        if mobile:
            self.model = create_hourglass_network(self.num_classes, self.num_stacks,
                                                  self.num_channels, self.inres, self.outres, bottleneck_mobile)
        else:
            self.model = create_hourglass_network(self.num_classes, self.num_stacks,
                                                  self.num_channels, self.inres, self.outres, bottleneck_block)
        # show model summary and layer name
        if show:
            self.model.summary()

    def train(self, batch_size, model_path, epochs):
        train_dataset = MPIIDataGen("../../data/mpii/mpii_annotations.json", "../../data/mpii/images",
                                    inres=self.inres, outres=self.outres, is_train=True)
        train_gen = train_dataset.generator(batch_size, self.num_stacks, sigma=2, is_shuffle=True,
                                            rot_flag=True, scale_flag=True, flip_flag=False, with_meta=False)
        
        filename = os.path.join(model_path, "csv_train_" + str(datetime.datetime.now().strftime('%H:%M')) + ".csv")
        open(filename,'w+').close()
        csvlogger = CSVLogger(filename)
        modelfile = os.path.join(model_path, 'weights_{epoch:02d}_{loss:.2f}.hdf5')
        
        open(modelfile,'w+').close()
        checkpoint = EvalCallBack(model_path, self.inres, self.outres)

        lr_reducer = ReduceLROnPlateau(monitor='loss', 
                    factor=0.8,
                    patience=5, 
                    verbose=1,
                    cooldown=10,
                    min_lr=0.0001,
                    mode='auto')
                
        logdir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=logdir)
        
        xcallbacks = [csvlogger, checkpoint, lr_reducer, tensorboard_callback]

        self.model.fit_generator(generator=train_gen, steps_per_epoch=(train_dataset.get_dataset_size() // batch_size)*2,
                                 epochs=epochs, callbacks=xcallbacks)



    def resume_train(self, batch_size, model_json, model_weights, init_epoch, epochs):

        self.load_model(model_json, model_weights)
        self.model.compile(optimizer=Adam(lr=0.01), loss=mean_squared_error, metrics=["accuracy"])

        train_dataset = MPIIDataGen("../../data/mpii/mpii_annotations.json", "../../data/mpii/images",
                                    inres=self.inres, outres=self.outres, is_train=True)

        train_gen = train_dataset.generator(batch_size, self.num_stacks, sigma=2, is_shuffle=True,
                                            rot_flag=True, scale_flag=True, flip_flag=False)

        model_dir = os.path.dirname(os.path.abspath(model_json))
        csvlogger = CSVLogger(
            os.path.join(model_dir, "csv_train_" + str(datetime.datetime.now().strftime('%H:%M')) + ".csv"))

        lr_reducer = ReduceLROnPlateau(monitor='loss', 
                    factor=0.9,
                    patience=10, 
                    verbose=1,
                    cooldown=10,
                    min_lr=0.0001,
                    mode='auto')
        
        checkpoint = EvalCallBack(model_dir, self.inres, self.outres)
        
        logdir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=logdir)
        
        xcallbacks = [csvlogger, checkpoint, lr_reducer, tensorboard_callback]

        self.model.fit_generator(generator=train_gen, steps_per_epoch=(train_dataset.get_dataset_size() // batch_size)*5,
                                 initial_epoch=init_epoch, epochs=epochs, callbacks=xcallbacks)

    def load_model(self, modeljson, modelfile):
        with open(modeljson) as f:
            self.model = model_from_json(f.read())
        self.model.load_weights(modelfile)

    def inference_rgb(self, rgbdata, orgshape, mean=None):

        scale = (orgshape[0] * 1.0 / self.inres[0], orgshape[1] * 1.0 / self.inres[1])
        imgdata = scipy.misc.imresize(rgbdata, self.inres)

        if mean is None:
            # mean = np.array([0.4404, 0.4440, 0.4327], dtype=np.float) #todo
            mean = np.array([0.1804, 0.1804, 0.1804])

        imgdata = normalize(imgdata, mean)

        input = imgdata[np.newaxis, :, :, :]

        out = self.model.predict(input)
        return out[-1], scale #todo pre 1 stack

    def inference_file(self, imgfile, mean=None):
        # imgdata = scipy.misc.imread(imgfile)
        imgdata = np.array(Image.open(imgfile).convert('RGB'))
        ret = self.inference_rgb(imgdata, imgdata.shape, mean)
        return ret
