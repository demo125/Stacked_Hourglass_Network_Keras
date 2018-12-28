import sys

sys.path.insert(0, "../data_gen/")
sys.path.insert(0, "../net/")
sys.path.insert(0, "../eval/")

import os
import numpy as np
import scipy.misc
from mpii_datagen import MPIIDataGen
from eval_heatmap import get_predicted_kp_from_htmap
from hourglass import HourglassNet
import argparse
from pckh import run_pckh
import cv2

def get_final_pred_kps(valkps, preheatmap, metainfo, outres):
    for i in range(preheatmap.shape[0]):
        prehmap = preheatmap[i, :, :, :]
        meta = metainfo[i]
        sample_index = meta['sample_index']
        kps = get_predicted_kp_from_htmap(prehmap, meta, outres)
        valkps[sample_index, :, :] = kps[:, 0:2]  # ignore the visibility

def flip_out_heatmap(flipout):

    # mpii matched parts
    matchedParts = (
        [0, 5], [1, 4], [2, 3],
        [10, 15], [11, 14], [12, 13]
    )

    outmap = np.zeros(flipout.shape, dtype=np.float)

    # flip all of channels
    for i in range(flipout.shape[-1]):
        _map = np.copy(flipout[:, :, i])
        outmap[:, :, i] = cv2.flip(_map, flipCode=1)

    # exchange right-left channels
    for pair in matchedParts:
        tmp = np.copy(outmap[:,  :, pair[0]])
        outmap[:,  :, pair[0]] = outmap[:, :, pair[1]]
        outmap[:,  :, pair[1]] = tmp

    return outmap


def inference_filpped_image(org_images, net):

    flip_images = np.zeros(shape=org_images.shape)
    for i in range(flip_images.shape[0]):
        flip_images[i,:,:,:] = cv2.flip(org_images[i,:,:,:], flipCode=1)

    flip_outputs = net.model.predict(flip_images)
    flip_outputs = flip_outputs[-1]

    flip_back_outputs =  np.zeros(shape=flip_outputs.shape)
    for i in range(flip_back_outputs.shape[0]):
        flip_back_outputs[i, :, :, :] = flip_out_heatmap(flip_outputs[i,:,:,:])

    return flip_back_outputs


def main_eval(model_json, model_weights, num_stack, num_class, matfile, tiny, flip=True):
    inres = (192, 192) if tiny else (256, 256)
    outres = (48, 48) if tiny else (64, 64)
    num_channles = 128 if tiny else 256

    xnet = HourglassNet(num_classes=num_class, num_stacks=num_stack, num_channels=num_channles, inres=inres,
                        outres=outres)

    xnet.load_model(model_json, model_weights)

    valdata = MPIIDataGen("../../data/mpii/mpii_annotations.json", "../../data/mpii/images",
                          inres=inres, outres=outres, is_train=False)

    print 'val data size', valdata.get_dataset_size()

    valkps = np.zeros(shape=(valdata.get_dataset_size(), 16, 2), dtype=np.float)

    count = 0
    batch_size = 6
    for _img, _meta in valdata.generator_val_data(batch_size, sigma=1):

        if count > valdata.get_dataset_size():
            break

        if flip:
            flipout = inference_filpped_image(_img, xnet)
            orgout = xnet.model.predict(_img)[-1]
            out = (flipout + orgout)/2
        else:
            out = xnet.model.predict(_img)
            out = out[-1]

        get_final_pred_kps(valkps, out, _meta, outres)

        count += batch_size

    scipy.io.savemat(matfile, mdict={'preds': valkps})

    run_pckh(model_json, matfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpuID", default=0, type=int, help='gpu id')
    parser.add_argument("--model_json", help='path to store trained model')
    parser.add_argument("--model_weights", help='path to store trained model')
    parser.add_argument("--mat_file", help='path to store trained model')
    parser.add_argument("--num_stack", type=int, help='num of stack')
    parser.add_argument("--tiny", default=False, type=bool, help="tiny network for speed, inres=[192x128], channel=128")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuID)

    main_eval(model_json=args.model_json, model_weights=args.model_weights, matfile=args.mat_file,
              num_stack=args.num_stack, num_class=16, tiny=args.tiny)
