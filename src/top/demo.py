import sys

sys.path.insert(0, "../data_gen/")
sys.path.insert(0, "../net/")
sys.path.insert(0, "../eval/")

import os
import numpy as np
import scipy.misc
from heatmap_process import post_process_heatmap
from hourglass import HourglassNet
import argparse
from pckh import run_pckh
from mpii_datagen import MPIIDataGen
import cv2


def render_joints(cvmat, joints, conf_th=0.2):
    for _joint in joints:
        _x, _y, _conf = _joint
        if _conf > conf_th:
            cv2.circle(cvmat, center=(int(_x), int(_y)), color=(255, 0, 0), radius=7, thickness=2)

    return cvmat

def inference_folder(model_json, model_weights, num_stack, num_class, input_folder, output_folder, confth):
    # xnet = HourglassNet(num_classes=num_class, num_stacks=args.num_stack, num_channels=256, inres=(256, 256),
    #                         outres=(64, 64))
    # xnet.load_model(model_json, model_weights)

    for path, _, files in os.walk(input_folder):
        for file in files:
            pathToFile = os.path.join(path, file)
            
            # out, scale = xnet.inference_file(pathToFile)
            print("inverence", pathToFile)
            # kps = post_process_heatmap(out[0, :, :, :])
            
            # kp_keys = MPIIDataGen.get_kp_keys()
            # mkps = list()
            # for i, _kp in enumerate(kps):
            #     _conf = _kp[2]
            #     mkps.append((_kp[0] * scale[1] * 4, _kp[1] * scale[0] * 4, _conf))

            # cvmat = render_joints(cv2.imread(imgfile), mkps, confth)
            out_file = os.path.join(output_folder, file)
            print('pred', out_file)
            # cv2.imwrite(out_folder, cvmat)


def main_inference(model_json, model_weights, num_stack, num_class, imgfile, confth, tiny):
    if tiny:
        xnet = HourglassNet(num_classes=num_class, num_stacks=args.num_stack, num_channels=128, inres=(192, 192),
                            outres=(48, 48))
    else:
        xnet = HourglassNet(num_classes=num_class, num_stacks=args.num_stack, num_channels=256, inres=(256, 256),
                            outres=(64, 64))

    xnet.load_model(model_json, model_weights)

    out, scale = xnet.inference_file(imgfile)
    kps = post_process_heatmap(out[0, :, :, :])

    ignore_kps = ['plevis', 'thorax', 'head_top']
    kp_keys = MPIIDataGen.get_kp_keys()
    mkps = list()
    for i, _kp in enumerate(kps):
        _conf = _kp[2]
        mkps.append((_kp[0] * scale[1] * 4, _kp[1] * scale[0] * 4, _conf))

    cvmat = render_joints(cv2.imread(imgfile), mkps, confth)

    cv2.imwrite('demo.png', cvmat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpuID", default=0, type=int, help='gpu id')
    parser.add_argument("--model_json", help='path to store trained model')
    parser.add_argument("--model_weights", help='path to store trained model')
    parser.add_argument("--num_stack", type=int, help='num of stack')
    parser.add_argument("--input_folder", help='input image folder')
    parser.add_argument("--output_folder", help='output image folder')
    parser.add_argument("--conf_threshold", type=float, default=0.2, help='confidence threshold')
    parser.add_argument("--tiny", default=False, type=bool, help="tiny network for speed, inres=[192x128], channel=128")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuID)
    inference_folder(model_json=args.model_json, 
                     model_weights=args.model_weights, 
                     num_stack=args.num_stack,
                     num_class=4,
                     input_folder=args.input_folder, 
                     output_folder = args.output_folder,
                     confth=args.conf_threshold)
    # main_inference(model_json=args.model_json, model_weights=args.model_weights, num_stack=args.num_stack,
    #                num_class=4, imgfile=args.input_image, confth=args.conf_threshold, tiny=args.tiny)
