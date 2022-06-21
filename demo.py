import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm

from person_detector import PersonDetector

def parse_args():
    parser = argparse.ArgumentParser(
        description='run person detector on example images')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--data_dir', default='demo', help='a directory containing input images')
    parser.add_argument('--threshold', default=0.5, help='the score threshold for detection results')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument('--save_root', default=None, help='a root directory for saving cropped person images')
    args = parser.parse_args()

    return args

def main():
    # get the arguments
    args = parse_args()

    # initialize a detector
    detector = PersonDetector(cfg_path=args.config, ckpt_path=args.checkpoint, is_fuse_conv_bn=args.fuse_conv_bn, gpu_idx=0)

    # list the names of images in the data_dir
    files = [ name for name in os.listdir(args.data_dir) if '.png' in name[-4:].lower() ]
    # load images from the data_dir
    for file in tqdm(files):
        ## load the image
        img = cv2.imread(os.path.join(args.data_dir, file))
        ## set the save_dir for curent image
        save_dir = os.path.join(args.save_root, file[:-4])
        ## run the detector on the image
        dets = detector.detect(img, ths=args.threshold, save_dir=save_dir)

if __name__ == '__main__':
    main()