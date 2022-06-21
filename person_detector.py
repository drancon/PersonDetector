import os
import warnings
import json
import cv2
import numpy as np

import mmcv
import torch
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmdet.apis import inference_detector
from mmdet.models import build_detector

class PersonDetector():
    """ Person Detector based on Swin Transformer(backbone) + Mask R-CNN(detection pipeline)
    Initialize the person detection model based on Swin Transformer
    and crop out the detected people from the input image
    
    Args:
        cfg_path (str): path to the configuration file of object detection model, the configuration follows the format of backend packages(mmcv and mmdet)
        ckpt_path (str): path to the checkpoint(traiend weights) of the model
        is_fuse_conv_bn (bool): a bool variable to determine if to fuse conv and bn, this will slightly increase the inference speed
        gpu_idx (int): gpu index for running this model
    """
    def __init__(self, cfg_path, ckpt_path, is_fuse_conv_bn=True, gpu_idx=0):
        # check if input variables are valid
        if is_fuse_conv_bn is not None:
            if not isinstance(is_fuse_conv_bn, bool):
                raise ValueError('The input variable "is_fuse_conv_bn" must be a bool.')
        if not isinstance(gpu_idx, int):
            raise ValueError('The input variable "gpu_idx" must be an integer.')

        # store the input variables into class variables
        self.cfg = Config.fromfile(cfg_path)
        self.ckpt_path = ckpt_path
        self.is_fuse_conv_bn = is_fuse_conv_bn
        self.device = torch.device("cuda:%d"%gpu_idx if torch.cuda.is_available() else "cpu")
        # initialize the detector model
        self._init_model()

    def _init_model(self):
        """ Initialize the detection model and load checkpoint file """
        # build detector
        self.cfg.model.train_cfg = None
        model = build_detector(self.cfg.model)
        # convert the datatype of weights in the detector into fp16
        fp16_cfg = self.cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        # load the saved model weights
        checkpoint = load_checkpoint(model, self.ckpt_path)
        if self.is_fuse_conv_bn:
            model = fuse_conv_bn(model)

        # old versions did not save class info in checkpoints, 
        # this walkaround is for backward compatibility
        model.CLASSES = checkpoint['meta']['CLASSES']
        # save the config in the model for convenience
        model.cfg = self.cfg
        # transfer the model into gpu
        model.to(self.device)
        # set the model as evaluation mode
        model.eval()

        # store the initialized model into class variable
        self.model = model

    def detect(self, img, ths=0.5, min_size=(30,80), save_dir=None, save_ext='png'):
        """ Detect person from the input image and crop out the detected people

        Args:
            img (numpy array, shape:[H, W, 3]): an input image
            ths (float, range:[0, 1]): a threshold value for prediction scores of detected boxes
            min_size (tuple, (W_min, H_min)): the minimum size of cropped image, detection boxes smaller than these values are discarded
            save_dir (str): a directory for saving the cropped images
            save_ext (str): the file extension of cropped images to be saved

        Returns:
            person_imgs (list): a list that contains images(numpy array, shape:[h, w, 3]) of person cropped out of the input image
        """
        # check if input variables are valid
        if ths < 0. or ths > 1.:
            raise ValueError('The input variable "ths" should be in the range of [0, 1].')
        if save_dir is not None:
            if not isinstance(save_dir, str):
                raise ValueError('The input variable "save_dir" should be in the type of "str" or "None".')
        if save_ext.lower() not in ['png', 'jpg', 'jpeg', 'bmp']:
            raise ValueError('The input variable "save_ext" should be either one of "png", "jpg", "jpeg", or "bmp".')

        # get the size of input image
        H, W, _ = img.shape
        # check if the minimum size input is valid
        if min_size[0] < 1 or min_size[0] > W:
            raise ValueError('The minimum width of a cropped image is given as %d. It should be smaller than the input image width %d'%(min_size[0], W))
        if min_size[1] < 1 or min_size[1] > H:
            raise ValueError('The minimum height of a cropped image is given as %d. It should be smaller than the input image height %d'%(min_size[1], H))

        # detect objects in the input image
        results = inference_detector(self.model, img)
        # extract detection results for the class "person"
        boxes = results[0][0]
        # initialize a list to store cropped person images
        person_imgs = []
        # crop out detected people from the input image
        for box in boxes:
            # if the prediction score of current detection result is under the threshold, 
            # just skip the current detection result
            if box[4] < ths:
                continue
            # get upper left, bottom right coordinates of box
            x1, y1, x2, y2 = box[:4]
            # if the size of current detection box is smaller than the minimum box size,
            # skip the current detection box
            if (x2 - x1) < min_size[0] or (y2 - y1) < min_size[1]:
                continue
            # convert the coordinates into integer indices,
            # and check if they are inside the image
            x1 = min(max(int(x1), 0), W-1)
            y1 = min(max(int(y1), 0), H-1)
            x2 = min(max(int(x2), 0), W-1)
            y2 = min(max(int(y2), 0), H-1)
            # crop out the detection box from the input image
            person_img = img[y1:y2, x1:x2]
            # put the cropped image into the image list
            person_imgs.append(person_img)

        # save the cropped images
        if save_dir is not None:
            # if the directory does not exist, create the directory
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # save the images into the directory
            for idx, img in enumerate(person_imgs):
                cv2.imwrite(os.path.join(save_dir, "%d.%s"%(idx, save_ext)), img)
        return person_imgs