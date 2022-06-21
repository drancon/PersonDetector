# PersonDetector
detect people and crop out the detected people's images.  
The detection model is made based on SwinTransformer(https://github.com/SwinTransformer/Swin-Transformer-Object-Detection).  
This repository uses mask R-CNN object detection module with Swin Transformer backbone provided by MMDetection(https://github.com/open-mmlab/mmdetection/tree/master/configs/swin).
# Pretrained Weights
- download *.pth file from https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth
- move the downloaded *.pth file to ckpts/mask_rcnn_swin-s.pth
