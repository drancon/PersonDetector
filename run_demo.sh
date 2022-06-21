python demo.py \
./configs/swin/mask_rcnn_swin-s.py \
./ckpts/mask_rcnn_swin-s.pth \
--data_dir ./demo \
--fuse-conv-bn \
--save_root ./demo/results