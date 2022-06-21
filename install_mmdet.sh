CUDA_VERSION=cu102
TORCH_VERSION=torch1.8.0

pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/$CUDA_VERSION/$TORCH_VERSION/index.html
pip install mmdet