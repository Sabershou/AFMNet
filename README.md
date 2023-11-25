# AFMNet
PyTorch implementation of the paper: Point Cloud Registration based on Adaptively Fused Multimodal Features
## Introduction
Our research focuses on point cloud registration in scenes with weak geometric structures and the challenge of reproducible structural patches. Recent point cloud registration methods have mainly focused on learning geometric features, while neglecting semantic information. However, in situations where there is insufficient structural information within the scene, semantic information can help us find the exact correspondence between point clouds. For this, we propose an **A**daptively **F**used **M**ultimodal method (AFMNet). This method assigns weights to both geometric and semantic features, resulting in more distinct feature descriptors. Additionally, we introduce a multilevel joint point pair filtering method that utilizes semantic information to select more accurate correspondence, further improving the alignment process. Our approach achieves a FMR at the state-of-the-art level on 3DMatch and 3DLoMatch benchmarks.
![](/data/overview.png)
## Installation
Please use the following command for installation.
```
# It is recommended to create a new environment
conda create -n AFMNet python==3.7
conda activate AFMNet

# If you are using CUDA 10.2 or newer, please install `torch==1.10.0+cu102`
pip install torch==1.10.0+cu102 torchvision==0.11.0+cu102 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install packages and other dependencies
pip install -r requirements.txt
python setup.py build develop
```
Code has been tested with Ubuntu 16.04, Python 3.7, PyTorch 1.10.0, CUDA 10.2.

## 3DMatch
### Dataset
The point cloud dataset can be downloaded from [PREDATOR](https://github.com/prs-eth/OverlapPredator), and the image dataset can be downloaded from [3DMatch](https://3dmatch.cs.princeton.edu/).
Our constructed dataset can be downloaded from [here]().
The data should be organized as follows:
```
--data--3DMatch--metadata
              |--data--train--7-scenes-chess--camera-intrinsics.txt
                    |      |               |--cloud_bin_0.pth
                    |      |               |--cloud_bin_0.info.txt
                    |      |               |--cloud_bin_0_color.png
                    |      |--...          |--...
                    |--test--7-scenes-redkitchen--camera-intrinsics.txt
                          |                    |--cloud_bin_0.pth
                          |                    |--cloud_bin_0.info.txt
                          |                    |--cloud_bin_0_color.png
                          |                    |--...
                          |--...
```
### Training
The code for 3DMatch is in afmnet_3dmatch. Use the following command for training.
```
python trainval.py
```
### Testing
Use the following command for testing.
```
# 3DMatch
python test.py --benchmark=3DMatch
python eval.py --benchmark=3DMatch --method=ransac
# 3DLoMatch
python test.py --benchmark=3DLoMatch
python eval.py --benchmark=3DLoMatch --method=ransac
```
## KITTI
Coming soon...
