# AFMNet
PyTorch implementation of the paper: Point Cloud Registration based on Adaptively Fused Multimodal Features
## Introduction
Point cloud registration is a fundamental task in 3D vision, which plays an important role in various fields but faces challenges in geometrically weak or repetitive scenes. Traditional geometric-based methods struggle in these cases, while recent multimodal approaches improve robustness in weak scenes but rely on precise point cloud-image alignment, which is difficult in real-world, low-alignment environments. To address the above challenge, we propose the Adaptively Fused Multimodal Network (AFMNet). AFMNet establishes point-to-pixel correspondences at the sparse superpoint level and assigns weights to geometric and texture features, creating more distinct feature descriptors and reducing the impact of misalignment. Additionally, we introduce an image-guided confidence estimation strategy that assigns higher confidence levels to points within the alignment region, prioritizing their selection for registration. To better evaluate the robustness of point cloud registration methods in geometrically weak scenes, we build new benchmarks, 3DWeakMacth and 3DLoWeakMatch, based on 3DMatch and 3DLoMatch. Reasonable multimodal fusion enables our method to achieve state-of-the-art performance on both indoor 3DMatch, 3DLoMatch, 3DWeakMacth, and 3DLoWeakMatch benchmarks, as well as the outdoor KITTI benchmark with low alignment.
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
Our constructed dataset can be downloaded from [here](https://drive.google.com/file/d/15O5VsbYPLLQOzxtdf1NMMpIdTzmt-FBE/view?usp=drive_link).
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
