# Overview

The goal of this workshop is to set up a new GPU-enabled cloud instance and run this set of scripts to fine-tune a Mask R-CNN model to detect people and run predictions on new data from image or video sources. All of the necessary materials including the training dataset are included.

The training and evaluation script is based on this guide from PyTorch: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
That guide uses the Penn-Fudan Database, which is included in this repository: https://www.cis.upenn.edu/~jshi/ped_html/

## Instructions

This will first require a GPU-enabled system with CUDA drivers installed

1. From an SSH terminal, Install git and miniconda
   
```
sudo yum install git
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Hit enter to scroll down to the end and then type 'yes'
Confirm the default location
At the end of the installation type 'yes'
If you didn't say yes at the end type ~/miniconda3/bin/conda init
Exit and SSH into the instance again to have miniconda loaded

2. Install the necessary packages to run the scripts
```
pip install cython
pip install matplotlib
pip install numpy==1.17.0
pip install opencv-python
pip install pafy
pip install pycocotools
pip install youtube-dl
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```
3. Clone this repository, run the training, and run the evaluation against new data

```
git clone https://github.com/BlueCloudDev/gpu-workshop.git
python train.py
```
To run using CPU for comparison: 
```
python train.py --device=cpu --epochs=1
```
Evaluate new data. Change the value for source to image URLs or youtube URLs
```
python evaluate.py --source=https://i.imgur.com/E6kSIKJ.jpg
```
