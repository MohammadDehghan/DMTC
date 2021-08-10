# DMTC: Mutual Dynamic Training based on contrastive learning for semi-supervised semantic segmentation

This repo contains the official implementation of our paper: Mutual Dynamic Training based on contrastive learning for semi-supervised semantic segmentation, which is  an concise and effective method for semi-supervised learning semantic segmentation. 

# Requirements
This repo was tested with Ubuntu 18.04.3 LTS, Python 3.7, PyTorch 1.7, and CUDA ?. we have used automatic mixed precision training which is availabel for PyTorch versions >=1.7 .

The required packages are pytorch and torchvision, together with PIL and opencv for data-preprocessing and tqdm for showing the training progress. To setup the necessary modules, simply run:

# Datasets
DMTC is evaluated with tow common datasets: CityScapes and PASCAL VOC. The Cityscapes dataset can be downloaded in their [official website](https://pages.github.com/).
