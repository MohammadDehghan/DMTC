# DMTC: Dynamic Mutual Training based on Contrastive learning for semi-supervised semantic segmentation

This repo contains the official implementation of our paper: Dynamic Mutual Training based on Contrastive learning for semi-supervised semantic segmentation, which is a concise and effective method for semi-supervised learning semantic segmentation. 

![GitHub Logo](/DMTC_pics/DMTC_diagram.png)

## Results
On Cityscapes:
Labeled data | Architecture | Backbone | mIoU
------------ | ------------- | ------------ | -------------
Full (2975) | DeepLabV2 | ResNet-101 | 68.9
1/4 (744) | DeepLabV2 | ResNet-101 | 66.37
1/8 (372) | DeepLabV2 | ResNet-101 | 65.33
1/30 (100) | DeepLabV2 | ResNet-101 | 59.54

On PASCAL VOC:
Labeled data | Architecture | Backbone | mIoU
------------ | ------------- | ------------ | -------------
Full (10582) | DeepLabV2 | ResNet-101 | 75.50
1/8 | DeepLabV2 | ResNet-101 | 72.50
1/20 | DeepLabV2 | ResNet-101 | 70.2
1/50 | DeepLabV2 | ResNet-101 | 68.1
1/100 | DeepLabV2 | ResNet-101 | 65.45

## Requirements
This repo was tested with Ubuntu 18.04.3 LTS, Python 3.7, PyTorch 1.7, and CUDA 11.1. we have used automatic mixed precision training which is available for PyTorch versions >=1.7 .

The required packages are PyTorch  and Torchvision, together with PIL and OpenCV for data-preprocessing and tqdm for showing the training progress. To set up the necessary modules, simply run:
```
pip install -r requirements.txt 
```
## Datasets
DMTC is evaluated with two common datasets in semi-supervised semantic segmentation: **CityScapes** and **PASCAL VOC**. 
* Cityscapes:
  - First download the original dataset from the [official website](https://www.cityscapes-dataset.com/), leftImg8bit_trainvaltest.zip, and gtFine_trainvaltest.zip.  then Create and extract them to the corresponding dataset/cityscapes folder. 

* PASCAL VOC:
  - First download the [original dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar), after extracting the files we'll end up with VOCtrainval_11-May-2012/VOCdevkit/VOC2012 containing the image sets, the XML annotation for both object detection and segmentation, and JPEG images. The second step is to augment the dataset using the additional annotations provided by Semantic Contours from Inverse Detectors. Download the rest of the annotations SegmentationClassAug and add them to the path VOCtrainval_11-May-2012/VOCdevkit/VOC2012, now we're set, for training use the path to VOCtrainval_11-May-2012.

## Run the code
* Set dataset paths:
Set the directories of your datasets [here](https://github.com/MD-IUST/DMTC/blob/main/utils/common.py)
 
For example, for running DMTC on Cityscapes dataset with 1/8 total labeled data:
```
./dmtc-city-8-1.sh
```

## Pre-trained weights 
Pre-trained weights  can be downloaded [here](https://drive.google.com/file/d/1sQ_FmyHrWmqJLCrW9vJTnv61Sp4hn-9m/view?usp=sharing).

## Citation :pencil:
If you find this repo/article useful for your research, please consider citing the paper as follows (we will put it here):
```
@article{xxxxxx,
  title={DMTC: Dynamic Mutual Training based on Contrastive learning for Semi-Supervised Semantic Segmentation},
  author={},
  journal={},
  year={}
}
```
## To do list:
- [ ] Prepare pre-trained weights 
- [ ] 
- [ ] 

## Contact:
If you have any questions which can not find in Google, feel free to contact us via m.dehghan9975@gmail.com.

## Acknowledgements

The overall implementation is based on TorchVision and PyTorch and DMT github repo.

The contrastive part is adapted from ReCo.
