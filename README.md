# DMTC: Mutual Dynamic Training based on Contrastive learning for semi-supervised semantic segmentation

This repo contains the official implementation of our paper: Mutual Dynamic Training based on Contrastive learning for semi-supervised semantic segmentation, which is  an concise and effective method for semi-supervised learning semantic segmentation. 

## Requirements
This repo was tested with Ubuntu 18.04.3 LTS, Python 3.7, PyTorch 1.7, and CUDA 11.1. we have used automatic mixed precision training which is availabel for PyTorch versions >=1.7 .

The required packages are pytorch and torchvision, together with PIL and opencv for data-preprocessing and tqdm for showing the training progress. To setup the necessary modules, simply run:
```
pip install -r requirements.txt 
```
## Datasets
DMTC is evaluated with tow common datasets in semi-supervised semantic segmentation: **CityScapes** and **PASCAL VOC**. 
* Cityscapes:
  - First download the original dataset from the [official website](https://www.cityscapes-dataset.com/), leftImg8bit_trainvaltest.zip and gtFine_trainvaltest.zip.  then Create and extract them to the corresponding dataset/cityscapes folder. 

* Pascal VOC:
  - First download the [original dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar), after extracting the files we'll end up with VOCtrainval_11-May-2012/VOCdevkit/VOC2012 containing the image sets, the XML annotation for both object detection and segmentation, and JPEG images. The second step is to augment the dataset using the additionnal annotations provided by Semantic Contours from Inverse Detectors. Download the rest of the annotations SegmentationClassAug and add them to the path VOCtrainval_11-May-2012/VOCdevkit/VOC2012, now we're set, for training use the path to VOCtrainval_11-May-2012.


## Pre-trained models
Pre-trained models can be downloaded [here](https://drive.google.com/file/d/1sQ_FmyHrWmqJLCrW9vJTnv61Sp4hn-9m/view?usp=sharing).

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
- [ ] prepare pretrained models
- [ ] 
- [ ] 

## Acknowledgements

The overall implementation is based on TorchVision and PyTorch and DMT github repo.

The contrastive part is adapted from ReCo.
