# DMTC: Mutual Dynamic Training based on contrastive learning for semi-supervised semantic segmentation

This repo contains the official implementation of our paper: Mutual Dynamic Training based on contrastive learning for semi-supervised semantic segmentation, which is  an concise and effective method for semi-supervised learning semantic segmentation. 

# Requirements
This repo was tested with Ubuntu 18.04.3 LTS, Python 3.7, PyTorch 1.7, and CUDA ?. we have used automatic mixed precision training which is availabel for PyTorch versions >=1.7 .

The required packages are pytorch and torchvision, together with PIL and opencv for data-preprocessing and tqdm for showing the training progress. To setup the necessary modules, simply run:

# Datasets
DMTC is evaluated with tow common datasets in semi-supervised semantic segmentation: **CityScapes** and **PASCAL VOC**. 
* Cityscapes:
  - First download the original dataset from the [official website](https://www.cityscapes-dataset.com/), leftImg8bit_trainvaltest.zip and gtFine_trainvaltest.zip.  then Create and extract them to the corresponding dataset/cityscapes folder. 

* Pascal VOC:
  - First download the original dataset, after extracting the files we'll end up with VOCtrainval_11-May-2012/VOCdevkit/VOC2012 containing the image sets, the XML annotation for both object detection and segmentation, and JPEG images. The second step is to augment the dataset using the additionnal annotations provided by Semantic Contours from Inverse Detectors. Download the rest of the annotations SegmentationClassAug and add them to the path VOCtrainval_11-May-2012/VOCdevkit/VOC2012, now we're set, for training use the path to VOCtrainval_11-May-2012.


# download pre-trained weights:

To downlaod COCO pre-trained weights, run:

# Pre-trained models
Pre-trained models can be downloaded [here](https://www.cityscapes-dataset.com/).

# Citation 
If you find this repo useful for your research, please consider citing the paper as follows:
# Acknowledgements

The overall implementation is based on TorchVision and PyTorch.

Pseudo-labels generation is based on Jiwoon Ahn's implementation
