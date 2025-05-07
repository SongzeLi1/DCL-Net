# Image Tampering Localization Using a Dense Fully Convolutional Network

# Overview
This is the implementation of the method proposed in "Image Tampering Localization Using a Dense Fully Convolutional Network" with tensorflow(1.10.0, gpu version). The aim of this repository is to achieve image tampering localization.

## Usage
```
train.py (training binary positioning network, tampering and desensitizing to the same category)

test_withoutComputeMetrics.py (testing the generation of positioning probability map and binary map)

train_densefcn_nclass3.py (training three-class positioning network)

infer_3labels_merge_calmetrics.py (testing the generation of three-class positioning probability map and prediction map, and calculating the tampering positioning and desensitization positioning indicators)

train_crop_denseFCN_cls.py (training classification network)

infer_denseFCN_cls.py (testing classification)
```

# Citation
If you use our code please cite: 
```
@ARTICLE{9393396,  author={P. {Zhuang} and H. {Li} and S. {Tan} and B. {Li} and J. {Huang}},  
journal={IEEE Transactions on Information Forensics and Security},   
title={Image Tampering Localization Using a Dense Fully Convolutional Network},   
year={2021},  
volume={16},  
number={},  
pages={2986-2999},  
doi={10.1109/TIFS.2021.3070444}}
```
