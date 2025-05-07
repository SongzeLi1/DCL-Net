# MVSS-Net

**Code and models for ICCV 2021 paper: *Image Manipulation Detection by Multi-View Multi-Scale Supervision***

![Image text](https://raw.githubusercontent.com/dong03/picture/main/framework.jpg)
### Update
- ***22.02.17***, Pretrained model for [Real-World Image Foregery Localization Challange](https://tianchi.aliyun.com/competition/entrance/531945/introduction)

To Be Done.
- ***21.12.17***,  Something new: MVSS-Net++

We now have an improved version of MVSS-Net, denoted as MVSS-Net++. Check [here](mvssnetplus.md).

### Environment

+ Ubuntu 16.04.6 LTS
+ Python 3.6
+ cuda10.1+cudnn7.6.3

  

### Requirements
+ Install [nvidia-apex](https://github.com/NVIDIA/apex) and move it to current directory.
+ pip install [requirements.txt](requirements.tx)



### Usage
```
train_demo.py (training binary positioning network, tampering and desensitizing to the same category)

infer_mvss_calmetrics.py (testing the generation of positioning probability maps and binary maps)

train_demo_nclass3.py (training three-class positioning network)

infer_3labels_merge_calmetrics.py (testing the generation of three-class positioning probability maps and prediction maps, and calculating the tampering positioning and desensitization positioning indicators)

train_demo_cls.py (training classification network)

infer_mvss_cls.py (testing classification)
```

### Citation
If you find this work useful in your research, please consider citing:
```
@InProceedings{MVSS_2021ICCV,  
author = {Chen, Xinru and Dong, Chengbo and Ji, Jiaqi and Cao, juan and Li, Xirong},  
title = {Image Manipulation Detection by Multi-View Multi-Scale Supervision},  
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},  
year = {2021}  
}
```

### Acknowledgments
- [Nvidia-apex](https://github.com/NVIDIA/apex) is adopted for semi-precision training/inferencing.
- The implement of DA module is based on the  [awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch).
### Contact

If you enounter any issue when running the code, please feel free to reach us either by creating a new issue in the github or by emailing

+ Xinru Chen (chen_xinru1999@163.com)
+ Chengbo Dong (dongchengbo@ruc.edu.cn)
