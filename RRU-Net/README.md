## Requirements
- Python 3.7
- PyTorch 1.0+ 
- CUDA 10.0+

## Details
 - './unet/unet-parts.py': it includes detailed implementations of 'U-Net', 'RU-Net' and 'RRU-Net'
 - 'train.py': you can use it to train your model
 - 'predict.py': you can use it to test

## Citation
Please add following information if you cite the paper in your publication:
```shell
@inproceedings{bi2019rru,
  title={RRU-Net: The Ringed Residual U-Net for Image Splicing Forgery Detection},
  author={Bi, Xiuli and Wei, Yang and Xiao, Bin and Li, Weisheng},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  pages={0--0},
  year={2019}
}
```

### Usage
```
train_rru.py (training binary positioning network, tampering and desensitizing to the same category)

infer_rru.py (testing the generation of positioning probability map and binary map)

merge.py (merging into the final positioning probability map and binary map)

train_rru_nclass3.py (training three-class positioning network)

infer_3labels_merge_calmetrics.py (testing the generation of three-class positioning probability map and prediction map, and calculating the tampering positioning and desensitization positioning indicators)

train_crop_rru_cls.py (training classification network)

infer_rru_cls.py (testing classification)
```