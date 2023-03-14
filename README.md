# BackRazor For Segmentation

## Environment Setting

1. Install the packages required by [backRazor](https://github.com/VITA-Group/BackRazor_Neurips22)
2. Install packages
```bash
pip install visdom matplotlib

# install actnn
git clone git@github.com:ucbrise/actnn.git
cd actnn/actnn
pip install -v -e .
```

3. Prepare datasets
* Standard Pascal VOC
You can run train.py with "--download" option to download and extract pascal voc dataset. The defaut path is './datasets/data':

```
/datasets
    /data
        /VOCdevkit 
            /VOC2012 
                /SegmentationClass
                /JPEGImages
                ...
            ...
        /VOCtrainval_11-May-2012.tar
        ...
```

*  Pascal VOC trainaug: 
*./datasets/data/train_aug.txt* includes the file names of 10582 trainaug images (val images are excluded). Please to download their labels from [Dropbox](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0) or [Tencent Weiyun](https://share.weiyun.com/5NmJ6Rk). Those labels come from [DrSleep's repo](https://github.com/DrSleep/tensorflow-deeplab-resnet).

Extract trainaug labels (SegmentationClassAug) to the VOC2012 directory.

```
/datasets
    /data
        /VOCdevkit  
            /VOC2012
                /SegmentationClass
                /SegmentationClassAug  # <= the trainaug labels
                /JPEGImages
                ...
            ...
        /VOCtrainval_11-May-2012.tar
        ...
```

## Runing cmds

Baseline
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --model deeplabv3_mobilenet --vis_port 23632 --gpu_id 0 --year 2012_aug --crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16
```

BackRazor
```bash
python main.py --model deeplabv3_mobilenet --vis_port 23632 --gpu_id 1 --year 2012_aug \
--crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --backRazorR 0.7
```

## Acknowledge
The partial code of this implement comes from [DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch)

## Cite
```
@inproceedings{
jiang2022back,
title={Back Razor: Memory-Efficient Transfer Learning by Self-Sparsified Backpropogation},
author={Jiang, Ziyu and Chen, Xuxi and Huang, Xueqin and Du, Xianzhi and Zhou, Denny and Wang, Zhangyang},
booktitle={Advances in Neural Information Processing Systems 36},
year={2022}
}
```