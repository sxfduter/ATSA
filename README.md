# ASTA
This codebase implements the system described in the paper:

Asymmetric Two-Stream Architecture for Accurate RGB-D Saliency Detection

Miao Zhang, Sun Xiao Fei, Jie Liu, Shuang Xu, [Yongri Piao](http://ice.dlut.edu.cn/yrpiao/), [Huchuan Lu](http://ice.dlut.edu.cn/lu/publications.html).
In ECCV 2020.

# Prerequisites
+ Ubuntu 18
+ PyTorch 1.0
+ CUDA 9.0
+ Cudnn 7.6.0
+ Python 3.6.5
+ Numpy 1.16.4

# Training and Testing Datasets

## Training dataset
[Download Link](). Code: 0fj8

## Testing dataset
[Download Link](). Code: f7vk

# Train/Test
## test
Once the TestData are prepared, you need to set dataset path and checkpoint name correctly. and set the param '--phase' as "test" and '--param' as 'True' in demo.py. 
```shell
python demo.py
```
## train
Once the train-augment dataset are prepared,you need to set dataset path and checkpoint name correctly. and set the param '--phase' as "train" and '--param' as 'True'(loading checkpoint) or 'False'(no loading checkpoint) in demo.py. 

```shell
python demo.py
```

# Contact Us
If you have any questions, please contact us (xiaofeisun@mail.dlut.edu.cn).


