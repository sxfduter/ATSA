# ASTA
This codebase implements the system described in the paper:

Asymmetric Two-Stream Architecture for Accurate RGB-D Saliency Detection

Miao Zhang, Sun Xiao Fei, Jie Liu, Shuang Xu, [Yongri Piao](http://ice.dlut.edu.cn/yrpiao/), [Huchuan Lu](http://ice.dlut.edu.cn/lu/publications.html).
In ECCV 2020.

# Prerequisites
+ Ubuntu 18
+ PyTorch 1.3.1
+ CUDA 10.1
+ Cudnn 7.5.1
+ Python 3.7
+ Numpy 1.17.3

# Training and Testing Datasets

## Training dataset
[Download Link](https://pan.baidu.com/s/1rduZEEo3HRq5HqQeXxuX-A ). Code: nx8x

## Testing dataset
[Download Link](). Code: f7vk

# Train/Test
## test
Firstly, you need to download the 'Testing dataset' and the pretraind checpoint we provided ([Baidu Pan](https://pan.baidu.com/s/1xPH1AzInc1JAMq4Vq7UxGg). Code: d2o0). Then, you need to set dataset path and checkpoint name correctly. and set the param '--phase' as "test" and '--param' as 'True' in demo.py. 
```shell
python demo.py
```
## train
Once the train-augment dataset are prepared,you need to set dataset path and checkpoint name correctly. and set the param '--phase' as "train" and '--param' as 'True'(loading checkpoint) or 'False'(not loading checkpoint) in demo.py. 

```shell
python demo.py
```

# Contact Us
If you have any questions, please contact us (xiaofeisun@mail.dlut.edu.cn; 1605721375@mail.dlut.edu.cn).


