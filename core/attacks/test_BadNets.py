'''
This is the test code of poisoned training under BadNets.
'''
import os.path as osp
'''
OpenCV是一个计算机视觉和机器学习软件库,实现了图像处理和计算机视觉方面的很多通用算法。
http://www.woshicver.com/
'''
import cv2
import torch

'''
nn全称为neural network,意思是神经网络，是torch中构建神经网络的模块。 
该模块包含构建神经网络需要的函数，包括卷积层、池化层、激活函数、损失函数、全连接函数等。
https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/
'''
import torch.nn as nn
'''
torchvision包 包含了目前流行的数据集，模型结构和常用的图片转换工具。
torchvision.datasets
torchvision.models
torchvision.transforms
torchvision.utils:(1)制作雪碧图：把N多个小图标放到一张大图上去；(2)将给定的Tensor保存成image文件
'''
import torchvision
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize
import os
import sys
print(__file__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
# print(sys.path)

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(BASE_DIR)
print(sys.path)

from attacks import BadNets
from models import BaselineMNISTNetwork


# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
CUDA_VISIBLE_DEVICES = '1'
datasets_root_dir = '../datasets'


# ========== BaselineMNISTNetwork_MNIST_BadNets ==========

dataset = torchvision.datasets.MNIST

transform_train = Compose([
    ToTensor()
])
trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)

transform_test = Compose([
    ToTensor()
])
testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)

pattern = torch.zeros((28, 28), dtype=torch.uint8)
pattern[-3:, -3:] = 255
weight = torch.zeros((28, 28), dtype=torch.float32)
weight[-3:, -3:] = 1.0

attack_config ={
    'name': 'BadNets',
    'benign_dataset': None,
    'y_target': 1,
    'poisoned_rate': 0.05,
    'pattern': pattern,
    'weight': weight,
    'poisoned_transform_index': 0,
    'poisoned_transform_test_index': 0,
    'poisoned_target_transform_index': 0
}
# Train Attacked Model (schedule is set by yamengxi)
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 2,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],

    'epochs': 200,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'BaselineMNISTNetwork_MNIST_package'
}
badnets = BadNets(
    attack_config,
    train_dataset=trainset,
    test_dataset=testset,
    model= BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    schedule=schedule,
    seed=global_seed,
    deterministic=deterministic
)
badnets.attack()
