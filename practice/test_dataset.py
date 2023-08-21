# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/08/17 14:10:49
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : tmp.py

import io
import os
import pandas as pd
import numpy as np
from torch.utils import data

# from __future__ import print_function, division
import os
import torch
import pandas as pd              #用于更容易地进行csv解析
from skimage import io, transform    #用于图像的IO和变换
# import skimage
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


# 忽略警告
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

#1.读书数据集信息

landmarks_frame = pd.read_csv('/home/zzq/CreatingSpace/BackdoorToolbox/datasets/faces/faces/face_landmarks.csv')

n = 65
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:].values
landmarks = landmarks.astype('float').reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))

#2.展示一张图片和它对应的标注点作为例子
def show_landmarks(image, landmarks):
    """显示带有地标的图片"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

plt.figure()

show_landmarks(io.imread(os.path.join('/home/zzq/CreatingSpace/BackdoorToolbox/datasets/faces/faces/', img_name)),
               landmarks)
plt.show()

#3.自定义数据集类并遍历
class FaceLandmarksDataset(data.Dataset):
    """面部标记数据集."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        csv_file（string）：带注释的csv文件的路径。
        root_dir（string）：包含所有图像的目录。
        transform（callable， optional）：一个样本上的可用的可选变换
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

face_dataset = FaceLandmarksDataset(csv_file='/home/zzq/CreatingSpace/BackdoorToolbox/datasets/faces/faces/face_landmarks.csv',
                                    root_dir='/home/zzq/CreatingSpace/BackdoorToolbox/datasets/faces/faces/',transform = transforms.ToTensor())

fig = plt.figure()


for i in range(len(face_dataset)):
    sample = face_dataset[i]
    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break

#4.使用pytorch的DataLoader 来进行batch-size遍历

from torch.utils.data import DataLoader 
from torch.utils.data import TensorDataset

loader= DataLoader(face_dataset, batch_size=10, shuffle=True, sampler=None, 
                    num_workers=5, pin_memory=False, drop_last=False)
for epoch in range(3):   # 训练所有!整套!数据 3 次
    for step, (batch_x, batch_y) in enumerate(loader): 
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())
        
#5.在DataLoader使用进行min-batch遍历时，设置采样器
from torch.utils.data import sampler
# class torch.utils.data.sampler.SequentialSampler(data_source)
# 样本元素顺序排列，始终以相同的顺序。

# 参数： - data_source (Dataset) – 采样的数据集。

# class torch.utils.data.sampler.RandomSampler(data_source)
# 样本元素随机，没有替换。

# 参数： - data_source (Dataset) – 采样的数据集。

# class torch.utils.data.sampler.SubsetRandomSampler(indices)
# 样本元素从指定的索引列表中随机抽取，没有替换。

# 参数： - indices (list) – 索引的列表

# class torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples, replacement=True)
# 样本元素来自于[0,..,len(weights)-1]，给定概率（weights）。

# 参数： - weights (list) – 权重列表。没必要加起来为1 - num_samples (int) – 抽样数量

#6.数据变换和组合变换
class Rescale(object):
    """将样本中的图像重新缩放到给定大小。.

    Args:
        output_size（tuple或int）：所需的输出大小。 如果是元组，则输出为
         与output_size匹配。 如果是int，则匹配较小的图像边缘到output_size保持纵横比相同。
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}





#7.通用的数据加载器ImageFolder定义数据集对象
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# 对数据进行随机旋转和裁剪
train_transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = datasets.ImageFolder('./train', transform=train_transform)














