# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/08/19 15:49:57
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : test_AdaptiveBlend.py
# @Description : This is the test code of poisoned training under Adaptive-Blend.
#                The main logic is to define parameters about task, attack strategy and training schedule;
#                If Interaction is needed during training, the tester could define the related functions and the which
#                as parameters passed Badnets.interactions(list).
import os.path as osp
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize
import os
import sys
import numpy as np
from torchvision import transforms 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
# print(sys.path)
from core.attacks import AdaptiveBlend
from core.attacks import BackdoorAttack
from models import ResNet
from models import BaselineMNISTNetwork
import time
import datetime
import os.path as osp
import random
from core.base import Observer
from core.base import Base
from PIL import Image
from utils import Log
from utils import show_img
from utils import save_img
from utils import compute_accuracy
from utils import get_latent_rep
from utils import plot_2d
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn import manifold
import numpy as np
# ========== Set global settings ==========
global_seed = None
deterministic = False
# torch.manual_seed(global_seed)
CUDA_VISIBLE_DEVICES = '1'
datasets_root_dir = os.path.join(BASE_DIR + '/datasets/')
date = datetime.date.today()
work_dir = os.path.join(BASE_DIR,'experiments/ResNet-18_CIFAR-10_No_Attack')
work_dir = os.path.join(work_dir,'datasets')
model_dir = os.path.join(work_dir,'model')
latents_dir = os.path.join(work_dir,"latents")
show_dir = os.path.join(work_dir,"show")
dirs = [work_dir,work_dir,model_dir,latents_dir,show_dir]
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

# ========== ResNet-18_CIFAR-10_No_Attack ==========
# The basic data type in torch is "tensor". In order to be computed, other data type, like PIL Image or numpy.ndarray,
#  must be Converted to "tensor".

dataset = torchvision.datasets.CIFAR10
transform_train = Compose([
    # transforms.Resize((32, 32)),
    ToTensor()
])
trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
transform_test = Compose([
    # transforms.Resize((32, 32)),
    ToTensor()
])
testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
classes = trainset.classes
optimizer = torch.optim.SGD


schedule = {
    'experiment': 'ResNet-18_CIFAR-10_No_Attack',
    # Settings for reproducible/repeatable experiments
    'seed': global_seed,
    'deterministic': deterministic,
    
    # Settings related to device
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    # Settings related to tarining 
    'pretrain': None,
    'epochs': 200,   
    'batch_size': 128,
    'num_workers': 2,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],
    # When this parameter is given,the model is saved after trianed
    'model_path':None,

    # Settings aving model,data and logs
    'work_dir': 'experiments',
    'log_iteration_interval': 100,
}

task = {
    'train_dataset': trainset,
    'test_dataset' : testset,
    'model': ResNet(18),
    'optimizer': optimizer,
    'loss' : nn.CrossEntropyLoss()
}

if __name__ == "__main__":
    
    log = Log(osp.join(work_dir, 'log.txt'))
    experiment = schedule['experiment']
    t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    # msg = "\n\n\n==========Start {0} at {1}==========\n".format(experiment,t)
    # log(msg)
    # # task: ResNet-18 CIFAR-10
    # base = Base(task,schedule)
    # # Show the structure of the model
    # log(task['model'])
   
    # # 1. Train and get model
    # # print(os.path.join(work_dir, 'model/backdoor_model.pth'))
    # log("\n==========Train ResNet-18 on cifar-10 and get model==========\n")
    # base.train()
    # model = base.get_model()
    # torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
    # log("Save model to" + os.path.join(model_dir, 'model.pth'))

    #2.Test model
    """
    确任'ResNet-18_CIFAR-10_No_Attack'实验以下几条效果：
    1.数据集在模型的潜在表示是否由按标签聚类的效应
    """
    # 2.1 The effect of model.

    # Alreadly exsiting trained model
    # model = nn.DataParallel(ResNet(18))
    # base.test(schedule=schedule,model=model,test_dataset=testset)
    # predict_digits, labels = model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')),strict=False)
    # predict_digits, labels = base.test()
    # accuracy = compute_accuracy(predict_digits,labels,topk=(1,3,5))
    # log("Total samples:{0}, accuracy:{1}".format(len(testset),accuracy))

    # #2.2 Verify that the latent representations of the dataset is clustered by label
    log("\n==========Verify that the latent representations of the dataset is clustered by label==========\n")
    
    # #Alreadly exsiting trained model
    # device = torch.device("cuda:0")
    # model = nn.DataParallel(ResNet(18),device_ids=[0],output_device=device)
    # model.to(device)
    # print(model)
    # layer = "linear"
    # latents, y_labels = get_latent_rep(model, layer, trainset, device=device)

    latents_path =os.path.join(latents_dir,"cifar10_train_latents.npz")
    # np.savez(latents_path, latents=latents, y_labels=y_labels)

    #get low-dimensional data points by t-SNE
    data = np.load(latents_path)
    latents,y_labels = data["latents"],data["y_labels"]
    # print(type(latents))
    print(latents.shape)
    # print(latents)

    n_components = 2 # number of coordinates for the manifold
    t_sne = manifold.TSNE(n_components=n_components, perplexity=300, early_exaggeration=120, init="random", n_iter=250, random_state=0)
    points = t_sne.fit_transform(latents)
    # points = points*1000
    print(type(points))
    print(points.shape)
    print(points)
    print(t_sne.kl_divergence_)
    #Display data clusters for all category by scatter plots
    # Custom color mapping
   
    colors = [plt.cm.tab10(i) for i in range(len(classes))]
    # Create a ListedColormap object
    cmap = mcolors.ListedColormap(colors)
    title = "t-SNE diagram of latent representation"
    path = os.path.join(show_dir,"latent_2d_all_clusters.png")
    plot_2d(points, y_labels, title=title, cmap=cmap, path=path)