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
work_dir = os.path.join(BASE_DIR,'experiments/ResNet-18_CIFAR-10_Adaptive-Blend')
poison_datasets_dir = os.path.join(work_dir,'datasets/ResNet-18_CIFAR-10_Adaptive-Patch_'+ str(date) + '/poisonedCifar-10')
dirs = [poison_datasets_dir,work_dir]
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

# ========== ResNet-18_CIFAR-10_AdaptiveBlend ==========
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

trigger_path = os.path.join(datasets_root_dir, "triggers/hellokitty_32.png")
# #PIL.Image.Image
# trigger = Image.open(trigger_path).convert("RGB") 

schedule = {
    'experiment': 'ResNet-18_CIFAR-10_Adaptive-Blend',
    # Settings for reproducible/repeatable experiments
    'seed': global_seed,
    'deterministic': deterministic,
    
    # Settings related to device
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    # Settings related to tarining 
    'pretrain': None,
    'epochs': 100,   
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

# Parameters are needed according to attack strategy
attack_schedule ={
    'experiment': 'ResNet-18_CIFAR-10_Adaptive-Blend',
    'attack_strategy': 'Adaptive-Blend',
    
    # attack config
    'y_target': 1,
    'poisoning_rate': 0.05,
    # conservatism ratio:balance the number of payload and regularization samples
    'cover_rate' : 0.01,
    # trigger
    'pattern': trigger_path,
    # diverse asymmetry
    
    'pieces':16,
    'train_mask_rate':0.5,
    'test_mask_rate':1.0,
    # asymmetric design:The trigger uses different transparency during the training phase and testing phase.
    'train_alpha':0.15,
    'test_alpha':0.2,

    'poisoned_transform_index': 2,
    'poisoned_target_transform_index': 2,

    # device config
    'device': None,
    'CUDA_VISIBLE_DEVICES': None,
    'GPU_num': None,
    'batch_size': None,
    'num_workers': None,
    # Settings related to saving model, data and logs
    'work_dir': 'experiments',
    'train_schedule':schedule,
}
if __name__ == "__main__":
    """
    Users can choose the tasks they want to perform by passing parameters.
    (1)The task of generating and showing backdoor samples
    (2)The task of training backdoor model
    (3)The task of testing backdoor model
    """
    os.makedirs(work_dir, exist_ok=True)
    log = Log(osp.join(work_dir, 'log.txt'))
    experiment = attack_schedule['experiment']
    t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    msg = "\n\n\n==========Start {0} at {1}==========\n".format(experiment,t)
    log(msg)

    adaptive_blend = AdaptiveBlend(
        task,
        attack_schedule
    )
    backdoor = BackdoorAttack(adaptive_blend)
    # Show the structure of the model
    # print(task['model'])
    # print(task['model'].layer4)

    # 1. Generate and show backdoor sample
    # # Alreadly exsiting dataset and trained model.
    # # poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt')) 
    # # poisoned_test_dataset = torch.load(os.path.join(poison_datasets_dir,'test.pt'))

    # log("\n==========Generate backdoor samples==========\n")
    # poisoned_train_dataset = backdoor.get_poisoned_train_dataset()
    # poisoned_test_dataset = backdoor.get_poisoned_test_dataset()
    # # #Save poisoned dataset
    # torch.save(poisoned_train_dataset, os.path.join(poison_datasets_dir,'train.pt'))
    # torch.save(poisoned_test_dataset, os.path.join(poison_datasets_dir,'test.pt'))
   
    # poison_indices = poisoned_train_dataset.get_poison_indices()
    # benign_indexs = list(set(range(len(poisoned_train_dataset))) - set(poison_indices))

    # log("Total samples:{0}, poisoning samples:{1},  benign samples:{2}".format(len(poisoned_train_dataset),\
    #     len(poison_indices),len(benign_indexs)))

    # # # Show posioning train sample
    # # poison_indices = poisoned_train_dataset.get_poison_indices()
    # # index = poison_indices[random.choice(range(len(poison_indices)))]
    # # print(len(poison_indices))
    # # image, label = poisoned_train_dataset[index] 
    # # # Outside of neural networks, packages including numpy and matplotlib are usually used for data operations, 
    # # # so the type of data is usually converted to np.ndrray()
    # # image = image.numpy()
    # # backdoor_sample_path = os.path.join(work_dir, "show/backdoor_train_sample.png")
    # # title = "label: " + str(label)
    # # save_image(image, title=title, path=backdoor_sample_path)

    # # # Show posioning test sample
    # # poison_indices = poisoned_test_dataset.get_poison_indices()
    # # index = poison_indices[random.choice(range(len(poison_indices)))]
    # # print(len(poison_indices))
    # # image, label = poisoned_test_dataset[index] 
    # # image = image.numpy()
    # # backdoor_sample_path = os.path.join(work_dir, "show/backdoor_test_sample.png")
    # # title = "label: " + str(label)
    # # save_image(image, title=title, path=backdoor_sample_path)
    
    # #2. Train and get backdoor model
    # print(os.path.join(work_dir, 'model/backdoor_model.pth'))
    # log("\n==========Train on poisoned_train_dataset and get backdoor model==========\n")
    # poisoned_model = backdoor.get_backdoor_model()
    # torch.save(poisoned_model.state_dict(), os.path.join(work_dir, 'model/backdoor_model.pth'))
    # log("Save backdoor model to" + os.path.join(work_dir, 'model/backdoor_model.pth'))

    # #3.Test backdoor model
    # """
    # 确任adaptive-blend以下几条效果：
    # 1.后门模型在原后门数据集上的效果(测试和训练数据集)的效果
    # 2.干净模型在后门数据集上的效果
    # 3.后门模型在新生成的后门数据集上的效果(测试和训练数据集)的效果
    # 4.后门模型在干净数据集上的效果
    # """
    # # 3.1 The attack effect of backdoor model on new backdoor datasets.
    # log("\n==========Test the effect of backdoor attack on poisoned_test_dataset==========\n")
    # testset = poisoned_test_dataset
    # poisoned_test_indexs = list(testset.get_poison_indices())
    # benign_test_indexs = list(set(range(len(testset))) - set(poisoned_test_indexs))
    
    # #Alreadly exsiting trained model
    # # model = nn.DataParallel(ResNet(18))
    # # model.load_state_dict(torch.load(os.path.join(work_dir, 'model/backdoor_model.pth')),strict=False)
    # # predict_digits, labels = backdoor.test(model=model, test_dataset=testset)
    # predict_digits, labels = backdoor.test()

    # benign_accuracy = compute_accuracy(predict_digits[benign_test_indexs],labels[benign_test_indexs],topk=(1,3,5))
    # poisoning_accuracy = compute_accuracy(predict_digits[poisoned_test_indexs],labels[poisoned_test_indexs],topk=(1,3,5))
    # log("Total samples:{0}, poisoning samples:{1},  benign samples:{2}".format(len(testset),len(poisoned_test_indexs),\
    #                                                                            len(benign_test_indexs)))
    # log("Benign_accuracy:{0}, poisoning_accuracy:{1}".format(benign_accuracy,poisoning_accuracy))

    # #3.2 The attack effect of backdoor model on new backdoor datasets.
    # # log("\n==========Test The attack effect of backdoor model on new backdoor datasets.==========\n")
    # # poisoned_test_dataset = backdoor.get_poisoned_test_dataset()
    # # testset = poisoned_test_dataset
    # # poisoned_test_indexs = list(testset.get_poison_indices())
    # # benign_test_indexs = list(set(range(len(testset))) - set(poisoned_test_indexs))
    # # #Alreadly exsiting trained model
    # # model = nn.DataParallel(ResNet(18))
    # # model.load_state_dict(torch.load(os.path.join(work_dir, 'model/backdoor_model.pth')),strict=False)
    # # predict_digits, labels = backdoor.test(model=model, test_dataset=testset)

    # # benign_accuracy = compute_accuracy(predict_digits[benign_test_indexs],labels[benign_test_indexs],topk=(1,3,5))
    # # poisoning_accuracy = compute_accuracy(predict_digits[poisoned_test_indexs],labels[poisoned_test_indexs],topk=(1,3,5))
    # # log("Total samples:{0}, poisoning samples:{1},  benign samples:{2}".format(len(poisoned_test_indexs),\
    # #     len(poisoned_test_indexs),len(benign_test_indexs)))
    # # log("Benign_accuracy:{0}, poisoning_accuracy:{1}".format(benign_accuracy,poisoning_accuracy))

    # # The attack effect of backdoor model on clean datasets.
    # # log("\n==========Test The attack effect of backdoor model on clean datasets.==========\n")
    # # #Alreadly exsiting trained model
    # # model = nn.DataParallel(ResNet(18))
    # # model.load_state_dict(torch.load(os.path.join(work_dir, 'model/backdoor_model.pth')),strict=False)
    # # predict_digits, labels = backdoor.test(model=model, test_dataset=testset)
    # # accuracy = compute_accuracy(predict_digits[benign_test_indexs],labels[benign_test_indexs],topk=(1,3,5))
    # # log("Total samples:{0},accuracy:{1}".format(len(testset),accuracy))

    # 4. 
    """
    4.1 对于有毒数据集，可视化所有类别的潜在聚类(验证所有类别之间的分离特性，以及同时标注有毒数据)
    4.2 仅仅可视化目标标签的有毒数据集(验证目标标签下干净和有毒样本表示的分离特性)
    4.3 对于干净模型，可视化所有类别的潜在表示(作为以上的对比实验，验证所有类别之间的分离特性)
    对于不同的类别需要用不同的颜色着色,有毒数据用红色着色
    """
    log("\n==========Verify the assumption of latent separability.==========\n")
   
    # #Alreadly exsiting trained model
    device = torch.device("cuda:0")
    model = nn.DataParallel(ResNet(18),device_ids=[0], output_device=device )
    # model.to(device)
    print(model)
    # print(trainset[0][1])
    poison_datasets_dir = os.path.join(work_dir,'datasets/ResNet-18_CIFAR-10_Adaptive-Patch_2023-10-08/poisonedCifar-10')
    poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt'))
    poison_indices = poisoned_train_dataset.get_poison_indices()
    model.load_state_dict(torch.load(os.path.join(work_dir, 'model/backdoor_model.pth')),strict=False)
    #Get the latent representation in the middle layer of the model.
    layer = "linear"
    device = torch.device("cuda:0")
    latents,y_labels = get_latent_rep(model, layer, poisoned_train_dataset, device=device)
    latents_path =os.path.join(work_dir,"datasets/latents/poisoned_cifar10_latents.npy")
    np.save(latents_path,latents)


    #get low-dimensional data points by t-SNE
    latents = np.load(latents_path)
    n_components = 2 # number of coordinates for the manifold
    t_sne = manifold.TSNE(n_components=n_components, perplexity=30, early_exaggeration=120, init="pca", n_iter=250, random_state=0 )
    points = t_sne.fit_transform(latents)
    # points = np.load(latents_path)
    print(type(points))
    print(points.shape)
    #Display data clusters for all category by scatter plots
    num = len(classes)
    # Custom color mapping
    colors = [plt.cm.tab10(i) for i in range(num)]
    colors.append("red")  
    y_labels[poison_indices] = num
    # Create a ListedColormap object
    cmap = mcolors.ListedColormap(colors)
    title = "t-SNE diagram of latent representation"
    plot_2d(points, y_labels, title=title, cmap=cmap)











