# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/09/06 13:59:58
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : test_Spectral.py
# @Description  : This is the test code of Spectral defense.
import os
import sys
from copy import deepcopy
import os.path as osp
from cv2 import transform
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.nn import DataParallel
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from torch.utils.data import Subset
from torchvision.transforms import Compose, RandomCrop,RandomHorizontalFlip, ToTensor, ToPILImage, Resize,Normalize
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import time
from models import BaselineMNISTNetwork,ResNet
from core.base import Base
from core.attacks.BadNets import BadNets
from core.attacks.BackdoorAttack import BackdoorAttack
from core.defenses.BackdoorDefense import BackdoorDefense
from core.defenses import BUNL
import random
from utils import Log, parser, save_img, get_latent_rep
from utils import compute_confusion_matrix, compute_indexes,compute_accuracy,SCELoss
from utils import plot_hist
"""
调用 --->接口---> 实现 

1.调用
1.1 对于调用者应该尽可能的简单
2.接口：
2.1 仅提供全面的功能
2.2 命名简单直观
2.3 对外一致
3.实现
3.1 代码复用
3.2 支持接口
"""
# ========== Set global settings ==========
global_seed = 333
deterministic = True
torch.manual_seed(global_seed)
#"0,1,2,3,4,5", "1,2,3,4,5"
CUDA_VISIBLE_DEVICES = "0,1,2,3,4,5"
datasets_root_dir = BASE_DIR + '/datasets/'

# ========== BaselineMNISTNetwork_MNIST_BadNets ==========
# The basic data type in torch is "tensor". In order to be computed, other data type, like PIL Image or numpy.ndarray,
#  must be Converted to "tensor"
#{model}_{datasets}_{defense}_for_{attack} 
args = parser.parse_args()
if args.dataset == "MNIST":
    experiment = 'BaselineMNISTNetwork_MNIST_BadNets'
    project = 'BaselineMNISTNetwork_MNIST'
    dataset = torchvision.datasets.MNIST
    transform_train = Compose([
        ToTensor(),
        RandomHorizontalFlip()
    ])
    trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
    transform_test = Compose([
        ToTensor()
    ])
    testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
    num_classes = 10
    model = BaselineMNISTNetwork()
    optimizer = torch.optim.SGD
    lr = 0.1
elif args.dataset == "CIFAR10":
    experiment = 'ResNet-18_CIFAR-10_Mine_for_BadNets'
    project = 'ResNet-18_CIFAR-10'
    dataset = torchvision.datasets.CIFAR10
    transform_train = Compose([
        RandomCrop(32, padding=4, padding_mode="reflect"),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])
    ])
    trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
    transform_test = Compose([
        ToTensor(),
        Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])
    ])
    testset = dataset(datasets_root_dir, train=True, transform=transform_test, download=True)
    num_classes = 10
    model = ResNet(18,num_classes = num_classes)
    # optimizer = torch.optim.SGD
    optimizer = torch.optim.Adam
    lr = 0.002
    

elif args.dataset == "CIFAR100":
    experiment = 'ResNet-18_CIFAR-100_Mine_for_BadNets'
    project = 'ResNet-18_CIFAR-100'
    dataset = torchvision.datasets.CIFAR100
    transform_train = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0),
    ])

    trainset = dataset(root=datasets_root_dir, train=True, download=True, transform=transform_train)
    testset = dataset(root=datasets_root_dir,train=False, download=True,transform=transform_test)
    num_classes = 100
    
elif args.dataset == "ImageNet":
    pass
    # dataset = torchvision.datasets.ImageNet
    # transform_train = Compose([
    #     ToTensor()
    # ])
    
attack = 'BUNL_for_BadNets'
work = attack
work_dir = os.path.join(BASE_DIR,'experiments/' + project + '/' + work)
datasets_dir = os.path.join(work_dir,'datasets')
poison_datasets_dir = os.path.join(datasets_dir,'poisoned_data')
latents_dir = os.path.join(datasets_dir,'latents')
predict_dir = os.path.join(datasets_dir,'predict')
model_dir = os.path.join(work_dir,'model')
show_dir = os.path.join(work_dir,'show')
defense_object_dir = os.path.join(work_dir,'defense_object')
dirs = [work_dir, datasets_dir, poison_datasets_dir, latents_dir, predict_dir, model_dir, show_dir,defense_object_dir]
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)
"""
network:
  resnet18_cifar:
    num_classes: 10
sync_bn: True  # synchronized batch normalization

optimizer:
  Adam:
    lr: 0.002
lr_scheduler: null
num_epochs: 120

# """

poisoned_trainset = torch.load(os.path.join(poison_datasets_dir,"train.pt")) 
poisoned_testset = torch.load(os.path.join(poison_datasets_dir,"test.pt"))

task = {
    'train_dataset': poisoned_trainset,
    'test_dataset' : poisoned_testset,
    'model' : model,
    'optimizer': optimizer,
    "loss":nn.CrossEntropyLoss()
}
layer = "linear"

"""
1.这种方法的训练过程有两个特点：
    2.1.初始训练阶段,样本数量很小,存在冷启动问题
    2.2.训练过程中训练数据集动态更新,存在类不平衡问题

2.无监督学习的主要作用为促进样本回归真实类,因此原理上无监督项的促进了hard样本的学习。同时应该也有加快训练的作用。

3.数据过滤的关键点：

3.1 过滤时的模型稳定性要求
这种基于SCE过滤的方法的一个关键点就在于对于样本的预测是可靠的,如果不可靠则该阈值的意义不大。 
因此这种方法对训练过程的稳定性要求很高,在不稳定的情况下样本很容易被误预测到其它标签,对于后门样本若被误分类到目标标签，
则数据的过滤效果就会很差。

3.2 阈值的精确选择要求
过滤时的模型是否满足两个假设：
(1) 每个阶段的模型来说,模型对于样本x的预测有偏向,即为1 of k中的最大值。
(2) 在1成立的情况下，在最后阶段的模型是否满足将hard样本和后门样本分开的SKL的临界点条件。

"""

schedule = {
    # experiment
    'experiment': experiment,
    # defense config:
    'defense_strategy':"BUNL",
    'layer':layer,
    #train config:
    'seed': global_seed,
    'deterministic': deterministic,
    # defense config:
    # 'invert_label_strategy':"random",
    'invert_label_strategy':"real",
    'noise_label_ratio':0.5,
    'y_target':1,
    # related to device
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,
    # related to tarining 
    'pretrain': None,
    # 250, 300
    'epochs': 100,
    'batch_size': 128,
    'num_workers': 4,
    'lr': lr,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],
    'work_dir': work_dir,
    'log_iteration_interval': 100,
}

if __name__ == "__main__":

    # Users can select the task to execute by passing parameters.


    # 1. The task of showing backdoor samples
    #  python test_BUNL.py  --task "show backdoor samples" --dataset "MNIST" 
   
        
    # 2. The task of defense backdoor attack
    #  python test_BUNL.py  --task "repair" --dataset "MNIST" 

    #   python test_BUNL.py --task "repair" --dataset "CIFAR100"

    # 3.The task of testing defense effect
    #   python test_BUNL.py --task "test" --dataset "MNIST" 

    # 4.The task of evaluating data filtering
    #   python test_BUNL.py --task "evaluate data filtering"

    # 5.The task of comparing sce scores of hard and poisoned samples
    # python test_BUNL.py --task  "compare sce scores of hard and poisoned samples"

    # 5.The task of visualizing latents by t-sne
    #     python test_BUNL.py --task "visualize latents by t-sne"
    #     python test_BUNL.py --task "visualize latents for target class by t-sne"

    # 6.The task of comparing predict_digits
    #     python test_BUNL.py --task "generate predict_digits"
    #     python test_BUNL.py --task "compare predict_digits"

    log = Log(osp.join(work_dir, 'log.txt'))
    experiment = schedule['experiment']
    t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    msg = "\n\n\n==========Start {0} at {1}==========\n".format(experiment,t)
    log(msg)
    bunl = BUNL(
        task, 
        schedule)
    # Show the structure of the model
    print(task["model"])
    defense = BackdoorDefense(bunl)
    if args.task == "show backdoor samples":
        log("\n==========Show posioning train sample==========\n")
        # Alreadly exsiting dataset and trained model.
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt')) 
        poison_indices = poisoned_train_dataset.get_poison_indices()
        print(len(poison_indices))
        index = poison_indices[random.choice(range(len(poison_indices)))]
        print(f"index:{index}")
        # print(poisoned_train_dataset[index])
        image,label,_ = poisoned_train_dataset[index]
        image = image.numpy()
        backdoor_sample_path = os.path.join(show_dir, "backdoor_train_sample.png")
        title = "label: " + str(label)
        save_img(image, title=title, path=backdoor_sample_path)
        log("Save backdoor_train_sample to" + backdoor_sample_path)

        poisoned_test_dataset = torch.load(os.path.join(poison_datasets_dir,'test.pt'))
        poisoned_test_indices = poisoned_test_dataset.get_poison_indices()
        benign_test_indexs = list(set(range(len(poisoned_test_dataset))) - set(poisoned_test_indices))

        real_targets = np.array(poisoned_test_dataset.get_real_targets())
        labels = real_targets[poisoned_test_indices]
        # for i, label in enumerate(poisoned_test_dataset.classes):
        #     print(f"the number of sample with label:{label} in poisoned_train_dataset:{labels.tolist().count(i)}\n")

        index = poisoned_test_indices[random.choice(range(len(poisoned_test_indices)))]
        print(f"index:{index}")
        image, label, _ = poisoned_test_dataset[index] 
        image = image.numpy()
        backdoor_sample_path = os.path.join(show_dir, "backdoor_test_sample.png")
        title = "label: " + str(label)
        save_img(image, title=title, path=backdoor_sample_path)
        log("Save backdoor_test_sample to" + backdoor_sample_path)

    elif args.task == "repair":
        # get backdoor sample
        log("\n==========get poisoning train_dataset and test_dataset dataset and repair model ==========\n")
        # Alreadly exsiting dataset and trained model.
        # poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt')) 
        # poisoned_test_dataset = torch.load(os.path.join(poison_datasets_dir,'test.pt'))
        # poisoned_train_indices = poisoned_train_dataset.get_poison_indices()
        # poisoned_test_indices = poisoned_test_dataset.get_poison_indices()
        # repaired_model = defense.get_repaired_model(dataset, schedule)
        # train_dataset, random_indices = defense.invert_random_labels()
        # poisoned_train_indices = train_dataset.get_poison_indices()
        # indces = np.intersect1d(random_indices, poisoned_train_indices)
        # print(f"indces number:{len(indces)},poisoned_train_indices number:{len(poisoned_train_indices)},indces:{indces},\n")
        # _,labels,_ = train_dataset[indces[2]]
        # print(f"indces number:{len(indces)},poisoned_train_indices number:{len(poisoned_train_indices)},indces:{indces},labels:{labels}\n")

        repaired_model = defense.get_repaired_model()
        torch.save(repaired_model.state_dict(), os.path.join(model_dir, 'repaired_model.pth'))
        # log("Save repaired model to" + os.path.join(model_dir, 'repaired_model.pth'))

        # torch.save(defense, os.path.join(defense_object_dir, 'Mine_object.pth'))
        # log("Save Mine object to" + os.path.join(defense_object_dir, 'Mine_object.pth'))
        
        

    elif args.task == "test":
        # Test the attack effect of backdoor model on backdoor datasets.
        log("\n==========Test the effect of defense on poisoned_test_dataset==========\n")

        poisoned_test_dataset = torch.load(os.path.join(poison_datasets_dir,'test.pt'))
        testset = poisoned_test_dataset
        poisoned_test_indexs = list(testset.get_poison_indices())
        benign_test_indexs = list(set(range(len(testset))) - set(poisoned_test_indexs))
        #Alreadly exsiting trained model
        model = task['model']
        #print(os.path.join(model_dir, 'backdoor_model.pth'))
        model.load_state_dict(torch.load(os.path.join(model_dir, 'repaired_model.pth')),strict=False)
        predict_digits, labels = defense.test(model=model, test_dataset=testset)

        benign_accuracy = compute_accuracy(predict_digits[benign_test_indexs],labels[benign_test_indexs],topk=(1,3,5))
        poisoning_accuracy = compute_accuracy(predict_digits[poisoned_test_indexs],labels[poisoned_test_indexs],topk=(1,3,5))
        log("Total samples:{0}, poisoning samples:{1},  benign samples:{2}".format(len(testset),len(poisoned_test_indexs),\
                                                                                len(benign_test_indexs)))                                                                                                                                                
        log("Benign_accuracy:{0}, poisoning_accuracy:{1}".format(benign_accuracy,poisoning_accuracy))

    elif args.task == "evaluate data filtering":
        # Evaluate the effectiveness of data filtering
        log("\n==========Evaluate the effectiveness of data filtering.==========\n")
        defense = torch.load(os.path.join(defense_object_dir, 'Mine_object.pth'))

        # log("Save pred_poisoned_sample_dist to" + os.path.join(predict_dir, 'pred_poisoned_sample_dist.pth'))
        target_label = defense.get_target_label()
        predicted_dist = defense.get_pred_poisoned_sample_dist()
        real_dist = defense.get_real_poisoned_sample_dist()
        tp, fp, tn, fn = compute_confusion_matrix(predicted_dist, real_dist)
        accuracy, precision, recall, F1 = compute_indexes(tp, fp, tn, fn)
        log(f"Results of data filtering:target_label:{target_label}, tp:{tp} fp:{fp}, tn:{tn}, fn:{fn},accuracy:{accuracy}, precision:{precision}, recall:{recall}, F1:{F1}")

        
    elif args.task == "compare sce scores of hard and poisoned samples":
        log("\n==========Compare sce scores of hard and poisoned samples.==========\n")
        defense = torch.load(os.path.join(defense_object_dir, 'Mine_object.pth'))
        predicted_dist = defense.get_pred_poisoned_sample_dist()
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt')) 
        poison_indices = poisoned_train_dataset.get_poison_indices()
        remain_sample_indces = np.arange(len(predicted_dist))[predicted_dist == 1]
        remain_sample_set = Subset(poisoned_train_dataset,remain_sample_indces)
        targets = [1]
        for item in remain_sample_set:
            # print(item)
            targets.append(item[1]) 
        for i, label in enumerate(poisoned_train_dataset.classes):
            print(f"the number of sample with label:{label} in poisoned_train_dataset:{targets.count(i)}\n")


    #    #Alreadly exsiting trained model
    #     model = task['model']
    #     #print(os.path.join(model_dir, 'backdoor_model.pth'))
    #     model.load_state_dict(torch.load(os.path.join(model_dir, 'repaired_model.pth')),strict=False)
    #     sceloss = SCELoss(reduction='none')
       
    #     with torch.no_grad(): 
    #         poisoned_sample_set = Subset(poisoned_train_dataset,np.intersect1d(poison_indices,remain_sample_indces))
    #         predict_digits, labels = defense.test(model=model, test_dataset=poisoned_sample_set)
    #         # print(f"the shape of labels:{labels.shape}")
    #         poisoned_sample_scores = sceloss(predict_digits,labels).numpy()
            
    #         clean_sample_set = Subset(poisoned_train_dataset,np.setdiff1d(remain_sample_indces,np.intersect1d(poison_indices,remain_sample_indces)))
    #         predict_digits, labels = defense.test(model=model, test_dataset=clean_sample_set)
    #         # print(f"the shape of labels:{labels.shape}")
    #         clean_sample_scores = sceloss(predict_digits,labels).numpy()
    #         # print(f"predict_digits:{predict_digits},shape:{predict_digits.shape},labels:{labels},shape:{labels.shape}\n")
       
    #     scores = [poisoned_sample_scores,clean_sample_scores]
    #     labels=["Clean samples","Poisoned samples"]
    #     colors = ["res","blue"]
    #     title = "Compare sce scores of hard and poisoned samples"
    #     xlabel = "SCE loss"
    #     ylabel = "Proportion (%)"
    #     plot_hist(*scores, colors=colors, title=title, xlabel=xlabel, ylabel=ylabel)


    elif args.task == "generate latents":
        log("\n==========Get the latent representation in the middle layer of the model.==========\n")
        # Alreadly exsiting trained model and poisoned datasets
        # device = torch.device("cuda:1")
        # model = BaselineMNISTNetwork()
        device = torch.device("cpu")
        model = task["model"]
        # model.to(device)
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt'))
        poison_indices = poisoned_train_dataset.get_poison_indices()
        model.load_state_dict(torch.load(os.path.join(work_dir, 'model/repaired_model.pth')),strict=False)
        # Get the latent representation in the middle layer of the model.
        layer = "fc2"
        latents,y_labels = get_latent_rep(model, layer, poisoned_train_dataset, device=device)
        latents_path = os.path.join(latents_dir,"MNIST_train_latents.npz")
        print(type(latents))
        print(latents.shape)
        np.savez(latents_path, latents=latents, y_labels=y_labels)
 




