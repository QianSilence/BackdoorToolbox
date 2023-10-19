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
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, ToPILImage, Resize
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models import BaselineMNISTNetwork
from core.base import Base
from core.attacks.BadNets import BadNets
from core.attacks.BackdoorAttack import BackdoorAttack
from core.defenses.BackdoorDefense import BackdoorDefense
from core.defenses.Spectral import Spectral
from utils import compute_confusion_matrix, compute_indexes,compute_accuracy
# ========== Set global settings ==========
global_seed = 333
deterministic = False
torch.manual_seed(global_seed)
CUDA_VISIBLE_DEVICES = '1'
datasets_root_dir = BASE_DIR + '/datasets/'
batch_size = 128
num_workers = 4

# ========== BaselineMNISTNetwork_MNIST_BadNets ==========
# The basic data type in torch is "tensor". In order to be computed, other data type, like PIL Image or numpy.ndarray,
#  must be Converted to "tensor".
dataset = torchvision.datasets.MNIST
transform_train = Compose([
    ToTensor()
])
trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)

transform_test = Compose([
    ToTensor()
])
testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
optimizer = torch.optim.SGD

# model = BaselineMNISTNetwork()
# model = nn.DataParallel(model)
# path = "/home/zzq/CreatingSpace/BackdoorToolbox/experiments/BaselineMNISTNetwork_MNIST_package_2023-09-08_17:10:29/Poisoned_model.pth"
# parameter=torch.load(path)
# # print(parameter)
# model.load_state_dict(parameter)
poisoned_trainset = torch.load("/home/zzq/CreatingSpace/BackdoorToolbox/datasets/PoisonedMNIST/training.pt") 
poisoned_testset = torch.load("/home/zzq/CreatingSpace/BackdoorToolbox/datasets/PoisonedMNIST/test.pt")
model_name = 'BaselineMNISTNetwork'
dataset_name= 'MNIST'
attack_name = 'Badnets'
defense_name = 'Spectral-Signature'

task = {
    'train_dataset': trainset,
    'test_dataset' : testset,
    'model' : BaselineMNISTNetwork(),
    'optimizer': optimizer,
    'loss' : nn.CrossEntropyLoss()
}
# Config related to model training
schedule = {
    # experiment
    'experiment': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}',
    # 'benign_training': False,
    'seed': global_seed,
    'deterministic': deterministic,
    
    # related to device
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    # related to tarining 
    'pretrain': None,
    'epochs': 5,
    'batch_size': 128,
    'num_workers': 2,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],

    'work_dir': 'experiments',
    'log_iteration_interval': 100,
}

defense_shedule ={
    'defense_strategy':"Spectral",
    'experiment': f'{model_name}_{dataset_name}_{attack_name}_{defense_name}',

    #Defense config 
    # Is training required to get poisoned models when performing data filtering defense?
    'train':True, 
    # The saving path of the backdoor model. When train = True, it is the path to save 
    # the trained model. When train = False, it is the path to load the model.
    "trained_model":None,
    'backdoor_model_path':None, 
    'y_target': 0,
    'poisoned_trainset':poisoned_trainset,
    'poisoned_testset':poisoned_testset,
    'percentile':85,

    # Device config
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,
    'batch_size': batch_size,
    'num_workers': num_workers,

    # Settings related to saving model data and logs
    'work_dir': 'experiments',
    'train_schedule':schedule,
}

# badnets = BadNets(
#     task,
#     attack_config,
#     schedule=schedule,  
# )
if __name__ == "__main__":

    spectral = Spectral(
        task, 
        defense_shedule)
    # Show the structure of the model
    print(task["model"])

    defense = BackdoorDefense(spectral)
    # 1.filter out poisoned samples
    removed_inds, left_inds, target_label_inds = defense.filter()
    precited = np.array(range(len(poisoned_trainset)))
    precited[removed_inds] = 0
    precited[np.delete(target_label_inds, removed_inds)] = 1

    poisoned_train_indexs = list(poisoned_trainset.get_poisoned_set())
    benign_train_indexs = list(set(range(len(poisoned_trainset))) - set(poisoned_train_indexs))
    expected = np.array(range(len(poisoned_trainset)))
    expected[poisoned_train_indexs] = 0
    expected[benign_train_indexs] = 1
    tp, fp, tn, fn = compute_confusion_matrix(precited,expected)
    accuracy, precision, recall, F1 = compute_indexes(tp, fp, tn, fn)
    print(accuracy, precision, recall, F1)


    # 2.重新练模型得到模型并保存参数,并输出准确率和中毒率

    defense_model = defense.repair()
    predict_digits, labels = defense.test()
    # print(predict_digits.shape)
    # print(labels.shape)
    # print(predict_digits[0:3])
    # print(labels[0:3])
   
    poisoned_test_indexs = list(poisoned_testset.get_poisoned_set())
    benign_test_indexs = list(set(range(len(poisoned_testset))) - set(poisoned_test_indexs))

    # print(len(predict_digits))
    # print(len(poisoned_testset))
    # print(len(poisoned_test_indexs))
    # print(len(benign_test_indexs))
    benign_accuracy = compute_accuracy(predict_digits[benign_test_indexs],labels[benign_test_indexs])
    poisoning_rate = compute_accuracy(predict_digits[poisoned_test_indexs],labels[poisoned_test_indexs])
    print(benign_accuracy)
    print(poisoning_rate)

