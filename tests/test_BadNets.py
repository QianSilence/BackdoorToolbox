# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/08/19 15:49:57
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : test_BadNets.py
# @Description : This is the test code of poisoned training under BadNets.
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
from torchvision import transforms 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
# print(sys.path)
from core.attacks import BadNets
from core.attacks import BackdoorAttack
from models import BaselineMNISTNetwork
import time
from utils import show_image
from utils import accuracy
import os.path as osp
import random
from core.base import Observer
from core.base import Base

# ========== Set global settings ==========
global_seed = 333
deterministic = False
torch.manual_seed(global_seed)
CUDA_VISIBLE_DEVICES = '1'
datasets_root_dir = BASE_DIR + '/datasets/'
# print(datasets_root_dir)

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
pattern = torch.zeros((28, 28), dtype=torch.uint8)
pattern[-3:, -3:] = 255
weight = torch.zeros((28, 28), dtype=torch.float32)
weight[-3:, -3:] = 1.0
schedule = {
    'experiment': 'BaselineMNISTNetwork_MNIST_package',

    # Settings for reproducible/repeatable experiments
    'seed': global_seed,
    'deterministic': deterministic,
    

    # Settings related to device
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    # Settings related to tarining 
    'pretrain': None,
    'epochs': 10,
    'batch_size': 128,
    'num_workers': 2,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],
    # When this parameter is given,the model is saved after trianed
    'model_path':'/Poisoned_model.pth',

    # Settings aving model,data and logs
    'save_dir': 'experiments',
    'log_iteration_interval': 100,
    # 日志的保存路径 save_dir+experiment+当前时间
}

task = {
    'train_dataset': trainset,
    'test_dataset' : testset,
    'model' : BaselineMNISTNetwork(),
    'optimizer': optimizer,
    'loss' : nn.CrossEntropyLoss()
}

# # Parameters are needed according to attack strategy
attack_schedule ={ 
    'experiment': 'BaselineMNISTNetwork_MNIST_package',
    'attack_strategy': 'BadNets',
    # attack config
    'y_target': 1,
    'poisoned_rate': 0.05,
    'pattern': pattern,
    'weight': weight,
    'poisoned_transform_index': 0,
    'poisoned_target_transform_index': 0,
    # device config
    'device': None,
    'CUDA_VISIBLE_DEVICES': None,
    'GPU_num': None,
    'batch_size': None,
    'num_workers': None,
    # Settings related to saving model,data and logs
    'save_dir': 'experiments',
    'train_schedule':schedule,
}
work_dir = attack_schedule['save_dir'] + '/' + attack_schedule['experiment']

if __name__ == "__main__":
    badnets = BadNets(
        task,
        attack_schedule
    )
    backdoor = BackdoorAttack(badnets)

    # 1. show backdoor sample
    # Alreadly exsiting dataset and trained model.
    # poisoned_train_dataset = torch.load("/home/zzq/CreatingSpace/BackdoorToolbox/datasets/PoisonedMNIST/training.pt") 
    # poisoned_test_dataset = torch.load("/home/zzq/CreatingSpace/BackdoorToolbox/datasets/PoisonedMNIST/test.pt")

    poisoned_train_dataset = backdoor.get_poisoned_train_dataset()
    poisoned_test_dataset = backdoor.get_poisoned_test_dataset()
    torch.save(poisoned_train_dataset, datasets_root_dir + '/PoisonedMNIST/training.pt')
    torch.save(poisoned_test_dataset, datasets_root_dir + '/PoisonedMNIST/test.pt')
    # poisoned_set = poisoned_train_dataset.get_poisoned_set()
    # index = poisoned_set[random.sample(range(len(poisoned_set)))]
    # # print(index)
    # image, label = poisoned_train_dataset[index]
    # image = image.squeeze().numpy()
    # # print(image.shape)
    # show_image(image, "label: " + str(label))

    #2. test backdoor model
    poisoned_model = backdoor.get_backdoor_model()
    torch.save(poisoned_model.state_dict(), work_dir + '/Poisoned_model.pth')


    #3.test benign_accuracy and poisoning_rate
    testset = poisoned_test_dataset
    poisoned_test_indexs = list(testset.get_poisoned_set())
    benign_test_indexs = list(set(range(len(testset))) - set(poisoned_test_indexs))
    predict_digits, labels = backdoor.test()
    benign_accuracy = accuracy(predict_digits[benign_test_indexs],labels[benign_test_indexs])
    poisoning_rate = accuracy(predict_digits[poisoned_test_indexs],labels[poisoned_test_indexs])
    print(benign_accuracy)
    print(poisoning_rate)

    #4.Alreadly exsiting trained model
    # model = BaselineMNISTNetwork()
    # model = nn.DataParallel(model)
    # path = "/home/zzq/CreatingSpace/BackdoorToolbox/experiments/BaselineMNISTNetwork_MNIST_package_2023-09-08_17:10:29/Poisoned_model.pth"
    # parameter=torch.load(path)
    # # # print(parameter)
    # model.load_state_dict(parameter)
    # trainset = torch.load("/home/zzq/CreatingSpace/BackdoorToolbox/datasets/PoisonedMNIST/training.pt") 
    # testset = torch.load("/home/zzq/CreatingSpace/BackdoorToolbox/datasets/PoisonedMNIST/test.pt")
    # predict_digits, labels = backdoor.test(schedule, model, testset)
    # poisoned_train_indexs = list(trainset.get_poisoned_set())
    # poisoned_test_indexs = list(testset.get_poisoned_set())
    # benign_test_indexs = list(set(range(len(testset))) - set(poisoned_test_indexs))
    # benign_accuracy = accuracy(predict_digits[benign_test_indexs],labels[benign_test_indexs])
    # poisoning_rate = accuracy(predict_digits[poisoned_test_indexs],labels[poisoned_test_indexs])
    # print(benign_accuracy)
    # print(poisoning_rate)


