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
from torchvision import transforms 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
# print(sys.path)
from core.attacks import AdaptivePatch
from core.attacks import BackdoorAttack
from models import ResNet
import time
import datetime
from utils import show_image
from utils import save_image
from utils import compute_accuracy
from utils import Log
import os.path as osp
import random
from core.base import Observer
from core.base import Base
from PIL import Image
# ========== Set global settings ==========
global_seed = None
deterministic = False
# torch.manual_seed(global_seed)
CUDA_VISIBLE_DEVICES = '1'
datasets_root_dir = BASE_DIR + '/datasets/'
work_dir = os.path.join(BASE_DIR+"/experiments",'ResNet-18_CIFAR-10_Adaptive-Patch')
date = datetime.date.today()
poison_datasets_dir = os.path.join(work_dir,'datasets/ResNet-18_CIFAR-10_Adaptive-Patch_'+ str(date) + '/poisonedCifar-10')
dirs = [work_dir,poison_datasets_dir]
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

# ========== ResNet-18_CIFAR-10_AdaptiveBlend ==========
# The basic data type in torch is "tensor". 
# In order to be computed, other data type, like PIL Image or numpy.ndarray, must be converted to "torch.tensor".
dataset = torchvision.datasets.CIFAR10
transform_train = Compose([
    ToTensor()
])
trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)

transform_test = Compose([
    ToTensor()
])
testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
optimizer = torch.optim.SGD
schedule = {
    'experiment': 'ResNet-18_CIFAR-10_Adaptive-Patch',
    "train_strategy": BaseTrainer,

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
    'model_path':'/backdoor_model.pth',

    # Settings aving model,data and logs
    'work_dir': 'experiments',
    'log_iteration_interval': 100,
    # 日志的保存路径 work_dir+experiment+当前时间
}

task = {
    'train_dataset': trainset,
    'test_dataset' : testset,
    'model': ResNet(18),
    'optimizer': optimizer,
    'loss' : nn.CrossEntropyLoss()
}

# Parameters are needed according to attack strategy
patterns =  ['phoenix_corner_32.png', 'firefox_corner_32.png', 'badnet_patch4_32.png', 'trojan_square_32.png']
masks = ["mask_" + item   for item in patterns ]
train_alphas = [  0.5, 0.2, 0.5, 0.3]
test_alphas = [  0.5, 0.2, 0.5, 0.3]
# train_alphas = [1.0, 1.0, 1.0, 1.0]
# test_alphas = [1.0, 1.0, 1.0, 1.0]
attack_schedule ={
    'experiment': 'ResNet-18_CIFAR-10_Adaptive-Patch',
    'attack_strategy': 'Adaptive-Patch',
    
    # attack config
    'y_target': 1,
    'poisoning_rate': 0.05,
    # trigger and opacitys
    'trigger_dir': datasets_root_dir + 'triggers/',
    'patterns': patterns,
    'masks': masks,
    'train_alphas': train_alphas,
    'test_alphas': test_alphas,
    'num_compose': 2,
    # conservatism ratio
    'cover_rate' : 0.01,
    'poisoned_transform_index': 0,
    'poisoned_target_transform_index': 0,

    # device config
    'device': None,
    'CUDA_VISIBLE_DEVICES': None,
    'GPU_num': None,
    'batch_size': None,
    'num_workers': None,
    # Settings related to saving model,data and logs
    'work_dir': 'experiments',
    'train_schedule':schedule,
}
if __name__ == "__main__":
    # # print("Start to test")
    adaptive_patch = AdaptivePatch(
        task,
        attack_schedule
    )
    backdoor = BackdoorAttack(adaptive_patch)

    os.makedirs(work_dir, exist_ok=True)
    log = Log(osp.join(work_dir, 'log.txt'))
    t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    msg = "Start experiment {0} at {1}\n".format(attack_schedule['experiment'],t)
    log(msg)
    # 1. generate and show backdoor sample

    # Alreadly exsiting datasets.
    # poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir, 'training.pt')) 
    # poisoned_test_dataset = torch.load(os.path.join(poison_datasets_dir, 'test.pt'))
    
    poisoned_train_dataset = backdoor.get_poisoned_train_dataset()
    poisoned_test_dataset = backdoor.get_poisoned_test_dataset()
    # Save posioning sample
    torch.save(poisoned_train_dataset, os.path.join(poison_datasets_dir, 'training.pt'))
    torch.save(poisoned_test_dataset, os.path.join(poison_datasets_dir, 'test.pt') )
    log("Save backdoor datasets to " + os.path.join(work_dir, 'experiments/ResNet-18_CIFAR-10_Adaptive-Patch/datasets/\n'))

    # Show posioning train sample
    # poison_indices = poisoned_train_dataset.get_poison_indices()
    # index = poison_indices[random.choice(range(len(poison_indices)))]
    # image, label = poisoned_train_dataset[index]
    # image = image.numpy()
    # backdoor_sample_path = os.path.join(work_dir, "show/backdoor_train_sample.png")
    # title = "label: " + str(label)
    # save_image(image,title=title,path=backdoor_sample_path)

    # Show posioning test sample
    # poison_indices = poisoned_test_dataset.get_poison_indices()
    # index = poison_indices[random.choice(range(len(poison_indices)))]
    # image, label = poisoned_test_dataset[index]
    # image = image.numpy()
    # backdoor_sample_path = os.path.join(work_dir, "show/backdoor_test_sample.png")
    # title = "label: " + str(label)
    # save_image(image,title=title,path=backdoor_sample_path)

    #2. Train and get backdoor model

    log("\n==========Train on poisoned_train_dataset and get backdoor model==========\n")
    poisoned_model = backdoor.get_backdoor_model()
    torch.save(poisoned_model.state_dict(), os.path.join(work_dir, 'model/backdoor_model.pth'))
    log("Save backdoor model to " + os.path.join(work_dir, 'model/backdoor_model.pth\n'))

    #3.Test backdoor attack effect
    log("\n==========Test the effect of backdoor attack on poisoned_test_dataset==========\n")
    testset = poisoned_test_dataset
    poisoned_test_indexs = list(testset.get_poison_indices())
    benign_test_indexs = list(set(range(len(testset))) - set(poisoned_test_indexs))

    #Alreadly exsiting trained model
    # model = nn.DataParallel(ResNet(18))
    # model.load_state_dict(torch.load(os.path.join(work_dir, 'model/backdoor_model.pth')),strict=False)
    # predict_digits, labels = backdoor.test(model=model, test_dataset=testset)
    predict_digits, labels = backdoor.test()
    benign_accuracy = compute_accuracy(predict_digits[benign_test_indexs],labels[benign_test_indexs],topk=(1,3,5))
    poisoning_accuracy = compute_accuracy(predict_digits[poisoned_test_indexs],labels[poisoned_test_indexs],topk=(1,3,5))
    log("Total samples:{0}, poisoning samples:{1},benign samples:{2}\n".format(len(poisoned_test_indexs),\
        len(poisoned_test_indexs),len(benign_test_indexs)))
    log("Benign_accuracy:{0}, poisoning_accuracy:{1}\n".format(benign_accuracy,poisoning_accuracy))

