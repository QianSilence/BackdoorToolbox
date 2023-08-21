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
from models import BaselineMNISTNetwork
import time
from utils import Log

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))


# ========== Set global settings ==========
global_seed = 666
deterministic = True
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

# task: including datasets,model, Optimizer algorithm and loss function.
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
    'experiment_name': 'BaselineMNISTNetwork_MNIST_package',
    # 'benign_training': False,
    'seed': global_seed,
    'deterministic': deterministic,
    
    # related to device
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,

    # related to tarining 
    'epochs': 200,
    'batch_size': 128,
    'num_workers': 2,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],

    # related to interaction
    'interact_in_training': True,
    'save_dir': 'experiments',
    'log_iteration_interval': 100
}
# Parameters are needed according to attack strategy
attack_config ={
    'attack_strategy': 'BadNets',
    'y_target': 1,
    'poisoned_rate': 0.05,
    'pattern': pattern,
    'weight': weight,
    'poisoned_transform_index': 0,
    'poisoned_transform_test_index': 0,
    'poisoned_target_transform_index': 0
}

def test_in_train(model, epoch, device):
    
    import os
    import os.path as osp
    import time
    from utils import Log
    save_dir = 'experiments'
    experiment_name = 'BaselineMNISTNetwork_MNIST_package'
    test_epoch_interval =  10

    i = epoch
    work_dir = osp.join(save_dir, experiment_name + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
    os.makedirs(work_dir, exist_ok=True)
    log = Log(osp.join(work_dir, 'log.txt'))

    if (i + 1) % test_epoch_interval == 0:
        # test result on benign test dataset
        predict_digits, labels = model.test2(model.test_dataset, device, model.current_schedule['batch_size'], model.current_schedule['num_workers'])
        total_num = labels.size(0)
        prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
        top1_correct = int(round(prec1.item() / 100.0 * total_num))
        top5_correct = int(round(prec5.item() / 100.0 * total_num))
        msg = "==========Test result on benign test dataset==========\n" + \
            time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
            f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n"
        log(msg)

        # test result on poisoned test dataset
        # if self.current_schedule['benign_training'] is False:
        predict_digits, labels = model.test2(model.poisoned_test_dataset, device, model.current_schedule['batch_size'], model.current_schedule['num_workers'])
        total_num = labels.size(0)
        prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
        top1_correct = int(round(prec1.item() / 100.0 * total_num))
        top5_correct = int(round(prec5.item() / 100.0 * total_num))
        msg = "==========Test result on poisoned test dataset==========\n" + \
        time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
        f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n"
            
        log(msg)
        model.model = model.model.to(device)
        model.model.train()

def save_model_in_train(model, epoch, device):
    import os.path as osp
    import torch
    save_dir = 'experiments'
    experiment_name = 'BaselineMNISTNetwork_MNIST_package'
    save_epoch_interval =  10
    i = epoch
    work_dir = osp.join(save_dir, experiment_name + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
    if (i + 1) % save_epoch_interval == 0:
        model.model.eval()
        model.model = model.model.cpu()
        ckpt_model_filename = "ckpt_epoch_" + str(i+1) + ".pth"
        ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
        torch.save(model.model.state_dict(), ckpt_model_path)
        model.model = model.model.to(device)
        model.model.train()

# 1.Attack training example and Interaction example:
badnets = BadNets(
    task,
    attack_config,
    schedule=schedule,  
)
badnets.interactions.append(test_in_train)
badnets.interactions.append(save_model_in_train)
badnets.attack()

# 2.Normal training example:
# badnets = BadNets(
#     task,
#     attack_config,
#     schedule=schedule,  
# )
# badnets.train()
