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
import time
from models import BaselineMNISTNetwork
from core.base import Base
from core.attacks.BadNets import BadNets
from core.attacks.BackdoorAttack import BackdoorAttack
from core.defenses.BackdoorDefense import BackdoorDefense
from core.defenses import Mine
import random
from utils import Log, parser, save_img, get_latent_rep
from utils import compute_confusion_matrix, compute_indexes,compute_accuracy
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
CUDA_VISIBLE_DEVICES = '0'
datasets_root_dir = BASE_DIR + '/datasets/'
batch_size = 128
num_workers = 4

work_dir = os.path.join(BASE_DIR,'experiments/Mine/BaselineMNISTNetwork_MNIST_BadNets_Mine')
datasets_dir = os.path.join(work_dir,'datasets')
poison_datasets_dir = os.path.join(datasets_dir,'poisoned_MNIST')
latents_dir = os.path.join(datasets_dir,'latents')
predict_dir = os.path.join(datasets_dir,'predict')
model_dir = os.path.join(work_dir,'model')
show_dir = os.path.join(work_dir,'show')
defense_object_dir = os.path.join(work_dir,'defense_object')
dirs = [work_dir, datasets_dir, poison_datasets_dir, latents_dir, predict_dir, model_dir, show_dir,defense_object_dir]
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

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
poisoned_trainset = torch.load(os.path.join(poison_datasets_dir,"train.pt")) 
poisoned_testset = torch.load(os.path.join(poison_datasets_dir,"test.pt"))
task = {
    'train_dataset': poisoned_trainset,
    'test_dataset' : poisoned_testset,
    'model' : BaselineMNISTNetwork(),
    'optimizer': optimizer,
    'loss' : nn.CrossEntropyLoss()
}
# Config related to model training
"""
这里注意训练的初始阶段：
'beta':0.01,
"init_size_clean_data_pool": 2000
对训练启动至关重要
"""
schedule = {
    # experiment
    'experiment': f'BaselineMNISTNetwork_MNIST_Badnets_Mine',
    # defense config:
    'defense_strategy':"Mine",
    'beta':0.0,
    'poison_rate':0.05,
    "threshold_prob":0.8, #  (0.5,8.0337),(0.6,7.8915),(0.7,7.7391),(0.8,7.5765),(0.9,7.4035)
    #这里其实可以设置为动态的阈值，比如：0.9-0.8衰减

    "init_size_clean_data_pool": 2500,
    "mean_disp":100.0,
    'layer':'fc2',
    #train config:
    'seed': global_seed,
    'deterministic': deterministic,
    
    # related to device
    'device': None,
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': None,

    # related to tarining 
    'pretrain': None,
    'epochs':100,
    'batch_size': 256,
    'num_workers': 2,
    'lr': 0.1,
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
    #     python test_Mine.py --task "show backdoor samples"
   
        
    # 2. The task of defense backdoor attack
    #     python test_Mine.py --task "repair"

    # 3.The task of testing defense effect
    #     python test_Mine.py --task "test"

    # 4.The task of evaluating data filtering
    #     python test_Mine.py --task "evaluate data filtering"

    # 5.The task of visualizing latents by t-sne
    #     python test_Mine.py --task "visualize latents by t-sne"
    #     python test_Mine.py --task "visualize latents for target class by t-sne"

    # 6.The task of comparing predict_digits
    #     python test_Mine.py --task "generate predict_digits"
    #     python test_Mine.py --task "compare predict_digits"

    log = Log(osp.join(work_dir, 'log.txt'))
    experiment = schedule['experiment']
    t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    msg = "\n\n\n==========Start {0} at {1}==========\n".format(experiment,t)
    log(msg)
    mine = Mine(
        task, 
        schedule)
    # Show the structure of the model
    print(task["model"])
    defense = BackdoorDefense(mine)
    args = parser.parse_args()
    if args.task == "show backdoor samples":
        # log("\n==========Show posioning train sample==========\n")
        # # Alreadly exsiting dataset and trained model.
        # poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt')) 
        # poison_indices = poisoned_train_dataset.get_poison_indices()
        # print(len(poison_indices))
        # index = poison_indices[random.choice(range(len(poison_indices)))]
        # print(f"index:{index}")
        # # print(poisoned_train_dataset[index])
        # image,label,_ = poisoned_train_dataset[index]
        # image = image.numpy()
        # backdoor_sample_path = os.path.join(show_dir, "backdoor_train_sample.png")
        # title = "label: " + str(label)
        # save_img(image, title=title, path=backdoor_sample_path)


        poisoned_test_dataset = torch.load(os.path.join(poison_datasets_dir,'test.pt'))
        poisoned_test_indices = poisoned_test_dataset.get_poison_indices()
        benign_test_indexs = list(set(range(len(poisoned_test_dataset))) - set(poisoned_test_indices))

        real_targets = poisoned_test_dataset.get_real_targets()
        labels = real_targets[poisoned_test_indices]
        log(f"Total samples:{len(poisoned_test_dataset)}, poisoning samples:{len(poisoned_test_indices)}, benign samples:{len(benign_test_indexs)}\n")
        for i, label in enumerate(poisoned_test_dataset.classes):
            print(f"the number of sample with label:{label} in poisoned_train_dataset:{labels.tolist().count(i)}\n")
        
        # Outside of neural networks, packages including numpy and matplotlib are usually used for data operations, 
        # so the type of data is usually converted to np.ndrray()
        # print(len(poisoned_test_indices))
        index = poisoned_test_indices[random.choice(range(len(poisoned_test_indices)))]
        # print(f"index:{index}")
        image, label,_ = poisoned_test_dataset[index] 
        # Statistically generated poisoned datasets information
        image = image.numpy()
        backdoor_sample_path = os.path.join(show_dir, "backdoor_test_sample.png")
        title = "label: " + str(label)
        save_img(image, title=title, path=backdoor_sample_path)
        
    elif args.task == "repair":
        # get backdoor sample
        log("\n==========get poisoning train_dataset and test_dataset dataset and repair model ==========\n")
        # Alreadly exsiting dataset and trained model.
        # poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt')) 
        # poisoned_test_dataset = torch.load(os.path.join(poison_datasets_dir,'test.pt'))
        # poisoned_train_indices = poisoned_train_dataset.get_poison_indices()
        # poisoned_test_indices = poisoned_test_dataset.get_poison_indices()
        # print(len(poisoned_train_indices))
        # repaired_model = defense.get_repaired_model(dataset, schedule)
        repaired_model = defense.get_repaired_model()
        torch.save(repaired_model.state_dict(), os.path.join(model_dir, 'repaired_model.pth'))
        log("Save repaired model to" + os.path.join(model_dir, 'repaired_model.pth'))

        target_label = defense.get_target_label()
        pred_poisoned_sample_dist = defense.get_pred_poisoned_sample_dist()
        torch.save(defense, os.path.join(defense_object_dir, 'Mine_object.pth'))
        log("Save Mine object to" + os.path.join(defense_object_dir, 'Mine_object.pth'))
        torch.save(pred_poisoned_sample_dist, os.path.join(defense_object_dir, 'pred_poisoned_sample_dist.pth'))
        log("Save pred_poisoned_sample_dist to" + os.path.join(predict_dir, 'pred_poisoned_sample_dist.pth'))
        

    elif args.task == "test":
        # Test the attack effect of backdoor model on backdoor datasets.
        log("\n==========Test the effect of defense on poisoned_test_dataset==========\n")

        poisoned_test_dataset = torch.load(os.path.join(poison_datasets_dir,'test.pt'))
        testset = poisoned_test_dataset
        poisoned_test_indexs = list(testset.get_poison_indices())
        benign_test_indexs = list(set(range(len(testset))) - set(poisoned_test_indexs))
        #Alreadly exsiting trained model
        model = BaselineMNISTNetwork()
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


    elif args.task == "generate latents":
        log("\n==========Get the latent representation in the middle layer of the model.==========\n")
        #Alreadly exsiting trained model and poisoned datasets
        # device = torch.device("cuda:1")
        # model = BaselineMNISTNetwork()
        device = torch.device("cpu")
        model = BaselineMNISTNetwork()
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
 




