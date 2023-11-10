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
from models import baseline_MNIST_network, ResNet
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
from utils import parser
# ========== Set global settings ==========
global_seed = 333
deterministic = True
# torch.manual_seed(global_seed)
CUDA_VISIBLE_DEVICES = '1'
datasets_root_dir = BASE_DIR + '/datasets/'
# ========== BaselineMNISTNetwork_MNIST_BadNets ==========
# The basic data type in torch is "tensor". In order to be computed, other data type, like PIL Image or numpy.ndarray,
#  must be Converted to "tensor"

#{model}_{datasets}_{attack}_{defense} 
# experiment = 'BaselineMNISTNetwork_MNIST_No_Attack'
# project = 'BaselineMNISTNetwork_MNIST'
experiment = 'ResNet-18_CIFAR-10_No_Attack'
project = 'ResNet-18_CIFAR-10'
attack = 'No_Attack'
work = attack

dataset = torchvision.datasets.CIFAR10
# transform_train = Compose([
#     ToTensor()
# ])
transform_train = Compose([
    RandomHorizontalFlip(),
    ToTensor()
])
trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
transform_test = Compose([
    ToTensor()
])
testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
optimizer = torch.optim.SGD
task = {
    'train_dataset': trainset,
    'test_dataset' : testset,
    'model' :  ResNet(18),
    'optimizer': optimizer,
    'loss' : nn.CrossEntropyLoss(),
}
layer = "fc2"


work_dir = os.path.join(BASE_DIR,'experiments/'+ project +'/'+ work)
# work_dir = os.path.join(BASE_DIR,'experiments/Mine/BaselineMNISTNetwork_MNIST_BadNets_Mine')
datasets_dir = os.path.join(work_dir,'datasets')
poison_datasets_dir = os.path.join(datasets_dir,'poisoned_data')
predict_dir = os.path.join(datasets_dir,'predict')
latents_dir = os.path.join(datasets_dir,'latents')
model_dir = os.path.join(work_dir,'model')
show_dir = os.path.join(work_dir,'show')

dirs = [work_dir, datasets_dir, poison_datasets_dir, predict_dir, latents_dir, model_dir, show_dir]
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

# ========== BaselineMNISTNetwork_MNIST_No_Attack ==========
# The basic data type in torch is "tensor". In order to be computed, other data type, like PIL Image or numpy.ndarray,
#  must be Converted to "tensor".

schedule = {
    'experiment': 'BaselineMNISTNetwork_MNIST_No_Attack',
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
    'work_dir': work_dir,
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
    """
    Users can select the task to execute by passing parameters.

    1. The task of training model
        python test_Mnist.py --task "train"

    2.The task of testing backdoor model
        python test_Mnist.py --task "test model"

    3.The task of testing model on poisoned dataset
        python test_Mnist.py --task "test model on poisoned dataset"

    4.The task of generating latents
        python test_Mnist.py --task "test model"

    5.The task of visualizing latents by t-sne
        python test_Mnist.py --task "visualize latents by t-sne"
    """
   
    log = Log(osp.join(work_dir, 'log.txt'))
    experiment = schedule['experiment']
    t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    msg = "\n\n\n==========Start {0} at {1}==========\n".format(experiment,t)
    log(msg)
    # Show the structure of the model
    log(str(task['model']))

    base = Base(task,schedule)
    args = parser.parse_args()
    if args.task == "train":
        #Train and get backdoor model
        log("\n==========Train on dataset and get model==========\n")
        base.train()
        model = base.get_model()
        torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
        log("Save model to" + os.path.join(model_dir, 'model.pth'))
    elif args.task == "test model":
        log("\n==========test the effect of model==========\n")
        # Alreadly exsiting trained model
        model = task['model']
        model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')),strict=False)
        # test on clean testset
        test_dataset=testset
        base.test(schedule=schedule,model=model,test_dataset=test_dataset)
        predict_digits, labels = base.test()
        accuracy = compute_accuracy(predict_digits,labels,topk=(1,3,5))
        log("Total samples:{0}, accuracy:{1}".format(len(testset),accuracy))
    
    elif args.task == "test model on poisoned dataset":
        # Test the attack effect of backdoor model on backdoor datasets.
        log("\n==========Test the effect of clean model on poisoned_test_dataset==========\n")
        poisoned_test_dataset = torch.load(os.path.join(poison_datasets_dir,'test.pt'))
        testset = poisoned_test_dataset
        poisoned_test_indexs = list(testset.get_poison_indices())
        benign_test_indexs = list(set(range(len(testset))) - set(poisoned_test_indexs))
        #Alreadly exsiting trained model
        model = task['model']
        # model.load_state_dict(torch.load(os.path.join(work_dir, 'model/backdoor_model.pth')),strict=False)
        model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')),strict=False)
        predict_digits, labels = base.test(model=model, test_dataset=testset)
        benign_acc = compute_accuracy(predict_digits[benign_test_indexs],labels[benign_test_indexs],topk=(1,3,5))
        poisoned_acc = compute_accuracy(predict_digits[poisoned_test_indexs],labels[poisoned_test_indexs],topk=(1,3,5))
        log(f"Total samples:{len(testset)}, poisoning samples:{len(poisoned_test_indexs)},  benign samples:{ len(benign_test_indexs)},Benign_accuracy:{benign_acc}, poisoning_accuracy:{poisoned_acc}\n")

    elif args.task == "generate latents":
        log("\n==========Get the latent representation in the middle layer of the model.==========\n")
        #Alreadly exsiting trained model and poisoned datasets
        device = torch.device("cpu")
        model = task['model']
        model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')),strict=False)
       
        # Get the latent representation in the middle layer of the model.
        latents,y_labels = get_latent_rep(model, layer, trainset, device=device)
        latents_path = os.path.join(latents_dir,"latents.npz")
        np.savez(latents_path, latents=latents, y_labels=y_labels)

    elif args.task == "visualize latents by t-sne":
        log("\n========== Clusters of latent representations for all classes.==========\n")  
        # # Alreadly exsiting latent representation
        data = np.load(os.path.join(latents_dir,"latents.npz"))
        latents,y_labels = data["latents"],data["y_labels"]
   
        # get low-dimensional data points by t-SNE
        n_components = 2 # number of coordinates for the manifold
        t_sne = manifold.TSNE(n_components=n_components, perplexity=30, early_exaggeration=120, init="pca", n_iter=250, random_state=0 )
        points = t_sne.fit_transform(latents)

        #Display data clusters for all category by scatter plots 
     
        # Custom color mapping
        colors = [plt.cm.tab10(i) for i in range(len(trainset.classes))]
        # Create a ListedColormap objectall_
        cmap = mcolors.ListedColormap(colors)
        title = "t-SNE diagram of latent representation"
        path = os.path.join(show_dir,"latent_2d_all_clusters.png")
        plot_2d(points, y_labels, title=title, cmap=cmap, path=path)
           
    