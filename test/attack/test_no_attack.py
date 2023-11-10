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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from utils import parser
from config import get_task_config,get_task_schedule,get_untransformed_dataset
# ========== Set global settings ==========
datasets_root_dir = os.path.join(BASE_DIR + '/datasets/')
args = parser.parse_args()
dataset = args.dataset

if dataset == "MNIST":
    # ========== BaselineMNISTNetwork_MNIST ==========
    #{model}_{datasets}
    experiment = 'BaselineMNISTNetwork_MNIST'
    task = 'BaselineMNISTNetwork_MNIST'
    dir = 'No_attack'
    layer = "fc2"
    poison_datasets_dir = ""

elif dataset == "CIFAR-10":
    # ========== ResNet-18_CIFAR-10 ==========
    experiment = 'ResNet-18_CIFAR-10' 
    task = 'ResNet-18_CIFAR-10'
    dir = 'No_attack'
    layer = "linear"
    poison_datasets_dir = ""
    
elif dataset == "CIFAR-100":
    # ========== ResNet-18_CIFAR-100 ==========
    experiment = 'ResNet-18_CIFAR-100'
    task = 'ResNet-18_CIFAR-100'
    dir = 'No_attack'
    layer = "layer"
    
elif dataset == "ImageNet":
    pass

work_dir = os.path.join(BASE_DIR,'experiments/' + task + '/'+ dir)
# work_dir = os.path.join(BASE_DIR,'experiments/Mine/BaselineMNISTNetwork_MNIST_BadNets_Mine')
datasets_dir = os.path.join(work_dir,'datasets')
predict_dir = os.path.join(datasets_dir,'predict')
latents_dir = os.path.join(datasets_dir,'latents')
model_dir = os.path.join(work_dir,'model')
show_dir = os.path.join(work_dir,'show')

dirs = [work_dir, datasets_dir, predict_dir, latents_dir, model_dir, show_dir]
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

task_config = get_task_config(task = task)
schedule = get_task_schedule(task = task)
schedule['experiment'] = experiment
schedule['work_dir'] = work_dir
schedule['layer'] = layer
# print(f"schedule:{schedule}, momentum:{type(schedule['momentum'])}\n")

if __name__ == "__main__":
    """
    Users can select the task to execute by passing parameters.
    1. The task of Training and getting backdoor samples
        python test_no_attack.py --dataset "MNIST"  --subtask "Train and get model" 
        python test_no_attack.py --dataset "CIFAR-10"  --subtask "Train and get model" 

    2. The task of showing samples
        python test_no_attack.py --dataset "MNIST" --subtask "show train samples"
        python test_no_attack.py --dataset "CIFAR-10" --subtask "show train samples"

        python test_no_attack.py --dataset "MNIST" --subtask "show test samples"
        python test_no_attack.py --dataset "CIFAR-10" --subtask "show test samples"
        
       
    3. The task of "clean test"
        python test_no_attack.py --dataset "MNIST" --subtask "clean test"
        python test_no_attack.py --dataset "CIFAR-10" --subtask "clean test"

    4.The task of "poisoned test"
        python test_no_attack.py --dataset "MNIST" --subtask "poisoned test"
        python test_no_attack.py --dataset "CIFAR-10" --subtask "clean test"    

    5.The task of generating latents
        python test_no_attack.py --dataset "MNIST" --subtask "generate latents"
        python test_no_attack.py --dataset "CIFAR-10" --subtask "generate latents"

    6.The task of visualizing latents by t-sne
        python test_no_attack.py --dataset "MNIST" --subtask "visualize latents"
        python test_no_attack.py --dataset "CIFAR-10" --subtask "visualize latents"

        python test_no_attack.py --dataset "MNIST" --subtask "visualize latents for target class"
        python test_no_attack.py --dataset "CIFAR-10" --subtask "visualize latents for target class"

    """
    log = Log(osp.join(work_dir, 'log.txt'))
    t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    msg = "\n\n\n==========Start {0} at {1}==========\n".format(experiment,t)
    log(msg)
    # Show the structure of the model
    log(str(task_config['model']))

    base = Base(task_config,schedule)
 
    if args.subtask == "Train and get model":
        log("\n========== Train and get model==========\n")
        base.train()
        model = base.get_model()
        torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
        log("Save model to" + os.path.join(model_dir, 'model.pth'))

    elif args.subtask == "show train samples":
        log("\n==========Show train sample==========\n")
        # Alreadly exsiting dataset and trained model.
        train_dataset, test_dataset, _ =get_untransformed_dataset(task=task)
        index = random.choice(range(len(train_dataset)))
        
        log(f"Random index:{index}") 
        # Outside of neural networks, packages including numpy and matplotlib are usually used for data operations, 
        # so the type of data is usually converted to np.ndrray()
        image, label, _ = train_dataset[index]

        image = image.numpy()
        sample_path = os.path.join(show_dir, "train_sample.png")
        title = "label: " + str(label)
        save_img(image, title=title, path=sample_path)

    elif args.subtask == "show test samples":
        log("\n==========Show posioning test sample==========\n")
        # Alreadly exsiting dataset and trained model.
        train_dataset, test_dataset, _ = get_untransformed_dataset(task=task)
        index = random.choice(range(len(test_dataset)))
        log(f"Random index:{index}") 

        image, label, _ = test_dataset[index] 
        image = image.numpy()
        sample_path = os.path.join(show_dir, "test_sample.png")
        title = "label: " + str(label)
        save_img(image, title=title, path=sample_path)
    
    elif args.subtask == "clean test":
        # Test the effect of model on clean datasets.
        log("\n==========Test the effect of model on clean test_dataset==========\n")
        #Alreadly exsiting trained model
        model = task_config['model']
        model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')),strict=False)
        test_dataset = task_config['test_dataset']

        predict_digits, labels = base.test(model=model, test_dataset=test_dataset)
        acc= compute_accuracy(predict_digits, labels, topk=(1,3,5))
        log(f"Total samples:{0}, Benign_accuracy:{1}".format(len(test_dataset),acc))  

    elif args.subtask == "poisoned test": 
        # Test the effect of model on clean datasets.
        log("\n==========Test the effect of model on clean test_dataset==========\n")
        #Alreadly exsiting trained model
        model = task_config['model']
        model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')),strict=False)

        poisoned_test_dataset = torch.load(os.path.join(poison_datasets_dir,'test.pt'))
        test_dataset = poisoned_test_dataset
        predict_digits, labels = base.test(model = model, test_dataset = test_dataset)
        
        poisoned_test_indexs = test_dataset.get_poison_indices()
        benign_test_indexs = list(set(range(len(test_dataset))) - set(poisoned_test_indexs))
        
        benign_acc= compute_accuracy(predict_digits[benign_test_indexs],labels[benign_test_indexs],topk=(1,3,5))
        poisoned_acc = compute_accuracy(predict_digits[poisoned_test_indexs],labels[poisoned_test_indexs],topk=(1,3,5))
        log("Total samples:{0}, poisoning samples:{1},  benign samples:{2}".format(len(test_dataset),len(poisoned_test_indexs),\
                                                                                len(benign_test_indexs)))                                                                                                                                                
        log("Benign_accuracy:{0}, poisoning_accuracy:{1}".format(benign_acc, poisoned_acc))
    
    elif args.subtask == "generate latents":
        log("\n==========Get the latent representation in the middle layer of the model.==========\n")
        #Alreadly exsiting trained model and poisoned datasets
        # device = torch.device("cuda:1")
        # model = nn.DataParallel(BaselineMNISTNetwork(), output_device=device)

        device = torch.device("cpu")
        model = task_config['model']
        model.to(device)
        model.load_state_dict(torch.load(os.path.join(work_dir, 'model/backdoor_model.pth')),strict=False)

        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt'))
        poison_indices = poisoned_train_dataset.get_poison_indices()
        
        # Get the latent representation in the middle layer of the model.
        latents,y_labels = get_latent_rep(model, layer, poisoned_train_dataset, device=device)
        latents_path = os.path.join(latents_dir,"latents.npz")

        np.savez(latents_path, latents=latents, y_labels=y_labels)

    elif args.subtask == "visualize latents":
        log("\n========== Clusters of latent representations for all classes.==========\n")  
        # # Alreadly exsiting latent representation
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt'))
        poison_indices = poisoned_train_dataset.get_poison_indices()
        latents_path = os.path.join(latents_dir,"latents.npz")
        data = np.load(latents_path)
        latents,y_labels = data["latents"],data["y_labels"]
   
        # get low-dimensional data points by t-SNE
        n_components = 2 # number of coordinates for the manifold
        t_sne = manifold.TSNE(n_components=n_components, perplexity=30, early_exaggeration=120, init="pca", n_iter=250, random_state=0 )
        points = t_sne.fit_transform(latents)

        # points = points*1000
        # print(t_sne.kl_divergence_)
        
        #Display data clusters for all category by scatter plots
        num = len(poisoned_train_dataset.classes)
        # Custom color mapping
        colors = [plt.cm.tab10(i) for i in range(num)]
        colors.append("red") 
        y_labels[poison_indices] = num
        # Create a ListedColormap objectall_
        cmap = mcolors.ListedColormap(colors)
        title = "t-SNE diagram of latent representation"
        path = os.path.join(show_dir,"latent_2d_all_clusters.png")
        plot_2d(points, y_labels, title=title, cmap=cmap, path=path)
           
    elif args.subtask == "visualize latents for target class":
        #Clusters of latent representations for target class
        log("\n==========Verify the assumption of latent separability.==========\n")  
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt'))
        poison_indices = poisoned_train_dataset.get_poison_indices()
        y_target = poisoned_train_dataset.get_y_target()

        # Alreadly exsiting latent representation
        data = np.load(os.path.join(latents_dir,"latents.npz"))
        latents,y_labels = data["latents"],data["y_labels"]
        indexs = np.where(y_labels == y_target)[0]

        color_numebers = [0 if index in poison_indices else 1 for index in indexs]
        color_numebers = np.array(color_numebers)

        # get low-dimensional data points by t-SNE
        n_components = 2 # number of coordinates for the manifold
        t_sne = manifold.TSNE(n_components=n_components, perplexity=30, early_exaggeration=120, init="pca", n_iter=250, random_state=0 )
        points = t_sne.fit_transform(latents[indexs])
        points = points*100

        colors = ["blue","red"]
        cmap = mcolors.ListedColormap(colors)
        title = "t-SNE diagram of latent representation"
        path = os.path.join(show_dir,"latent_2d_clusters.png")
        plot_2d(points, color_numebers, title=title, cmap=cmap, path=path)

        
   