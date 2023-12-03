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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
# print(sys.path)
from core.attacks import AdaptivePatch
from core.attacks import BackdoorAttack
from models import ResNet
import time
import datetime
import os.path as osp
import random
import numpy as np
from core.base import Observer
from core.base import Base
from PIL import Image
from utils import parser
from config import get_task_config, get_task_schedule, get_attack_config 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn import manifold
from utils import show_img,save_img,accuracy,compute_accuracy,get_latent_rep, plot_2d,count_model_predict_digits
from utils import Log,parser
# ========== Set global settings ==========
args = parser.parse_args()
dataset = args.dataset
attack = "Adaptive-Patch"

if dataset == "MNIST":
    # ========== BaselineMNISTNetwork_MNIST_BadNets ==========
    #{model}_{datasets}_{defense}_for_{attack} 
    experiment = f'BaselineMNISTNetwork_MNIST_{attack}'
    task = 'BaselineMNISTNetwork_MNIST'
    attack = 'Adaptive-Patch'
    defense = None
    dir = 'Adaptive-Patch'
    layer = "fc2"

elif dataset == "CIFAR10":
    #{model}_{datasets}_{attack}_{defense}
    experiment = f'ResNet-18_CIFAR-10_{attack}' 
    task = 'ResNet-18_CIFAR-10'
    attack = 'Adaptive-Patch'
    defense = None
    dir = 'Adaptive-Patch'
    layer = "linear"
    datasets_root_dir = '/home/zzq/CreatingSpace/BackdoorToolbox/datasets'

elif dataset == "CIFAR100":
    experiment = f'ResNet-18_CIFAR-100_{attack}'
    task = 'ResNet-18_CIFAR-100'
    attack = 'Adaptive-Patch'
    defense = None
    dir = 'Adaptive-Patch'
    
elif dataset == "ImageNet":
    pass

work_dir = os.path.join(BASE_DIR,'experiments/' + task + '/'+ dir)

task_config = get_task_config(task = task)
schedule = get_task_schedule(task = task)
attack_schedule = get_attack_config(attack_strategy= attack, dataset = dataset)
# work_dir = os.path.join(BASE_DIR,'experiments/Mine/BaselineCIFAR10Network_CIFAR10_BadNets_Mine')
datasets_dir = os.path.join(work_dir,'datasets')
schedule['experiment'] = experiment
schedule['work_dir'] = work_dir

attack_schedule['train_schedule'] = schedule
attack_schedule['trigger_dir'] = os.path.join(datasets_dir, 'triggers/')
attack_schedule['work_dir'] = work_dir

poisoning_rate = attack_schedule["poisoning_rate"]
cover_rate = attack_schedule["cover_rate"]

poison_datasets_dir = os.path.join(datasets_dir, 'poisoned_data')
poison_datasets_dir = os.path.join(poison_datasets_dir, f"poison_{poisoning_rate}_cover_{cover_rate}/")

predict_dir = os.path.join(datasets_dir,'predict')
predict_dir= os.path.join(predict_dir, f"poison_{poisoning_rate}_cover_{cover_rate}/")

latents_dir = os.path.join(datasets_dir,'latents')
latents_dir = os.path.join(latents_dir, f"poison_{poisoning_rate}_cover_{cover_rate}/")

model_dir = os.path.join(work_dir,'model')
model_dir = os.path.join(model_dir, f"poison_{poisoning_rate}_cover_{cover_rate}/")

show_dir = os.path.join(work_dir,'show')
show_dir = os.path.join(show_dir, f"poison_{poisoning_rate}_cover_{cover_rate}/")

dirs = [work_dir, datasets_dir, poison_datasets_dir, predict_dir, latents_dir, model_dir, show_dir]
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == "__main__":
    """
    Users can select the task to execute by passing parameters.
    1. The task of generating and showing backdoor samples
        python test_AdaptivePatch.py --dataset "CIFAR10"  --subtask "generate train backdoor samples" 
        python test_AdaptivePatch.py --dataset "CIFAR10"  --subtask "generate test backdoor samples" 

    2. The task of showing backdoor samples
        python test_AdaptivePatch.py --dataset "CIFAR10" --subtask "show train backdoor samples"
        python test_AdaptivePatch.py --dataset "CIFAR10" --subtask "show test backdoor samples"
        python test_AdaptivePatch.py --dataset "CIFAR10" --subtask "show cover samples"
        
    3. The task of training backdoor model
        python test_AdaptivePatch.py --dataset "CIFAR10" --subtask "attack"

    4.The task of testing backdoor model
        python test_AdaptivePatch.py --dataset "CIFAR10" --subtask "test"

    5.The task of generating latents
        python test_AdaptivePatch.py --dataset "CIFAR10" --subtask "generate latents"
        
    6.The task of visualizing latents by t-sne
        python test_AdaptivePatch.py --dataset "CIFAR10" --subtask "visualize latents by t-sne"
        python test_AdaptivePatch.py --subtask "visualize latents for target class by t-sne"

    7.The task of comparing predict_digits
        python test_AdaptivePatch.py --subtask "generate predict_digits"
        python test_AdaptivePatch.py --subtask "compare predict_digits"

    """

    os.makedirs(work_dir, exist_ok=True)
    log = Log(osp.join(work_dir, 'log.txt'))
    t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    msg = "Start experiment {0} at {1}\n".format(experiment,t)
    log(msg)
    # # print("Start to test")
    adaptive_patch = AdaptivePatch(
        task_config,
        attack_schedule
    )
    backdoor = BackdoorAttack(adaptive_patch)

    if args.subtask == "generate train backdoor samples":
        log("\n==========Generate backdoor samples==========\n")
        poisoned_train_dataset = backdoor.create_poisoned_train_dataset()
        poison_indices = poisoned_train_dataset.get_poison_indices()
        benign_indexs = list(set(range(len(poisoned_train_dataset))) - set(list(poison_indices)))
            
        # Statistically generated poisoned datasets information
        real_targets = np.array(poisoned_train_dataset.get_real_targets())
        labels = real_targets[poison_indices]
        log(f"Total samples:{len(poisoned_train_dataset)}, poisoning samples:{len(poison_indices)}, benign samples:{len(benign_indexs)}\n")
        for i, label in enumerate(poisoned_train_dataset.classes):
            print(f"the number of sample with label:{label} in poisoned_train_dataset:{labels.tolist().count(i)}\n")
        torch.save(poisoned_train_dataset, os.path.join(poison_datasets_dir,'train.pt'))
        # Save poisoned dataset
        log("Save generated train_datasets to" + os.path.join(poison_datasets_dir,'train.pt'))
    
    elif args.subtask == "generate test backdoor samples":   
        poisoned_test_dataset = backdoor.create_poisoned_test_dataset(poisoning_rate = 0.1)
        poison_test_indices =  poisoned_test_dataset.get_poison_indices()
        
        benign_test_indexs = list(set(range(len(poisoned_test_dataset))) - set(poison_test_indices))
        real_targets = np.array(poisoned_test_dataset.get_real_targets())
        labels = real_targets[poison_test_indices]
        
        log(f"Total samples:{len(poisoned_test_dataset)}, poisoning samples:{len(poison_test_indices)}, benign samples:{len(benign_test_indexs)}\n")
        for i, label in enumerate(poisoned_test_dataset.classes):
            print(f"the number of sample with label:{label} in poisoned_train_dataset:{labels.tolist().count(i)}\n")

        torch.save(poisoned_test_dataset, os.path.join(poison_datasets_dir,'test.pt'))
        log("Save generated test_datasets to" + os.path.join(poison_datasets_dir,'test.pt'))

    elif args.subtask == "show train backdoor samples":
        log("\n==========Show posioning train sample==========\n")
        # Alreadly exsiting dataset and trained model.
        # poison_datasets_dir = '/home/zzq/CreatingSpace/BackdoorToolbox/experiments/ResNet-18_CIFAR-10/Adaptive-Patch/datasets/poisonedCifar-10/'
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt')) 
        poison_indices = poisoned_train_dataset.get_poison_indices()
        index = poison_indices[random.choice(range(len(poison_indices)))]
        
        log(f"Random index:{index}") 
        # Outside of neural networks, packages including numpy and matplotlib are usually used for data operations, 
        # so the type of data is usually converted to np.ndrray()
        image, label = poisoned_train_dataset.get_sample_by_index(index)
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        backdoor_sample_path = os.path.join(show_dir, "backdoor_train_sample.png")
        title = "label: " + str(label)
        # print(f"image:{image}")
        save_img(image, title=title, path=backdoor_sample_path)

    elif args.subtask == "show cover samples":
        log("\n==========Show cover samples==========\n")
        # Alreadly exsiting dataset and trained model.
        # poison_datasets_dir = '/home/zzq/CreatingSpace/BackdoorToolbox/experiments/ResNet-18_CIFAR-10/Adaptive-Patch/datasets/poisonedCifar-10/'
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt')) 
        cover_indices = poisoned_train_dataset.get_cover_indices()
        
        index = cover_indices[random.choice(range(len(cover_indices)))]
        
        log(f"Random index:{index}") 
        # Outside of neural networks, packages including numpy and matplotlib are usually used for data operations, 
        # so the type of data is usually converted to np.ndrray()
        image, label = poisoned_train_dataset.get_sample_by_index(index)
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        cover_sample_path = os.path.join(show_dir, "cover_sample.png")
        title = "label: " + str(label)
        # print(f"image:{image}")
        save_img(image, title=title, path=cover_sample_path)

    elif args.subtask == "show test backdoor samples":
        log("\n==========Show posioning test sample==========\n")
        # Alreadly exsiting dataset and trained model.
        poisoned_test_dataset = torch.load(os.path.join(poison_datasets_dir,'test.pt'))
        poison_indices = poisoned_test_dataset.get_poison_indices()
        index = poison_indices[random.choice(range(len(poison_indices)))]
        log(f"Random index:{index}") 

        image, label = poisoned_test_dataset.get_sample_by_index(index)
        image = image.numpy()
        backdoor_sample_path = os.path.join(show_dir, "backdoor_test_sample.png")
        title = "label: " + str(label)
        save_img(image, title=title, path=backdoor_sample_path)

    elif args.subtask == "attack":
        #Train and get backdoor model
        log("\n==========Train on poisoned_train_dataset and get backdoor model==========\n")
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt')) 
        backdoor.attack(train_dataset = poisoned_train_dataset)
        poisoned_model = backdoor.get_backdoor_model()
        torch.save(poisoned_model.state_dict(), os.path.join(model_dir, 'backdoor_model.pth'))
        log("Save backdoor model to" + os.path.join(model_dir, 'backdoor_model.pth'))
    
    
    elif args.subtask == "test":
        # Test the attack effect of backdoor model on backdoor datasets.
        log("\n==========Test the effect of backdoor attack on poisoned_test_dataset==========\n")
        poisoned_test_dataset = torch.load(os.path.join(poison_datasets_dir,'test.pt'))
        testset = poisoned_test_dataset
        poisoned_test_indexs = testset.get_poison_indices()
        benign_test_indexs = list(set(range(len(testset))) - set(poisoned_test_indexs))

        #Alreadly exsiting trained model
        model = task_config['model']
        model.load_state_dict(torch.load(os.path.join(model_dir, 'backdoor_model.pth')),strict=False)

        predict_digits, labels = backdoor.test(model=model, test_dataset=testset)

        benign_acc= compute_accuracy(predict_digits[benign_test_indexs],labels[benign_test_indexs],topk=(1,3,5))
        poisoned_acc = compute_accuracy(predict_digits[poisoned_test_indexs],labels[poisoned_test_indexs],topk=(1,3,5))
        log("Total samples:{0}, poisoning samples:{1},  benign samples:{2}".format(len(testset),len(poisoned_test_indexs),\
                                                                                len(benign_test_indexs)))                                                                                                                                                
        log("Benign_accuracy:{0}, poisoning_accuracy:{1}".format(benign_acc, poisoned_acc))
    
    elif args.subtask == "generate latents":
        log("\n==========Get the latent representation in the middle layer of the model.==========\n")
        #Alreadly exsiting trained model and poisoned datasets
        device = torch.device("cuda:1")
        # model = nn.DataParallel(BaselineCIFAR10Network(), output_device=device)
        model = task_config['model']
        model.to(device)
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt'))
        poison_indices = poisoned_train_dataset.get_poison_indices()
        #'backdoor_model.pth'
        file = "backdoor_model.pth"
        model.load_state_dict(torch.load(os.path.join(model_dir,file)),strict=False)
        # Get the latent representation in the middle layer of the model.
        latents, y_labels = get_latent_rep(model, layer, poisoned_train_dataset, device=device)
        latents_path = os.path.join(latents_dir,"latents.npz")

        np.savez(latents_path, latents=latents, y_labels=y_labels)
        log(f"Save latents to {latents_path}\n")

    elif args.subtask == "visualize latents by t-sne":
        log("\n========== Clusters of latent representations for all classes.==========\n")  
        # # Alreadly exsiting latent representation
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt'))
        poison_indices = poisoned_train_dataset.get_poison_indices()
        latents_path = os.path.join(latents_dir,"latents.npz")
        data = np.load(latents_path)
        latents,y_labels = data["latents"],data["y_labels"]
   
        # get low-dimensional data points by t-SNE
        n_components = 2 # number of coordinates for the manifold
        
        #perplexity=50
        t_sne = manifold.TSNE(n_components=n_components, perplexity=100, early_exaggeration=120, init="pca", n_iter=1000, random_state=0)
        points = t_sne.fit_transform(latents)

        #Display data clusters for all category by scatter plots
        num = len(poisoned_train_dataset.classes)
        # Custom color mapping
        colors = [plt.cm.tab10(i) for i in range(num)]
        colors.append("black") 
        y_labels[poison_indices] = num
        # Create a ListedColormap objectall_
        cmap = mcolors.ListedColormap(colors)
        title = "t-SNE diagram of latent representation"
        path = os.path.join(show_dir,"latent_2d_all_clusters.png")
        plot_2d(points, y_labels, title=title, cmap=cmap, path=path)

    elif args.subtask == "visualize latents for target class by t-sne":
        #Clusters of latent representations for target class
        log("\n==========Verify the assumption of latent separability.==========\n")  
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt'))
        poison_indices = poisoned_train_dataset.get_poison_indices()
        latents_path = os.path.join(latents_dir,"latents.npz")

        # Alreadly exsiting latent representation
        data = np.load(latents_path)
        latents,y_labels = data["latents"],data["y_labels"]

        indexs = np.where(y_labels == attack_schedule['y_target'])[0]
        color_numebers = [0 if index in poison_indices else 1 for index in indexs]
        color_numebers = np.array(color_numebers)

        # get low-dimensional data points by t-SNE
        n_components = 2 # number of coordinates for the manifold
        t_sne = manifold.TSNE(n_components=n_components, perplexity=100, early_exaggeration=120, init="pca", n_iter=1000, random_state=0 )
        points = t_sne.fit_transform(latents[indexs])

        colors = ["black","blue"]
        cmap = mcolors.ListedColormap(colors)
        title = "t-SNE diagram of latent representation"
        path = os.path.join(show_dir,"latent_2d_clusters.png")
        plot_2d(points, color_numebers, title=title, cmap=cmap, path=path)
