# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/08/19 15:49:57
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : test_BadNets.py
# @Description : This is the test code of BadNets.              
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, Normalize, ToPILImage, Resize
from torchvision import transforms 
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
# print(sys.path)
from core.attacks import BadNets,BackdoorAttack
from core.base import Observer, Base
from models import BaselineMNISTNetwork, ResNet

import random
import time
import datetime
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn import manifold
from utils import show_img,save_img,accuracy,compute_accuracy,get_latent_rep, plot_2d,count_model_predict_digits
from utils import Log,parser
from config import get_task_config,get_task_schedule,get_attack_config
import pdb

args = parser.parse_args()
dataset = args.dataset

if dataset == "MNIST":
    # ========== BaselineMNISTNetwork_MNIST_BadNets ==========
    #{model}_{datasets}_{defense}_for_{attack} 
    experiment = 'BaselineMNISTNetwork_MNIST_BadNets'
    task = 'BaselineMNISTNetwork_MNIST'
    attack = 'BadNets'
    defense = None
    dir = 'BadNets'
    layer = "fc2"

elif dataset == "CIFAR10":
    #{model}_{datasets}_{attack}_{defense}
    experiment = 'ResNet-18_CIFAR-10_BadNets' 
    task = 'ResNet-18_CIFAR-10'
    attack = 'BadNets'
    defense = None
    dir = 'BadNets'
    layer = "linear"
    
elif dataset == "CIFAR100":
    experiment = 'ResNet-18_CIFAR-100_BadNets'
    task = 'ResNet-18_CIFAR-100'
    attack = 'BadNets'
    defense = None
    dir = 'BadNets'
    
elif dataset == "ImageNet":
    pass
   
work_dir = os.path.join(BASE_DIR,'experiments/' + task + '/'+ dir)
# work_dir = os.path.join(BASE_DIR,'experiments/Mine/BaselineMNISTNetwork_MNIST_BadNets_Mine')
datasets_dir = os.path.join(work_dir,'datasets')
poison_datasets_dir = os.path.join(datasets_dir, 'poisoned_data')
predict_dir = os.path.join(datasets_dir,'predict')
latents_dir = os.path.join(datasets_dir,'latents')
model_dir = os.path.join(work_dir,'model')
show_dir = os.path.join(work_dir,'show')

dirs = [work_dir, datasets_dir, poison_datasets_dir, predict_dir, latents_dir, model_dir, show_dir]
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)
# pdb.set_trace()

task_config = get_task_config(task = task)
schedule = get_task_schedule(task = task)
schedule['experiment'] = experiment
schedule['work_dir'] = work_dir
attack_schedule = get_attack_config(attack_strategy= attack, dataset = dataset)
attack_schedule['work_dir'] = work_dir
attack_schedule['train_schedule'] = schedule
#  0.4, 0.6, 0.8
cover_rate = 0.8

if __name__ == "__main__":
    """
    Users can select the task to execute by passing parameters.
    1. The task of generating and showing backdoor samples
        python test_BadNets.py --dataset "MNIST"  --subtask "generate backdoor samples" 
        python test_BadNets.py --dataset "CIFAR10"  --subtask "generate backdoor samples" 

    2. The task of showing backdoor samples
        python test_BadNets.py --dataset "MNIST" --subtask "show train backdoor samples"
        python test_BadNets.py --dataset "CIFAR10" --subtask "show train backdoor samples"
        python test_BadNets.py --dataset "MNIST" --subtask "show test backdoor samples"
        python test_BadNets.py --dataset "CIFAR10" --subtask "show test backdoor samples"
        
    3. The task of training backdoor model
        3.1 python test_BadNets.py --dataset "MNIST" --subtask "attack"
        3.2 python test_BadNets.py --dataset "MNIST" --subtask "adptive attack"

    4.The task of testing backdoor model
        python test_BadNets.py --dataset "MNIST" --subtask "test"

    5.The task of generating latents
        python test_BadNets.py --dataset "MNIST" --subtask "generate latents"
        
    6.The task of visualizing latents by t-sne
        python test_BadNets.py --dataset "MNIST" --subtask "visualize latents by t-sne"

        python test_BadNets.py --subtask "visualize latents for target class by t-sne"

    7.The task of comparing predict_digits
        python test_BadNets.py --subtask "generate predict_digits"
        python test_BadNets.py --subtask "compare predict_digits"

    """
    log = Log(osp.join(work_dir, 'log.txt'))
    t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    msg = "\n\n\n==========Start {0} at {1}==========\n".format(experiment,t)
    log(msg)
    # Show the structure of the model
    log(str(task_config['model']))

    badnets = BadNets(
        task_config,
        attack_schedule
    )
    backdoor = BackdoorAttack(badnets)
 
    if args.subtask == "generate backdoor samples":
        # Generate backdoor sample
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
        
        #Save poisoned dataset
        torch.save(poisoned_train_dataset, os.path.join(poison_datasets_dir,'train.pt'))
        log("Save generated train_datasets to" + os.path.join(poison_datasets_dir,'train.pt'))

        poisoned_test_dataset = backdoor.create_poisoned_test_dataset()
        poison_test_indices =  poisoned_test_dataset.get_poison_indices()
        benign_test_indexs = list(set(range(len(poisoned_test_dataset))) - set(poison_test_indices))
        log("Total samples:{0}, poisoning samples:{1}, benign samples:{2}".format(len(poisoned_test_dataset),\
        len(poison_test_indices),len(benign_test_indexs)))
        torch.save(poisoned_test_dataset, os.path.join(poison_datasets_dir,'test.pt'))
        log("Save generated test_datasets to" + os.path.join(poison_datasets_dir,'test.pt'))

    elif args.subtask == "show train backdoor samples":
        log("\n==========Show posioning train sample==========\n")
        # Alreadly exsiting dataset and trained model.
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt')) 
        poison_indices = poisoned_train_dataset.get_poison_indices()
        index = poison_indices[random.choice(range(len(poison_indices)))]
        
        log(f"Random index:{index}") 
        # Outside of neural networks, packages including numpy and matplotlib are usually used for data operations, 
        # so the type of data is usually converted to np.ndrray()
        image, label, _ = poisoned_train_dataset[index]

        image = image.numpy()
        backdoor_sample_path = os.path.join(show_dir, "backdoor_train_sample.png")
        title = "label: " + str(label)
        save_img(image, title=title, path=backdoor_sample_path)

    elif args.subtask == "show test backdoor samples":
        log("\n==========Show posioning test sample==========\n")
        # Alreadly exsiting dataset and trained model.
        poisoned_test_dataset = torch.load(os.path.join(poison_datasets_dir,'test.pt'))
        poison_indices = poisoned_test_dataset.get_poison_indices()
        index = poison_indices[random.choice(range(len(poison_indices)))]
        log(f"Random index:{index}") 

        image, label, _ = poisoned_test_dataset[index] 
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

    elif args.subtask == "adptive attack":
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt'))
        poison_indices = poisoned_train_dataset.get_poison_indices()
        indces = np.random.choice(np.arange(len(poison_indices)),size = int(len(poison_indices) * cover_rate))
        real_targets = poisoned_train_dataset.get_real_targets()
        log(f"The number of poisoned data:{len(poison_indices)}, modify indexes:{len(indces)}, modify indexes:{indces}, real targets:{real_targets[indces]}\n")
        poisoned_train_dataset.modify_targets(indces, real_targets[indces])
        backdoor.attack(dataset = poisoned_train_dataset)
        poisoned_model = backdoor.get_backdoor_model()
        torch.save(poisoned_model.state_dict(), os.path.join(model_dir, f'backdoor_model_cover_{cover_rate}.pth'))
        log("Save backdoor model to" + os.path.join(model_dir, f'backdoor_model_cover_{cover_rate}.pth'))
    
    elif args.subtask == "test":
        # Test the attack effect of backdoor model on backdoor datasets.
        log("\n==========Test the effect of backdoor attack on poisoned_test_dataset==========\n")
        poisoned_test_dataset = torch.load(os.path.join(poison_datasets_dir,'test.pt'))
        testset = poisoned_test_dataset
        poisoned_test_indexs = testset.get_poison_indices()
        benign_test_indexs = list(set(range(len(testset))) - set(poisoned_test_indexs))
       
        #Alreadly exsiting trained model
        model = task_config['model']
        model.load_state_dict(torch.load(os.path.join(model_dir, f'backdoor_model.pth')),strict=False)
        predict_digits, labels = backdoor.test(model=model, test_dataset=testset)

        benign_acc= compute_accuracy(predict_digits[benign_test_indexs],labels[benign_test_indexs],topk=(1,3,5))
        poisoned_acc = compute_accuracy(predict_digits[poisoned_test_indexs],labels[poisoned_test_indexs],topk=(1,3,5))
        log("Total samples:{0}, poisoning samples:{1},  benign samples:{2}".format(len(testset),len(poisoned_test_indexs),\
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
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt'))
        poison_indices = poisoned_train_dataset.get_poison_indices()
        model.load_state_dict(torch.load(os.path.join(work_dir, f'model/backdoor_model.pth')),strict=False)
        # Get the latent representation in the middle layer of the model.
        latents,y_labels = get_latent_rep(model, layer, poisoned_train_dataset, device=device)
        latents_path = os.path.join(latents_dir, f"latents.npz")

        np.savez(latents_path, latents=latents, y_labels=y_labels)

    elif args.subtask == "visualize latents by t-sne":
        log("\n========== Clusters of latent representations for all classes.==========\n")  
        # # Alreadly exsiting latent representation
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt'))
        poison_indices = poisoned_train_dataset.get_poison_indices()
        latents_path = os.path.join(latents_dir, f"latents.npz")
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
        path = os.path.join(show_dir, f"latent_2d_all_clusters_poison_rate_0.1.png")
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
        t_sne = manifold.TSNE(n_components=n_components, perplexity=30, early_exaggeration=120, init="pca", n_iter=250, random_state=0 )
        points = t_sne.fit_transform(latents[indexs])
        points = points*100

        colors = ["blue","red"]
        cmap = mcolors.ListedColormap(colors)
        title = "t-SNE diagram of latent representation"
        path = os.path.join(show_dir,"latent_2d_clusters.png")
        plot_2d(points, color_numebers, title=title, cmap=cmap, path=path)

    elif args.subtask == "generate predict_digits":
        log("\n==========generate predict_digits.==========\n") 
        device = torch.device("cpu")
        model = BaselineMNISTNetwork()
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'test.pt'))
        poison_indices = poisoned_train_dataset.get_poison_indices()
        model.load_state_dict(torch.load(os.path.join(work_dir, 'model/backdoor_model.pth')),strict=False)
        testset = poisoned_train_dataset
        predict_digits, labels = backdoor.test(model=model, test_dataset=testset)
        predict_digits_path = os.path.join(predict_dir,"MNIST_test_predict_digits.npz")
        # print(type(predict_digits))
        # print(predict_digits.shape)
        np.savez(predict_digits_path, predict_digits=predict_digits.numpy(), y_labels=labels.numpy())
        
    elif args.subtask == "compare predict_digits": 
        log("\n==========compare predict_digits.==========\n") 
       
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'test.pt'))
        poison_indices = poisoned_train_dataset.get_poison_indices()
        predict_digits_path = os.path.join(predict_dir,"MNIST_test_predict_digits.npz")
        data = np.load(predict_digits_path)
        predict_digits,y_labels = data["predict_digits"],data["y_labels"]

        save_path = os.path.join(predict_dir,"MNIST_test_predict_digits_statistics.npz")
        y_target = attack_schedule["y_target"]
        res = count_model_predict_digits(predict_digits,y_labels,poison_indices,y_target,save_path)
        y_target = res["y_target"]
        predicts, y_post_prob_matrix, entropy_matrix, entropy_vector = res[y_target]["predicts"], res[y_target]["y_post_prob_matrix"],\
            res[y_target]["entropy_matrix"],res[y_target]["entropy_vector"]
        poison_predicts, poison_y_post_prob_matrix, poison_entropy_matrix, poison_entropy_vector = res[y_target]["poison_predicts"],\
            res[y_target]["poison_y_post_prob_matrix"], res[y_target]["poison_entropy_matrix"], res[y_target]["poison_entropy_vector"]
        
        log("y_target:{0}\n".format(y_target)) 

        log("\n==========predicts vs poison_predicts.==========\n") 
        log("predict:{0}\n".format(predicts[0]))
        log("poison predict:{0}\n".format(poison_predicts[0])) 

        log("\n==========y_post_prob_matrix vs poison_y_post_prob_matrix.==========\n") 
        log("p(y|x):{0}\n".format(y_post_prob_matrix[0]))
        log("poison p(y|x):{0}\n".format(poison_y_post_prob_matrix[0])) 
        
        log("\n==========entropy vs poison_entropy.==========\n") 
        log("entropy:{0},sum:{1}\n".format(entropy_matrix[0],entropy_vector[0]))
        log("poison entropy:{0},sum:{1}\n".format(poison_entropy_matrix[0],poison_entropy_vector[0])) 







        

       
    
    
 

 
    
  


    
  

 

  
  

