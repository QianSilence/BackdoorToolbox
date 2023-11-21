# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/11/14 10:33:51
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : MIMRL.py
# @Description : This is the test code of MIMRL backdoor defense.

import os
import sys
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Subset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
import time
from core.defenses.BackdoorDefense import BackdoorDefense
from core.defenses.MIMRL import MIMRL
import random
from utils import Log, parser, save_img, get_latent_rep, get_latent_rep_without_detach
from utils import compute_accuracy,SCELoss
from utils import show_img,save_img, compute_accuracy, plot_2d
from config import get_task_config,get_task_schedule,get_attack_config,get_defense_config
from core import SCELoss
from core import InfomaxLoss
from sklearn import manifold
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# ========== Set global settings ==========
datasets_root_dir = BASE_DIR + '/datasets/'

args = parser.parse_args()
defense = 'MIMRL'
if args.attack is None: 
    attack = args.attack
else:
    attack = 'BadNets'  # the default attack is BadNets

dir = f'{defense}_for_{attack}'
if args.dataset == "MNIST": 
    # ========== BaselineMNISTNetwork_MNIST_MIMRL_for_BadNets ==========
    task = 'BaselineMNISTNetwork_MNIST' # {model}_{dataset}
    experiment = f'BaselineMNISTNetwork_MNIST_{defense}_for_{attack}' # {task}_{defense}_for_{attack} 
    layer = "fc2"

elif args.dataset == "CIFAR-10":
    # ========== ResNet-18_CIFAR-10_MIMRL_for_BadNets ==========
    experiment = f'ResNet-18_CIFAR-10_{defense}_for_{attack}'
    task = 'ResNet-18_CIFAR-10'
    layer = "linear"
    
elif args.dataset == "CIFAR-100":
    # ========== ResNet-18_CIFAR-100_MIMRL_for_BadNets ==========
    experiment = f'ResNet-18_CIFAR-100_{defense}_for_{attack}'
    task = 'ResNet-18_CIFAR-100'
    layer = "linear"
    
elif args.dataset == "ImageNet":
    pass
   
work_dir = os.path.join(BASE_DIR,'experiments/' + task + '/' + dir)
poison_datasets_dir = os.path.join(BASE_DIR,f'experiments/{task}/{attack}/datasets/poisoned_data')
datasets_dir = os.path.join(work_dir,'datasets')
latents_dir = os.path.join(datasets_dir,'latents')
predict_dir = os.path.join(datasets_dir,'predict')
model_dir = os.path.join(work_dir,'model')
show_dir = os.path.join(work_dir,'show')
defense_object_dir = os.path.join(work_dir,'defense_object')
dirs = [work_dir, datasets_dir, poison_datasets_dir, latents_dir, predict_dir, model_dir, show_dir,defense_object_dir]
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

poisoned_trainset = torch.load(os.path.join(poison_datasets_dir,"train.pt")) 
poisoned_testset = torch.load(os.path.join(poison_datasets_dir,"test.pt"))

task_config = get_task_config(task=task)
task_config['train_dataset'] = poisoned_trainset
task_config['test_dataset'] = poisoned_testset

schedule = get_task_schedule(task = task)
schedule['experiment'] = experiment
schedule['work_dir'] = work_dir

defense_schedule = get_defense_config(defense_strategy = defense)
defense_schedule["layer"] = layer
schedule = {**schedule,**defense_schedule}

if __name__ == "__main__":

    # Users can select the task to execute by passing parameters.

    # 1. The task of showing backdoor samples
    #   python test_MIMRL.py --subtask "show backdoor samples" --dataset "MNIST"

    # 2. The task of defense backdoor attack
    #   python test_MIMRL.py --subtask "repair" --dataset "MNIST"

    # 3.The task of testing defense effect
    #   python test_MIMRL.py --subtask "test" --dataset "MNIST"

    # 4.The task of evaluating data filtering
    #   python test_MIMRL.py --subtask "evaluate data filtering" --dataset "MNIST"

    # 5.The task of generating latents
    #   python test_MIMRL.py  --subtask "generate latents" --dataset "MNIST"

    # 6.The task of visualizing latents by t-sne
    #   python test_MIMRL.py --subtask "visualize latents by t-sne" --dataset "MNIST"
    #   python test_MIMRL.py --subtask "visualize latents for target class by t-sne"

    # 7.The task of comparing predict_digits
    #   python test_MIMRL.py --subtask "generate predict_digits"
    #   python test_MIMRL.py --subtask "compare predict_digits"

    log = Log(osp.join(work_dir, 'log.txt'))
    t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    msg = "\n\n\n==========Start {0} at {1}==========\n".format(experiment,t)
    log(msg)
    mimrl = MIMRL(
        task_config, 
        schedule)
    # Show the structure of the model
    print(task_config["model"])
    defense = BackdoorDefense(mimrl)

    if args.subtask == "show backdoor samples":
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

    elif args.subtask == "repair":
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

        torch.save(defense, os.path.join(defense_object_dir, 'MIMRL_object.pth'))
        log("Save defense object to" + os.path.join(defense_object_dir, 'MIMRL_object.pth')) 

    elif args.subtask == "test":
        # Test the attack effect of backdoor model on backdoor datasets.
        log("\n==========Test the effect of defense on poisoned_test_dataset==========\n")

        poisoned_test_dataset = torch.load(os.path.join(poison_datasets_dir,'test.pt'))
        testset = poisoned_test_dataset
        poisoned_test_indexs = list(testset.get_poison_indices())
        benign_test_indexs = list(set(range(len(testset))) - set(poisoned_test_indexs))
        #Alreadly exsiting trained model
        model = task_config['model']
        #print(os.path.join(model_dir, 'backdoor_model.pth'))
        model.load_state_dict(torch.load(os.path.join(model_dir, 'repaired_model.pth')),strict=False)
        predict_digits, labels = defense.test(model=model, test_dataset=testset)

        benign_accuracy = compute_accuracy(predict_digits[benign_test_indexs],labels[benign_test_indexs],topk=(1,3,5))
        poisoning_accuracy = compute_accuracy(predict_digits[poisoned_test_indexs],labels[poisoned_test_indexs],topk=(1,3,5))
        log("Total samples:{0}, poisoning samples:{1},  benign samples:{2}".format(len(testset),len(poisoned_test_indexs),\
                                                                                len(benign_test_indexs)))                                                                                                                                                
        log("Benign_accuracy:{0}, poisoning_accuracy:{1}".format(benign_accuracy,poisoning_accuracy))

    elif args.subtask == "evaluate data filtering":
        # Evaluate the effectiveness of data filtering
        log("\n==========Evaluate the effectiveness of data filtering.==========\n")
        defense = torch.load(os.path.join(defense_object_dir, 'MIMRL_object.pth'))
        
        poisoned_test_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt'))
        testset = poisoned_test_dataset
        device = torch.device("cuda:0")
        # predict_digits, labels = defense.defense_method.calculate_loss(testset, device=device)
       
        # sceloss = SCELoss(alpha=0.1, beta=1, num_classes=10,)
        # losses = sceloss(predict_digits, labels)
        x_true, labels, predict_digits, z = defense.defense_method.calculate_info_xz(testset, device=device)
        log(f"x_true, type:{type}, len:{len(x_true)}, z,type:{type(z)}, len:{len(z)}\n")
        
        infomax_loss = InfomaxLoss(x_dim = 784, dim = 10)
        infomax_loss.disc.load_state_dict(defense.defense_method.get_discriminator_state_dict())
        losses = infomax_loss(x_true, z)

        poisoned_test_indexs = list(testset.get_poison_indices())
        top_loss_indices = np.argsort(losses.detach().numpy())
        indices = top_loss_indices[0:len(poisoned_test_indexs)]
        print(f"losses[indices]:{losses[indices]}\n")
        # indices = top_loss_indices[-1*len(poisoned_test_indexs):]
        # print(f"losses[indices]:{losses[indices]}\n")
        # print(f"losses[poisoned_test_indexs]:{losses[indices]}\n")

        intersection = np.intersect1d(indices, poisoned_test_indexs)
        total = len(testset)
        poisoned_sample_num = len(poisoned_test_indexs)
        identified_poisoned_sample_num = len(intersection)

        log(f"Total samples:{total}, poisoning samples:{poisoned_sample_num},top losses samples from poisoned_test_dataset:{identified_poisoned_sample_num},proportion:{identified_poisoned_sample_num/poisoned_sample_num}\n")

    elif args.subtask == "compare sce scores of hard and poisoned samples":
        log("\n==========Compare sce scores of hard and poisoned samples.==========\n")
        defense = torch.load(os.path.join(defense_object_dir, 'MIMRL_object.pth'))
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

    elif args.subtask == "generate latents":
        log("\n==========Get the latent representation in the middle layer of the model.==========\n")
        # Alreadly exsiting trained model and poisoned datasets
        # device = torch.device("cuda:1")
        # model = BaselineMNISTNetwork()
        device = torch.device("cpu")
        model = task_config["model"]
        # model.to(device)
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt'))
        poison_indices = poisoned_train_dataset.get_poison_indices()
        model.load_state_dict(torch.load(os.path.join(model_dir, 'repaired_model.pth')),strict=False)
        # Get the latent representation in the middle layer of the model.

        latents, y_labels = get_latent_rep(model, layer, poisoned_train_dataset, device=device)
        latents_path = os.path.join(latents_dir,"train_latents.npz")
        print(type(latents))
        print(latents.shape)
        np.savez(latents_path, latents=latents, y_labels=y_labels)

    elif args.subtask == "visualize latents by t-sne":
        log("\n========== Clusters of latent representations for all classes.==========\n")  
        # # Alreadly exsiting latent representation
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt'))
        poison_indices = poisoned_train_dataset.get_poison_indices()
        latents_path = os.path.join(latents_dir,"train_latents.npz")
        data = np.load(latents_path)
        latents, y_labels = data["latents"], data["y_labels"]
   
        # get low-dimensional data points by t-SNE
        n_components = 2 # number of coordinates for the manifold
        t_sne = manifold.TSNE(n_components=n_components, perplexity=30, early_exaggeration=120, init="pca", n_iter=250, random_state=0 )
        points = t_sne.fit_transform(latents)

        points = points*100
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