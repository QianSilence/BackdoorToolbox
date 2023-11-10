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
import torch
from torch.utils.data import Subset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import time
from models import BaselineMNISTNetwork,ResNet
from core.base import Base
from core.attacks.BadNets import BadNets
from core.attacks.BackdoorAttack import BackdoorAttack
from core.defenses.BackdoorDefense import BackdoorDefense
from core.defenses import Mine
import random
from utils import Log, parser, save_img, get_latent_rep
from utils import compute_confusion_matrix, compute_indexes,compute_accuracy,SCELoss
from utils import plot_hist
from config import get_task_config,get_task_schedule,get_attack_config,get_defense_config

# ========== Set global settings ==========
datasets_root_dir = BASE_DIR + '/datasets/'

defense = 'Mine'
args = parser.parse_args()
if args.attack is None: 
    attack = args.attack
else:
    # the default attack is BadNets
    attack = 'BadNets'
dir = f'Mine_for_{attack}'
if args.dataset == "MMNIST": 
    # ========== BaselineMNISTNetwork_MNIST_Mine_for_BadNets ==========
    #{model}_{datasets}_{defense}_for_{attack} 
    experiment = f'BaselineMNISTNetwork_MNIST_Mine_for_{attack}'
    task = 'BaselineMNISTNetwork_MNIST'
    layer = "fc2"

elif args.dataset == "CIFAR-10":
    # ========== ResNet-18_CIFAR-10_Mine_for_BadNets ==========
    experiment = 'ResNet-18_CIFAR-10_Mine_for_{attack}'
    task = 'ResNet-18_CIFAR-10'
    layer = "linear"
    
elif args.dataset == "CIFAR-100":
    # ========== ResNet-18_CIFAR-100_Mine_for_BadNets ==========
    experiment = 'ResNet-18_CIFAR-100_Mine_for_{attack}'
    task = 'ResNet-18_CIFAR-100'
    layer = "linear"
    
elif args.dataset == "ImageNet":
    pass
   
work_dir = os.path.join(BASE_DIR,'experiments/' + task + '/' + dir)
datasets_dir = os.path.join(work_dir,'datasets')
poison_datasets_dir = os.path.join(datasets_dir,'poisoned_data')
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

task_config = get_task_config(task = task)
task_config['train_dataset'] = poisoned_trainset
task_config['test_dataset'] = poisoned_testset
task_config['loss'] = "MineLoss"

schedule = get_task_schedule(task = task)
schedule['experiment'] = experiment
schedule['work_dir'] = work_dir

defense_schedule = get_defense_config(defense_strategy = defense)
defense_schedule["layer"] = layer
schedule ={**schedule,**defense_schedule}

if __name__ == "__main__":

    # Users can select the task to execute by passing parameters.


    # 1. The task of showing backdoor samples
    #     python test_Mine.py --task "show backdoor samples"
   
        
    # 2. The task of defense backdoor attack
    #   python test_Mine.py --task "repair" --dataset "CIFAR100"

    # 3.The task of testing defense effect
    #   python test_Mine.py --task "test"

    # 4.The task of evaluating data filtering
    #   python test_Mine.py --task "evaluate data filtering"

    # 5.The task of comparing sce scores of hard and poisoned samples
    # python test_Mine.py --task  "compare sce scores of hard and poisoned samples"

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
    if args.task == "show backdoor samples":
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
        model = task['model']
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

        
    elif args.task == "compare sce scores of hard and poisoned samples":
        log("\n==========Compare sce scores of hard and poisoned samples.==========\n")
        defense = torch.load(os.path.join(defense_object_dir, 'Mine_object.pth'))
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


    #    #Alreadly exsiting trained model
    #     model = task['model']
    #     #print(os.path.join(model_dir, 'backdoor_model.pth'))
    #     model.load_state_dict(torch.load(os.path.join(model_dir, 'repaired_model.pth')),strict=False)
    #     sceloss = SCELoss(reduction='none')
       
    #     with torch.no_grad(): 
    #         poisoned_sample_set = Subset(poisoned_train_dataset,np.intersect1d(poison_indices,remain_sample_indces))
    #         predict_digits, labels = defense.test(model=model, test_dataset=poisoned_sample_set)
    #         # print(f"the shape of labels:{labels.shape}")
    #         poisoned_sample_scores = sceloss(predict_digits,labels).numpy()
            
    #         clean_sample_set = Subset(poisoned_train_dataset,np.setdiff1d(remain_sample_indces,np.intersect1d(poison_indices,remain_sample_indces)))
    #         predict_digits, labels = defense.test(model=model, test_dataset=clean_sample_set)
    #         # print(f"the shape of labels:{labels.shape}")
    #         clean_sample_scores = sceloss(predict_digits,labels).numpy()
    #         # print(f"predict_digits:{predict_digits},shape:{predict_digits.shape},labels:{labels},shape:{labels.shape}\n")
       
    #     scores = [poisoned_sample_scores,clean_sample_scores]
    #     labels=["Clean samples","Poisoned samples"]
    #     colors = ["res","blue"]
    #     title = "Compare sce scores of hard and poisoned samples"
    #     xlabel = "SCE loss"
    #     ylabel = "Proportion (%)"
    #     plot_hist(*scores, colors=colors, title=title, xlabel=xlabel, ylabel=ylabel)


    elif args.task == "generate latents":
        log("\n==========Get the latent representation in the middle layer of the model.==========\n")
        # Alreadly exsiting trained model and poisoned datasets
        # device = torch.device("cuda:1")
        # model = BaselineMNISTNetwork()
        device = torch.device("cpu")
        model = task["model"]
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
 




