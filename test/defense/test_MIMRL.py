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
import torchvision
from torch.utils.data import Subset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
import time
from core.defenses.BackdoorDefense import BackdoorDefense
from core.defenses.MIMRL import MIMRL
import random
from utils import  parser, save_img, get_latent_rep, Log, log
from utils import compute_accuracy,SCELoss
from utils import show_img,save_img, compute_accuracy, plot_2d
from utils import compute_confusion_matrix, compute_indexes, compute_accuracy,save_img
from utils.compute import cluster_metrics
from utils import cal_cos_sim
from config import get_task_config,get_task_schedule,get_attack_config,get_defense_config
from core import SCELoss
from core import InfomaxLoss
from core.defenses.MIMRL import whitening_dataset
from sklearn import manifold
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ========== Set global settings ==========
datasets_root_dir = BASE_DIR + '/datasets/'
args = parser.parse_args()

#['BadNets', 'Adaptive-Blend', 'Adaptive-Patch']
attack = 'BadNets'
defense = 'MIMRL'
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

attack_schedule = get_attack_config(attack_strategy= attack, dataset = args.dataset)
poisoning_rate = attack_schedule["poisoning_rate"]

if attack == "BadNets":
    data_path = f'poison_{poisoning_rate}'
elif attack == "Adaptive-Blend" or  attack == "Adaptive-Patch":
    cover_rate = attack_schedule["cover_rate"]
    data_path = f'poison_{poisoning_rate}_cover_{cover_rate}'

work_dir = os.path.join(BASE_DIR, 'experiments/' + task + '/' + dir)
datasets_dir = os.path.join(work_dir, 'datasets')

poison_datasets_dir = os.path.join(BASE_DIR, f'experiments/{task}/{attack}/datasets/poisoned_data')
poison_datasets_dir = os.path.join(poison_datasets_dir, data_path)

latents_dir = os.path.join(datasets_dir, 'latents')
latents_dir = os.path.join(latents_dir, data_path)

whitened_dataset_dir = os.path.join(datasets_dir, 'whitened_dataset')
whitened_dataset_dir = os.path.join(whitened_dataset_dir, data_path)

predict_dir = os.path.join(datasets_dir, 'predict')
predict_dir = os.path.join(predict_dir, data_path)

filter_dir = os.path.join(datasets_dir, 'filter')
predict_dir = os.path.join(filter_dir, data_path)

model_dir = os.path.join(work_dir, 'model')
model_dir = os.path.join(model_dir, data_path)

show_dir = os.path.join(work_dir, 'show')
show_dir = os.path.join(show_dir, data_path)

defense_object_dir = os.path.join(work_dir,'defense_object')
defense_object_dir = os.path.join(defense_object_dir, data_path)

dirs = [work_dir, datasets_dir, poison_datasets_dir, latents_dir, whitened_dataset_dir,
        predict_dir, model_dir, show_dir, defense_object_dir]
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

task_config = get_task_config(task=task)

# poisoned_trainset = torch.load(os.path.join(poison_datasets_dir,"train.pt")) 
# poisoned_testset = torch.load(os.path.join(poison_datasets_dir,"test.pt"))
# task_config['train_dataset'] = poisoned_trainset
# task_config['test_dataset'] = poisoned_testset

schedule = get_task_schedule(task = task)
schedule['experiment'] = experiment

defense_schedule = get_defense_config(defense_strategy = defense)
defense_schedule["layer"] = layer
defense_schedule["schedule"] = schedule 
defense_schedule["filter_config"]["filter"]["layer"] = layer

log.set_log_path(osp.join(work_dir, 'log.txt'))

if __name__ == "__main__":

    # Users can select the task to execute by passing parameters.
    # nohup  python test_MIMRL.py --subtask "show train backdoor samples" --dataset "MNIST" &

    # 1. The task of showing backdoor samples
    #   python test_MIMRL.py --subtask "show train backdoor samples" --dataset "MNIST"
    #   python test_MIMRL.py --subtask "show train backdoor samples" --dataset "CIFAR-10"

    #   python test_MIMRL.py --subtask "show test backdoor samples" --dataset "MNIST"
    #   python test_MIMRL.py --subtask "show test backdoor samples" --dataset "CIFAR-10"
    
    # 2. The task of whitening dataset

    #   python test_MIMRL.py --subtask "whiten dataset" --dataset "MNIST"
    #   python test_MIMRL.py --subtask "whiten dataset" --dataset "CIFAR-10"
    
    #   python test_MIMRL.py --subtask "compare whitened sample vector" --dataset "MNIST"
    #   python test_MIMRL.py --subtask "compare whitened sample vector" --dataset "CIFAR-10"
    #   python test_MIMRL.py --subtask "compare whitened sample vector in the diffirent class" --dataset "CIFAR-10"


    

    # 3. The task of defense backdoor attack
    #   python test_MIMRL.py --subtask "repair" --dataset "MNIST"
    #   nohup python test_MIMRL.py --subtask "repair" --dataset "CIFAR-10" &
   
    # 4.The task of testing defense effect
    #   python test_MIMRL.py --subtask "test" --dataset "MNIST"
    #   python test_MIMRL.py --subtask "test" --dataset "CIFAR-10"

    # 5.The task of generating latents
    #   python test_MIMRL.py  --subtask "generate latents" --dataset "MNIST"
    #   python test_MIMRL.py  --subtask "generate latents" --dataset "CIFAR-10"

    # 6.The task of visualizing latents by t-sne
    #   python test_MIMRL.py --subtask "visualize latents by t-sne" --dataset "MNIST"
    #   python test_MIMRL.py --subtask "visualize latents for target class by t-sne" --dataset "MNIST"

    #   python test_MIMRL.py --subtask "visualize latents by t-sne" --dataset "CIFAR-10"
    #   python test_MIMRL.py --subtask "visualize latents for target class by t-sne" --dataset "CIFAR-10"

    # 7.The task of filtering data 
    #   python test_MIMRL.py --subtask "filter" --dataset "MNIST"
    #   python test_MIMRL.py --subtask "filter" --dataset "CIFAR-10"
    
    # 8.The task of computing condition number
    #   python test_MIMRL.py --subtask "compute condition number" --dataset "MNIST"
    #   python test_MIMRL.py --subtask "compute condition number" --dataset "CIFAR-10"

    t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    msg = "\n\n\n==========Start {0} at {1}==========\n".format(experiment,t)
    log(msg)
    mimrl = MIMRL(
        task_config, 
        defense_schedule)
    
    defense = BackdoorDefense(mimrl)

    if args.subtask == "show train backdoor samples":
        log("\n==========Show posioning train sample==========\n")
        # Alreadly exsiting dataset and trained model.
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt')) 
        poison_indices = poisoned_train_dataset.get_poison_indices()
        print(len(poison_indices))
        index = poison_indices[random.choice(range(len(poison_indices)))]
        print(f"index:{index}")
        # print(poisoned_train_dataset[index])
        image,label,_ = poisoned_train_dataset.get_sample_by_index(index)
        image = image.numpy()
        backdoor_sample_path = os.path.join(show_dir, "backdoor_train_sample.png")
        title = "label: " + str(label)
        save_img(image, title=title, path=backdoor_sample_path)
        log(f"Save backdoor train sample to {backdoor_sample_path}\n")

    elif args.subtask == "show test backdoor samples":
        poisoned_test_dataset = torch.load(os.path.join(poison_datasets_dir,'test.pt'))
        poisoned_test_indices = poisoned_test_dataset.get_poison_indices()
        benign_test_indexs = list(set(range(len(poisoned_test_dataset))) - set(poisoned_test_indices))

        real_targets = np.array(poisoned_test_dataset.get_real_targets())
        labels = real_targets[poisoned_test_indices]
        # for i, label in enumerate(poisoned_test_dataset.classes):
        #     print(f"the number of sample with label:{label} in poisoned_train_dataset:{labels.tolist().count(i)}\n")

        index = poisoned_test_indices[random.choice(range(len(poisoned_test_indices)))]
        print(f"index:{index}")
        image, label, _ = poisoned_test_dataset.get_sample_by_index(index)
        image = image.numpy()
        backdoor_sample_path = os.path.join(show_dir, "backdoor_test_sample.png")
        title = "label: " + str(label)
        save_img(image, title=title, path=backdoor_sample_path)
        log(f"Save backdoor test sample to {backdoor_sample_path}\n")

    elif args.subtask == "whiten dataset":
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt')) 
        poisoned_test_dataset = torch.load(os.path.join(poison_datasets_dir,'test.pt'))
        whitened_res = whitening_dataset(poisoned_train_dataset)
        whitened_dataset_path = os.path.join(whitened_dataset_dir, f'whiten_{args.dataset}.pt')
        torch.save(whitened_res, whitened_dataset_path)
        log(f"Save whitened sample to {whitened_dataset_path}\n")

    elif args.subtask == "compare whitened sample vector":

        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt'))
        poison_indices = poisoned_train_dataset.get_poison_indices()
        whitened_dataset = torch.load(os.path.join(whitened_dataset_dir, f'whiten_{args.dataset}.pt'))
        n_classes = poisoned_train_dataset.get_classes()
        class_means = whitened_dataset["class_means"]
        class_samples = whitened_dataset["class_samples"]
        class_singular_values = whitened_dataset["class_singular_values"] 
        class_covariances = whitened_dataset["class_covariances"] 
        class_whitening_matrix = whitened_dataset["class_whitening_matrix"]
        
        y_target = attack_schedule['y_target']
        class_indices = whitened_dataset["class_indices"] 
        y_target_indices = class_indices[y_target]
        clean_y_target_indices = np.setdiff1d(y_target_indices,poison_indices)
        indexes = np.random.choice(clean_y_target_indices,size=2,replace=False)
        sample_1, label_1, _ = poisoned_train_dataset[indexes[0]]
        sample_2, label_2, _= poisoned_train_dataset[indexes[1]]
        index = np.random.choice(poison_indices)
        poison_sample, y_label, _ = poisoned_train_dataset[index]
        
        print(f"clean_y_target_indices:{len(clean_y_target_indices)}, indexes:{indexes}, label_1:{label_1}, label_2:{label_1},y_label:{y_label},poison_index:{index}\n")
       
        # 在样本空间，样本之间的相似性
        log(f"In the sample space, the similarity between samples\n")

        delta_1 = sample_1.view(-1) - class_means[label_1]
        delta_2 = sample_2.view(-1) - class_means[label_2]
        poison_delta = poison_sample.view(-1) - class_means[y_label]

        cos_sim = cal_cos_sim(sample_1.view(-1).numpy(),sample_2.view(-1).numpy())
        cos_sim_mean = cal_cos_sim(sample_1.view(-1).numpy(),class_means[label_1].numpy())
        cos_sim_delta_mean = cal_cos_sim(delta_1.view(-1).numpy(),class_means[label_1].numpy())
        
        poison_cos_sim = cal_cos_sim(poison_sample.view(-1).numpy(), sample_1.view(-1).numpy())
        poison_cos_sim_mean = cal_cos_sim(poison_sample.view(-1).numpy(),class_means[y_label].numpy())
        poison_cos_sim_delta_mean = cal_cos_sim(poison_delta.view(-1).numpy(),class_means[y_label].numpy())

        print(f"class_mean_norm:{torch.norm(class_means[label_1])}\n")
        print(f"sample_1_norm:{torch.norm(sample_1)},sample_2_norm:{torch.norm(sample_2)},poison_sample_norm:{torch.norm(poison_sample)},delta_1_norm:{torch.norm(delta_1)},delta_2_norm:{torch.norm(delta_2)},poison_delta_norm:{torch.norm(poison_delta)}\n")
        print(f"cos_sim:{cos_sim},cos_sim_mean:{cos_sim_mean},cos_sim_delta_mean:{cos_sim_delta_mean}\n")
        print(f"poison_cos_sim:{ poison_cos_sim},poison_cos_sim_mean:{poison_cos_sim_mean},poison_cos_sim_delta_mean:{poison_cos_sim_delta_mean}\n")
        
        # Whthening Method 1:\Sigma^{-1/2}(X-\mu) + \mu, 样本之间的相似性
        log(f"In the sample space, the similarity between Whthened samples by Whthening Method 1\n")

        delta_1 = sample_1.view(-1) - class_means[label_1]
        delta_2 = sample_2.view(-1) - class_means[label_2]
        poison_delta = poison_sample.view(-1) - class_means[y_label]

        whitened_sample_1 =  1.0 * torch.matmul(class_whitening_matrix[label_1], delta_1) + class_means[label_1]
        whitened_sample_2 =  1.0 * torch.matmul(class_whitening_matrix[label_2], delta_2) + class_means[label_2]
        poison_whitened_sample =  1.0 * torch.matmul(class_whitening_matrix[y_label], poison_delta) + class_means[y_label]

        whitened_delta_1 = whitened_sample_1 - class_means[label_1]
        whitened_delta_2 = whitened_sample_2 - class_means[label_2]
        poison_whitened_delta = poison_whitened_sample - class_means[y_label]

        whiten_cos_sim = cal_cos_sim(whitened_sample_1.numpy(),whitened_sample_2.numpy())
        whiten_cos_sim_mean = cal_cos_sim(whitened_sample_1.numpy(),class_means[label_1].numpy())
        whiten_cos_sim_delta_mean = cal_cos_sim(whitened_delta_1.view(-1).numpy(),class_means[label_1].numpy())

        poison_whitened_cos_sim = cal_cos_sim(poison_whitened_sample.view(-1).numpy(), whitened_sample_1.view(-1).numpy())
        poison_whitened_cos_sim_mean = cal_cos_sim(poison_whitened_sample.view(-1).numpy(), class_means[y_label].numpy())
        poison_whitened_cos_sim_delta_mean = cal_cos_sim(poison_whitened_delta.view(-1).numpy(), class_means[y_label].numpy())
        print(f"class_mean_norm:{torch.norm(class_means[label_1])}\n")
        print(f"whitened_sample_1_norm:{torch.norm(whitened_sample_1)},whitened_sample_2_norm:{torch.norm(whitened_sample_2)},poison_whitened_sample_norm:{torch.norm(poison_whitened_sample)},whitened_delta_1_norm:{torch.norm(whitened_delta_1)}, whitened_delta_2_norm:{torch.norm(whitened_delta_2)},poison_whitened_delta_norm:{torch.norm(poison_whitened_delta)}\n")
        print(f"whiten_cos_sim:{whiten_cos_sim}, whiten_cos_sim_mean:{whiten_cos_sim_mean},whiten_cos_sim_delta_mean:{whiten_cos_sim_delta_mean}\n")
        print(f"poison_whitened_cos_sim:{poison_whitened_cos_sim},poison_whitened_cos_sim_mean:{poison_whitened_cos_sim_mean}, poison_whitened_cos_sim_delta_mean:{poison_whitened_cos_sim_delta_mean}\n")
        
        # Whthening Method 1:\Sigma^{-1/2}X, 样本之间的相似性
        log(f"In the sample space, the similarity between Whthened samples by Whthening Method 2\n")

        whitened_sample_1 = torch.matmul(class_whitening_matrix[label_1], sample_1.view(-1)) 
        whitened_sample_2 = torch.matmul(class_whitening_matrix[label_2], sample_2.view(-1)) 
        poison_whitened_sample =  torch.matmul(class_whitening_matrix[y_label], poison_sample.view(-1)) 
        whitened_mean = torch.matmul(class_whitening_matrix[label_1], class_means[label_1]) 

        whitened_delta_1 = whitened_sample_1 - whitened_mean
        whitened_delta_2 = whitened_sample_2 - whitened_mean
        poison_whitened_delta = poison_whitened_sample - whitened_mean

        whiten_cos_sim = cal_cos_sim(whitened_sample_1.numpy(),whitened_sample_2.numpy())
        whiten_cos_sim_mean = cal_cos_sim(whitened_sample_1.numpy(),whitened_mean.numpy())
        whiten_cos_sim_delta_mean = cal_cos_sim(whitened_delta_1.view(-1).numpy(),whitened_mean.numpy())

        poison_whitened_cos_sim = cal_cos_sim(poison_whitened_sample.view(-1).numpy(), whitened_sample_1.view(-1).numpy())
        poison_whitened_cos_sim_mean = cal_cos_sim(poison_whitened_sample.view(-1).numpy(), whitened_mean.numpy())
        poison_whitened_cos_sim_delta_mean = cal_cos_sim(poison_whitened_delta.view(-1).numpy(),whitened_mean.numpy())
        
        print(f"class_mean_norm:{torch.norm(whitened_mean)}\n")
        print(f"whitened_sample_1_norm:{torch.norm(whitened_sample_1)},whitened_sample_2_norm:{torch.norm(whitened_sample_2)},poison_whitened_sample_norm:{torch.norm(poison_whitened_sample)},whitened_delta_1_norm:{torch.norm(whitened_delta_1)}, whitened_delta_2_norm:{torch.norm(whitened_delta_2)},poison_whitened_delta_norm:{torch.norm(poison_whitened_delta)}\n")
        print(f"whiten_cos_sim:{whiten_cos_sim}, whiten_cos_sim_mean:{whiten_cos_sim_mean},whiten_cos_sim_delta_mean:{whiten_cos_sim_delta_mean}\n")
        print(f"poison_whitened_cos_sim:{poison_whitened_cos_sim},poison_whitened_cos_sim_mean:{poison_whitened_cos_sim_mean}, poison_whitened_cos_sim_delta_mean:{poison_whitened_cos_sim_delta_mean}\n")

    elif args.subtask == "compare whitened sample vector in the diffirent class":
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt'))
        poison_indices = poisoned_train_dataset.get_poison_indices()
        n_classes = poisoned_train_dataset.get_classes()
        whitened_dataset = torch.load(os.path.join(whitened_dataset_dir, f'whiten_{args.dataset}.pt'))
        class_indices = whitened_dataset["class_indices"] 
        class_samples = whitened_dataset["class_samples"]
        class_means = whitened_dataset["class_means"]
        class_singular_values = whitened_dataset["class_singular_values"] 
        class_covariances = whitened_dataset["class_covariances"] 
        class_whitening_matrix = whitened_dataset["class_whitening_matrix"]

        indexes = np.random.choice(np.arange(len(n_classes)),size=2,replace=False)
        index_1 = np.random.choice(class_indices[indexes[0]])
        index_2 = np.random.choice(class_indices[indexes[1]])

        sample_1, label_1, _ = poisoned_train_dataset[index_1]
        sample_2, label_2, _ = poisoned_train_dataset[index_2]
        
        # delta_1 = sample_1.view(-1) - class_means[label_1]
        # delta_2 = sample_2.view(-1) - class_means[label_2]
        # whitened_sample_1 = torch.matmul(class_whitening_matrix[label_1],delta_1)
        # whitened_sample_2 = torch.matmul(class_whitening_matrix[label_2],delta_2)
        # cos_sim_2 = cal_cos_sim(whitened_sample_1.numpy(),whitened_sample_2.numpy())

        # print(f"label_1:{label_1}, label_2:{label_2}, index_1:{index_1}, index_2:{index_2},cos_sim_2:{cos_sim_2}")
        cos_sim_between_mean = cal_cos_sim(class_means[label_1],class_means[label_2])
        log(f"label_1:{label_1},label_2:{label_2},cos_sim_between_mean:{cos_sim_between_mean}")
        # log(f"the cos similarity of whitened sample vector in the diffirent class is:{cos_sim_2}")

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
        # Show the structure of the model
        print(task_config["model"])

        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt'))

        defense.repair(dataset = poisoned_train_dataset)
        repaired_model = defense.get_repaired_model()
        torch.save(repaired_model.state_dict(), os.path.join(model_dir, 'repaired_model.pth'))
        log("Save repaired model to" + os.path.join(model_dir, 'repaired_model.pth'))

        torch.save(defense, os.path.join(defense_object_dir, 'MIMRL_object.pth'))
        log(f"Save defense object to {os.path.join(defense_object_dir, 'MIMRL_object.pth')}\n" ) 

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

    elif args.subtask == "generate latents":
        log("\n==========Get the latent representation in the middle layer of the model.==========\n")
        # Alreadly exsiting trained model and poisoned datasets
        # device = torch.device("cuda:1")
        # model = BaselineMNISTNetwork()

        device = torch.device("cuda:4")
        model = task_config["model"]
        model.to(device)
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt'))
        poison_indices = poisoned_train_dataset.get_poison_indices()
        model.load_state_dict(torch.load(os.path.join(model_dir, 'repaired_model.pth')),strict=False)
       
        # Get the latent representation in the middle layer of the model.
        latents, y_labels = get_latent_rep(model, layer, poisoned_train_dataset, device=device)
        latents_path = os.path.join(latents_dir,"latents.npz")
       
        np.savez(latents_path, latents=latents, y_labels=y_labels)
        log(f"Latents shape:{latents.shape},Save latents to {latents_path}\n")

    elif args.subtask == "visualize latents by t-sne":
        log("\n========== Clusters of latent representations for all classes.==========\n")  
        # # Alreadly exsiting latent representation
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt'))
        poison_indices = poisoned_train_dataset.get_poison_indices()
        latents_path = os.path.join(latents_dir,"latents.npz")
        data = np.load(latents_path)
        latents, y_labels = data["latents"], data["y_labels"]
   
        # get low-dimensional data points by t-SNE
        n_components = 2 # number of coordinates for the manifold
        t_sne = manifold.TSNE(n_components=n_components, perplexity=100, early_exaggeration=120, init="pca", n_iter=1000, random_state=0)
        points = t_sne.fit_transform(latents)

        # print(t_sne.kl_divergence_)
        
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

        poison_latents = torchvision.transforms.ToTensor()(latents[poison_indices]).squeeze()
        clean_latents = torchvision.transforms.ToTensor()(latents[np.setdiff1d(indexs,poison_indices)]).squeeze()
       
        silhouette_score , intra_clust_dists, inter_clust_dists= cluster_metrics(clean_latents, poison_latents)
        print(f"clean_latents:{clean_latents.shape},poison_latents:{poison_latents.shape},silhouette_score:{silhouette_score},intra_clust_dists:{intra_clust_dists},inter_clust_dists:{inter_clust_dists}\n")
        color_numebers = [0 if index in poison_indices else 1 for index in indexs]
        color_numebers = np.array(color_numebers)

        # get low-dimensional data points by t-SNE
        n_components = 2 # number of coordinates for the manifold
        t_sne = manifold.TSNE(n_components=n_components, perplexity=100, early_exaggeration=120, init="pca", n_iter=1000, random_state=0)
        points = t_sne.fit_transform(latents[indexs])

        colors = ["black","blue"]
        cmap = mcolors.ListedColormap(colors)
        title = f"silhouette_score:{silhouette_score}"
        path = os.path.join(show_dir,"latent_2d_clusters.png")
        plot_2d(points, color_numebers, title=title, cmap=cmap, path=path)

    elif args.subtask == "filter":
        
        model = task_config['model']
        model.load_state_dict(torch.load(os.path.join(model_dir, 'repaired_model.pth')),strict=False)
        print(os.path.join(poison_datasets_dir,'train.pt'))
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt'))
        
        poison_indices = poisoned_train_dataset.get_poison_indices()
        log(f"y_target:{poisoned_train_dataset.get_y_target()},poison_indices:{len(poison_indices)}\n")
        benign_indices = list(set(range(len(poisoned_train_dataset))) - set(poison_indices))
        
        # # 1.filter out poisoned samples
        # predict_poisoned_indices, predict_clean_indices = defense.filter(model=model, dataset=poisoned_train_dataset,schedule=defense_schedule["filter_config"])

        latents_path = os.path.join(latents_dir,"latents.npz")
        predict_poisoned_indices, predict_clean_indices = defense.filter_with_latents(latents_path=latents_path,schedule=defense_schedule["filter_config"])
     
        precited = np.zeros(len(poisoned_train_dataset))
        precited[predict_poisoned_indices] = 1
        expected = np.zeros(len(poisoned_train_dataset))
        expected[poison_indices] = 1
        tp, fp, tn, fn = compute_confusion_matrix(precited,expected)
        log(f"tp:{tp}, fp:{fp}, tn:{tn}, fn:{fn}\n")
        accuracy, precision, recall, F1 = compute_indexes(tp, fp, tn, fn)

        log(f"accuracy:{accuracy}, precision:{precision}, recall:{recall}, F1:{F1}\n")

        filter_res = {"predict_poisoned_indices":predict_poisoned_indices, "predict_clean_indices":predict_clean_indices,\
         "poison_indices":poison_indices,"benign_indices":benign_indices}
        filter_path = os.path.join(filter_dir,"filter.npz")
        np.savez(filter_path, **filter_res)
        log(f"Save filter results to {filter_path}\n")

    elif args.subtask == "compute condition number":
        
        latents_path = os.path.join(latents_dir,"latents.npz")
        data = np.load(latents_path)
        latents, y_labels = data["latents"], data["y_labels"]
        log(f"The shape of the 'latents' matrix is {latents.shape}\n")
        cur_indices = np.where(y_labels == defense_schedule["filter_config"]["filter"]["y_target"])[0]
        
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt'))
        poison_indices = poisoned_train_dataset.get_poison_indices()
        clean_indices = np.setdiff1d(cur_indices, poison_indices)

        mix_latents = latents[cur_indices]
        mix_mean = np.mean(mix_latents, axis=0, keepdims=True) 
        mix_centered_cov = mix_latents - mix_mean
        # Use Frobenius norm
        mix_cond_number = np.linalg.cond(np.dot(mix_centered_cov.T,mix_centered_cov), p='fro')
        mix_u, mix_s, mix_v = np.linalg.svd(mix_centered_cov, full_matrices=False)

        clean_latents = latents[clean_indices]
        clean_mean = np.mean(clean_latents, axis=0, keepdims=True) 
        centered_cov = clean_latents - clean_mean
        # Use Frobenius norm
        cond_number = np.linalg.cond(np.dot(centered_cov.T,centered_cov), p='fro')
        u, s, v = np.linalg.svd(centered_cov, full_matrices=False)

        log(f'mix_cov_cond_number:{mix_cond_number},Top 10 Singular Values:{mix_s[0:20]},cond_number:{cond_number},Top 10 Singular Values:{s[0:20]}\n')
        log(f'Mix Singular Values:{mix_s}, Singular Values:{s}\n')

        
     

    