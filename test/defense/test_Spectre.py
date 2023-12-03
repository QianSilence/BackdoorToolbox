# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/09/06 13:59:58
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : test_Spectre.py
# @Description  : This is the test code of Spectre defense.
import os
import sys
from copy import deepcopy
import os.path as osp
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from core.attacks.BackdoorAttack import BackdoorAttack
from core.defenses.BackdoorDefense import BackdoorDefense
from core.defenses.Spectre import Spectre
from utils import compute_confusion_matrix, compute_indexes, compute_accuracy,save_img
from utils import parser,Log
from config import get_task_config, get_task_schedule, get_attack_config, get_defense_config
import time
import random

# ========== Set global settings ==========
datasets_root_dir = BASE_DIR + '/datasets/'
args = parser.parse_args()
defense = 'Spectre'
#["BadNets","Adaptive-Blend","Adaptive-Patch"]
attack = 'Adaptive-Patch'
dir = f'{defense}_for_{attack}'
if args.dataset == "MNIST": 
    # ========== BaselineCIFAR-10Network_CIFAR-10_Spectre_for_BadNets ==========
    experiment = f'BaselineMNISTNetwork_MNIST_{defense}_for_{attack}' # {task}_{defense}_for_{attack} 
    task = 'BaselineMNISTNetwork_MNIST' # {model}_{dataset}
    layer = "fc2"

elif args.dataset == "CIFAR-10":
    # ========== ResNet-18_CIFAR-10_Spectre_for_BadNets ==========
    experiment = f'ResNet-18_CIFAR-10_{defense}_for_{attack}'
    task = 'ResNet-18_CIFAR-10'
    layer = "linear"
    
elif args.dataset == "CIFAR-100":
    # ========== ResNet-18_CIFAR-100_Spectre_for_BadNets ==========
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

work_dir = os.path.join(BASE_DIR,'experiments/' + task + '/' + dir)
datasets_dir = os.path.join(work_dir,'datasets')

poison_datasets_dir = os.path.join(BASE_DIR,f'experiments/{task}/{attack}/datasets/poisoned_data')
poison_datasets_dir = os.path.join(poison_datasets_dir,f'{data_path}/')

latents_dir = os.path.join(BASE_DIR,f'experiments/{task}/{attack}/datasets/latents')
latents_dir = os.path.join(latents_dir,f'{data_path}/')

filter_dir = os.path.join(datasets_dir,'filter')
filter_dir = os.path.join(filter_dir,f'{data_path}/')

predict_dir = os.path.join(datasets_dir,'predict')
predict_dir = os.path.join(predict_dir,f'{data_path}/')

model_dir = os.path.join(BASE_DIR,f'experiments/{task}/{attack}/model')
# model_dir = os.path.join(BASE_DIR,f'experiments/{task}/{dir}/model')
model_dir = os.path.join(model_dir,f'{data_path}/')

show_dir = os.path.join(work_dir,'show')
show_dir = os.path.join(show_dir,f'{data_path}/')

defense_object_dir = os.path.join(work_dir,'defense_object')
defense_object_dir  = os.path.join(defense_object_dir ,f'{data_path}/')

dirs = [work_dir, datasets_dir, poison_datasets_dir, latents_dir, filter_dir, predict_dir, model_dir, show_dir,defense_object_dir]
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

task_config = get_task_config(task=task)

schedule = get_task_schedule(task = task)
schedule['experiment'] = experiment
schedule['work_dir'] = work_dir

defense_schedule = get_defense_config(defense_strategy = defense)
defense_schedule["filter"]["layer"] = layer
defense_schedule['device'] = "cuda:4"
defense_schedule['work_dir'] = work_dir
defense_schedule['schedule'] = schedule

if __name__ == "__main__":

    # Users can select the task to execute by passing parameters.

    # 1. The task of showing backdoor samples
    #   python test_Spectre.py --subtask "show train backdoor samples" --dataset "CIFAR-10"
    #   python test_Spectre.py --subtask "show test backdoor samples" --dataset "CIFAR-10"

    # 2. The task of filtering backdoor samples
    #   python test_Spectre.py --subtask "filter" --dataset "CIFAR-10"

    # 3. The task of filtering with latents
    #   python test_Spectre.py --subtask "filter with latents" --dataset "CIFAR-10"

    # 4. The task of defense backdoor attack
    #   python test_Spectre.py --subtask "repair" --dataset "CIFAR-10"

    # 5.The task of testing defense effect
    #   python test_Spectre.py --subtask "test" --dataset "CIFAR-10"

    # 6.The task of identifying target label 
    #   python test_Spectre.py --subtask "identify target label " --dataset "CIFAR-10"

    log = Log(osp.join(work_dir, 'log.txt'))
    t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    msg = "\n\n\n==========Start {0} at {1}==========\n".format(experiment,t)
    log(msg)
    Spectre = Spectre(task_config, defense_schedule)
    defense = BackdoorDefense(Spectre)

    if args.subtask == "show train backdoor samples":
        log("\n==========Show posioning train sample==========\n")
        # Alreadly exsiting dataset and trained model.
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt')) 
        poison_indices = poisoned_train_dataset.get_poison_indices()
        
        index = poison_indices[random.choice(range(len(poison_indices)))]
        log(f"total:{len(poison_indices)},index:{index}\n")
        image, label = poisoned_train_dataset.get_sample_by_index(index)
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        backdoor_sample_path = os.path.join(show_dir, "backdoor_train_sample.png")
        title = f"label:{label},class:{poisoned_train_dataset.classes[label]}"
        save_img(image, title=title, path=backdoor_sample_path)
        log("Save backdoor_train_sample to" + backdoor_sample_path)

    elif args.subtask == "show test backdoor samples":
        poisoned_dataset = torch.load(os.path.join(poison_datasets_dir,'test.pt'))
        poisoned_indices = poisoned_dataset.get_poison_indices()
        benign_indexs = list(set(range(len(poisoned_dataset))) - set(poisoned_indices))
      
        index = poisoned_indices[random.choice(range(len(poisoned_indices)))]
        log(f"total:{len(poisoned_indices)},index:{index}\n")

        image, label = poisoned_dataset.get_sample_by_index(index)
        if isinstance(image, torch.Tensor):
            image = image.numpy()
    
        backdoor_sample_path = os.path.join(show_dir, "backdoor_test_sample.png")
        title = f"label:{label},class:{poisoned_dataset.classes[label]}"
        save_img(image, title=title, path=backdoor_sample_path)
        log("Save backdoor_test_sample to" + backdoor_sample_path)
    
    elif args.subtask == "identify target label ":
        pass

    elif args.subtask == "filter": 
        model = task_config['model']
        model.load_state_dict(torch.load(os.path.join(model_dir, 'backdoor_model.pth')),strict=False)
        # model.load_state_dict(torch.load(os.path.join(model_dir, 'repaired_model.pth')),strict=False)

        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt'))
        
        poison_indices = poisoned_train_dataset.get_poison_indices()
        benign_indices = list(set(range(len(poisoned_train_dataset))) - set(poison_indices))
        log(f"y_target:{poisoned_train_dataset.get_y_target()}, poison_indices:{len(poison_indices)},benign_indices:{len(benign_indices)}\n")
        
        # 1.filter out poisoned samples
        predict_poisoned_indices, predict_clean_indices = defense.filter(model=model, dataset=poisoned_train_dataset)
     
        precited = np.zeros(len(poisoned_train_dataset))
        precited[predict_poisoned_indices] = 1
        expected = np.zeros(len(poisoned_train_dataset))
        expected[poison_indices] = 1

        tp, fp, tn, fn = compute_confusion_matrix(precited,expected)
        log(f"tp:{tp}, fp:{fp}, tn:{tn}, fn:{fn}\n")
        accuracy, precision, recall, F1 = compute_indexes(tp, fp, tn, fn)
        log(f"accuracy:{accuracy}, precision:{precision}, recall:{recall}, F1:{F1}\n")

        filter_path = os.path.join(filter_dir,"filter.npz")
        filter_res = {"predict_poisoned_indices":predict_poisoned_indices, "predict_clean_indices":predict_clean_indices,\
         "poison_indices":poison_indices,"benign_indices":benign_indices}
        np.savez(filter_path, **filter_res)
        log(f"Save filter results to {filter_path}\n")
    #filter_with_latents(self, latents_path=None, schedule=None):
    elif args.subtask == "filter with latents": 
        model = task_config['model']
        model.load_state_dict(torch.load(os.path.join(model_dir, 'backdoor_model.pth')),strict=False)
       
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt'))
        poison_indices = poisoned_train_dataset.get_poison_indices()
        benign_indices = list(set(range(len(poisoned_train_dataset))) - set(poison_indices))
        log(f"y_target:{poisoned_train_dataset.get_y_target()}, poison_indices:{len(poison_indices)},benign_indices:{len(benign_indices)}\n")
        
        latents_path = os.path.join(latents_dir,"latents.npz")
        # 1.filter out poisoned samples
        predict_poisoned_indices, predict_clean_indices = defense.filter_with_latents(latents_path = latents_path)
     
        precited = np.zeros(len(poisoned_train_dataset))
        precited[predict_poisoned_indices] = 1
        expected = np.zeros(len(poisoned_train_dataset))
        expected[poison_indices] = 1

        tp, fp, tn, fn = compute_confusion_matrix(precited,expected)
        log(f"tp:{tp}, fp:{fp}, tn:{tn}, fn:{fn}\n")
        accuracy, precision, recall, F1 = compute_indexes(tp, fp, tn, fn)
        log(f"accuracy:{accuracy}, precision:{precision}, recall:{recall}, F1:{F1}\n")

        filter_path = os.path.join(filter_dir,"filter.npz")
        filter_res = {"predict_poisoned_indices":predict_poisoned_indices, "predict_clean_indices":predict_clean_indices,\
         "poison_indices":poison_indices,"benign_indices":benign_indices}
        np.savez(filter_path, **filter_res)
        log(f"Save filter results to {filter_path}\n")

     

    elif args.subtask == "repair":
        # get backdoor sample
        log("\n==========get poisoning train_dataset and test_dataset dataset and repair model ==========\n")
        
       
        model = task_config['model']
        model.load_state_dict(torch.load(os.path.join(model_dir, 'backdoor_model.pth')),strict=False)
        poisoned_train_dataset = torch.load(os.path.join(poison_datasets_dir,'train.pt'))

        predict_poisoned_indices, predict_clean_indices = defense.filter(model=model,dataset=poisoned_train_dataset)
        dataset = Subset(poisoned_train_dataset, predict_clean_indices)
        
        defense.repair(dataset = dataset)

        repaired_model = defense.get_repaired_model()
        torch.save(repaired_model.state_dict(), os.path.join(model_dir, 'repaired_model.pth'))
        log("Save repaired model to" + os.path.join(model_dir, 'repaired_model.pth'))

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


