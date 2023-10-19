# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/09/04 14:40:01
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : Spectral.py
# @Description :This is the implement of spectral signatures backdoor defense.
#               (link: https://github.com/MadryLab/backdoor_data_poisoning)
#               Reference:[1] Spectral Signatures in Backdoor Attacks. NeurIPS, 2018.
import os
import os.path as osp
from copy import deepcopy
import time
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.base import Base
from torch.utils.data import DataLoader, Subset
from utils.interact.log import Log
from utils import compute_confusion_matrix, compute_indexes
from .Defense import Defense
from multiprocessing.sharedctypes import Value
from tqdm import trange    

class Spectral(Base,Defense):
    """Filtering the training samples spectral signatures (Spectral).
    Args:
        task(dict):The defense strategy is used for the task, refrence core.Base.Base.
        schedule=None(dict): Config related to model training,refrence core.Base.Base.
        defense_schedule(dict): Parameters related to defense tasks, mainly including poisoned datasets and equipment parameters,includeing:
            defense_strategy (str):The name of defense_strategy
            backdoor_model (torch.nn.Module): Network.
            poisoned_trainset (types in support_list):training dataset.
            poisoned_testset (types in support_list): testing dataset.
            target_label(int):The target label of backdoor.
            percentile(int):0-to-100. Default: 85
            # related to device
            device (str): 'GPU',
            CUDA_VISIBLE_DEVICES: CUDA_VISIBLE_DEVICES,
            GPU_num (int): 1,
            work_dir (str): 'experiments',
            log_iteration_interval(int): 100
    """

    def __init__(self, task, defense_schedule):
        schedule = None
        if 'train_schedule' in defense_schedule:
            schedule = defense_schedule['train_schedule']
        Base.__init__(self, task, schedule)   
        Defense.__init__(self)

        self.defense_strategy = defense_schedule["defense_strategy"]
        self.global_defense_schedule = defense_schedule
        self.poisoned_trainset = defense_schedule["poisoned_trainset"]
        self.poisoned_testset = defense_schedule["poisoned_testset"] 
        self.target_label = defense_schedule["y_target"]
        self.percentile = defense_schedule["percentile"]
        self.backdoor_model = None

    def get_defense_strategy(self):
        return self.defense_strategy

       
    def repair(self, transform=None, schedule=None):
        """Perform Spectral defense method based on attacked dataset and retrain with the filtered dataset. 
        The repaired model will be stored in self.model
        
        Args:
            schedule (dict): Schedule for Spectral. Contraining sub-schedule for pre-isolatoin training, clean training, unlearning and test phrase.
            transform (classes in torchvison.transforms): Transform for poisoned trainset in filter phrase
        """
        if schedule is not None:
            current_schedule = deepcopy(schedule)
        elif self.global_defense_schedule is not None:
            # print(self.global_defense_schedule)
            current_schedule = deepcopy(self.global_defense_schedule)
        else: 
            raise AttributeError("Training schedule is None, please check your schedule setting.")
        
        # get logger
        work_dir = osp.join(current_schedule['work_dir'], current_schedule['experiment'])
       
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))
        experiment = current_schedule['experiment']
        t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        msg = "\n\n\n==========Execute {experiment} at {time}==========\n".format(experiment=experiment, time=t)
        log(msg)

        if current_schedule['train']:
            # train the model with poisoned dataset
            msg = "==========Start training with poisoned dataset==========\n"
            log(msg)
            log('the length of poisoned dataset:{0}.\n'.format(len(self.poisoned_trainset)))
            self.train(self.poisoned_trainset,current_schedule["train_schedule"])
            self.backdoor_model = deepcopy(self.get_model())
        elif current_schedule['trained_model'] is not None:
            self.backdoor_model = current_schedule['trained_model']
        elif current_schedule['backdoor_model_path'] is not None:
            self.backdoor_model = deepcopy(self.model.load_state_dict(torch.load(current_schedule['backdoor_model_path'])))
        else:
            raise AttributeError("Training schedule is None, please check your schedule setting.")
        assert isinstance(self.backdoor_model, nn.Module), "the tyoe of backdoor_model must be nn.module"

        # filter out poisoned samples
        msg = "==========Start filtering the poisoned samples==========\n"
        log(msg) 
        if transform is not None:     
            train_transform = self.poisoned_trainset.transform
            self.poisoned_trainset.transform = transform
        current_schedule['train'] = False
        current_schedule["backdoor_model_path"] = None 
        current_schedule["trained_model"] = self.backdoor_model 
        

        removed_indices, left_indices, cur_indices = self.filter(self.poisoned_trainset, current_schedule)
        
        # self.poisoned_trainset.transform = train_transform
        # torch.utils.data中的类
        removed_dataset = Subset(self.poisoned_trainset, removed_indices) 
        left_dataset = Subset(self.poisoned_trainset, left_indices) 
        
        # self.poisoned_dataset = removed_dataset

        log('the length of removed samples:{0}.\n'.format(len(removed_dataset)))
        log('the length of left samples:{0}.\n'.format(len(left_dataset)))
        # torch.save(removed_dataset, os.path.join(work_dir, "selected_poison.pth"))
        log("Select %d poisoned data\n"%len(removed_indices))

        # Train the model with filtered dataset
        msg = "==========Training with selected clean data==========\n"
        log(msg)
        self.train(left_dataset, current_schedule['train_schedule'])
        return self.get_model()
        

    def filter(self, dataset=None, schedule=None):

        """filter out poisoned samples from poisoned dataset. 
        
        Args:
            dataset (torch.utils.data.Dataset): The dataset to filter.
            schedule (dict): defense schedule filteringing the dataset.           
        """
        if dataset is None:
            if self.poisoned_trainset is not None:
                dataset = self.poisoned_trainset
            else: 
                raise AttributeError("Dataset is None, please check your dataset setting.")

        if schedule is not None:
            current_schedule = deepcopy(schedule)
        elif self.global_defense_schedule is not None:
            current_schedule = deepcopy(self.global_defense_schedule)
        else: 
            raise AttributeError("Training schedule is None, please check your schedule setting.")
        
        work_dir = osp.join(current_schedule['work_dir'], current_schedule['experiment'])
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))
        experiment = current_schedule['experiment']
        t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        msg = "\n\n\n==========Execute {experiment} at {time}==========\n".format(experiment=experiment, time=t)
        log(msg)
            
        if current_schedule['train']:
            # train the model with poisoned dataset
            msg = "==========Start training with poisoned dataset==========\n"
            log(msg)
            log('the length of poisoned dataset:{0}.\n'.format(len(dataset)))
            self.train(dataset,current_schedule["train_schedule"])
            self.backdoor_model = deepcopy(self.get_model())
        elif current_schedule['trained_model'] is not None:
            self.backdoor_model = current_schedule['trained_model']
        elif current_schedule['backdoor_model_path'] is not None:
            self.backdoor_model = deepcopy(self.model.load_state_dict(torch.load(current_schedule['backdoor_model_path'])))
        else:
            raise AttributeError("Training schedule is None, please check your schedule setting.")
        assert isinstance(self.backdoor_model, nn.Module), "the tyoe of backdoor_model must be nn.module"
        # Use GPU
        if 'device' in current_schedule and current_schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in current_schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = current_schedule['CUDA_VISIBLE_DEVICES']

            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert current_schedule['GPU_num'] > 0, 'GPU_num should be a positive integer'
            assert torch.cuda.device_count() >= current_schedule['GPU_num'] , 'This machine has {0} cuda devices, and use {1} of them to train'.format(torch.cuda.device_count(), current_schedule['GPU_num'])
            device = torch.device("cuda:0")
            gpus = list(range(current_schedule['GPU_num']))
            if isinstance(self.backdoor_model,nn.DataParallel):
                self.backdoor_model = self.backdoor_model.module
            self.backdoor_model = nn.DataParallel(self.backdoor_model, device_ids=gpus, output_device=device)
            
            log(f"This machine has {0} cuda devices, and use {1} of them to train.\n".format(torch.cuda.device_count(), current_schedule['GPU_num']))
        # Use CPU
        else:
            device = torch.device("cpu")

        self.backdoor_model = self.backdoor_model.to(device)

        ### a. prepare the model and dataset
        model = self.backdoor_model
        lbl = self.target_label
        dataset_y = []

        for i in range(len(dataset)):
            dataset_y.append(dataset[i][1])
        # print(dataset_y)
        cur_indices = [i for i,v in enumerate(dataset_y) if v==lbl]
        # benign_id = [i for j,v in enumerate(dataset_y) if v!=lbl]
        cur_example_num = len(cur_indices)
        # print(cur_indices)
        
        log('Target label {0}, Number of samples with  label {1} is {2}\n'.format(lbl, lbl ,cur_example_num))
        model.eval()
        msg = "==========Start filtering poisoned dataset==========\n"
        log(msg)

        ### b. get the activation as representation for each data
        for iex in trange(cur_example_num):
            cur_im = cur_indices[iex]
            x_batch = dataset[cur_im][0].unsqueeze(0).to(device)
            y_batch = dataset[cur_im][1]
            inps,outs = [],[]
            def layer_hook(module, inp, out):
                outs.append(out.data)
            """
            这里需要明确的是hook函数是什么？怎么使用？
            中间层输出的所在设备，数据类型，形状是什么？
            中间层表示结果的保存
            """
            # hook = model.module.layer4.register_forward_hook(layer_hook)
            if isinstance(model, nn.DataParallel):
                hook = model.module.fc2.register_forward_hook(layer_hook)
            else:
                hook = model.fc2.register_forward_hook(layer_hook)

            _ = model(x_batch)
            batch_grads = outs[0].view(outs[0].size(0), -1).squeeze(0)
            hook.remove()
            if iex==0:
                full_cov = np.zeros(shape=(cur_example_num, len(batch_grads)))
            full_cov[iex] = batch_grads.detach().cpu().numpy()
       
        ### c. detect the backdoor data by the SVD decomposition
        total_p = self.percentile            
        full_mean = np.mean(full_cov, axis=0, keepdims=True) 
        
        centered_cov = full_cov - full_mean
        #（√）奇异值分解算法直接调用的np的
        u,s,v = np.linalg.svd(centered_cov, full_matrices=False)
        
        log("The shape of the 'full_cov' matrix is {0}\n".format(full_cov.shape))
        log('Top 7 Singular Values:{0}\n'.format(s[0:7]))

        eigs = v[0:1]  
        p = total_p
        #shape num_top, num_active_indices
        corrs = np.matmul(eigs, np.transpose(full_cov)) 
        #shape num_active_indices
        scores = np.linalg.norm(corrs, axis=0) 
        p_score = np.percentile(scores, p)
       
        top_scores = np.where(scores>p_score)[0]
        removed_inds = np.copy(top_scores)
        
        re = [cur_indices[v] for i,v in enumerate(removed_inds)]
        left_inds = np.delete(range(len(dataset)), re)
        
        log('Length Scores:{0}\n'.format(len(scores)))
        log('The value at {0}% of the Scores list is {1}\n'.format(total_p, p_score))
        log('removed:{0}\n'.format(len(removed_inds)))
        log('left:{0}\n'.format(len(left_inds)))       
        log("The number of Samples with target label {0}:{1}\n".format(self.target_label,len(cur_indices)))
        return removed_inds, left_inds, cur_indices

    # def save_ckpt(self, ckpt_name):
    #     ckpt_model_path = os.path.join(self.work_dir, ckpt_name)
    #     torch.save(self.model.cpu().state_dict(), ckpt_model_path)
    #     return 


 