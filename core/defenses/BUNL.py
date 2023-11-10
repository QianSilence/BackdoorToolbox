# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/10/13 17:35:23
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : test_Mine.py
# @Description  : backdoor unLearning with Noisy Labels  
import torch
import torch.nn as nn
import os
import os.path as osp
import sys
from tqdm import trange
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import numpy as np
import random
import time
from collections import Counter
from core.base import Base
from core.defenses import Defense 
from copy import deepcopy
from copy import deepcopy
from utils  import Log, get_latent_rep_without_detach
from utils import SCELoss
from utils import compute_accuracy
from utils.compute import is_singular_matrix
import torch
from torch.utils.data import DataLoader, Subset, IterableDataset  
import torch.nn.functional as F
class BUNL(Base,Defense):
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
        Base.__init__(self, task, defense_schedule)   
        Defense.__init__(self)
        self.global_defense_schedule = defense_schedule
        self.defense_strategy = defense_schedule["defense_strategy"]
        self.num_classes = len(self.train_dataset.classes)
        self.invert_label_strategy = defense_schedule["invert_label_strategy"]
        self.noise_label_ratio =  defense_schedule["noise_label_ratio"]
        self.y_target = defense_schedule["y_target"]
        self.noisy_label_pool = None
        self.acc=np.array([[],[]])
    
    def get_defense_strategy(self):
        return self.defense_strategy
    
    def get_target_label(self):
        pass
    def get_clean_data_pool(self):
        pass
    def get_poison_data_pool(self):
        pass
         
    def repair(self, dataset=None, schedule=None):
        """
        Perform Mine defense method based on poisoning train dataset,the repaired model will be stored in self.model
        
        Args:
            schedule (dict): Schedule for Spectral. Contraining sub-schedule for pre-isolatoin training, clean training, unlearning and test phrase.
            transform (classes in torchvison.transforms): Transform for poisoned trainset in filter phrase
        """
        """Perform Spectral defense method based on attacked dataset and retrain with the filtered dataset. 
        The repaired model will be stored in self.model
        
        Args:
            schedule (dict): Schedule for Spectral. Contraining sub-schedule for pre-isolatoin training, clean training, unlearning and test phrase.
            transform (classes in torchvison.transforms): Transform for poisoned trainset in filter phrase
        """
        
        self.train(dataset=dataset, schedule=schedule)

    def filter(self, dataset=None, schedule=None):
        if dataset is not None:
            self.repair(dataset=dataset, schedule=schedule)
        return self._get_filter_indices()
        
      
    def _get_filter_indices(self):
        dataset_size = len(self.poison_data_pool)
        poisoned_indices = np.arange(dataset_size)[self.poison_data_pool==1]
        clean_indices = np.arange(dataset_size)[self.clean_data_pool==1]
        return poisoned_indices, clean_indices
    
    def invert_random_labels(self, dataset = None):
        if dataset is not None:
            self.train_dataset = dataset
        else:
            train_dataset = self.train_dataset
        num = int(len(train_dataset) * self.noise_label_ratio)
        random_indices = np.random.choice(np.arange(len(train_dataset)),num,replace=False)
        self.noisy_label_pool = np.zeros(len(train_dataset))
        self.noisy_label_pool[random_indices] = 1

        work_dir = self.global_schedule['work_dir']  
        log_path = osp.join(work_dir, 'log.txt')
        log = Log(log_path)
        labels = []
        for index in random_indices:
            _,label,_ = train_dataset[index]
            random_label = np.random.choice(np.delete(np.arange(self.num_classes),label),size=1,)[0]
            labels.append(random_label)
        train_dataset.modify_targets(random_indices, np.array(labels))
        return train_dataset,random_indices
    
    def invert_order_labels(self, dataset = None):
        if dataset is not None:
            self.train_dataset = dataset
        else:
            train_dataset = self.train_dataset
        num = int(len(train_dataset) * self.noise_label_ratio)
        random_indices = np.random.choice(np.arange(len(train_dataset)),num,replace=False)
        self.noisy_label_pool = np.zeros(len(train_dataset))
        self.noisy_label_pool[random_indices] = 1

        work_dir = self.global_schedule['work_dir']  
        log_path = osp.join(work_dir, 'log.txt')
        log = Log(log_path)
        labels = []
        for index in random_indices:
            _,label,_ = train_dataset[index]      
            labels.append((int(label) + 1) % self.num_classes)
        train_dataset.modify_targets(random_indices, np.array(labels))
        return train_dataset,random_indices
    def invert_target_labels(self, dataset = None):
        if dataset is not None:
            self.train_dataset = dataset
        else:
            train_dataset = self.train_dataset
        num = int(len(train_dataset) * self.noise_label_ratio)
        random_indices = np.random.choice(np.arange(len(train_dataset)),num,replace=False)
        self.noisy_label_pool = np.zeros(len(train_dataset))
        self.noisy_label_pool[random_indices] = 1

        work_dir = self.global_schedule['work_dir']  
        log_path = osp.join(work_dir, 'log.txt')
        log = Log(log_path)
        labels = np.zeros(len(random_indices))
        labels.fill(self.y_target)
        train_dataset.modify_targets(random_indices, np.array(labels))
        return train_dataset,random_indices
    
    def invert_real_labels(self, dataset = None):
        if dataset is not None:
            self.train_dataset = dataset
        else:
            train_dataset = self.train_dataset
        work_dir = self.global_schedule['work_dir']  
        log_path = osp.join(work_dir, 'log.txt')
        log = Log(log_path)
        
        poison_indices = self.train_dataset.get_poison_indices()
        num = int(len(poison_indices) * self.noise_label_ratio)
        random_indices = np.random.choice(poison_indices,num,replace=False)
        self.noisy_label_pool = np.zeros(len(train_dataset))
        self.noisy_label_pool[random_indices] = 1

        real_targets = train_dataset.get_real_targets()
        labels = real_targets[random_indices]
        train_dataset.modify_targets(random_indices, np.array(labels))
        return train_dataset,random_indices
    
    def repair(self, dataset=None, schedule=None):
        if self.invert_label_strategy == "random":
            train_dataset,_ = self.invert_random_labels(dataset=dataset)
        elif self.invert_label_strategy == "order":
            train_dataset,_ = self.invert_order_labels(dataset=dataset)
        elif self.invert_label_strategy == "real":
            train_dataset,_ = self.invert_real_labels(dataset=dataset)
       
        self.train(dataset=train_dataset, schedule=schedule)
     