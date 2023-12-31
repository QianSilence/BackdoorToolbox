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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.base import Base
from torch.utils.data import DataLoader, Subset
from utils.interact.log import Log, log
from .Defense import Defense
from tqdm import trange    
from utils import get_latent_rep

class Spectral(Base,Defense):
    """Filtering the training samples spectral signatures (Spectral).
    Args:
        task(dict):The defense strategy is used for the task, refrence core.Base.Base.
        schedule=None(dict): Config related to model training,refrence core.Base.Base.
        defense_schedule(dict): Parameters related to defense tasks.    
    """

    def __init__(self, task=None, defense_schedule=None):
        if task is not None:
            Base.__init__(self, task=task, schedule=defense_schedule['schedule'])   
        else:
            Base.__init__(self)

        Defense.__init__(self)

        self.global_defense_schedule = defense_schedule
        

    def get_defense_strategy(self):
        return self.global_defense_schedule["defense_strategy"]

    def repair(self, model=None, dataset=None, schedule=None):
        """Perform Spectral defense method based on attacked dataset and retrain with the filtered dataset. 
        The repaired model will be stored in self.model
        
        Args:
            schedule (dict): Schedule for Spectral. Contraining sub-schedule for pre-isolatoin training, clean training, unlearning and test phrase.
            transform (classes in torchvison.transforms): Transform for poisoned trainset in filter phrase
        """
        assert dataset is not None, "dataset is None"

        if model is None:
            model = self.model

        if schedule is not None:
            current_schedule = deepcopy(schedule)
        elif self.global_defense_schedule is not None:
            # print(self.global_defense_schedule)
            current_schedule = deepcopy(self.global_defense_schedule)
        else: 
            raise AttributeError("Training schedule is None, please check your schedule setting.")
    
        # filter before repair
        if current_schedule["repair"]["filter"]:
            msg = "==========Start filtering before repair==========\n"
            log(msg) 
            removed_indices, left_indices = self.filter(dataset=dataset, schedule=current_schedule)
            dataset = Subset(dataset, left_indices)
            log(f'The number of Samples with label:{current_schedule["filter"]["y_target"]} is {len(removed_indices) + len(left_indices)}, removed:{len(removed_indices)}, left:{len(left_indices)}\n')  
 
        # Train the model with filtered dataset
        msg = "==========Train the model with filtered dataset==========\n"
        log(msg)
        self.init_model()
        self.train(dataset, current_schedule['schedule'])

    def _get_latents(self, model=None, dataset=None, schedule=None):
        device = schedule["filter"]["device"]
        model = model.to(device)
        y_target = schedule["filter"]["y_target"]  

        # dataset_y = []
        # for i in range(len(dataset)):
        #     dataset_y.append(dataset[i][1])

        # cur_indices = [i for i,v in enumerate(dataset_y) if v==y_target]
        # benign_id = [i for j,v in enumerate(dataset_y) if v!=y_target]

        dataset_y = dataset.get_modified_targets()
        cur_indices = np.where(dataset_y == y_target)[0]
        msg = f'Target label {y_target}, Number of samples with  label {y_target} is {len(cur_indices)}\n'
        log(msg)
        
        model.eval()
        msg = "==========Start filtering poisoned dataset==========\n"
        log(msg)
        sub_dataset = Subset(dataset,cur_indices)
        latents, _ = get_latent_rep(model, schedule["filter"]['layer'], sub_dataset, device=device)
        return cur_indices, latents
    
    def detect_via_SVD(self, latents=None, percentile=None):
        log(f"The shape of the 'latents' matrix is {latents.shape}\n")
        full_mean = np.mean(latents, axis=0, keepdims=True) 
        centered_cov = latents - full_mean

        # Use Frobenius norm
        cond_number = np.linalg.cond(np.dot(centered_cov.T,centered_cov), p='fro')

        u,s,v = np.linalg.svd(centered_cov, full_matrices=False)
        log('Top 10 Singular Values:{0}\n'.format(s[0:10]))
        log('Singular Values:{0}\n'.format(s))

        eigs = v[0:1] 
        corrs = np.matmul(eigs, np.transpose(centered_cov)) 
        scores = np.linalg.norm(corrs, axis=0) 
        p_score = np.percentile(scores, percentile)
       
        top_scores = np.where(scores>p_score)[0]
        removed_inds = np.copy(top_scores)
        
        return removed_inds, p_score, cond_number

    def filter(self, model=None, dataset=None, schedule=None):
        """
        filter out poisoned samples from dataset with label y_target. 
        Args:
            dataset (torch.utils.data.Dataset): The dataset to filter.
            schedule (dict): defense schedule filteringing the dataset.           
        """
        if schedule is not None:
            current_schedule = deepcopy(schedule)
        elif self.global_defense_schedule is not None:
            current_schedule = deepcopy(self.global_defense_schedule)
        else: 
            raise AttributeError("Training schedule is None, please check your schedule setting.")
              
        if current_schedule['filter']['train']:
            # train the model with poisoned dataset
            msg = "==========Start training with poisoned dataset==========\n"
            log(msg)
            log('the length of poisoned dataset:{0}.\n'.format(len(dataset)))
            self.train(dataset, current_schedule["schedule"])
            model = deepcopy(self.get_model())
        else:
            assert model is not None, "model is None"

        cur_indices, latents = self._get_latents(model=model, dataset=dataset, schedule=current_schedule)
        removed_inds, p_score, cond_number= self.detect_via_SVD(latents=latents, percentile=current_schedule["filter"]["percentile"])
        ### c. detect the backdoor data by the SVD decomposition
        re = [cur_indices[v] for i,v in enumerate(removed_inds)]
        # print(f"cur_indices:{cur_indices}, latents:{len(latents)}")
        left_inds = np.delete(range(len(dataset)), re)
                   
        log(f'The value at {current_schedule["filter"]["percentile"]} of the Scores list is {p_score}, the condition number of the covariance matrix is:{cond_number}\n')
        log(f'The number of Samples with label:{current_schedule["filter"]["y_target"]} is {len(cur_indices)}, removed:{len(re)}, left:{len(left_inds)}\n') 

        return re, left_inds
    
    def filter_with_latents(self, latents_path=None, schedule=None):
        
        if schedule is not None:
            current_schedule = deepcopy(schedule)
        elif self.global_defense_schedule is not None:
            current_schedule = deepcopy(self.global_defense_schedule)
        else: 
            raise AttributeError("Training schedule is None, please check your schedule setting.")
     
        assert latents_path is not None, "the 'latents_path' must be not None, when dataset is not specified"
        data = np.load(latents_path)
        latents, y_labels = data["latents"], data["y_labels"]
        cur_indices = np.where(y_labels == current_schedule["filter"]["y_target"])[0]
        latents = latents[cur_indices]

        removed_inds, p_score, cond_number = self.detect_via_SVD(latents=latents, percentile=current_schedule["filter"]["percentile"])
        print(f"removed_inds:{len(removed_inds)}")
        ### c. detect the backdoor data by the SVD decomposition
        re = [cur_indices[v] for i,v in enumerate(removed_inds)]
        left_inds = np.delete(range(len(data["latents"])), re)
                   
        log(f'The value at {0} of the Scores list is {p_score}, the condition number of the covariance matrix is:{cond_number}\n')
        log(f'The number of Samples with label:{current_schedule["filter"]["y_target"]} is {len(cur_indices)}, removed:{len(re)}, left:{len(left_inds)}\n') 

        return re, left_inds
        

