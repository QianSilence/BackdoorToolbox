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
from utils import BeingRobust
from utils import robust_estimation
import torchvision
class Spectre(Base,Defense):
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
    
    def target_label_identifier(self, latent_feats, class_indices, max_k, alpha, varepsilon):
        """
        The idea is that the mean QUE scores obtained for the target label are clearly larger (for appropriate values of effective dimension k)
        than those obtained for untargeted labels       
        """
        num_classes = len(class_indices) 
        target_label  = None
        best_k = None
        max_q = 0.0
        best_to_be_removed = None
        for i in range(num_classes):
            if len(class_indices[i]) > 1:
                latents = latent_feats[class_indices[i]]
                k, q, removed = self.k_identifier(latents, max_k, alpha, varepsilon)
                if q > max_q:
                    max_q = q
                    target_label = i
                    best_k = k
                    best_to_be_removed = removed
        return target_label, best_k, best_to_be_removed
              
    def k_identifier(self, latents, max_k, alpha, varepsilon):
        """
        The idea is that if poisons were correctly identified,  then the mean QUE score will be large as poisons have strong spectral signature. 
        """
        # n * m
        feat_deviation = latents - latents.mean(dim=0) # centered data
        # m * m
        U, _, _ = torch.svd(feat_deviation.T)
        # m * max_k
        U = U[:, :max_k]
        max_q = 0.0
        best_n_dim = None
        best_to_be_removed = None
        for n_dim in range(2, max_k):
            S_removed, S_left,_ = self.SPECTRE(latents, n_dim, alpha, varepsilon)
            # len(S_left) * m
            left_feats = latents[S_left]
            # max_k*max_k
            # covariance = torch.cov(torch.matmul(left_feats, U))
            # L, V = torch.linalg.eig(covariance)
            # L, V = L.real, V.real
            # L = (torch.diag(L) ** (1 / 2) + 0.001).inverse()
            # # max_k*max_k
            # normalizer = torch.matmul(V, torch.matmul(L, V.T))

            # # n*m *  m*max_k --> n*max_k
            # projected_feats = torch.matmul(latents, U)
            # # n*max_k *  max_k*max_k --> n*max_k
            # whitened_feats = torch.matmul(projected_feats, normalizer)
            
            whitened_feats = self.whiten(torch.matmul(latents, U), oracle_clean_feats=torch.matmul(left_feats, U))
            q = self.QUEscore(whitened_feats, alpha=alpha).mean()
            if q > max_q:
                max_q = q
                best_n_dim = n_dim
                best_to_be_removed = S_removed

            return  best_n_dim, max_q, best_to_be_removed
               
    def SPECTRE(self, latents, n_dim, alpha, varepsilon):
        # n * m
        feat_deviation = latents - latents.mean(dim=0) # centered data
        # m * m
        U, _, _ = torch.svd(feat_deviation.T)
        # m * n_dim
        U = U[:, :n_dim]
        # print(f"latents:{latents.size()}, U:{U.size()}\n")
        # n*m *  m*n_dim --> n*n_dim
        projected_feats = torch.matmul(latents, U)
        # n*n_dim
        whitened_feats = self.whiten(projected_feats, oracle_clean_feats=None)
        taus = self.QUEscore(whitened_feats, alpha = alpha)

        p_score = np.percentile(taus, 1.5 * varepsilon)
       
        top_scores = np.where(taus > p_score)[0]
        S_removed = np.copy(top_scores)
        S_left = np.delete(range(len(latents)),S_removed)

        return S_removed, S_left, p_score 

    # filter criteria
    def QUEscore(sefl, temp_feats, alpha = 4.0):
        n_samples = temp_feats.shape[0]
        n_dim = temp_feats.shape[1]
        temp_feats = temp_feats - temp_feats.mean(dim=0)
        Sigma = torch.matmul(temp_feats, temp_feats.T) / (n_samples - 1) 
        I = torch.eye(n_dim).cuda()
        Q = torch.exp((alpha * (Sigma - I)) / (torch.linalg.norm(Sigma, ord=2) - 1))
        trace_Q = torch.trace(Q)

        taus = []
        for i in range(n_samples):
            h_i = temp_feats[i]
            tau_i = torch.matmul(h_i.T, torch.matmul(Q, h_i)) / trace_Q
            tau_i = tau_i.item()
            taus.append(tau_i)
        taus = np.array(taus)
        return taus

    def whiten(self, temp_feats, oracle_clean_feats=None):
        # whiten the data
        if oracle_clean_feats is None:
            estimator = robust_estimation.BeingRobust(random_state=0, keep_filtered=True).fit((temp_feats.T).cpu().numpy())
            clean_mean = torch.FloatTensor(estimator.location_).cuda()
            filtered_feats = (torch.FloatTensor(estimator.filtered_).cuda() - clean_mean).T
            # clean_covariance = torch.cov(filtered_feats)
            clean_covariance = torch.mm(filtered_feats, filtered_feats.t()) / (filtered_feats.size(0) - 1)
        else:
            clean_feats = oracle_clean_feats
            clean_mean = torch.FloatTensor(clean_feats.mean(dim = 1)).cuda()
            filtered_feats = (torch.FloatTensor(clean_feats).cuda() - clean_mean).T
            # clean_covariance = torch.cov(clean_feats)
            clean_covariance = torch.mm(filtered_feats, filtered_feats.t()) / (filtered_feats.size(0) - 1)
        temp_feats = temp_feats.cuda()
        temp_feats = (temp_feats.T - clean_mean).T
        #clean_covariance ^ (-1/2)
        L, V = torch.eig(clean_covariance,eigenvectors = True)
        # L, V = torch.linalg.eig(clean_covariance)
        L, V = L.real, V.real
        # L = L[:, 0]
        # print(f"L:{L},V:{V}\n")
        # print(torch.diag(L)**(1/2)+0.001)
        L = (torch.diag(L)**(1/2)+0.001).inverse()
        normalizer = torch.matmul(V, torch.matmul( L, V.T ) )
        temp_feats = torch.matmul(normalizer, temp_feats)
        return temp_feats

    def filter_via_spectre(self, latents=None, n_dim=512, alpha=4.0, varepsilon=0.01):
        log(f"The shape of the 'latents' matrix is {latents.shape}\n")
        if isinstance(latents, np.ndarray):
            latents = torchvision.transforms.ToTensor()(latents)
        if latents.dim() > 2:
            latents = latents.squeeze()
        S_removed, S_left, p_score = self.SPECTRE(latents, n_dim, alpha, varepsilon)
        return S_removed, S_left, p_score

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
        
        removed_inds, _, p_score = self.filter_via_spectre(latents=latents, n_dim=current_schedule["filter"]["n_dim"], alpha=current_schedule["filter"]["alpha"], varepsilon=current_schedule["filter"]["varepsilon"])
        ### c. detect the backdoor data by the SVD decomposition
        re = [cur_indices[v] for i,v in enumerate(removed_inds)]
        # print(f"cur_indices:{cur_indices}, latents:{len(latents)}")
        left_inds = np.delete(range(len(dataset)), re)
                   
        log(f'The value at {current_schedule["filter"]["percentile"]} of the Scores list is {p_score}\n')
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
        removed_inds, _, p_score = self.filter_via_spectre(latents=latents, n_dim=current_schedule["filter"]["n_dim"], alpha=current_schedule["filter"]["alpha"], varepsilon=current_schedule["filter"]["varepsilon"])
        print(f"removed_inds:{len(removed_inds)}")
        ### c. detect the backdoor data by the SVD decomposition
        re = [cur_indices[v] for i,v in enumerate(removed_inds)]
        left_inds = np.delete(range(len(data["latents"])), re)
                   
        log(f'The value at {0} of the Scores list is {p_score}\n')
        log(f'The number of Samples with label:{current_schedule["filter"]["y_target"]} is {len(cur_indices)}, removed:{len(re)}, left:{len(left_inds)}\n') 

        return re, left_inds
        

