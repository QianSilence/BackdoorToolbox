# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/11/14 10:27:44
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : IMaxRL.py
# @Description  : Mutual Information Maximization Representation Learning for Backdoor Defense
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
from models import discriminator

class InfomaxLoss(nn.Module):
    def __init__(self,x_dim = 1024,dim = 10):
        super(InfomaxLoss, self).__init__()
        self.disc = discriminator(z_dim=dim, x_dim=x_dim)
    def forward(self, x_true, z):
        # pass x_true and learned features z from the discriminator
        d_xz = self.disc(x_true, z)
        z_perm = self.permute_dims(z)
        d_x_z = self.disc(x_true, z_perm)
        info_xz = -(d_xz.mean() - (torch.exp(d_x_z - 1).mean()))
        return info_xz 
    
    def permute_dims(self, z):
        """
        function to permute z based on indicies
        """
        assert z.dim() == 2
        B, _ = z.size()
        perm = torch.randperm(B)
        perm_z = z[perm]
        return perm_z
    
class MIMRLLoss(nn.Module):
    def __init__(self, supervise_loss=None, beta=1.0, x_dim=1024, dim=10):
        super(MIMRLLoss, self).__init__()
        self.beta = beta
        self.x_dim = x_dim
        self.dim = dim 
        self.supervise_loss = supervise_loss
        self.infomax_loss = InfomaxLoss(x_dim = self.x_dim, dim = self.dim)

    def forward(self, out, labels, x, z):
        supervise_loss = self.supervise_loss(out, labels)
        info_xz = self.infomax_loss(x, z)
        loss = supervise_loss + self.beta * info_xz
        return loss, supervise_loss, info_xz    
    
    def reset_parameter(self, beta=1.0, x_dim=1024, dim=10):
        self.beta = beta
        self.x_dim = x_dim
        self.dim = dim
    

class MIMRL(Base,Defense):
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
        beta = self.global_defense_schedule['beta']
        x_dim=self.global_defense_schedule['x_dim']
        dim=self.global_defense_schedule['dim']
        self.loss = MIMRLLoss(supervise_loss=self.loss, beta=beta, x_dim=x_dim, dim=dim)
        
    def get_defense_strategy(self):
        return self.defense_strategy
    
    def get_target_label(self):
        subset_indices = np.arange(len(self.poison_data_pool))[self.poison_data_pool==1]
        sub_dataset = Subset(self.train_dataset,subset_indices)
        labels = [data[1] for data in sub_dataset]
        ##Find the number that appears the most, get (number, times) and return the number
        most_common_label = Counter(labels).most_common(1)[0]
        return most_common_label[0]

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
        
    #Override  function "train"  in further class "Base" according to the need of the defense strategy "Mine"
    def train(self, dataset=None, schedule=None):
        if dataset is not None:
            self.train_dataset = dataset
        
        if schedule is not None:
            current_schedule = deepcopy(schedule)
        elif self.global_schedule is not None:
            current_schedule = deepcopy(self.global_schedule)
        else: 
            raise AttributeError("Training schedule is None, please check your schedule setting.")
        
        work_dir = current_schedule['work_dir']  
        log_path = osp.join(work_dir, 'log.txt')
        log = Log(log_path)

        assert "loss" in current_schedule, "Schedule must contain 'loss' configuration!"
        supervise_loss = current_schedule['loss']
        assert "beta" in current_schedule, "Schedule must contain 'beta' configuration!"
        beta = current_schedule['beta']
        assert "x_dim" in current_schedule, "Schedule must contain 'x_dim' configuration!"
        x_dim = current_schedule['x_dim']
        assert "dim" in current_schedule, "Schedule must contain 'dim'configuration!"
        dim = current_schedule["dim"]
        assert "lr_dis" in current_schedule, "Schedule must contain 'lr_dis'configuration!"
        lr_dis = current_schedule["lr_dis"]
        assert "layer" in current_schedule, "Schedule must contain 'layer' configuration!"
        layer = current_schedule['layer']

        assert "poison_rate" in current_schedule, "Schedule must contain 'poison_rate' configuration!"
        poison_rate = current_schedule["poison_rate"]
        assert "threshold" in current_schedule, "Schedule must contain 'threshold' configuration!"
        threshold = current_schedule['threshold']
        
        if 'pretrain' in current_schedule and current_schedule['pretrain'] is not None:
            self.model.load_state_dict(torch.load(current_schedule['pretrain']), strict=False)

        #os.environ(mapping)：A variable that records information about the current code execution environment;
        # Use GPU
        if 'device' in current_schedule and current_schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in current_schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = current_schedule['CUDA_VISIBLE_DEVICES']
            # print(current_schedule['CUDA_VISIBLE_DEVICES'])
            # print(f"This machine has {torch.cuda.device_count()} cuda devices")
            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert current_schedule['GPU_num'] > 0, 'GPU_num should be a positive integer'
            assert torch.cuda.device_count() >= current_schedule['GPU_num'] , 'This machine has {0} cuda devices, and use {1} of them to train'.format(torch.cuda.device_count(), current_schedule['GPU_num'])
            device = torch.device("cuda:0")
            gpus = list(range(current_schedule['GPU_num']))
            self.model = nn.DataParallel(self.model, device_ids=gpus, output_device=device)
            log(f"This machine has {torch.cuda.device_count()} cuda devices, and use {current_schedule['GPU_num']} of them to train.")
        # Use CPU
        else:
            device = torch.device("cpu")
        
        train_loader = DataLoader(
                dataset=self.train_dataset,
                batch_size=current_schedule['batch_size'],
                shuffle=True,
                num_workers=current_schedule['num_workers'],
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker
            )
        
        self.loss.reset_parameter(beta=beta, x_dim=x_dim, dim=dim)
        self.model = self.model.to(device)
        # optimizer = self.optimizer(self.model.parameters(), lr=current_schedule['lr'])
        optimizer = self.optimizer(self.model.parameters(), lr=current_schedule['lr'], momentum=current_schedule['momentum'], weight_decay=current_schedule['weight_decay'])
        # assign selected device to discriminator
        disc = self.loss.infomax_loss.disc.to(device)
        # Adam optimzer for discriminator
        optim_disc = self.optimizer(disc.parameters(), lr=current_schedule['lr_dis'], momentum=current_schedule['momentum'], weight_decay=current_schedule['weight_decay'])
        # optim_disc = self.optimizer(disc.parameters(), lr=current_schedule['lr_dis'], betas=(current_schedule['beta1'], current_schedule['beta2']))
        
        self.model.train()
        disc.train()
        iteration = 0
        for i in range(current_schedule['epochs']):
            for batch_id, batch in enumerate(train_loader): 
                #(img,label,index)  
                batch_img = batch[0]
                batch_label = batch[1]
                batch_indices = batch[2]
                batch_img = batch_img.to(device)
                batch_label = batch_label.to(device)
                
                # predict_digits = self.model(batch_img)
                latents, predict_digits = get_latent_rep_without_detach(self.model, layer, batch_img, device)
                # print(f"latents:{latents},predict_digits:{predict_digits}")

                # "latents" are not the final output of the model, so when using nn.DataParallel() for multi-GPU training, 
                # you need to transfer the latents to the final output device of the device.
                loss, supervise_loss, info_xz = self.loss(predict_digits, batch_label, batch_img, latents.to(device))

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                optim_disc.zero_grad()
                info_xz.backward(retain_graph=True, inputs=list(disc.parameters()))
                optim_disc.step()

                iteration += 1

                if iteration % current_schedule['log_iteration_interval'] == 0:
                    last_time = time.time()
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S]", time.localtime()) +f"Epoch:{i+1}/{current_schedule['epochs']}, iteration:{batch_id + 1}\{len(self.train_dataset)//current_schedule['batch_size']},lr:{current_schedule['lr']}, ls_dist:{current_schedule['lr_dis']}, loss:{float(loss)}, supervise_loss:{float(supervise_loss)}, info_xz:{float(info_xz)}, time:{time.time()-last_time}\n"
                    log(msg)
                    # Check the predictions of samples in each batch during training
                    max_num, index = torch.max(predict_digits, dim=1)
                    equal_matrix = torch.eq(index,batch_label)
                    correct_num =torch.sum(equal_matrix)
                    msg =f"batch_size:{current_schedule['batch_size']},correct_num:{correct_num}\n"
                    log(msg)

        
    def _test2(self, dataset, device = None):    
        model = self.model
        if device is None:
            device = torch.device("cpu")
        with torch.no_grad():
            test_loader = DataLoader(
                dataset,
                batch_size=16,
                shuffle=False,
                num_workers=8,
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker
            )

            model = model.to(device)
            model.eval()
            predict_digits = []
            labels = []
            for batch in test_loader:   
                batch_img,batch_label,batch_indices = batch[0],batch[1],batch[2]
                batch_img = batch_img.to(device)
                batch_img = model(batch_img)
                batch_img = batch_img.cpu()
                predict_digits.append(batch_img)
                labels.append(batch_label)

            predict_digits = torch.cat(predict_digits, dim=0)
            labels = torch.cat(labels, dim=0)
            total_num = labels.size(0)
            prec1, prec5 = compute_accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num)) 
            return top1_correct, top5_correct
        
    def get_discriminator_state_dict(self):
        return self.loss.infomax_loss.disc.state_dict()
        
    def calculate_loss(self, dataset, device=None):
        model = self.model
        if device is None:
            device = torch.device("cpu")
        with torch.no_grad():
            test_loader = DataLoader(
                dataset,
                batch_size=16,
                shuffle=False,
                num_workers=8,
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker
            )

            model = model.to(device)
            model.eval()
            predict_digits = []
            labels = []
            loss_list = []
            supervise_loss_list = []
            info_xz_list = []
            for batch in test_loader:   
                batch_img,batch_label,batch_indices = batch[0],batch[1],batch[2]
                batch_img = batch_img.to(device)
                batch_predict_digits = model(batch_img)
                batch_predict_digits = batch_predict_digits.cpu()
                
                latents, predict_digits = get_latent_rep_without_detach(self.model, self.global_defense_schedule['layer'], batch_img, device)
                batch_loss, batch_supervise_loss, batch_info_xz =  self.loss(predict_digits, batch_label, batch_img, latents.to(device))

                predict_digits.append(batch_predict_digits)
                labels.append(batch_label)
                loss_list.append(batch_loss)
                supervise_loss_list.append(batch_supervise_loss)
                info_xz_list.append(batch_info_xz)

               
                # batch_loss = self.loss(batch_predict_digits,batch_label)
                # losses.append(batch_loss)
        
            predict_digits = torch.cat(predict_digits, dim=0)
            labels = torch.cat(labels, dim=0)
            losses = torch.cat(loss_list, dim=0)
            supervise_losses = torch.cat(supervise_loss_list, dim=0)
            info_xz = torch.cat(info_xz_list, dim=0)

        return  predict_digits, labels, losses, supervise_losses, info_xz
    
    def calculate_info_xz(self, dataset, device=None):
        model = self.model
        if device is None:
            device = torch.device("cpu")
        with torch.no_grad():
            test_loader = DataLoader(
                dataset,
                batch_size=16,
                shuffle=False,
                num_workers=8,
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker
            )

            model = model.to(device)
            model.eval()
            imgs = [] 
            predict_digits = []
            labels = []
            latents = []
            for batch in test_loader:   
                batch_img, batch_label, batch_indices = batch[0],batch[1],batch[2]
                batch_img = batch_img.to(device)
                batch_latents, batch_predict_digits = get_latent_rep_without_detach(model, self.global_defense_schedule['layer'], batch_img, device)
                imgs.append(batch_img)
                labels.append(batch_label)
                predict_digits.append(batch_predict_digits)
                latents.append(batch_latents)

            imgs = torch.cat(imgs, dim=0).to(torch.device("cpu"))
            predict_digits = torch.cat(predict_digits, dim=0).to(torch.device("cpu"))
            labels = torch.cat(labels, dim=0).to(torch.device("cpu"))
            latents = torch.cat(latents, dim=0).to(torch.device("cpu"))
          

        return  imgs, labels, predict_digits, latents






    

    
 

   
   
       

       


       




       
    
    
 

 
    
  


    
  

 

  
  


