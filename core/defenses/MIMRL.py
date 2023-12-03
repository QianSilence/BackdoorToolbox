# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/11/14 10:27:44
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : IMaxRL.py
# @Description  : Representation Learning by Mutual Information Maximization for Backdoor Defense
# Backdoor Defense via Adversarial Training by Mutual Information Maximization 
# InfoMAX Adversarial Training（InfoMAX-AT）

import torch
import torch.nn as nn
import os
import os.path as osp
import sys
from tqdm import trange
from tqdm import tqdm
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
from utils  import Log, log, get_latent_rep_without_detach
from utils import SCELoss
from utils import compute_accuracy
from utils.compute import is_singular_matrix
import torch
from torch.utils.data import DataLoader, Subset, IterableDataset  
import torch.nn.functional as F
from models import discriminator, Disc
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from core.Loss import RCELoss
from torch.distributions import MultivariateNormal

def whitening_dataset(dataset):
    n_classes = dataset.get_classes()
    targets = dataset.get_modified_targets()
    poison_indices = dataset.get_poison_indices()
    y_target = dataset.get_y_target()
    res = {} 
    class_indices = []
    class_samples = []
    class_means = []
    class_covariances = []
    class_singular_values = []
    class_whitening_matrix = []
    
    for i, label in enumerate(n_classes):
        indices = np.arange(len(dataset))[targets == i]
        if i == y_target:
            indices = np.setdiff1d(indices,poison_indices)

        data_loader = DataLoader(
            Subset(dataset, indices),
            batch_size=128,
            shuffle=False,
            num_workers=2,
            drop_last=False,
            pin_memory=True
        )
    
        samples = torch.cat([batch[0] for batch in tqdm(data_loader, desc="Processing data", unit="batch")],dim=0)
        # print(f"indices:{len(indices)},tyep:{type(samples)},len:{len(samples)},shape:{samples.size()}\n")
       
        # n * c * w * h  --> n * (c*w*h)---> 1 * (c*w*h)
        mean = torch.mean(samples.view(samples.size()[0],-1),dim=0)
        # n * (c*w*h) - 1 * (c*w*h) = n * (c*w*h)
        centered_samples = samples.view(samples.size()[0],-1) - mean
        covariances = torch.matmul(centered_samples.permute(1,0), centered_samples) / (len(samples) - 1)
        covariances = covariances + 0.1 * torch.eye(covariances.size()[0])

        # print(f"mean:{mean.size()},centered_samples:{centered_samples.size()},covariances.size():{covariances.size()},covariances:{covariances}\n")
        eigenvalues, eigenvectors = torch.symeig(covariances, eigenvectors=True)
        scaling_matrix = torch.diag(1.0 / torch.sqrt(eigenvalues))
        whitening_matrix = torch.matmul(torch.matmul(eigenvectors, scaling_matrix), eigenvectors.t()) 

        # print(f"eigenvalues:{eigenvalues.size()},{eigenvalues},eigenvectors:{eigenvectors.size()},scaling_matrix:{scaling_matrix}, whitening_matrix:{whitening_matrix.size()}\n") 

        class_indices.append(indices)
        class_samples.append(samples)
        class_means.append(mean)
        class_singular_values.append(eigenvalues) 
        class_covariances.append(covariances)
        class_whitening_matrix.append(whitening_matrix)  

    res["class_indices"] = class_indices
    res["class_samples"] = class_samples
    res["class_means"] = class_means
    res["class_singular_values"] = class_singular_values
    res["class_covariances"] = class_covariances
    res["class_whitening_matrix"] = class_whitening_matrix
    return  res

class WhitenLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(WhitenLoss, self).__init__()
        self.reduction = reduction

    def get_latents_with_label(self, latents_z, labels, label):
        indexs = torch.nonzero(labels == label).squeeze()
        # print(f"indexs:{indexs}\n")
        latents = latents_z[indexs]
        return latents
    
    def KL_Divergence(self,latents_z):
        n = latents_z.size()[0]
        # n*k ---> 1*k
        mu_matrix = torch.mean(latents_z, dim=0, keepdim=True)
        # n*k- 1*k ---->  n*k 
        delta_matrix = (latents_z - mu_matrix).to(dtype=torch.float64)

        # sigma_matrix: k*n * n*k ---> k*k
        sigma_matrix = 1.0 / (n - 1) * torch.matmul(torch.transpose(delta_matrix,0,1), delta_matrix).to(dtype=torch.float64)
        
        # Singular value decomposition is also a numerically stable method to estimate the determinant.
        #  When calculating the determinant, you can use the decomposed singular values to get the value of the determinant.
        diag_matrix = 0.1 * torch.eye(sigma_matrix.size()[0]).to(device=sigma_matrix.device,dtype=torch.float64)
        matrix = (sigma_matrix + diag_matrix).to(dtype=torch.float64)
        
        # torch.autograd.gradcheck(torch.det, (matrix,), raise_exception=True)
       
        # U, S, V = torch.svd(matrix) 
        # log_det = torch.sum(torch.log(S),dim=0)
      
        # eigenvalues = torch.eig(matrix)[0][:, 0]
        # log_det = torch.sum(torch.log(eigenvalues))

        sign, log_det = torch.slogdet(matrix)
        trace = torch.trace(matrix) 
        loss = trace - log_det
        # loss = 0.1 * trace + (torch.log(torch.tensor(10^5, device=log_sigma_det.device)) - log_sigma_det)
        return loss 
    
    def CELoss(self,latents_z):
        # n*k ---> 1*k
        mean = torch.mean(latents_z, dim=0)
        # print(mean.size())
        covariance_matrix = torch.eye(mean.size()[0]).to(device=mean.device)
        # define gaussian_distribution
        gaussian_distribution = MultivariateNormal(loc=mean, covariance_matrix=covariance_matrix)
        log_probs = gaussian_distribution.log_prob(latents_z)
        loss = -1.0 * torch.sum(log_probs, dim=0)
        # print(f"mean_size:{mean.size()}, mean:{mean}, covariance_matrix_size:{covariance_matrix.size()}, covariance_matrix:{covariance_matrix}, log_probs_size:{log_probs.size()},log_probs:{log_probs}, loss:{loss}\n")
        return loss

    def forward(self, latents, batch_label):   
        loss = 0.0
        for label in torch.unique(batch_label):
            indexs = torch.nonzero(batch_label == label).squeeze()
            latents_z = latents[indexs]
            if latents_z.dim() > 1 and latents_z.size()[0] > 1:
                loss = loss + self.KL_Divergence(latents_z)
            # if latents_z.dim() > 1 and latents_z.size()[0] > 1:
            #     loss = loss + self.CELoss(latents_z)
                
        # label = 0
        # indexs = torch.nonzero(batch_label == label).squeeze()
        # latents_z = latents[indexs]
        # if latents_z.dim() > 1 and latents_z.size()[0] > 10:
        #     loss = loss + self.KL_Divergence(latents_z)   
        return loss * 0.01

       
class InfomaxLoss(nn.Module):
    def __init__(self, z_dim = 10, x_dim = 1024, out_dim = 1):
        super(InfomaxLoss, self).__init__()
        self.disc = discriminator(z_dim=z_dim, x_dim=x_dim, out_dim=out_dim)
        # self.disc = Disc(z_dim=z_dim, x_dim=x_dim, out_dim=out_dim)
    def forward(self, x_true, z):
        # pass x_true and learned features z from the discriminator
        d_xz = self.disc(x_true, z)
        z_perm = self.permute_dims(z)
        d_x_z = self.disc(x_true, z_perm)
        # info_xz = self.distance_critera(d_xz, d_x_z, critera = "KL-Divergence")
        info_xz = -(d_xz.mean() - d_x_z.mean())
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
    
    def distance_critera(d_xz, d_x_z, critera = "KL-Divergence"):
        if critera == "KL-Divergence":
            info_xz = -(d_xz.mean() - (torch.exp(d_x_z - 1).mean()))
        elif critera == "Wasserstein":
            info_xz = -(d_xz.mean() - d_x_z.mean())
        return info_xz

class MIMRLLoss(nn.Module):
    def __init__(self, supervise_loss=None, alpha=0.001, beta=1.0, x_dim=1024, z_dim=10, n_classes=1, constraint = True, regular = "condition number"):
        super(MIMRLLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.x_dim = x_dim
        self.z_dim = z_dim 
        self.supervise_loss = supervise_loss
        self.constraint = constraint
        self.infomax_loss = InfomaxLoss(z_dim = self.z_dim, x_dim = self.x_dim, out_dim = n_classes)
        self.regular = regular
        if self.regular  == "condition number":
            self.whiten_loss = WhitenLoss()
        elif self.regular  == "RCELoss":
            self.whiten_loss = RCELoss()
            self.alpha = 1.0

    def forward(self, out, labels, x, z):
        supervise_loss = self.supervise_loss(out, labels)
        info_xz = self.infomax_loss(x, z)
        if self.constraint is False:
            whiten_loss = 0.0
        elif self.regular == "condition number":
            whiten_loss = self.whiten_loss(z, labels)
        elif self.regular =="RCELoss":
            whiten_loss = self.whiten_loss(out, labels)
        loss = supervise_loss + self.beta * info_xz + self.alpha * whiten_loss
        return loss, supervise_loss, info_xz, whiten_loss
      
    def reset_parameter(self, beta=1.0, x_dim=1024, z_dim=10):
        self.beta = beta
        self.x_dim = x_dim
        self.z_dim = z_dim
        
    def normalize(self,x, labels):
        
        # 定义CIFAR-10的数据转换
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # 加载CIFAR-10数据集
        cifar10_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

        # 计算每个类别的均值和标准差
        class_means = torch.zeros(10, 3)
        class_stds = torch.zeros(10, 3)

        for i in range(10):
            # 选择当前类别的样本
            class_samples = [img for img, label in cifar10_dataset if label == i]
            
            # 计算均值和标准差
            class_means[i] = torch.stack([img.mean(dim=(1, 2)) for img in class_samples]).mean(dim=0)
            class_stds[i] = torch.stack([img.std(dim=(1, 2)) for img in class_samples]).mean(dim=0)

        # 对每个图像进行每一类的归一化
        normalized_dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=class_means.mean(dim=0), std=class_stds.mean(dim=0))
        ]))

# 使用normalized_dataset进行训练...

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
        Base.__init__(self, task, defense_schedule["schedule"])   
        Defense.__init__(self)
        self.global_defense_schedule = defense_schedule
        self.defense_strategy = defense_schedule["defense_strategy"]
        alpha = self.global_defense_schedule['alpha']
        beta = self.global_defense_schedule['beta']
        x_dim = self.global_defense_schedule['x_dim']
        z_dim = self.global_defense_schedule['z_dim']
        n_classes = self.global_defense_schedule['n_classes']
        self.loss = MIMRLLoss(supervise_loss=self.loss, alpha=alpha, beta=beta, x_dim=x_dim, z_dim=z_dim, n_classes=1)
        self.filter_strategy = defense_schedule["filter_strategy"]
        self.data_filter_object = defense_schedule["filter_object"]

        # work_dir = defense_schedule["schedule"]['work_dir']  
        # log_path = osp.join(work_dir, 'log.txt')
        # log = Log(log_path)

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
        self.init_model()
        self.train(dataset=dataset, schedule=schedule)

    def filter(self, model = None, dataset=None, schedule=None):
        removed_inds, left_inds = self.data_filter_object.filter(model=model, dataset=dataset, schedule=schedule)
        return removed_inds, left_inds
    
    def filter_with_latents(self, latents_path=None, schedule=None):
        removed_inds, left_inds = self.data_filter_object.filter_with_latents(latents_path=latents_path, schedule=schedule)
        return removed_inds, left_inds
    
    def whiten(self, x=None, labels =None):
        # batch_size * x_dim * x_dim
        whitening_matrix = self.whitening_matrix[labels]
        batch_size = x.size()[0]
        # batch_size * x_dim * x_dim * batch_size * x_dim ---> batch_size * x_dim 
        whitened_x = torch.matmul(whitening_matrix, x.view(batch_size, self.x_dim))
        return whitened_x

    #Override  function "train"  in further class "Base" according to the need of the defense strategy "Mine"
    def train(self, dataset=None, schedule=None):
        if dataset is not None:
            self.train_dataset = dataset
        
        if schedule is not None:
            current_schedule = deepcopy(schedule)
        elif self.global_defense_schedule is not None:
            current_schedule = deepcopy(self.global_defense_schedule)
        else: 
            raise AttributeError("Training schedule is None, please check your schedule setting.")
        
        assert "loss" in current_schedule, "Schedule must contain 'loss' configuration!"
        supervise_loss = current_schedule['loss']
        assert "beta" in current_schedule, "Schedule must contain 'beta' configuration!"
        beta = current_schedule['beta']
        assert "x_dim" in current_schedule, "Schedule must contain 'x_dim' configuration!"
        x_dim = current_schedule['x_dim']
        assert "z_dim" in current_schedule, "Schedule must contain 'dim'configuration!"
        z_dim = current_schedule["z_dim"]
        assert "lr_dis" in current_schedule, "Schedule must contain 'lr_dis'configuration!"
        lr_dis = current_schedule["lr_dis"]
        assert "layer" in current_schedule, "Schedule must contain 'layer' configuration!"
        layer = current_schedule['layer']
        
        if 'pretrain' in current_schedule['schedule'] and current_schedule['schedule']['pretrain'] is not None:
            self.model.load_state_dict(torch.load(current_schedule['schedule']['pretrain']), strict=False)

        #os.environ(mapping)：A variable that records information about the current code execution environment;
        # Use GPU
        if 'device' in current_schedule['schedule'] and current_schedule['schedule']['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in current_schedule['schedule']:
                os.environ['CUDA_VISIBLE_DEVICES'] = current_schedule['schedule']['CUDA_VISIBLE_DEVICES']
            # print(current_schedule['CUDA_VISIBLE_DEVICES'])
            # print(f"This machine has {torch.cuda.device_count()} cuda devices")
            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert current_schedule['schedule']['GPU_num'] > 0, 'GPU_num should be a positive integer'
            assert torch.cuda.device_count() >= current_schedule['schedule']['GPU_num'] , 'This machine has {0} cuda devices, and use {1} of them to train'.format(torch.cuda.device_count(), current_schedule['schedule']['GPU_num'])
            device = torch.device("cuda:0")
            gpus = list(range(current_schedule['schedule']['GPU_num']))
            self.model = nn.DataParallel(self.model, device_ids=gpus, output_device=device)
            log(f"This machine has {torch.cuda.device_count()} cuda devices, and use {current_schedule['schedule']['GPU_num']} of them to train.")
        # Use CPU
        else:
            device = torch.device("cpu")
        
        train_loader = DataLoader(
                dataset=self.train_dataset,
                batch_size=current_schedule['schedule']['batch_size'],
                shuffle=True,
                num_workers=current_schedule['schedule']['num_workers'],
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker
            )
        
        self.loss.reset_parameter(beta=beta, x_dim=x_dim, z_dim=z_dim)
        self.model = self.model.to(device)
        # optimizer = self.optimizer(self.model.parameters(), lr=current_schedule['schedule']['lr'])
        optimizer = self.optimizer(self.model.parameters(), lr=current_schedule['schedule']['lr'], momentum=current_schedule['schedule']['momentum'], weight_decay=current_schedule['schedule']['weight_decay'])
        # assign selected device to discriminator
        disc = self.loss.infomax_loss.disc.to(device)
        # Adam optimzer for discriminator
        optim_disc = torch.optim.Adam(disc.parameters(), lr=lr_dis, betas=(0.5,0.999))
        # adv_scheduler = StepLR(optim_disc, step_size=10, gamma=0.9)
        # optim_disc = self.optimizer(disc.parameters(), lr=lr_dis, momentum=current_schedule['schedule']['momentum'], weight_decay=current_schedule['schedule']['weight_decay'])
        # optim_disc = self.optimizer(disc.parameters(), lr=lr_dis, betas=(current_schedule['schedule']['beta1'], current_schedule['schedule']['beta2']))
        
        self.model.train()
        disc.train()
        iteration = 0
        for i in range(current_schedule['schedule']['epochs']):
            for batch_id, batch in enumerate(train_loader): 
                #(img,label,index)  
                batch_img = batch[0]
                batch_label = batch[1]
                batch_indices =  batch[2]
                batch_img = batch_img.to(device)
                batch_label = batch_label.to(device)

                latents, predict_digits = get_latent_rep_without_detach(self.model, layer, batch_img, device)

                # print(f"latents:{latents},predict_digits:{predict_digits}")
                # "latents" are not the final output of the model, so when using nn.DataParallel() for multi-GPU training, 
                # you need to transfer the latents to the final output device of the device.
                # loss, supervise_loss, info_xz, _ = self.loss(predict_digits, batch_label, batch_img, latents.to(device))
                        
                loss, supervise_loss, info_xz, whiten_loss = self.loss(predict_digits, batch_label, batch_img, latents.to(device))

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                #Use the second cropping method.
                # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=current_schedule['schedule']['clip_grad_norm'], norm_type=2)
                optimizer.step()

                optim_disc.zero_grad()
                info_xz.backward(retain_graph=True, inputs=list(disc.parameters()))
                # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=current_schedule['schedule']['clip_grad_norm'], norm_type=2)
                optim_disc.step()
                disc.weight_norm()
                # adv_scheduler.step()
                
                iteration += 1

                if iteration % current_schedule['schedule']['log_iteration_interval'] == 0:
                    last_time = time.time()
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S]", time.localtime()) +f"Epoch:{i+1}/{current_schedule['schedule']['epochs']}, iteration:{batch_id + 1}\{len(self.train_dataset)//current_schedule['schedule']['batch_size']},lr:{current_schedule['schedule']['lr']}, ls_dist:{optim_disc.param_groups[0]['lr']}, loss:{float(loss)}, supervise_loss:{float(supervise_loss)}, info_xz:{float(info_xz)}, whiten_loss:{whiten_loss}, time:{time.time()-last_time}\n"
                    log(msg)
                    # Check the predictions of samples in each batch during training
                    max_num, index = torch.max(predict_digits, dim=1)
                    equal_matrix = torch.eq(index,batch_label)
                    correct_num =torch.sum(equal_matrix)
                    msg =f"batch_size:{current_schedule['schedule']['batch_size']},correct_num:{correct_num}\n"
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






    

    
 

   
   
       

       


       




       
    
    
 

 
    
  


    
  

 

  
  


