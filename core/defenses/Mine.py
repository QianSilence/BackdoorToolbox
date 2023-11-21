# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/10/13 17:35:23
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : test_Mine.py
# @Description  : Backdoor Defense via Mutual Information Neural Estimation 
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
from core.Loss import RKLLoss, SCELoss, CBCELoss, CBSCELoss
""".
需要进一步验证的实验：

1.展示现有防御方案的潜在分离效果，是否出现后门样本的真实回归？后门效果有没有得到抑制？

（1）

生成潜在表示并可视化

2.后门样本中主特征和后门特征是否存在tradeoff关系？softmax得到的p(y|x)是否有所体现？
需要对比展示一下正常样本和后门样本的p(y|x)方差的差异，主要是概率分布的差异，
如熵信息


3.在监督学习下，潜在表示各分类的聚类分布是如何变化的？方差是如何变化的？均值是如何变化的？

需要展示一下方差和均值的变化曲线

"""
class UnsupervisedLoss(nn.Module):
    def __init__(self, num_classes=10, reduction="mean", device=torch.device("cpu")):
        super(UnsupervisedLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.softmax = nn.Softmax()
        self.device=device

    """
    torch.mul(a, b)
    torch.mm(a, b)
    torch.matmul(a, b)
    
    n:the number of samples
    k: k classes

    gamma_matrix
    torch.Size([128, 10])
    pi_vector
    torch.Size([10])
    mu_matrix
    torch.Size([10, 10])
    middle_matrix
    torch.Size([10, 128, 10])
    sigma_matrix
    torch.Size([10, 10, 10])

    """
    def forward(self, latents_z, predict):
        # n*k ---> n*k
        gamma_matrix= self.softmax(predict)
        # n*k ---> k
        N_vector = torch.sum(gamma_matrix, 0) 
        N = gamma_matrix.size()[0]
        pi_vector  = N_vector / N
        # k*n * (n*m) ---> k*m
        mu_matrix = torch.mm(torch.transpose(gamma_matrix,0,1), latents_z) 

        # k*n*m - k*1*m  ---->  k*n*m 
        delta_matrix = latents_z.unsqueeze(0) - mu_matrix.unsqueeze(1)

        # gamma_matrix: n*k--> k*n*1 ---> k*n*1 *  k*n*m ---> k*n*m
        middle_matrix = torch.transpose(gamma_matrix,0,1).unsqueeze(2) * delta_matrix
        #  k*m*n * k*n*m ---> k*m*m
        sigma_matrix = torch.matmul(delta_matrix.permute(0, 2, 1),middle_matrix)

        # #m*m
        diag_matrix = 1.0 * torch.eye(sigma_matrix.size()[1],sigma_matrix.size()[2]).to(self.device)
        # # k*m*m + 1*m*m ---> k*m*m
        sigma_matrix = sigma_matrix + diag_matrix.unsqueeze(0)
        # # k*1 * k*1 --> 1
        # loss1 = torch.sum(pi_vector * torch.log(torch.det(sigma_matrix)))

        # Singular value decomposition is also a numerically stable method to estimate the determinant.
        #  When calculating the determinant, you can use the decomposed singular values to get the value of the determinant.
        
        # U:k *m *m ,  S: k*m  ,  V:k *m *m
        # print(f"sigma_matrix:{sigma_matrix.shape}，{sigma_matrix}")
        U, S, V = torch.svd(sigma_matrix)
        # print(f"U:{U[0]},shape:{U[0].shape}S:{S},shape:{S.shape},V:{V[0]},shape:{V[0].shape}")
        # diag_matrix = 1.0 * torch.ones(S.size()[1]).to(self.device)
        # matrix = S + diag_matrix.unsqueeze(0)
        # matrix = matrix.to(torch.float64)

        matrix= S.to(torch.float64)
        
        # loss1 = torch.sum(pi_vector * torch.log(torch.prod(matrix,dim=1)))
        loss1 = torch.sum(pi_vector * torch.sum(torch.log(matrix),dim=1))


        # (k*m)^T * k*1 ---> m*1
        overline_mu = torch.mm(torch.transpose(mu_matrix,0,1),pi_vector.unsqueeze(1)) 
      
        #  k*m  -  1*m ---> k*m
        delta_matrix2 = mu_matrix -  torch.transpose(overline_mu,0,1)

        # k*m  * k*1 ---> k*m
        middle_matrix2 = torch.mul(delta_matrix2, pi_vector.unsqueeze(1))

        #  (k*m)^T * k*m ---> m*m
        sigma_matrix2 = torch.mm(delta_matrix2.permute(1, 0),middle_matrix2) 


        # # m*m
        diag_matrix2 = 1.0 * torch.eye(sigma_matrix2.size()[0],sigma_matrix2.size()[1]).to(self.device)
        sigma_matrix2 = sigma_matrix2 + diag_matrix2
        # loss2 = torch.abs(60 - torch.log(torch.det(sigma_matrix2)))

        U2, S2, V2 = torch.svd(sigma_matrix2)
        # diag_matrix2 = 1.0 * torch.ones(S2.size()[0]).to(self.device)
        # matrix2 = S2 + diag_matrix2.unsqueeze(0)
        # matrix2= matrix2.to(torch.float64)
        matrix2= S2.to(torch.float64)
       
        # loss2 = torch.abs(60 - torch.log(torch.prod(matrix2,dim=1)))
        loss2 = torch.abs(80 - torch.sum(torch.log(matrix2)))

        if torch.isnan(loss1):
            is_singular_matrix(S)
        # if torch.isnan(loss2):
        #     is_singular_matrix(S2)
        # print(f"matrix:{torch.prod(matrix,dim=1)},matrix2:{torch.prod(matrix2,dim=1)}\n")
        # print(f" torch.prod(matrix2,dim=1):{torch.log(matrix2)}\n")
        # print(f"loss1:{loss1},loss2{loss2}\n")
        loss = loss1 + loss2
        return loss,loss1,loss2
    
class MineLoss(nn.Module):
    def __init__(self, supervised_loss_type="CrossEntropyLoss", volume=5000, beta=0.001, num_classes=10, labels=None, device=torch.device("cpu")):
        super(MineLoss,self).__init__()
        
        self.supervised_loss_type = supervised_loss_type
        self.volume = volume
        self.beta = beta
        self.clean_data_pool =None
        self.clean_indices = None
        self.labels= labels
        self.num_classes = num_classes
        self.device = device

        if supervised_loss_type == "CELoss":
            self.supervisedLoss = nn.CrossEntropyLoss()
        elif supervised_loss_type == "CBCELoss":
            self.supervisedLoss = CBCELoss(beta=(1-1.0/volume),num_classes=num_classes,n_arr=None,device=device)
        elif supervised_loss_type == "SCELoss":
            self.supervisedLoss = SCELoss(alpha=1.0, beta=1.0, num_classes=num_classes,device=device)
        elif supervised_loss_type == "CBSCELoss":
            self.supervisedLoss = CBSCELoss(beta=(1-1.0/volume), num_classes=num_classes,n_arr=None,device=device)
            
        self.unsupervisedLoss = UnsupervisedLoss(num_classes=num_classes, device=device)
     
    def set_clean_pool(self,clean_data_pool=None,batch_indices=None):
        self.clean_data_pool = clean_data_pool
        # indices = np.arange(len(clean_data_pool))[clean_data_pool==1]
        self.clean_indices = np.isin(batch_indices,np.arange(len(clean_data_pool))[clean_data_pool==1])
        # print(f"self.clean_indices:{self.clean_indices.shape}, self.labels:{self.labels.shape}")
        if self.supervised_loss_type == "CBCELoss" or self.supervised_loss_type == "CBSCELoss":
            n_arr = []
            for label in range(self.num_classes):
                clean_pool_labels = self.labels[clean_data_pool==1]
                n_arr.append(np.sum(clean_pool_labels == label))
            self.supervisedLoss.update_n_arr(n_arr) 

       
    def forward(self, latents_z, predict, batch_labels):
        clean_indices = self.clean_indices 
        clean_sample_predict = predict[clean_indices]
        clean_sample_labels = batch_labels[clean_indices]

        loss1 = 0.0
        if len(clean_sample_predict) > 0:
            loss1 = self.supervisedLoss(clean_sample_predict,clean_sample_labels)
            
        loss2,loss_post_prob_z, loss_prior_prob_z = self.unsupervisedLoss(latents_z, predict)
        loss = loss1 + self.beta * loss2
        # print(f"loss:{loss},loss1:{loss1}, loss2:{loss2}, loss_post_prob_z:{loss_post_prob_z}, loss_prior_prob_z:{loss_prior_prob_z}\n")
        return (loss,loss1,loss2,loss_post_prob_z, loss_prior_prob_z)
        # entropyLoss =  EntropyLoss()
        # loss3, _ = entropyLoss(predict)
        # loss = loss1 + self.beta * (loss2) + self.eta* loss3
        # return (loss,loss1,loss2,loss3) 

class Mine(Base,Defense):
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
        self.clean_data_pool = None
        self.buffer_data_pool = None
        self.poison_data_pool = None
        self.acc=np.array([[],[]])
    
    def get_defense_strategy(self):
        return self.defense_strategy
    
    def get_target_label(self):
        subset_indices = np.arange(len(self.poison_data_pool))[self.poison_data_pool==1]
        sub_dataset = Subset(self.train_dataset,subset_indices)
        labels = [data[1] for data in sub_dataset]
        ##Find the number that appears the most, get (number, times) and return the number
        most_common_label = Counter(labels).most_common(1)[0]
        return most_common_label[0]

    def get_clean_data_pool(self):
        if self.clean_data_pool is None:
            self.repair()
        return self.clean_data_pool
    def get_poison_data_pool(self):
        if self.poison_data_pool is None:
            self.repair()
        return self.poison_data_pool
         
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
    def _initialize_data_pool(self,init_size_clean_data_pool):
        train_dataset = self.train_dataset
        self.clean_data_pool = np.zeros(len(train_dataset)) 
        self.buffer_data_pool = np.zeros(len(train_dataset)) 
        self.poison_data_pool = np.ones(len(train_dataset)) 
        poison_indices = train_dataset.get_poison_indices()
        class_clean_indices = {}
        for i in range(len(train_dataset)):
            image,label,_ = train_dataset[i]
            if label not in class_clean_indices:
                class_clean_indices[label] = []
            if i not in poison_indices:
                class_clean_indices[label].append(i)

        # Randomly select 10 samples from each class
        random_class_indices = {}
        # torch.zeros(len(self.train_dataset.data.classes))
        num = int(init_size_clean_data_pool / len(train_dataset.classes))
        for class_label, indices in class_clean_indices.items():
            if len(indices) >= num:
                random_indices = random.sample(indices, num)
                random_class_indices[class_label] = random_indices
                self.clean_data_pool[random_indices] = 1
                self.poison_data_pool[random_indices] = 0 

        # for class_label, indices in random_class_indices.items():
        #     print(f"Class {class_label} indices:{indices}\n")
        return (random_class_indices)
    
    def _update_data_pool(self, method="ration", ratio=0.0, threshold = 0.0, times = 2,batch_size=16, num_workers=8,device=torch.device("cpu")):
        """
        Return index according to ration or threshold
        """
        model = self.model
        train_dataset = self.train_dataset
        # remain_sample_indixes = np.arange(len(self.poison_data_pool))[self.poison_data_pool]
        # print(f"self.poison_data_pool:{self.poison_data_pool ==1}\n")
        remain_sample_indixes = subset_indices = np.arange(len(self.poison_data_pool))[self.poison_data_pool==1]
        test_dataset = Subset(train_dataset,subset_indices)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
            worker_init_fn=self._seed_worker
        )
        with torch.no_grad(): 
            model = model.to(device)
            model.eval()
            predict_digits = []
            labels = []
            for _, batch in enumerate(test_loader):
                batch_img, batch_label = batch[0],batch[1]
                batch_img = batch_img.to(device)
                batch_img = model(batch_img)
                batch_img = batch_img.cpu()
                predict_digits.append(batch_img)
                labels.append(batch_label)
        
            
            predict_digits = torch.cat(predict_digits, dim=0)
            labels = torch.cat(labels, dim=0)
            # print(f"the shape of labels:{labels.shape}")
            
            sceloss = SCELoss(alpha=0.1, beta=1.0, reduction='none')
            
            """
             Here, notice the tyep and shape of data.
            """
            # print(f"predict_digits:{predict_digits},shape:{predict_digits.shape},labels:{labels},shape:{labels.shape}\n")
            scores = sceloss(predict_digits,labels)
            scores = scores.numpy()
            # Return the sorted index after sorting from small to large using np.argsort()
            if method == "ratio":
                sorted_indexes = scores.argsort()[: int(len(scores) * ratio)]
                indices = remain_sample_indixes[sorted_indexes]
            elif method == "threshold":
                less_threshold_indexes = np.arange(len(scores))[scores < threshold]
                indices = remain_sample_indixes[less_threshold_indexes]

            if len(indices) > 0:
                self.clean_data_pool[indices] = 1
                self.poison_data_pool[indices] = 0
            # selected_indices = []
            # if len(indices) > 0:
            #     self.buffer_data_pool[indices] = self.buffer_data_pool[indices] + 1
            #     selected_indices = np.arange(len(self.buffer_data_pool))[self.buffer_data_pool == times]
            #     self.clean_data_pool[selected_indices] = 1
            #     self.poison_data_pool[selected_indices] = 0
            #     self.buffer_data_pool[selected_indices] = 0

            return predict_digits, labels, scores, indices
        
    def unlearning(self,train_dataset,predict_digits,batch_labels,batch_indices, threshold):
        work_dir = self.global_schedule['work_dir']  
        log_path = osp.join(work_dir, 'log.txt')
        log = Log(log_path)
        rkl_loss = RKLLoss(alpha=1.0, beta=1.0,num_classes = self.num_classes, prob_min=1e-13, one_hot_min=1e-20,reduction="none")
        # k
        loss = rkl_loss(predict_digits,batch_labels)
        batch_labels = batch_labels.cpu().numpy()
        batch_indices = batch_indices.cpu().numpy()
        susp_poisoned_indces = batch_indices[loss.cpu() < threshold]
        labels = []
        if len(susp_poisoned_indces) > 0:
            for i in range(len(susp_poisoned_indces)):
                random_label = np.random.choice(np.delete(np.arange(self.num_classes),batch_labels[i]),size=1,)[0]
                labels.append(random_label)
            # log(f"susp_poisoned_indces:{susp_poisoned_indces},{len(susp_poisoned_indces)},labels:{labels},{len(labels)}\n")
            # log(f"loss:{loss}\n")
            train_dataset.modify_targets(susp_poisoned_indces,np.array(labels))
        return susp_poisoned_indces
       
    #Override  function "train"  in further class "Base" according to the need of the defense strategy "Mine"
    def train(self, dataset=None, schedule=None):
       
        if schedule is not None:
            current_schedule = deepcopy(schedule)
        elif self.global_schedule is not None:
            current_schedule = deepcopy(self.global_schedule)
        else: 
            raise AttributeError("Training schedule is None, please check your schedule setting.")
        
        if dataset is not None:
            self.train_dataset = dataset

        work_dir = current_schedule['work_dir']  
        log_path = osp.join(work_dir, 'log.txt')
        log = Log(log_path)
        assert "supervised_loss_type" in current_schedule, "Schedule must contain 'supervised_loss_type' configuration!"
        supervised_loss_type = current_schedule['supervised_loss_type']
        assert "volume" in current_schedule, "Schedule must contain 'volume' configuration!"
        volume = current_schedule['volume']
       
        
        assert "start_epoch" in current_schedule, "Schedule must contain 'start_epoch' configuration!"
        start_epoch = current_schedule['start_epoch']
        assert "beta" in current_schedule, "Schedule must contain 'beta' configuration!"
        beta = current_schedule['beta']
        assert "delta_beta" in current_schedule, "Schedule must contain 'delta_beta'configuration!"
        delta_beta = current_schedule["delta_beta"]
        assert "beta_threshold" in current_schedule, "Schedule must contain 'beta_threshold'configuration!"
        beta_threshold = current_schedule["beta_threshold"]

        assert "poison_rate" in current_schedule, "Schedule must contain 'poison_rate' configuration!"
        poison_rate = current_schedule['poison_rate']
        assert "init_size_clean_data_pool" in current_schedule, "Schedule must contain 'init_size_clean_data_pool' configuration!"
        init_size_clean_data_pool = current_schedule["init_size_clean_data_pool"]
        assert "layer" in current_schedule, "Schedule must contain 'layer' configuration!"
        layer = current_schedule['layer']
        
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

        self.model = self.model.to(device)
        self.model.train()
        optimizer = self.optimizer(self.model.parameters(), lr=current_schedule['lr'])
        labels = np.array(self.train_dataset.get_real_targets())
        self.loss = MineLoss(supervised_loss_type=supervised_loss_type,volume=volume,beta=beta,num_classes=self.num_classes,labels=labels, device=device)
        
        # optimizer = self.optimizer(self.model.parameters(), lr=current_schedule['lr'], momentum=current_schedule['momentum'], weight_decay=current_schedule['weight_decay'])
        experiment = current_schedule['experiment']
        t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        msg = "\n==========Execute model train in {experiment} at {time}==========\n".format(experiment=experiment, time=t)
        log(msg)

        iteration = 0
        last_time = time.time()
        
        msg = f"Total train samples: {len(self.train_dataset)}\nTotal test samples: {len(self.train_dataset)}\nBatch size: {current_schedule['batch_size']}\niteration every epoch: {len(self.train_dataset) // current_schedule['batch_size']}\nInitial learning rate: {current_schedule['lr']}\n"
        log(msg)

        random_class_indices = self._initialize_data_pool(init_size_clean_data_pool)
        accumulate = 0
        for key in random_class_indices.keys():
            log(f"Initialize_data_pool, the number of sample with label:{key} in poisoned_train_dataset:{len(random_class_indices[key])}\n")
        # print(f"random_class_indices:{random_class_indices}/n")
        total = 0
        from_poison_data_num = 0 
        for i in range(current_schedule['epochs']):
            unlearning_indces = np.array([])
            # self.adjust_learning_rate(optimizer, i, schedule)
            if i >= start_epoch and self.loss.beta < beta_threshold:
                self.loss.beta = self.loss.beta + delta_beta
            for batch_id, batch in enumerate(train_loader): 
                #(img,label,index)  
                batch_img = batch[0]
                batch_label = batch[1]
                batch_indices = batch[2]
                batch_img = batch_img.to(device)
                batch_label = batch_label.to(device)
                optimizer.zero_grad()
                # predict_digits = self.model(batch_img)
                latents, predict_digits = get_latent_rep_without_detach(self.model, layer, batch_img, device)
                # print(f"latents:{latents},predict_digits:{predict_digits}")
                indices = np.arange(len(self.clean_data_pool))[self.clean_data_pool==1]
                clean_indices = np.isin(batch_indices,indices)
                # "latents" are not the final output of the model, so when using nn.DataParallel() for multi-GPU training, 
                # you need to transfer the latents to the final output device of the device.
                self.loss.set_clean_pool(self.clean_data_pool,batch_indices)
                loss,loss1,loss2,loss_post_prob_z, loss_prior_prob_z = self.loss(latents.to(device), predict_digits, batch_label)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    susp_poisoned_indces = self.unlearning(self.train_dataset,predict_digits,batch_label,batch_indices,current_schedule["unlearning_threshold"])
                    unlearning_indces = np.concatenate((unlearning_indces,susp_poisoned_indces))
                
                iteration += 1

                if iteration % current_schedule['log_iteration_interval'] == 0:
                    last_time = time.time()
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) +f"Epoch:{i+1}/{current_schedule['epochs']}, iteration:{batch_id + 1}\{len(self.train_dataset)//current_schedule['batch_size']},lr: {current_schedule['lr']}, loss: {float(loss)}, loss1:{float(loss1)},loss2:{float(loss2)},loss_post_prob_z:{float(loss_post_prob_z)},loss_prior_prob_z:{float(loss_prior_prob_z)},time: {time.time()-last_time}\n"
                    log(msg)
                    # Check the predictions of samples in each batch during training
                    max_num, index = torch.max(predict_digits, dim=1)
                    equal_matrix = torch.eq(index,batch_label)
                    correct_num =torch.sum(equal_matrix)
                    msg =f"batch_size:{current_schedule['batch_size']},correct_num:{correct_num},num_selected_clean_sample:{np.sum(clean_indices, axis=0)}\n"
                    log(msg)

            poison_indices = self.train_dataset.get_poison_indices()
            from_poisoned_indces = np.intersect1d(unlearning_indces,poison_indices).tolist()
            total = total + len(unlearning_indces)
            from_poison_data_num = from_poison_data_num + len(from_poisoned_indces)
            log(f"total:{total}, from poisoned datasets:{from_poison_data_num}, new unlearning_indces:{len(unlearning_indces)},in which the number of those from poisoned datasets:{len(from_poisoned_indces)},{from_poisoned_indces}\n")
            # For each iteration, update 0.05% of the data to Dc
            if i > 0 and i % current_schedule["filter_epoch_interation"] == 0 and np.sum(self.poison_data_pool,0) > int(len(self.poison_data_pool) * poison_rate):
                threshold = current_schedule["threshold"]
                # times = current_schedule["times"]
                _, _, scores, indices = self._update_data_pool(method="threshold", threshold=threshold, times=1, device=device)
                poison_indices = self.train_dataset.get_poison_indices()
                filter_poisoned_indces = np.intersect1d(indices,poison_indices).tolist()
                accumulate = accumulate + len(filter_poisoned_indces)
                labels = np.array(self.train_dataset.get_real_targets())
                log(f"update_data_pool,threshold scores:{threshold},scores:{scores},shape:{scores.shape},add new clean samples:{len(indices)}, \
                    in which the number of those from poisoned datasets:{len(filter_poisoned_indces)},from:{labels[filter_poisoned_indces]},accumulate:{accumulate}, \
                        poison_data_pool:{np.sum(self.poison_data_pool,axis=0)}, clean_data_pool:{np.sum(self.clean_data_pool,axis=0)},buffer_data_pool:{np.sum(self.buffer_data_pool==1,axis=0)}\n")
           
            # redefine train_loader
            train_loader = DataLoader(
                dataset=self.train_dataset,
                batch_size=current_schedule['batch_size'],
                shuffle=True,
                num_workers=current_schedule['num_workers'],
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker
            )

            # # test on clean datasets during trianing
            # top1_correct,top5_correct = self._test2(dataset=self.test_dataset, device=device)
            # total_num = len(self.test_dataset)
            # self.acc[0] =  np.append(self.acc[0],top1_correct/total_num)
            # self.acc[1] =  np.append(self.acc[1],top5_correct/total_num)
            # msg = f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n"
            # log(msg)

        
    def _test2(self, dataset, device):    
        model = self.model
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






    

    
 

   
   
       

       


       




       
    
    
 

 
    
  


    
  

 

  
  

