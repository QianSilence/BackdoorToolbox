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
import sys

from tqdm import trange
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import numpy as np
from core.base import Base
from core.defenses import Defense 
from copy import deepcopy
from copy import deepcopy
import torch
from torch.utils.data import DataLoader, Subset
import os.path as osp
import time
from utils  import Log, get_latent_rep_without_detach
import numpy as np
import random
from utils import SCELoss
from utils import compute_accuracy
from collections import Counter
from utils.compute import is_singular_matrix
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
    def __init__(self, num_classes=10, reduction="mean"):
        super(UnsupervisedLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.softmax = nn.Softmax()

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
    def forward(self, latents_z, predict, device = torch.device("cpu")):
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
        # diag_matrix = 1.0 * torch.eye(sigma_matrix.size()[1],sigma_matrix.size()[2]).to(device)
        # # k*m*m + 1*m*m ---> k*m*m
        # sigma_matrix = sigma_matrix + diag_matrix.unsqueeze(0)
        # # k*1 * k*1 --> 1
        # loss1 = torch.sum(pi_vector * torch.log(torch.det(sigma_matrix)))

        # Singular value decomposition is also a numerically stable method to estimate the determinant.
        #  When calculating the determinant, you can use the decomposed singular values to get the value of the determinant.
        # U:k *m *m ,  S: k*m  ,  V:k *m *m
        U, S, V = torch.svd(sigma_matrix)
        # print(f"U:{U[0]},shape:{U[0].shape}S:{S},shape:{S.shape},V:{V[0]},shape:{V[0].shape}")
        # m 
        diag_matrix = 0.1 * torch.ones(S.size()[1]).to(device)
        loss1 = torch.sum(pi_vector * torch.log(torch.prod(S + diag_matrix.unsqueeze(0),dim=1)))
        if torch.isnan(loss1):
            is_singular_matrix(S)
        # (k*m)^ * k*1 ---> m*1
        overline_mu = torch.mm(torch.transpose(mu_matrix,0,1),pi_vector.unsqueeze(1)) 
      
        #  k*m  -  1*m ---> k*m
        delta_matrix2 = mu_matrix -  torch.transpose(overline_mu,0,1)

        # k*m  * k*1 ---> k*m
        middle_matrix2 = torch.mul(delta_matrix2, pi_vector.unsqueeze(1))
        
        #  (k*m)^T * k*m ---> m*m
        sigma_matrix2 = torch.mm(delta_matrix2.permute(0, 1),middle_matrix2) 

        # # m*m
        # diag_matrix2 = 0.5 * torch.eye(sigma_matrix2.size()[0],sigma_matrix2.size()[1]).to(device)
        # sigma_matrix2 = sigma_matrix2 + diag_matrix2
        # loss2 = torch.abs(60 - torch.log(torch.det(sigma_matrix2)))

        U2, S2, V2 = torch.svd(sigma_matrix2)
        diag_matrix2 = 0.1 * torch.ones(S2.size()[0]).to(device)
        loss2 = torch.abs(60 - torch.log(torch.prod(S2 + diag_matrix2.unsqueeze(0))))
        loss = loss1 + loss2
        return loss,loss1,loss2
    
class  EntropyLoss(nn.Module):
    def __init__(self, num_classes=10, reduction="mean"):
        super(EntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
    def forward(self,predict):
        prob_matrix = nn.Softmax()(predict)
        log_likelihood_matrix = nn.LogSoftmax()(predict)
        entropy_vector = -1.0* torch.sum(log_likelihood_matrix * prob_matrix,dim=1)
        vector = torch.max(torch.tensor(0), 0.5 - entropy_vector)
        loss = torch.sum(vector*vector).item()
        return loss,entropy_vector

class MineLoss(nn.Module):
    def __init__(self, beta=0.01, eta=0.0):
        super(MineLoss,self).__init__()
        self.beta = beta
        self.cross_entropy_Loss = nn.CrossEntropyLoss()
        self.unsupervised_loss = UnsupervisedLoss()
        self.eta= eta
       
       
    def forward(self, latents_z, predict, labels, batch_indices, clean_data_pool, device=torch.device("cpu")):
        indices = np.arange(len(clean_data_pool))[clean_data_pool==1]
        clean_indices = np.isin(batch_indices,indices)
        clean_sample_predict = predict[clean_indices]
        clean_sample_labels = labels[clean_indices]
        loss1 = 0.0
        if len(clean_sample_predict) > 0:
            loss1 = self.cross_entropy_Loss(clean_sample_predict,clean_sample_labels)

        loss2, loss_post_prob_z, loss_prior_prob_z = self.unsupervised_loss(latents_z, predict,device = device)
        
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
        self.beta = defense_schedule['beta']
        self.poison_rate = defense_schedule['poison_rate']
        self.poison_rate = defense_schedule['poison_rate']
        self.init_size_clean_data_pool = defense_schedule["init_size_clean_data_pool"]
        self.layer = defense_schedule['layer']
        self.loss = MineLoss(beta=self.beta, eta=0.0)
        self.clean_data_pool = None
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
        poisoned_indices = np.arange(dataset_size)[self.poison_data_pool]
        clean_indices = np.arange(dataset_size)[self.clean_data_pool]
        return poisoned_indices, clean_indices
    def _initialize_data_pool(self):
        train_dataset = self.train_dataset
        self.clean_data_pool = np.zeros(len(train_dataset)) 
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
        num = int(self.init_size_clean_data_pool / 10)
        for class_label, indices in class_clean_indices.items():
            if len(indices) >= num:
                indices= random.sample(indices, num)
                random_class_indices[class_label] = indices
                self.clean_data_pool[indices] = 1
                self.poison_data_pool[indices] = 0 

        # for class_label, indices in random_class_indices.items():
        #     print(f"Class {class_label} indices:{indices}\n")
        return (random_class_indices)
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
            print(f"This machine has {torch.cuda.device_count()} cuda devices, and use {current_schedule['GPU_num']} of them to train.")
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
        optimizer = self.optimizer(self.model.parameters(), lr=current_schedule['lr'], momentum=current_schedule['momentum'], weight_decay=current_schedule['weight_decay'])
        
        work_dir = current_schedule['work_dir']  
        log_path = osp.join(work_dir, 'log.txt')
        log = Log(log_path)
        experiment = current_schedule['experiment']
        t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        msg = "\n==========Execute model train in {experiment} at {time}==========\n".format(experiment=experiment, time=t)
        log(msg)

        iteration = 0
        last_time = time.time()
        
        msg = f"Total train samples: {len(self.train_dataset)}\nTotal test samples: {len(self.train_dataset)}\nBatch size: {current_schedule['batch_size']}\niteration every epoch: {len(self.train_dataset) // current_schedule['batch_size']}\nInitial learning rate: {current_schedule['lr']}\n"
        log(msg)

        random_class_indices = self._initialize_data_pool()
        # print(f"random_class_indices:{random_class_indices}/n")
        for i in range(current_schedule['epochs']):
            # self.adjust_learning_rate(optimizer, i, schedule)
            for batch_id, batch in enumerate(train_loader): 
                #(img,label,index)  
                batch_img = batch[0]
                batch_label = batch[1]
                batch_indices = batch[2]
                batch_img = batch_img.to(device)
                batch_label = batch_label.to(device)
                optimizer.zero_grad()
                predict_digits = self.model(batch_img)

                latents, predict_digits = get_latent_rep_without_detach(self.model, self.layer, batch_img)
                # print("latents:{0}".format(latents))
                indices = np.arange(len(self.clean_data_pool))[self.clean_data_pool==1]
                clean_indices = np.isin(batch_indices,indices)
                loss,loss1,loss2,loss_post_prob_z, loss_prior_prob_z = self.loss(latents, predict_digits, batch_label, batch_indices, self.clean_data_pool, device=device)
                loss.backward()
                optimizer.step()
                iteration += 1
                if iteration % current_schedule['log_iteration_interval'] == 0:
                    last_time = time.time()
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) +f"Epoch:{i+1}/{current_schedule['epochs']}, iteration:{batch_id + 1}\{len(self.train_dataset)//current_schedule['batch_size']},lr: {current_schedule['lr']}, loss: {float(loss)}, loss1:{float(loss1)},loss2:{float(loss2)},loss_post_prob_z:{float(loss_post_prob_z)},loss_prior_prob_z:{float(loss_prior_prob_z)},time: {time.time()-last_time}\n"
                    log(msg)
                    # Check the predictions of samples in each batch during training
                    max_num, index = torch.max(predict_digits, dim=1)
                    equal_matrix = torch.eq(index,batch_label)
                    correct_num =torch.sum(equal_matrix)
                    # msg =f"batch_size:{current_schedule['batch_size']},correct_num:{correct_num},num_selected_clean_sample:{np.sum(clean_indices, axis=0)},predict_digits:{predict_digits},softmax(predict_digits):{nn.Softmax()(predict_digits)},batch_label:{batch_label}\n"
                    # log(msg)
            # For each iteration, update 0.05% of the data to Dc
            if np.sum(self.poison_data_pool,0) > int(len(self.poison_data_pool) * self.poison_rate):
                # self._update_data_pool(method="ratio",  ratio=0.005)
                classes = len(self.train_dataset.classes)
                p = torch.zeros(1,classes)
                p[0][0] = current_schedule["threshold_prob"]
                # p[0][1] = 1-current_schedule["threshold_prob"] 
                # p[0][2:-1] = 0

                p[0][1:-1] = (1.0 - current_schedule["threshold_prob"]) / (classes - 1)
                q = torch.tensor([0])
                sceloss = SCELoss(reduction='sum')
                threshold = sceloss(p,q)
                _, _, scores, indices = self._update_data_pool(method="threshold", threshold=threshold.numpy(), device=device)
                log(f"update_data_pool,threshold scores:{threshold},scores:{scores},shape:{scores.shape},add new clean samples:{len(indices)},poison_data_pool:{np.sum(self.poison_data_pool,axis=0)},clean_data_pool:{np.sum(self.clean_data_pool,axis=0)}\n")
            
            # test on clean datasets during trianing
            # top1_correct,top5_correct = self._test2(dataset=self.test_dataset, device=device)
            # total_num = len(self.test_dataset)
            # self.acc[0] =  np.append(self.acc[0],top1_correct/total_num)
            # self.acc[1] =  np.append(self.acc[1],top5_correct/total_num)
            # msg = f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n"
            # log(msg)
            
    def _update_data_pool(self, method="ration", ratio=0.0, threshold = 0.0, batch_size=16, num_workers=8,device=torch.device("cpu")):
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
            
            sceloss = SCELoss(reduction='none')
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
            return predict_digits, labels, scores, indices
        
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






    

    
 

   
   
       

       


       




       
    
    
 

 
    
  


    
  

 

  
  

