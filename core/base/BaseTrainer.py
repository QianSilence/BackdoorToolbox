# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/10/19 20:40:21
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : BaseTrainer.py
# @Description  :
from copy import deepcopy
from .Trainer import Trainer 
import torch
import os.path as osp
from utils  import Log
import time
class BaseTrainer(Trainer):
    """
    Notice that the model objct passed in is a mutable object. If the object defined by this class changes the parameters of the model, 
    the object outside will also change, because the variables point to the same object.
    """
    def __init__(self, model, train_dataset, train_loader, loss_fn, optimizer,device) -> None:
        super(BaseTrainer,self).__init__(model, train_dataset, train_loader, loss_fn, optimizer, device)
    def train(self,schedule):
        model = self.model
        train_dataset = self.train_dataset
        train_loader = self.train_loader
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        device = self.device 
   
        work_dir = schedule['work_dir']  
        log_path = osp.join(work_dir, 'log.txt')
        log = Log(log_path)
        experiment = schedule['experiment']
        t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        msg = "\n==========Execute model train in {experiment} at {time}==========\n".format(experiment=experiment, time=t)
        log(msg)

        iteration = 0
        last_time = time.time()
        
        msg = f"Total train samples: {len(train_dataset)}\nTotal test samples: {len(train_dataset)}\nBatch size: {schedule['batch_size']}\niteration every epoch: {len(train_dataset) // schedule['batch_size']}\nInitial learning rate: {schedule['lr']}\n"
        log(msg)

        for i in range(schedule['epochs']):
            # self.adjust_learning_rate(optimizer, i, schedule)
            for batch_id, batch in enumerate(train_loader):
                batch_img = batch[0]
                batch_label = batch[1]
                batch_img = batch_img.to(device)
                batch_label = batch_label.to(device)
                optimizer.zero_grad()
                predict_digits = model(batch_img)
                loss = loss_fn(predict_digits, batch_label)
                # print("predict_digits:{0},batch_label:{1}".format(predict_digits.shape,batch_label.shape))
                loss.backward()
                optimizer.step()
                iteration += 1

                if iteration % schedule['log_iteration_interval'] == 0:
                    last_time = time.time()
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) +f"Epoch:{i+1}/{schedule['epochs']}, iteration:{batch_id + 1}\{len(train_dataset)//schedule['batch_size']},lr: {schedule['lr']}, loss: {float(loss)}, time: {time.time()-last_time}\n"
                    log(msg)
                    max_num , index = torch.max(predict_digits, dim=1)
                    equal_matrix = torch.eq(index,batch_label)
                    correct_num =torch.sum(equal_matrix)
                    msg ="batch_size:{0},correct_num:{1}\n".format(schedule["batch_size"],correct_num)
                    log(msg)
     
            # if len(training_observers) > 0:
            #     model = self.model.to(device)
            #     model.eval()
            #     context = {}
            #     context["model"] = deepcopy(self.model)
            #     context["epoch"] = i
            #     context["device"] = device
            #     context["work_dir"] = work_dir
            #     context["log_path"] = log_path
            #     context["last_time"] = last_time
            #     self._notify_training_observer(context)
            #     model = self.model.to(device)
            #     model.train()
 
        
