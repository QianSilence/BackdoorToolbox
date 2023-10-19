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
import time
from utils  import Log, get_latent_rep_without_detach

class MineTrainer(Trainer):
    """
    Notice that the model objct passed in is a mutable object. If the object defined by this class changes the parameters of the model, 
    the object outside will also change, because the variables point to the same object.
    """
    def __init__(self, model, train_dataset, train_loader, loss_fn, optimizer,device) -> None:
        super(MineTrainer,self).__init__(model, train_dataset, train_loader, loss_fn, optimizer, device)
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
                layer = loss_fn.regularize_layer
                # print(layer)
                latents, predict_digits = get_latent_rep_without_detach(self.model, layer, batch_img)
                # print("latents:{0}".format(latents))
                # loss,loss1,loss2 = self.loss_fn(latents, predict_digits, batch_label)
                loss,loss1,loss2,loss3,loss4,entropy_vector = self.loss_fn(latents, predict_digits, batch_label)
                # print("predict_digits:{0},batch_label:{1}".format(predict_digits.shape,batch_label.shape))
                loss.backward()
                optimizer.step()
                iteration += 1

                if iteration % schedule['log_iteration_interval'] == 0:
                    last_time = time.time()
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) +f"Epoch:{i+1}/{schedule['epochs']}, iteration:{batch_id + 1}\{len(self.train_dataset)//schedule['batch_size']},lr: {schedule['lr']}, loss: {float(loss)}, loss1:{float(loss1)},loss2:{float(loss2)},loss3:{float(loss3)},loss4:{float(loss4)},entropy_vector:{entropy_vector}time: {time.time()-last_time}\n"
                    log(msg)
                    max_num , index = torch.max(predict_digits, dim=1)
                    equal_matrix = torch.eq(index,batch_label)
                    correct_num =torch.sum(equal_matrix)
                    msg ="batch_size:{0},correct_num:{1}\n".format(schedule["batch_size"],correct_num)
                    log(msg)
     

        
