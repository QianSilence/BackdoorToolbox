# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/08/21 10:25:23
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : tmp.py
# @Description  : xxxx
from abc import ABC, abstractmethod
from copy import deepcopy
from .Attack import Attack
from .Base import *

class DataPoisoningAttack(Base, Attack):
    """
    the abstract class representing the data poisoning attack  strategy.
    It incldues a abstract method called as create_poisoned_dataset(),which  is overrided by its subclass
    according to the specific attack strategy.

    Args:
        task(dict):The attack strategy is used for the task, including datasets, model, Optimizer algorithm 
            and loss function.
        schedule=None(dict): Config related to model training 
 
    Attributes:
        poisoned_train_dataset(torch.utils.data.Dataset) : poisoning training dataset
        poisoned_test_dataset(torch.utils.data.Dataset) : poisoning test dataset
        poisoned_model(torch.nn.Module): The resulting poisoned model after training on the poisoning training dataset
    """
    def __init__(self, task, schedule= None, seed=0):
        
        self.poisoned_train_dataset = None
        self.poisoned_test_dataset = None
        self.poisoned_model = None
        # print(type(train_dataset))
        # print(type(test_dataset))
        Base.__init__(self, task, schedule = schedule)   
        Attack.__init__(self)
    
    def get_poisoned_train_dataset(self):
        if self.poisoned_train_dataset is not None:
            return self.poisoned_train_dataset
       
    def get_poisoned_test_dataset(self):
        if self.poisoned_test_dataset is not None:
            return self.poisoned_test_dataset
    def get_poisoned_model(self):
        if self.poisoned_model is not None:
            return self.poisoned_model 
        
    def attack(self):
        dataset = self.train_dataset 
        self.train_dataset = self.poisoned_train_dataset = self.create_poisoned_dataset(dataset)
        dataset = self.test_dataset 
        self.test_dataset = self.poisoned_test_dataset = self.create_poisoned_dataset(dataset) 
        self.train()
        self.poisoned_model = deepcopy(self.model)

    @abstractmethod
    def create_poisoned_dataset(self,dataset):
        raise NotImplementedError


 







 
