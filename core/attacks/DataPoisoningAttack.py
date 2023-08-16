# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/08/16 20:05:10
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : DataPoisoningAttack.py
from abc import ABC, abstractmethod
from copy import deepcopy
from Attack import Attack
from Base import Base 
class DataPoisoningAttack(Base, Attack):
    def __init__(self, attack_config, train_dataset, test_dataset, model, loss, schedule= None, seed=0, deterministic=False):
        self.attack_config = attack_config
        self.poisoned_train_dataset = None
        self.poisoned_test_dataset = None
        self.poisoned_model = None
        # print(type(train_dataset))
        # print(type(test_dataset))
        Base.__init__(self,train_dataset, test_dataset, model, loss, schedule = schedule, seed=seed, deterministic=deterministic )   
        Attack.__init__(self,attack_config['name'])
    
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
        self.train_dataset = self.poisoned_train_dataset = self.create_poisoned_dataset(dataset,attack_config)
        dataset = self.test_dataset 
        self.test_dataset = self.poisoned_test_dataset = self.create_poisoned_dataset(dataset,attack_config) 
        self.train()
        self.poisoned_model = deepcopy(self.model)

        
    "子类根据具体的数据投毒方法覆写该方法"
    @abstractmethod
    def create_poisoned_dataset(self,dataset,attack_config):
        pass
        ""


 







 
