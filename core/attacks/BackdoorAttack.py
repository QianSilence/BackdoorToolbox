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
from ..base.Base import *
from .BadNets import BadNets
#这里根据策略设计模式，也可以这样设计：
# （1）令一个个策略直接继承Attack和badnets，然后依据具体的攻击策略实现Attack接口的方法
# （2）设计一个backdoor类作为策略的context，聚合所有的策略，起到承上启下的封装作用，屏蔽高层模块的对策略、算法的直接访问，
# 封装可能的存在的变化

class BackdoorAttack(object):
    """
    As the context class of the strategy mode, this class aggregates all specific attack strategies, 
    acts as a link between the preceding and the following, and shields high-level modules
    from direct access to strategies and algorithms.

    Args:
        attack_method(Attack):a specific attack strategy object.
        
    Attributes:
        poisoned_train_dataset(torch.utils.data.Dataset) : poisoning training dataset
        poisoned_test_dataset(torch.utils.data.Dataset) : poisoning test dataset
        poisoned_model(torch.nn.Module): The resulting poisoned model after training on the poisoning training dataset
    """
    def __init__(self, attack_method):
        self.attack_method = attack_method
        self.attack_strategy = self.attack_method.get_attack_strategy()
        self.train_dataset, self.test_dataset = self.attack_method.get_dataset()
        self.model = self.attack_method.get_model()
        self.poisoned_train_dataset = None
        self.poisoned_test_dataset = None
        self.poisoned_model = None
        
        # print(type(train_dataset))
        # print(type(test_dataset))
    def get_attack_strategy(self):
        return self.attack_strategy
    
    def get_train_dataset(self):
        return self.train_dataset
    def get_test_dataset(self):
        return self.test_dataset

    def get_poisoned_train_dataset(self):
        if self.poisoned_train_dataset is None:
            self.poisoned_train_dataset = self.attack_method.create_poisoned_dataset(self.train_dataset) 
        return self.poisoned_train_dataset
    def get_poisoned_test_dataset(self):
        if self.poisoned_test_dataset is None:
            self.poisoned_test_dataset = self.attack_method.create_poisoned_dataset(self.test_dataset)
        return self.poisoned_test_dataset
    
    def get_model(self):
        return self.model

    def get_poisoned_model(self):
        if self.poisoned_model is None:
            self.attack()
        return self.poisoned_model
     
    # 可以指定寻览数据集，默认为原有的干净数据集训练
    def train(self,schedule=None,dataset=None):
        self.attack_method.train(schedule,dataset)
        self.model = self.attack_method.get_model()
        
    def test(self, schedule=None, model=None, test_dataset=None): 
        self.attack_method.test(schedule, model, test_dataset)

    def attack(self,schedule=None):
        if  self.poisoned_train_dataset is None:
            self.poisoned_train_dataset = self.attack_method.create_poisoned_dataset(self.train_dataset) 
        self.attack_method.train(self.poisoned_train_dataset)
        self.poisoned_model = self.attack_method.get_model()
    def addObserver(self,observer):
        self.attack_method.addObserver(observer)
    def deleteObserver(self,observer):
        self.attack_method.deleteObserver(observer)





 







 
