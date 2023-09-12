# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/08/21 10:25:23
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : tmp.py
# @Description  : xxxx
from abc import ABC, abstractmethod
from copy import deepcopy
from .Defense import Defense
from ..base.Base import *
from .Spectral import Spectral
#对外的功能：
#1.返回干净模型
#2.返回防御结果：比如准确率，中毒率
#3.如果是数据过滤的算法：则有有毒数据集，过滤后的数据集，以及准确率，假阳率，漏报率等

class BackdoorDefense(object):
    """
    As the context class of the strategy mode, this class aggregates all specific defense strategies, acts as a link between 
    the preceding and the following, and shields high-level modules from direct access to strategies and algorithms.
    It could have several methods:
        1. Return defense model.
        2. Return test results on  the defense model. 
        3. If it is an algorithm for data filtering, return the indexs of the removed samples,
           left samples and samples with target label.

    Args:
        defense_method(core.defenses.Defense):a specific defense strategy object.
        
    Attributes:
        defense_model(torch.nn.Module) : poisoning training dataset
        removed_inds(np.array):the indexs of the removed samples.
        left_inds(np.array):the indexs of the left samples.
        target_label_inds(np.array):the indexs of the samples with target label.
    """
    def __init__(self, defense_method):
        self.defense_method = defense_method
        self.defense_strategy = self.defense_method.get_defense_strategy()
        self.defense_model = None
        self.removed_inds =  None
        self.left_inds = None
        self.target_label_inds = None
    
    def get_defense_strategy(self):
        return self.attack_strategy
    
    def repair(self, transform=None, schedule=None):
        self.defense_model = self.defense_method.repair(transform, schedule)
        return self.defense_model
 
    def filter(self, dataset = None, schedule=None):
        self.removed_inds, self.left_inds, self.target_label_inds =  self.defense_method.filter(dataset, schedule)
        return self.removed_inds, self.left_inds, self.target_label_inds
    
    def test(self, schedule=None, model=None, test_dataset=None): 
        return self.defense_method.test(schedule, model, test_dataset)

    def add_training_observer(self,observer):
        self.attack_method.add_training_observer(observer)
    def delete_training_observer(self,observer):
        self.attack_method.delete_training_observer(observer)

    def add_post_training_observer(self, observer):
        print("add_post_training_observer")
        self.attack_method.add_post_training_observer(observer)
    def delete_post_training_observer(self,observer):
        self.attack_method.delete_post_training_observer(observer)







 
