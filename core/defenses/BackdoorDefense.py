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


"""
从防御的角度来看，测试时仅有两个任务

防御过程：

1.返回干净模型

防御效果：

1.模型的效果，如：准确率，中毒率

因此需要提供测试接口函数

具体的指标由tester根据接口函数来计算：


2.2 如果是数据过滤的算法，还要有数据过滤的效果：
更细节一点：
返回目标标签
返回目标标签在数据集中对应的索引
返回过滤的有毒数据索引和干净数据索引


综上计算：
具体的过滤指标由tester根据上述接口函数来计算

准确率，假阳率，漏报率等
"""

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
        self.poisoned_indices =  None
        self.clean_indices = None
        self.target_label_inds = None
        self.repaired_model =  None
    def get_defense_strategy(self):
        return self.defense_strategy
    
    def get_repaired_model(self, dataset=None, schedule=None):
        if self.repaired_model is None:
            self.defense_method.repair(dataset, schedule)
        self.repaired_model = self.defense_method.get_model()
        return self.repaired_model
    
    def get_target_label(self):
        return self.defense_method.get_target_label()

    def get_pred_poisoned_sample_dist(self):
        return  self.defense_method.get_poison_data_pool()
    
    def get_real_poisoned_sample_dist(self):
        """
        In the form of a bool array  of all samples in the dataset, 1 is a poisoning sample and 0 is a non-poisoning sample.
        """
        train_dataset, _= self.defense_method.get_dataset()
        poison_indices = train_dataset.get_poison_indices()
        expected = np.zeros(len(train_dataset))
        expected[poison_indices] = 1
        return expected

    def filter(self, dataset=None, schedule=None):
        """
        Return the results of dataset filtering
        """
        self.poisoned_indices, self.clean_indices =  self.defense_method.filter(dataset, schedule)
        return self.poisoned_indices, self.clean_indices
   
    def test(self, schedule=None, model=None, test_dataset=None): 
        return self.defense_method.test(schedule, model, test_dataset)
    
    def invert_random_labels(self):
        return self.defense_method.invert_random_labels()
    

    def add_training_observer(self,observer):
        self.attack_method.add_training_observer(observer)
    def delete_training_observer(self,observer):
        self.attack_method.delete_training_observer(observer)

    def add_post_training_observer(self, observer):
        print("add_post_training_observer")
        self.attack_method.add_post_training_observer(observer)
    def delete_post_training_observer(self,observer):
        self.attack_method.delete_post_training_observer(observer)







 
