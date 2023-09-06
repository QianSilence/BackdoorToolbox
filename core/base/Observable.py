# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/09/05 16:51:48
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : tmp.py
# @Description : # @Description  : Use the observer mode to realize the intervention of external classes 
# in the model training process

from abc import ABC, abstractmethod
class Observable(ABC):
    @abstractmethod
    def addObserver(self,observer):
        raise NotImplementedError
    @abstractmethod
    def deleteObserver(self,observer):
        raise NotImplementedError
    @abstractmethod
    def _notifyObservers(self,train_context):
        raise NotImplementedError
    