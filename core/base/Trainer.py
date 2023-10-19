# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/10/19 20:21:34
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : Trainer.py
# @Description  : Abstract class about training strategy, specific training strategies can be customized as needed

from abc import ABC, abstractmethod
class Trainer(ABC):
    def __init__(self, model, train_dataset, train_loader, loss_fn, optimizer, device) -> None:
        super().__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
    @abstractmethod
    def train(self,schedule):
        raise NotImplementedError
