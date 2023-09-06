# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/08/21 10:26:03
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : tmp.py
# @Description  : xxxx
from abc import abstractmethod
from .Observable import Observable
from .Observer import Observer
import os
import os.path as osp
import time
from copy import deepcopy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, MNIST, CIFAR10
import os 
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
# print(sys.path)
from utils import Log
support_list = (
    DatasetFolder,
    MNIST,
    CIFAR10
)
def check(dataset):
    return isinstance(dataset, support_list)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Base(Observable):
    """
    Base class for backdoor training and testing.
    Args:
        task(dict):The attack strategy is used for the task, including datasets, model, Optimizer algorithm 
            and loss function.
        schedule=None(dict): Config related to model training 

    Attributes:
        train_dataset (types in support_list): Benign training dataset.
        test_dataset (types in support_list): Benign testing dataset.
        model (torch.nn.Module): Network.
        loss (torch.nn.Module): Loss.
        optimizer(torch.optim.optimizer):optimizer.
        schedule (dict): Training or testing global schedule. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
        observers(list[observer]): Contains observer object that interfere with the model training process
    """
    def __init__(self, task, schedule=None):
        assert 'train_dataset' in task, "task must contain 'train_dataset' configuration! "
        assert isinstance(task['train_dataset'], support_list), 'train_dataset is an unsupported dataset type, train_dataset should be a subclass of our support list.'
        self.train_dataset = task['train_dataset']
        assert 'test_dataset' in task, "task must contain 'test_dataset' configuration! "
        assert isinstance(task['test_dataset'], support_list), 'test_dataset is an unsupported dataset type, test_dataset should be a subclass of our support list.'
        self.test_dataset = task['test_dataset']
        assert 'model' in task, "task must contain 'model' configuration! "
        self.model =  task['model'] 
        assert 'loss' in task, "task must contain 'loss' configuration! "
        self.loss = task['loss']

        assert 'optimizer' in task, "task must contain 'optimizer' configuration! "
        self.optimizer = task['optimizer']

        self.global_schedule = deepcopy(schedule)
        self.current_schedule = None

        assert 'seed' in schedule, "task must contain 'seed' configuration! "
        assert 'deterministic' in schedule, "task must contain 'deterministic' configuration! "
        self._set_seed(schedule['seed'], schedule['deterministic'])
        
        self.observers = []


    #根据seed设置训练结果的随机性
    """
        影响可复现的因素主要有这几个：
        1.随机种子 python，numpy,torch随机种子；环境变量随机种子；GPU随机种子
        2.训练使用不确定的算法
        2.1 CUDA卷积优化——CUDA convolution benchmarking
        2.2 Pytorch使用不确定算法——Avoiding nondeterministic algorithms
    """
    def _set_seed(self, seed, deterministic):
        # Use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA).
        torch.manual_seed(seed)

        # Set python seed
        random.seed(seed)

        # Set numpy seed (However, some applications and libraries may use NumPy Random Generator objects,
        # not the global RNG (https://numpy.org/doc/stable/reference/random/generator.html), and those will
        # need to be seeded consistently as well.)
        np.random.seed(seed)

        os.environ['PYTHONHASHSEED'] = str(seed)

        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            # Hint: In some versions of CUDA, RNNs and LSTM networks may have non-deterministic behavior.
            # If you want to set them deterministic, see torch.nn.RNN() and torch.nn.LSTM() for details and workarounds.

    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    def get_model(self):
        return deepcopy(self.model)
    def set_train_dataset(self, dataset):
        self.train_dataset = dataset
    def set_test_dataset(self,dataset):
        self.test_dataset = dataset

        
    def get_dataset(self):
        return self.train_dataset, self.test_dataset
    # def get_poisoned_dataset(self):
        # return self.poisoned_train_dataset, self.poisoned_test_dataset

    def adjust_learning_rate(self, optimizer, epoch):
        if epoch in self.current_schedule['schedule']:
            self.current_schedule['lr'] *= self.current_schedule['gamma']
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.current_schedule['lr']
    
    def train(self, schedule=None, dataset=None):
        if schedule is None and self.global_schedule is None:
            raise AttributeError("Training schedule is None, please check your schedule setting.")
        elif schedule is not None and self.global_schedule is None:
            self.current_schedule = deepcopy(schedule)
        elif schedule is None and self.global_schedule is not None:
            self.current_schedule = deepcopy(self.global_schedule)
        elif schedule is not None and self.global_schedule is not None:
            self.current_schedule = deepcopy(schedule)

        if 'pretrain' in self.current_schedule:
            self.model.load_state_dict(torch.load(self.current_schedule['pretrain']), strict=False)

        #os.environ(mapping)：A variable that records information about the current code execution environment;
        
        # Use GPU
        if 'device' in self.current_schedule and self.current_schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in self.current_schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = self.current_schedule['CUDA_VISIBLE_DEVICES']

            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert self.current_schedule['GPU_num'] > 0, 'GPU_num should be a positive integer'
            assert torch.cuda.device_count() >= self.current_schedule['GPU_num'] , 'This machine has {0} cuda devices, and use {1} of them to train'.format(torch.cuda.device_count(), self.current_schedule['GPU_num'])
            device = torch.device("cuda:0")
            gpus = list(range(self.current_schedule['GPU_num']))
            self.model = nn.DataParallel(self.model, device_ids=gpus, output_device=device)
            print(f"This machine has {torch.cuda.device_count()} cuda devices, and use {self.current_schedule['GPU_num']} of them to train.")
        # Use CPU
        else:
            device = torch.device("cpu")

        train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.current_schedule['batch_size'],
                shuffle=True,
                num_workers=self.current_schedule['num_workers'],
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker
            )

        self.model = self.model.to(device)
        self.model.train()
        optimizer = self.optimizer(self.model.parameters(), lr=self.current_schedule['lr'], momentum=self.current_schedule['momentum'], weight_decay=self.current_schedule['weight_decay'])
        
        work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))

        # log and output:
        # 1. ouput loss and time
        # 2. test and output statistics
        # 3. save checkpoint
        iteration = 0
        last_time = time.time()

        msg = f"Total train samples: {len(self.train_dataset)}\nTotal test samples: {len(self.test_dataset)}\nBatch size: {self.current_schedule['batch_size']}\niteration every epoch: {len(self.train_dataset) // self.current_schedule['batch_size']}\nInitial learning rate: {self.current_schedule['lr']}\n"
        log(msg)

        for i in range(self.current_schedule['epochs']):
            self.adjust_learning_rate(optimizer, i)
            for batch_id, batch in enumerate(train_loader):
                batch_img = batch[0]
                batch_label = batch[1]
                batch_img = batch_img.to(device)
                batch_label = batch_label.to(device)
                optimizer.zero_grad()
                predict_digits = self.model(batch_img)
                loss = self.loss(predict_digits, batch_label)
                loss.backward()
                optimizer.step()

                iteration += 1

                # if iteration % self.current_schedule['log_iteration_interval'] == 0:
                #     msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+1}/{self.current_schedule['epochs']}, iteration:{batch_id + 1}/{len(self.poisoned_train_dataset)//self.current_schedule['batch_size']}, lr: {self.current_schedule['lr']}, loss: {float(loss)}, time: {time.time()-last_time}\n"
                #     last_time = time.time()
                #     log(msg)

            if len(self.observers) > 0:
                train_text = {}
                train_text["model"] = self.model
                train_text["epoch"] = i
                train_text["device"] = device
                self._notifyObservers(train_text)
                   

        
            # if (i + 1) % self.current_schedule['test_epoch_interval'] == 0:
            #     # test result on benign test dataset
            #     predict_digits, labels = self._test(self.test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'])
            #     total_num = labels.size(0)
            #     prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            #     top1_correct = int(round(prec1.item() / 100.0 * total_num))
            #     top5_correct = int(round(prec5.item() / 100.0 * total_num))
            #     msg = "==========Test result on benign test dataset==========\n" + \
            #           time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
            #           f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n"
            #     log(msg)

            #     # test result on poisoned test dataset
            #     # if self.current_schedule['benign_training'] is False:
            #     predict_digits, labels = self._test(self.poisoned_test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'])
            #     total_num = labels.size(0)
            #     prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            #     top1_correct = int(round(prec1.item() / 100.0 * total_num))
            #     top5_correct = int(round(prec5.item() / 100.0 * total_num))
            #     msg = "==========Test result on poisoned test dataset==========\n" + \
            #           time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
            #           f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n"
            #     log(msg)

            #     self.model = self.model.to(device)
            #     self.model.train()
            # #保存模型
            # if (i + 1) % self.current_schedule['save_epoch_interval'] == 0:
            #     self.model.eval()
            #     self.model = self.model.cpu()
            #     ckpt_model_filename = "ckpt_epoch_" + str(i+1) + ".pth"
            #     ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
            #     torch.save(self.model.state_dict(), ckpt_model_path)
            #     self.model = self.model.to(device)
            #     self.model.train()
    # Template Method Pattern：
    # In order to realize that the subclass can interrupt or intervene in the execution process 
    # of the parent class function, use the template method pattern to define the skeleton of 
    # an algorithm in the parent class, but leave the implementation of some specific steps to the subclass, 
    # so that subclasses can customize and extend parts of the algorithm
    @abstractmethod
    def interact_in_training():
        raise NotImplementedError

    def _test(self, dataset, device, batch_size=16, num_workers=8, model=None):
        if model is None:
            model = self.model
        else:
            model = model

        with torch.no_grad():
            test_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker
            )

            model = model.to(device)
            model.eval()

            predict_digits = []
            labels = []
            for batch in test_loader:
                batch_img, batch_label = batch
                batch_img = batch_img.to(device)
                batch_img = model(batch_img)
                batch_img = batch_img.cpu()
                predict_digits.append(batch_img)
                labels.append(batch_label)
            predict_digits = torch.cat(predict_digits, dim=0)
            
            labels = torch.cat(labels, dim=0)
            return predict_digits, labels
        
    def test2(self, dataset, device, batch_size=16, num_workers=8, model=None):
        return self._test(dataset, device, batch_size=batch_size, num_workers=num_workers, model=model)
        

    def test(self, schedule=None, model=None, test_dataset=None, poisoned_test_dataset=None):
        if schedule is None and self.global_schedule is None:
            raise AttributeError("Test schedule is None, please check your schedule setting.")
        elif schedule is not None and self.global_schedule is None:
            self.current_schedule = deepcopy(schedule)
        elif schedule is None and self.global_schedule is not None:
            self.current_schedule = deepcopy(self.global_schedule)
        elif schedule is not None and self.global_schedule is not None:
            self.current_schedule = deepcopy(schedule)

        if model is None:
            model = self.model

        if 'test_model' in self.current_schedule:
            model.load_state_dict(torch.load(self.current_schedule['test_model']), strict=False)

        if test_dataset is None and poisoned_test_dataset is None:
            test_dataset = self.test_dataset
            poisoned_test_dataset = self.poisoned_test_dataset

        # Use GPU
        if 'device' in self.current_schedule and self.current_schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in self.current_schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = self.current_schedule['CUDA_VISIBLE_DEVICES']

            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert self.current_schedule['GPU_num'] >0, 'GPU_num should be a positive integer'
            print(f"This machine has {torch.cuda.device_count()} cuda devices, and use {self.current_schedule['GPU_num']} of them to train.")

            if self.current_schedule['GPU_num'] == 1:
                device = torch.device("cuda:0")
            else:
                gpus = list(range(self.current_schedule['GPU_num']))
                model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
                # TODO: DDP training
                pass
        # Use CPU
        else:
            device = torch.device("cpu")

        work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))

        if test_dataset is not None:
            last_time = time.time()
            # test result on benign test dataset
            predict_digits, labels = self._test(test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'], model)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on benign test dataset==========\n" + \
                  time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                  f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n"
            log(msg)

        if poisoned_test_dataset is not None:
            last_time = time.time()
            # test result on poisoned test dataset
            predict_digits, labels = self._test(poisoned_test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'], model)
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on poisoned test dataset==========\n" + \
                  time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                  f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n"
            log(msg)

    def addObserver(self, observer):
        self.observers.append(observer)

    def deleteObserver(self,observer):
        for item in self.observers:
            if item is observer:
                self.observers.remove(observer)
    def _notifyObservers(self, train_context):
        for observer in self.observers:
            assert callable(observer.work),"function {0} is not callable!".format(observer.work)
            observer.work(train_context)

        
    