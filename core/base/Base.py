# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/08/21 10:26:03
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : Base.py
# @Description  : Logical implementation of model training and testing
from abc import abstractmethod
from .TrainingObservable import TrainingObservable
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
from torchvision.datasets.vision import VisionDataset
from utils import compute_accuracy
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
# print(sys.path)
from utils import Log, get_latent_rep_without_detach
# ignore warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
support_list = (
    DatasetFolder,
    MNIST,
    CIFAR10,
    VisionDataset
)
def check(dataset):
    return isinstance(dataset, support_list)
"""
这里看可以依据训练策略自定义训练函数
"""
class Base(TrainingObservable):
    """
    Base class for training and testing.According to the principle of single function, this class is only responsible for model training 
    and can not perceive the attack and defense strategies of high-level modules.
    Args:
        task(dict): The training task, including datasets, model, Optimizer algorithm 
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
        current_schedule = None

        assert 'seed' in schedule, "task must contain 'seed' configuration! "
        assert 'deterministic' in schedule, "task must contain 'deterministic' configuration! "
        if 'seed' in schedule and schedule['seed'] is not None and 'deterministic' in schedule and schedule['deterministic']: 
            self._set_seed(schedule['seed'], schedule['deterministic'])
        self.training_observers = []
        self.post_training_observers = []


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
    #Here ensure that the type of the output model is nn.module 
    def get_model(self):
        if isinstance(self.model,nn.DataParallel):
            return deepcopy(self.model.module)
        else:
            return deepcopy(self.model)
    def set_train_dataset(self, dataset):
        self.train_dataset = dataset
    def set_test_dataset(self,dataset):
        self.test_dataset = dataset

        
    def get_dataset(self):
        return self.train_dataset, self.test_dataset

    def adjust_learning_rate(self, optimizer, epoch, current_schedule):
        if epoch in current_schedule['schedule']:
            current_schedule['lr'] *= current_schedule['gamma']
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_schedule['lr']
    
    def train(self, dataset=None, schedule=None):
        if schedule is not None:
            current_schedule = deepcopy(schedule)
        elif self.global_schedule is not None:
            current_schedule = deepcopy(self.global_schedule)
        else: 
            raise AttributeError("Training schedule is None, please check your schedule setting.")

        if dataset is not None:
            self.train_dataset = dataset
     

        if 'pretrain' in current_schedule and current_schedule['pretrain'] is not None:
            self.model.load_state_dict(torch.load(current_schedule['pretrain']), strict=False)

        #os.environ(mapping)：A variable that records information about the current code execution environment;
        # Use GPU
        if 'device' in current_schedule and current_schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in current_schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = current_schedule['CUDA_VISIBLE_DEVICES']
            # print(current_schedule['CUDA_VISIBLE_DEVICES'])
            # print(f"This machine has {torch.cuda.device_count()} cuda devices")
            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert current_schedule['GPU_num'] > 0, 'GPU_num should be a positive integer'
            assert torch.cuda.device_count() >= current_schedule['GPU_num'] , 'This machine has {0} cuda devices, and use {1} of them to train'.format(torch.cuda.device_count(), current_schedule['GPU_num'])
            device = torch.device("cuda:0")
            gpus = list(range(current_schedule['GPU_num']))
            self.model = nn.DataParallel(self.model, device_ids=gpus, output_device=device)
            print(f"This machine has {torch.cuda.device_count()} cuda devices, and use {current_schedule['GPU_num']} of them to train.")
        # Use CPU
        else:
            device = torch.device("cpu")

        train_loader = DataLoader(
                self.train_dataset,
                batch_size=current_schedule['batch_size'],
                shuffle=True,
                num_workers=current_schedule['num_workers'],
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker
            )

        self.model = self.model.to(device)
        self.model.train()
        optimizer = self.optimizer(self.model.parameters(), lr=current_schedule['lr'], momentum=current_schedule['momentum'], weight_decay=current_schedule['weight_decay'])
        
        work_dir = current_schedule['work_dir']  
        log_path = osp.join(work_dir, 'log.txt')
        log = Log(log_path)
        experiment = current_schedule['experiment']
        t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        msg = "\n==========Execute model train in {experiment} at {time}==========\n".format(experiment=experiment, time=t)
        log(msg)

        iteration = 0
        last_time = time.time()
        
        msg = f"Total train samples: {len(self.train_dataset)}\nTotal test samples: {len(self.train_dataset)}\nBatch size: {current_schedule['batch_size']}\niteration every epoch: {len(self.train_dataset) // current_schedule['batch_size']}\nInitial learning rate: {current_schedule['lr']}\n"
        log(msg)

        for i in range(current_schedule['epochs']):
            # self.adjust_learning_rate(optimizer, i, schedule)
            for batch_id, batch in enumerate(train_loader):
                batch_img = batch[0]
                batch_label = batch[1]
                batch_img = batch_img.to(device)
                batch_label = batch_label.to(device)
                optimizer.zero_grad()
                predict_digits = self.model(batch_img)
                loss = self.loss(predict_digits, batch_label)
                # print("predict_digits:{0},batch_label:{1}".format(predict_digits.shape,batch_label.shape))
                loss.backward()
                optimizer.step()
                iteration += 1

                if iteration % current_schedule['log_iteration_interval'] == 0:
                    last_time = time.time()
                    msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) +f"Epoch:{i+1}/{current_schedule['epochs']}, iteration:{batch_id + 1}\{len(self.train_dataset)//current_schedule['batch_size']},lr: {current_schedule['lr']}, loss: {float(loss)}, time: {time.time()-last_time}\n"
                    log(msg)
                    max_num , index = torch.max(predict_digits, dim=1)
                    equal_matrix = torch.eq(index,batch_label)
                    correct_num =torch.sum(equal_matrix)
                    msg ="batch_size:{0},correct_num:{1}\n".format(current_schedule["batch_size"],correct_num)
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
 
    # Template Method Pattern：
    # In order to realize that the subclass can interrupt or intervene in the train process of the parent class function,
    # use the template method pattern to define the skeleton of an algorithm in the parent class, but leave the implementation 
    # of some specific steps to the subclass, so that subclasses can customize and extend parts of the algorithm
   
    def interact_in_training():
        pass

    def _test(self, dataset, device, batch_size=16, num_workers=8, model=None, work_dir = None):
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
                batch_img, batch_label = batch[0],batch[1]
                batch_img = batch_img.to(device)
                batch_img = model(batch_img)
                batch_img = batch_img.cpu()
                predict_digits.append(batch_img)
                labels.append(batch_label)
                # if work_dir is not None:
                #     log_path = osp.join(work_dir, 'log.txt')
                #     log = Log(log_path)
                #     # print(torch.tensor(batch_img).shape)
                #     max_num , index = torch.max(batch_img, dim=1)
                #     equal_matrix = torch.eq(index,batch_label)
                #     correct_num =torch.sum(equal_matrix)
                #     msg ="batch_size:{0},correct_num:{1}\n".format(batch_label.size()[0],correct_num)
                #     log(msg)

            predict_digits = torch.cat(predict_digits, dim=0)
            
            labels = torch.cat(labels, dim=0)
            # print(labels)
            return predict_digits, labels
        
    # Here, new test models, datasets and schedules are allowed to be passed in, and by default,
    #  the models, datasets and schedules defined in this class are used.
    def test(self, schedule=None, model=None, test_dataset=None):
        if schedule is not None:
            current_schedule = schedule
        elif self.global_schedule is not None:
            current_schedule = self.global_schedule
        else:
            raise AttributeError("Test schedule is None, please check your schedule setting.")

        if model is None:
            model = self.model

        if test_dataset is None:
            test_dataset = self.test_dataset

        
        if 'device' in current_schedule and current_schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in current_schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = current_schedule['CUDA_VISIBLE_DEVICES']

            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert current_schedule['GPU_num'] > 0, 'GPU_num should be a positive integer'
            assert torch.cuda.device_count() >= current_schedule['GPU_num'] , 'This machine has {0} cuda devices, and use {1} of them to test'.format(torch.cuda.device_count(), current_schedule['GPU_num'])
            device = torch.device("cuda:0")
            gpus = list(range(current_schedule['GPU_num']))
            model = nn.DataParallel(model, device_ids=gpus, output_device=device)
            print(f"This machine has {torch.cuda.device_count()} cuda devices, and use {current_schedule['GPU_num']} of them to test.")
        # Use CPU
        else:
            device = torch.device("cpu")
            print(f"Use cpu to test.")

        work_dir = current_schedule['work_dir']
        log = Log(osp.join(work_dir, 'log.txt'))
        experiment = current_schedule['experiment']
        t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        msg = "\n==========Execute model test in {experiment} at {time}==========\n".format(experiment=experiment, time=t)
        log(msg)
        
        last_time = time.time()
        # test result on test dataset
        predict_digits, labels = self._test(test_dataset, device, batch_size=current_schedule['batch_size'], num_workers=current_schedule['num_workers'], model=model,work_dir=work_dir)

        total_num = labels.size(0)
        prec1, prec5 = compute_accuracy(predict_digits, labels, topk=(1, 5))
        top1_correct = int(round(prec1.item() / 100.0 * total_num))
        top5_correct = int(round(prec5.item() / 100.0 * total_num)) 
        
        msg = "\n==========Test result on test dataset==========\n"
        log(msg)
        msg = f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n"
        log(msg)
        return predict_digits, labels

    def add_training_observer(self, observer):
        self.training_observers.append(observer)

    def delete_training_observer(self,observer):
        for item in self.training_observers:
            if item is observer:
                self.training_observers.remove(observer)
    def _notify_training_observer(self, train_context):
        for observer in self.training_observers:
            assert callable(observer.work),"function {0} is not callable!".format(observer.work)
            observer.work(train_context)
    def add_post_training_observer(self, observer):
        self.post_training_observers.append(observer)

    def delete_post_training_observer(self,observer):
        for item in self.post_training_observers:
            if item is observer:
                self.post_training_observers.remove(observer)
    def _notify_post_training_observer(self, train_context):
        for observer in self.observers:
            assert callable(observer.work),"function {0} is not callable!".format(observer.work)
            observer.work(train_context)


        
    
