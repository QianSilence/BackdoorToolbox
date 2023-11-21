import yaml
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.transforms import Compose, RandomCrop,RandomHorizontalFlip, ToTensor, ToPILImage, Resize,Normalize
import torch
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from models import BaselineMNISTNetwork,ResNet
from config.load_config import load_config

config, inner_dir, config_name = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)),"task_config.yaml"))


support_tasks = ["BaselineMNISTNetwork_MNIST","ResNet-18_CIFAR-10"]
support_datasets = ["MNIST","CIFAR-10","CIFAR-100","ImageNet"]
support_models = ["BaselineMNISTNetwork","ResNet-18","ResNet-18","ResNet-18","ResNet-34","ResNet-50","ResNet-101","ResNet-152"]
support_optimizers = ["SGD","Adam"]
support_losses = ["CrossEntropyLoss"]

def get_dataset(dataset_info=None):
    dataset_type = dataset_info['type']
    assert dataset_type in support_datasets, f"{dataset_type} is not in support_datasets:{support_datasets}"
    datasets_root_dir = dataset_info["dataset_root_dir"]
    if dataset_type == "MNIST":
        transform_train = Compose([
            ToTensor()
        ])
        trainset = torchvision.datasets.MNIST(datasets_root_dir, train=True, transform=transform_train, download=True)
        transform_test = Compose([
            ToTensor()
        ])
        testset = torchvision.datasets.MNIST(datasets_root_dir, train=False, transform=transform_test, download=True)
        classes = trainset.classes
        num_classes = 10
    elif dataset_type == "CIFAR-10": 
        transform_train = Compose([
            RandomCrop(32, padding=4, padding_mode="reflect"),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])
        ])
        trainset = torchvision.datasets.CIFAR10(datasets_root_dir, train=True, transform=transform_train, download=True)
        transform_test = Compose([
            ToTensor(),
            Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])
        ])
        testset = torchvision.datasets.CIFAR10(datasets_root_dir, train=False, transform=transform_test, download=True)
        classes = trainset.classes
        num_classes = 10

    elif dataset_type == "CIFAR-100":
        transform_train = Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([125.3, 123.0, 113.9]) / 255.0,
                np.array([63.0, 62.1, 66.7]) / 255.0),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([125.3, 123.0, 113.9]) / 255.0,
                np.array([63.0, 62.1, 66.7]) / 255.0),
        ])
        trainset = torchvision.datasets.CIFAR100(datasets_root_dir, train=True, transform=transform_train, download=True)
        testset = torchvision.datasets.CIFAR100(datasets_root_dir, train=True, transform=transform_test, download=True)
        classes = trainset.classes
        num_classes = 100
    elif dataset_type == "ImageNet":
        pass
    return trainset, testset, classes, num_classes

def get_model(model_info=None):
    model_type = model_info['type']
    assert model_type in support_models, f"{model_type} is not in support_datasets:{support_models}"
    if model_type == "BaselineMNISTNetwork":
        model = BaselineMNISTNetwork()
    elif model_type == "ResNet-18":
        model = ResNet(18,num_classes=model_info['num_classes'])
    elif model_type == "ResNet-34":
        model = ResNet(34,num_classes=model_info['num_classes'])
    elif model_type == "ResNet-50":
        model = ResNet(50,num_classes=model_info['num_classes'])
    elif model_type == "ResNet-101":
        model = ResNet(101,num_classes=model_info['num_classes'])
    elif model_type == "ResNet-152":
        model = ResNet(152,num_classes=model_info['num_classes'])
    return model

def get_loss(loss_info=None):
    loss_type = loss_info['type']
    assert loss_type in support_losses, f"{loss_type} is not in support_datasets:{support_losses}"
    if loss_type == "CrossEntropyLoss":
        loss = torch.nn.CrossEntropyLoss()
    return loss

def get_optimizer(optimizer=None):
    assert optimizer in support_optimizers, f"{optimizer} is not in support_datasets:{support_optimizers}"
    if optimizer == "SGD":
        optimizer = torch.optim.SGD
    elif optimizer == "Adam":
        optimizer = torch.optim.Adam
    return optimizer

def get_task_config(task = None):
    assert task in support_tasks, f"{task} is not in support_datasets:{support_tasks}"
    task_config = {
        'train_dataset': None,
        'test_dataset' : None,
        'model' : None,
        'optimizer': None,
        "loss": None,
    }
    task_config['train_dataset'], task_config['test_dataset'], _,_ = get_dataset(dataset_info=config[task]["dataset"])
    task_config['model'] = get_model(model_info=config[task]["model"])
    task_config['loss'] = get_loss(loss_info=config[task]["loss"])
    task_config['optimizer'] = get_optimizer(optimizer=config[task]["optimizer"])
    return task_config

def get_task_schedule(task = None):
    assert task in support_tasks, f"{task} is not in support_datasets:{support_tasks}"
    schedule = {
        # related task, attack and defense
        'experiment': None,
        'work_dir': None,

        # Settings for reproducible/repeatable experiments
        'seed': None,
        'deterministic': None,
        # Settings related to device
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': None,
        'GPU_num': None,

        # Settings related to tarining 
        'pretrain': None,
        'epochs': None,
        'batch_size': None,
        'num_workers': None,
        'lr': None,
        'momentum': None,
        'weight_decay': None,
        'gamma': None,

        # Settings aving model,data and logs
        'log_iteration_interval': None,
    }
    
    schedule['experiment'] = config[task]['schedule']['experiment']

    # repeatability_setting:
    schedule['seed'] = config[task]['schedule']['seed']
    schedule['deterministic'] = config[task]['schedule']['deterministic']

    # related_device:
    schedule['device'] = config[task]['schedule']['device']
    schedule['CUDA_VISIBLE_DEVICES'] = config[task]['schedule']['CUDA_VISIBLE_DEVICES']
    schedule['GPU_num'] = config[task]['schedule']['GPU_num']

    # related_tarin: 
    schedule['pretrain'] = config[task]['schedule']['pretrain']
    schedule['epochs'] = config[task]['schedule']['epochs']
    schedule['batch_size'] = config[task]['schedule']['batch_size']
    schedule['num_workers'] = config[task]['schedule']['num_workers']

    # related_optimization:
    schedule['lr'] = config[task]['schedule']['lr']
    schedule['momentum'] = config[task]['schedule']['momentum']
    schedule['weight_decay'] = float(config[task]['schedule']['weight_decay'])
    schedule['gamma'] = config[task]['schedule']['gamma']
    #log:
    schedule['work_dir'] = config[task]['schedule']['work_dir']
    schedule['log_iteration_interval'] = config[task]['schedule']['log_iteration_interval']
    return schedule

def get_untransformed_dataset(task=None):
    assert task in support_tasks, f"{task} is not in support_datasets:{support_tasks}"
    datasets=config[task]["dataset"]
    datasets_root_dir=config[task]["data"]["dataset_root_dir"]
    assert datasets in support_datasets, f"{datasets} is not in support_datasets:{support_datasets}"
    if datasets == "MNIST":
        dataset = torchvision.datasets.MNIST
        transform_train = Compose([
            ToTensor()
        ])
        trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
        transform_test = Compose([
            ToTensor()
        ])
        testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
        num_classes = 10
    elif datasets == "CIFAR-10":
        dataset = torchvision.datasets.CIFAR10
        transform_train = Compose([
            ToTensor(),
        ])
        trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
        transform_test = Compose([
            ToTensor(),
        ])
        testset = dataset(datasets_root_dir, train=True, transform=transform_test, download=True)
        num_classes = 10

    elif datasets == "CIFAR-100":
        dataset = torchvision.datasets.CIFAR100
        transform_train = Compose([
            ToTensor(),
        ])
        transform_test = transforms.Compose([
            ToTensor(),
        ])
        trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
        testset = dataset(datasets_root_dir, train=True, transform=transform_test, download=True)
        num_classes = 100
    elif datasets == "ImageNet":
        pass
    return trainset, testset, num_classes

if __name__ == "__main__":
    """
    task = {
        'train_dataset': trainset,
        'test_dataset' : testset,
        'model' :  ResNet(18),
        'optimizer': optimizer,
        'loss' : nn.CrossEntropyLoss()
    }
    schedule = {
        'experiment': experiment,
        # Settings for reproducible/repeatable experiments
        'seed': global_seed,
        'deterministic': deterministic,
        # Settings related to device
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        # Settings related to tarining 
        'pretrain': None,
        'epochs': 200,
        'batch_size': 256,
        'num_workers': 2,
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        # Settings aving model,data and logs
        'work_dir': work_dir,
        'log_iteration_interval': 100,
    }
    """
    task = 'ResNet-18_CIFAR-10'
    task_config = get_task_config(task)
    task_schedule = get_task_schedule(task = task)
    print(config)
    print(task_schedule)
    print(type(task_schedule['GPU_num']))



    





  






    

