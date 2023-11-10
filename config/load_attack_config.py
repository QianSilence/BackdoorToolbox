import yaml
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.transforms import Compose, RandomCrop,RandomHorizontalFlip, ToTensor, ToPILImage, Resize,Normalize
import os
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from models import BaselineMNISTNetwork,ResNet
import torch
from config.load_config import load_config

config, inner_dir, config_name = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)),"attack_config.yaml"))

support_attack_strategies = ["BadNets"]

def get_BadNets_config(W=28,H=28):
    attack_config = {
        'attack_strategy':'BadNets',
        'y_target':None,
        'poisoning_rate':None,
        'pattern': None,
        'weight':None,
        'poisoned_transform_index': None,
        'train_schedule':None
    }
    
    attack_config['y_target'] = config['BadNets']['y_target']
    attack_config['poisoning_rate'] = config['BadNets']['poisoning_rate']

    pattern = torch.zeros((W, H), dtype=torch.uint8)
    pattern[-3:, -3:] = 255
    weight = torch.zeros((W, H), dtype=torch.float32)
    weight[-3:, -3:] = 1.0
    attack_config['pattern'] = pattern
    attack_config['weight'] = weight

    attack_config['poisoned_transform_index'] = config['BadNets']['poisoned_transform_index']
    return attack_config

def get_attack_config(attack_strategy = None, dataset = None):
    assert attack_strategy in support_attack_strategies, f"{attack_strategy} is not in support_datasets:{support_attack_strategies}"
    if attack_strategy == "BadNets":
        if dataset == "MNIST":
            attack_config = get_BadNets_config(W=28,H=28)
        elif dataset == "CIFAR-10":
            attack_config = get_BadNets_config(W=32,H=32)

    return attack_config
if __name__ == "__main__":
    """
    # Parameters are needed according to attack strategy
    attack_schedule = { 
        'experiment': experiment,
        'attack_strategy': attack,
        # attack config
        'y_target': 1,
        'poisoning_rate': 0.10,
        'pattern': pattern,
        'weight': weight,
        'poisoned_transform_index': 0,
        'train_schedule':schedule
    }
    """
    print(f"config:{config} inner_dir:{inner_dir}, config_name:{config_name}")
    attack_strategy = "BadNets"
    attack_config = get_attack_config(attack_strategy, dataset="MNIST")
    print(config)
    print(attack_config)
