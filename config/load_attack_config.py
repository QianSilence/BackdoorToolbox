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

support_attack_strategies = ["BadNets", "RegularBadNets","Adaptive-Patch","Adaptive-Blend"]

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

def get_RegularBadNets_config(W=28,H=28):
    attack_config = {
            'attack_strategy':'RegularBadNets',
            'y_target':None,
            'poisoning_rate':None,
            'cover_rate':None,
            'pattern': None,
            'weight':None,
            'poisoned_transform_index': None,
            'train_schedule':None
        }
        
    attack_config['y_target'] = config['RegularBadNets']['y_target']
    attack_config['poisoning_rate'] = config['RegularBadNets']['poisoning_rate']

    pattern = torch.zeros((W, H), dtype=torch.uint8)
    pattern[-3:, -3:] = 255
    weight = torch.zeros((W, H), dtype=torch.float32)
    weight[-3:, -3:] = 1.0
    attack_config['pattern'] = pattern
    attack_config['weight'] = weight
    attack_config['poisoned_transform_index'] = config['RegularBadNets']['poisoned_transform_index']
    attack_config['cover_rate'] = config['RegularBadNets']['cover_rate']

    return attack_config

def get_AdaptivePatch_config():
    attack_config ={
        'attack_strategy': 'Adaptive-Patch',
        # attack config
        'y_target': 0,
        'poisoning_rate': 0.05,
        # trigger and opacitys
        'trigger_dir': '/home/zzq/CreatingSpace/BackdoorToolbox/experiments/ResNet-18_CIFAR-10/Adaptive-Patch/datasets/triggers',
        'patterns': None,
        'masks': None,
        'train_alphas': None,
        'test_alphas': None,
        'test_sample_trigger_num': 2,
        # conservatism ratio
        'cover_rate' : 0.05,
        'poisoned_transform_index':0,
        'train_schedule':None,
        'work_dir': None
    }
    attack_config['y_target'] = config['Adaptive-Patch']['y_target']
    attack_config['poisoning_rate'] = config['Adaptive-Patch']['poisoning_rate']
    attack_config['trigger_dir'] = config['Adaptive-Patch']['trigger_dir']
    attack_config['patterns'] = config['Adaptive-Patch']['patterns']
    attack_config['masks'] = config['Adaptive-Patch']['masks']
    attack_config['train_alphas'] = [float(element) for element in config['Adaptive-Patch']['train_alphas']] 
    attack_config['test_alphas'] = [float(element) for element in config['Adaptive-Patch']['test_alphas']] 
    attack_config['test_sample_trigger_num'] = config['Adaptive-Patch']['test_sample_trigger_num']
    attack_config['cover_rate'] = config['Adaptive-Patch']['cover_rate']

    return attack_config

def get_AdaptiveBlend_config():

    attack_config ={
        'attack_strategy': 'Adaptive-Blend',
        # attack config
        'y_target': 0,
        'poisoning_rate': 0.05,
        "cover_rate" : 0.05,
        "pattern": 'trigger_path',
        "pieces": 16,
        # diverse asymmetry
        "train_mask_rate": 0.5,
        "test_mask_rate": 1.0,
        # asymmetric design:The trigger uses different transparency during the training phase and testing phase.
        "train_alpha": 0.15,
        "test_alpha": 0.2,
        'train_schedule':None,
        'work_dir': None
    }
    attack_config['y_target'] = config['Adaptive-Blend']['y_target']
    attack_config['poisoning_rate'] = config['Adaptive-Blend']['poisoning_rate']
    attack_config['cover_rate'] = config['Adaptive-Blend']['cover_rate']
    attack_config['pattern'] = config['Adaptive-Blend']['pattern']
    attack_config['pieces'] = config['Adaptive-Blend']['pieces']
    attack_config['train_mask_rate'] = config['Adaptive-Blend']['train_mask_rate']
    attack_config['test_mask_rate'] = config['Adaptive-Blend']['test_mask_rate'] 
    attack_config['train_alpha'] = config['Adaptive-Blend']['train_alpha']
    attack_config['test_alpha'] = config['Adaptive-Blend']['test_alpha']

    return attack_config


def get_attack_config(attack_strategy = None, dataset = None):
    assert attack_strategy in support_attack_strategies, f"{attack_strategy} is not in support_datasets:{support_attack_strategies}"
    if attack_strategy == "BadNets":
        if dataset == "MNIST":
            attack_config = get_BadNets_config(W=28,H=28)
        elif dataset == "CIFAR-10":
            attack_config = get_BadNets_config(W=32,H=32)
    elif attack_strategy == "RegularBadNets":
        if dataset == "MNIST":
            attack_config = get_RegularBadNets_config(W=28,H=28)
    elif attack_strategy == "Adaptive-Patch":
        attack_config = get_AdaptivePatch_config()
    elif attack_strategy == "Adaptive-Blend":
        attack_config = get_AdaptiveBlend_config()
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
