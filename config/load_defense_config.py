import yaml
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.transforms import Compose, RandomCrop,RandomHorizontalFlip, ToTensor, ToPILImage, Resize,Normalize
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
from models import BaselineMNISTNetwork,ResNet
import torch
from utils import parser
from .load_config import load_config

config, inner_dir, config_name = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)),"defense_config.yaml"))

support_defense_strategies = ["Spectral","Mine"]

def get_Spectral_config():
    defense_config = {
        'defense_strategy': 'Spectral',
        'train':None,
        'trained_model':None,
        'backdoor_model_path': None,
        'y_target':None,
        'poisoned_trainset': None,
        'poisoned_testset': None,
        'percentile': None,
    }
    defense_config['train'] = config['Spectral']['train']
    defense_config['trained_model'] = config['Spectral']['trained_model']
    defense_config['backdoor_model_path'] = config['Spectral']['backdoor_model_path']
    defense_config['y_target'] = config['Spectral']['y_target']
    defense_config['poisoned_trainset'] = config['Spectral']['poisoned_trainset']
    defense_config['poisoned_testset'] = config['Spectral']['poisoned_testset']
    defense_config['percentile'] = config['Spectral']['percentile']
    return defense_config

def get_Mine_config():
    defense_config = {
        # defense config:
        'defense_strategy':"Mine",
        'supervised_loss_type':None,
        'volume':None,
        "start_epoch":None,
        'beta':None,
        'delta_beta':None,
        'beta_threshold':None,
        'poison_rate':None,
        "filter_epoch_interation":None,
        #1.4172, 1.8931
        "threshold":None,
        "unlearning_threshold":None,
        # "times":2,
        "init_size_clean_data_pool":None,
        'layer':"layer"
    }
    defense_config['defense_strategy'] = config['Mine']['defense_strategy']
    defense_config['supervised_loss_type'] = config['Mine']['supervised_loss_type']
    defense_config['volume'] = config['Mine']['volume']
    defense_config["start_epoch"] = config['Mine']["start_epoch"]
    defense_config['beta'] = config['Mine']['beta']
    defense_config['delta_beta'] = config['Mine']['delta_beta']
    defense_config['beta_threshold'] = config['Mine']['beta_threshold']
    defense_config['poison_rate'] = config['Mine']['poison_rate']
    defense_config["filter_epoch_interation"] = config['Mine']["filter_epoch_interation"]
    defense_config['threshold'] = config['Mine']['threshold']
    defense_config['unlearning_threshold'] = config['Mine']['unlearning_threshold']
    defense_config['init_size_clean_data_pool'] = config['Mine']['init_size_clean_data_pool']
    defense_config['unlearning_threshold'] = config['Mine']['unlearning_threshold']
    defense_config['layer'] = config['Mine']['layer']

    return defense_config
    
def get_defense_config(defense_strategy = None):
    assert defense_strategy in support_defense_strategies, f"{defense_strategy} is not in support_datasets:{support_defense_strategies}"
    if defense_strategy == "Spectral":
        defense_config = get_Spectral_config()
    if defense_strategy == "Mine":
        defense_config = get_Mine_config()

    return defense_config

if __name__ == "__main__":
    print(f"inner_dir:{inner_dir}, config_name:{config_name}\n")
    defense_strategy = "Spectral"
    defense_config = get_defense_config(defense_strategy)
    print(config)
    print(defense_config)
