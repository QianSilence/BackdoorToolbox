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
from core.defenses import Spectral
import torch
from utils import parser
from .load_config import load_config

config, inner_dir, config_name = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)),"defense_config.yaml"))

support_defense_strategies = ["Spectral", "Spectre", "Mine", "MIMRL"]

def get_filter_object(filter_strategy="Spectral",schedule=None):
    if filter_strategy == "Spectral":
        filter_object = Spectral(task=None, defense_schedule=schedule)
    return filter_object

def get_Spectral_config(default=False):
    defense_config = {
        'defense_strategy': 'Spectral',
        # The saving path of the backdoor model. When train = True, it is the path to save 
        # the trained model. When train = False, it is the path to load the model.
        'filter':{
            'train':False,
            'layer':None,
            'y_target':None,
            'percentile':None,
            'latents_path': None,
            'device': 'cuda:0'
        },
        'repair':{'filter': True},
        'schedule':None
    }
    if default is False:
        defense_config['defense_strategy'] = config['Spectral']['defense_strategy']

        defense_config['filter']['train'] = config['Spectral']['filter']['train']
        defense_config['filter']['layer'] = config['Spectral']['filter']['layer']
        defense_config['filter']['y_target'] = config['Spectral']['filter']['y_target']
        defense_config['filter']['percentile'] = config['Spectral']['filter']['percentile']
        defense_config['filter']['latents_path'] = config['Spectral']['filter']['latents_path']
        defense_config['filter']['device'] = config['Spectral']['filter']['device']

        defense_config['repair']['filter'] = config['Spectral']['repair']['filter']
        defense_config['device'] = config['Spectral']['device']
        defense_config['schedule'] = config['Spectral']['schedule']

    return defense_config

def get_Spectre_config(default=False):
    defense_config = {
        'defense_strategy': 'Spectre',
        # The saving path of the backdoor model. When train = True, it is the path to save 
        # the trained model. When train = False, it is the path to load the model.
        'filter':{
            'train':False,
            'layer':None,
            'y_target':None,
            'n_dim':None,
            'alpha':None,
            'varepsilon':None,
            'latents_path': None,
            'device': 'cuda:0'
        },
        'repair':{'filter': True},
        'schedule':None
    }
    if default is False:
        defense_config['defense_strategy'] = config['Spectre']['defense_strategy']

        defense_config['filter']['train'] = config['Spectre']['filter']['train']
        defense_config['filter']['layer'] = config['Spectre']['filter']['layer']
        defense_config['filter']['y_target'] = config['Spectre']['filter']['y_target']
        defense_config['filter']['n_dim'] = config['Spectre']['filter']['n_dim']
        defense_config['filter']['alpha'] = config['Spectre']['filter']['alpha']
        defense_config['filter']['varepsilon'] = config['Spectre']['filter']['varepsilon']
        defense_config['filter']['latents_path'] = config['Spectre']['filter']['latents_path']
        defense_config['filter']['device'] = config['Spectre']['filter']['device']

        defense_config['repair']['filter'] = config['Spectre']['repair']['filter']
        defense_config['device'] = config['Spectre']['device']
        defense_config['schedule'] = config['Spectre']['schedule']

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

def get_MIMRL_config():
    defense_config = {
        # defense config:
        'defense_strategy':"MIMRL",
        'loss':'MIMRLLoss',
        'alpha':0.01,
        'beta': 0.2,
        'x_dim': 784,
        'z_dim': 10,
        'n_classes':10,
        # related Infor-max   
        'lr_dis': 0.00001,
        'layer': None,
        'schedule':None,
        #related filtering config
        'filter_object':None,
        'filter_config':None
        
    }
   
    defense_config['defense_strategy'] = config['MIMRL']['defense_strategy']
    defense_config['loss'] = config['MIMRL']['loss']
    defense_config['alpha'] = config['MIMRL']['alpha']
    defense_config['beta'] = config['MIMRL']['beta']
    defense_config['x_dim'] = config['MIMRL']['x_dim']
    defense_config["z_dim"] = config['MIMRL']["z_dim"]
    defense_config["n_classes"] = config['MIMRL']["n_classes"]
    defense_config['lr_dis'] = config['MIMRL']['lr_dis']
    defense_config['layer'] = config['MIMRL']['layer']
    
    filter_config = get_Spectral_config(default=True)
    filter_config["filter"]["y_target"] = config['MIMRL']["filter"]["y_target"]
    filter_config["filter"]["percentile"] = config['MIMRL']["filter"]["percentile"]
    filter_config["filter"]["device"] = config['MIMRL']["filter"]["device"]

    defense_config['filter_strategy'] = config['MIMRL']['filter_strategy']
    defense_config['filter_object'] = get_filter_object(filter_strategy=config['MIMRL']['filter_strategy'])
    defense_config['filter_config'] = filter_config

    return defense_config
    
    
def get_defense_config(defense_strategy = None):
    assert defense_strategy in support_defense_strategies, f"{defense_strategy} is not in support_datasets:{support_defense_strategies}"
    if defense_strategy == "Spectral":
        defense_config = get_Spectral_config()
    elif defense_strategy == "Spectre":
        defense_config = get_Spectre_config()
    elif defense_strategy == "Mine":
        defense_config = get_Mine_config()
    elif defense_strategy == "MIMRL":
        defense_config = get_MIMRL_config()

    return defense_config

if __name__ == "__main__":
    print(f"inner_dir:{inner_dir}, config_name:{config_name}\n")
    defense_strategy = "Spectral"
    defense_config = get_defense_config(defense_strategy)
    print(config)
    print(defense_config)
