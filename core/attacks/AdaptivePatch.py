# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/08/21 10:24:42
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : AdaptiveBlend.py
# @Description :This is the implement of Adaptive-Blend [1].
#               Reference:[1] Qi X, Xie T, Li Y, et al. Revisiting the assumption of latent separability for backdoor defenses[C]// \
#                             The eleventh international conference on learning representations. 2023.

import copy
import random
import numpy as np
import PIL
from PIL import Image
import torchvision
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import functional as F
from torchvision.transforms import Compose, ToTensor
from ..base.Base import *
from torchvision.datasets import DatasetFolder, MNIST, CIFAR10
from .Attack import Attack
from ..base.Base import *
import math
support_list = (
    DatasetFolder,
    MNIST,
    CIFAR10
)

class PoisonedVisionDataset(VisionDataset):
    """
    Poisoned VisionDataset : inherit torchvision.datasets.vision.VisionDataset and add the trigger generation logic 
    to the transform logic of the sample in function:__getitem__().   
    Args:
        pattern (None | torch.Tensor): shape (1, W, H) or (W, H).
        pieces (int): the number of pieces which the full trigger image is divided into
        mask_rate(float): the proportion of peices which would be selected as trigger in the full trigger iamge.
        mask (None | torch.Tensor): shape (1, W, H) or (W, H).
        train_alpha(float):The opacity of the poisoning sample's trigger in test datasets.
        test_alpha(float):The opacity of the poisoning sample's trigger in train datasets.
        cover_rate(float):The proportion of regularization samples in the dataset.
        poisoned_transform_index(int):the index of function which can transform samples into poisoning samples
        in self.poisoned_transform.transforms(list) 
        poisoned_target_transform_index(int):the index of function in which can transform label into target label
        self.poisoned_target_transform(list) 

    Attributes:
    self.poison_indices(frozenset):The index of the poisoning samples in the dataset.
    self.cover_indices(frozenset):The index of the regularization samples in the clean dataset.
    self.poisoned_transform(Optional[Callable]):Compose which contains operation which can transform 
    samples into poisoning samples. 
    self.poisoned_target_transform(Optional[Callable]):Compose which contains operation which can transform 
    label into poisoning label. 

    """
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoning_rate,
                 trigger_dir, patterns, masks,
                 alphas, compose,
                 test_sample_trigger_num,cover_rate):
        super(PoisonedVisionDataset, self).__init__(benign_dataset.root, transform = benign_dataset.transform,
            target_transform = benign_dataset.target_transform)
        
        # self.data = benign_dataset.data
        # self.targets = benign_dataset.targets
        if isinstance(benign_dataset.data,torch.Tensor):
            self.data = deepcopy(benign_dataset.data.numpy()) 
        elif isinstance(benign_dataset.data,list):
            self.data = deepcopy(np.array(benign_dataset.data))
        else:
            self.data = deepcopy(benign_dataset.data)

        if isinstance(benign_dataset.targets,torch.Tensor):
            self.targets = deepcopy(benign_dataset.targets.numpy())
        elif isinstance(benign_dataset.targets,list):
            self.targets = deepcopy(np.array(benign_dataset.targets))
        else:
            self.targets = deepcopy(benign_dataset.targets)
        
        self.classes = deepcopy(benign_dataset.classes)
        self.y_target = y_target
        self.compose = compose

        # load patterns and transform torch.tensor,the Normalize the value of pattern.
        self.trigger_class_num = len(patterns)
        self.trigger_dir = trigger_dir
        self.patterns = [F.pil_to_tensor(Image.open(trigger_dir+pattern).convert("RGB")).float() / 255.0  for pattern in patterns]
        self.masks = [self.get_trigger_mask(self.patterns[i], trigger_dir+masks[i]) for i in range(len(patterns))]

        self.poisoning_rate = poisoning_rate
        self.cover_rate = cover_rate
        self.alphas = alphas
        self.compose = compose
        self.test_sample_trigger_num = test_sample_trigger_num

        self.poison_indices = None
        self.cover_indices = None
        self.modified_targets = None
        self.set_poisoned_subdatasets(y_target=self.y_target, poisoning_rate=self.poisoning_rate, cover_rate=self.cover_rate )
       
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img, target = self.get_sample_by_index(index)
        if img.size(0)== 1:
            img = Image.fromarray(img.squeeze().numpy(), mode='L')
        # 3 x H x W
        elif img.size(0) == 3:
            img = img.permute(1, 2, 0).numpy()
            img = Image.fromarray((img * 255).astype('uint8'))
      
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    
    def get_sample_by_index(self,index):
        img, target = self.data[index], int(self.modified_targets[index])
        img = ToTensor()(img)
        '''
        Add watermarked trigger to VisionDataset. 
        the objects  'img' and 'pattern' must has same type(torch.tensor), shape(C,W,H) and data type(float:0-1)
        '''
        num_pattern= len(self.patterns)
        if index in self.poison_indices:
            if self.compose:
                indexes = random.choices(range(len(self.patterns)), k=self.test_sample_trigger_num)
                for j in indexes:
                    alpha = self.alphas[j]
                    mask = self.masks[j]
                    pattern = self.patterns[j]
                    img = img + alpha * mask * (pattern - img)
            else:
                k = len(self.patterns)
                j = (self.poison_indices.index(index)) % num_pattern 
                alpha = self.alphas[j]
                mask = self.masks[j]
                pattern = self.patterns[j]
                img = img + alpha * mask * (pattern - img)
            target = self.y_target
        elif index in self.cover_indices:
            j = self.cover_indices.index(index) % num_pattern 
            alpha = self.alphas[j]
            mask = self.masks[j]
            pattern = self.patterns[j]
            img = img + alpha * mask * (pattern - img)
        return img, target
    
    def get_trigger_mask(self, trigger, mask_path = None):
        '''
        Return the mask corresponding to trigger, if mask_path is None, then compute mask according to 'trigger', 
        otherwise, load directly mask.     
        '''
        if mask_path is not None and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("RGB")
            mask = transforms.ToTensor()(mask)[0]  # only use 1 channel
        else: # by default, all black pixels are masked with 0's
            # print("trigger:" + str(type(trigger)) + " " + str(trigger.shape))
            # print(trigger) 
            # print(torch.logical_or(trigger[0] > 0, trigger[1] > 0))
            mask = torch.logical_or(torch.logical_or(trigger[0] > 0, trigger[1] > 0), trigger[2] > 0)
        return mask
    
    def add_trigger(self, img, alpha, mask, pattern):
        """Add watermarked trigger to image.
        Args:
            img (torch.Tensor): shape (C, H, W).

        Returns:
            torch.Tensor: Poisoned image, shape (C, H, W).
        """
        img = img + alpha * mask * (pattern - img)
        return  img
    
    def set_poisoned_subdatasets(self, y_target=None, poisoning_rate=None, cover_rate=None):
        total_num = len(self.data)
        poisoned_num = int(total_num * poisoning_rate)
        cover_num = int(total_num * cover_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = np.arange(len(self.data))[~np.array(self.targets == y_target)]
        random.shuffle(tmp_list)

        self.poison_indices = sorted(list(tmp_list[:poisoned_num]))
        self.cover_indices = sorted(list(tmp_list[poisoned_num:poisoned_num + cover_num]))
        self.modified_targets = np.array(deepcopy(self.targets))
        self.modified_targets[self.poison_indices] = y_target  

    def set_trigger_compose(self, compose=True, trigger_num=2):
        self.compose = compose
        self.test_sample_trigger_num = trigger_num

    def get_classes(self):
        return self.classes
    def get_y_target(self):
        return self.y_target
    def get_poisoning_rate(self):
        return self.poisoning_rate
    
    def modify_targets(self, indces, labels):
        self.modified_targets[indces] = labels
    def get_real_targets(self):
        return self.targets
    def get_modified_targets(self):
        return self.modified_targets
    
    def get_poison_indices(self):
        return self.poison_indices
    def get_cover_indices(self):
        return self.cover_indices
    

class AdaptivePatch(Base, Attack):
    """
    According to the specific attack strategy, override the create_poisoned_dataset() function of the parent class 
    to realize the algorithmic logic of generating the poisoned dataset.

    Args:
        task(dict):The attack strategy is used for the task, including datasets, model, Optimizer algorithm 
            and loss function.
        attack_schedule(dict): Parameters are needed according to attack strategy
        schedule=None(dict): Config related to model training
 
    Attributes:
        self.attack_schedule(dict): Initialized by the incoming  parameter "attack_schedule".
        self.attack_strategy(string): The name of attack_strategy.
    """
    def __init__(self, task, attack_schedule):
        schedule = None
        if 'train_schedule' in attack_schedule:
            schedule = attack_schedule['train_schedule']
        Base.__init__(self, task, schedule = schedule)   
        Attack.__init__(self)
        self.attack_schedule = attack_schedule
        assert 'attack_strategy' in self.attack_schedule, "Attack_config must contain 'attack_strategy' configuration! "
        self.attack_strategy = attack_schedule['attack_strategy']
    
    def get_attack_strategy(self):
        return self.attack_strategy
  
    def create_poisoned_dataset(self, dataset, y_target=None, poisoning_rate=None):
        benign_dataset = dataset
        if y_target is not None:
            y_target = y_target
        else:
            assert 'y_target' in self.attack_schedule, "Attack_config must contain 'y_target' configuration! "
            y_target = self.attack_schedule['y_target']
        if poisoning_rate is not None:
            poisoning_rate = poisoning_rate
        else:
            assert 'poisoning_rate' in self.attack_schedule, "Attack_config must contain 'poisoning_rate' configuration! "
            poisoning_rate = self.attack_schedule['poisoning_rate']
        assert 'cover_rate' in self.attack_schedule, "Attack_config must contain 'cover_rate' configuration! "
        cover_rate = self.attack_schedule['cover_rate']
        assert 'trigger_dir' in self.attack_schedule, "Attack_config must contain 'trigger_dir' configuration! "
        trigger_dir = self.attack_schedule['trigger_dir']
        assert 'patterns' in self.attack_schedule, "Attack_config must contain 'patterns' configuration! "
        patterns = self.attack_schedule['patterns']
        assert 'masks' in self.attack_schedule, "Attack_config must contain 'patterns' configuration! "
        masks = self.attack_schedule['masks']
           
        assert 'train_alphas' in self.attack_schedule, "Attack_config must contain 'train_alphas' configuration! "
        train_alphas = self.attack_schedule['train_alphas']
        assert 'test_alphas' in self.attack_schedule, "Attack_config must contain 'test_alphas' configuration! "
        test_alphas = self.attack_schedule['test_alphas']
        assert 'test_sample_trigger_num' in self.attack_schedule, "Attack_config must contain 'test_sample_trigger_num' configuration! "
        test_sample_trigger_num = self.attack_schedule['test_sample_trigger_num']
        work_dir = self.attack_schedule['work_dir']

        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))
        msg = "\n\n\n==========Start creating poisoned_dataset==========\n"
        log(msg)
        msg = f"Total samples: {len(benign_dataset)},Among the poisoned samples:{int(len(benign_dataset) * poisoning_rate)}\n"
        log(msg)

        if benign_dataset.train:
            compose = False
            alphas = train_alphas 
        else: 
            compose = True
            alphas = test_alphas 

        if isinstance(benign_dataset,VisionDataset): 
            return PoisonedVisionDataset(benign_dataset, y_target, poisoning_rate, trigger_dir, patterns, masks, alphas, compose, test_sample_trigger_num, \
                             cover_rate)
        else:
            log("Dataset must be the instance of 'DatasetFolder' or  'VisionDataset'")



