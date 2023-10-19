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
from torchvision.transforms import Compose

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
class AddTrigger:
    def __init__(self):
        pass

    def add_trigger(self, img):
        """Add watermarked trigger to image.

        Args:
            img (torch.Tensor): shape (C, H, W).

        Returns:
            torch.Tensor: Poisoned image, shape (C, H, W).
        """

        return (self.mask * img + self.res).type(torch.uint8)


class AddDatasetFolderTrigger(AddTrigger):
    """Add watermarked trigger to DatasetFolder images.

    Args:
        pattern (torch.Tensor): shape (C, H, W) or (H, W).
        mask (torch.Tensor): shape (C, H, W) or (H, W).
    """

    def __init__(self, pattern, mask):
        super(AddDatasetFolderTrigger, self).__init__()

        if pattern is None:
            raise ValueError("Pattern can not be None.")
        else:
            self.pattern = pattern
            if self.pattern.dim() == 2:
                self.pattern = self.pattern.unsqueeze(0)

        if mask is None:
            raise ValueError("Weight can not be None.")
        else:
            self.mask = mask
            if self.mask.dim() == 2:
                self.mask = self.mask.unsqueeze(0)

        # Accelerated calculation
        self.res = self.mask * self.pattern
        self.mask = 1.0 - self.mask

    def __call__(self, img):
        """Get the poisoned image.

        Args:
            img (PIL.Image.Image | numpy.ndarray | torch.Tensor): If img is numpy.ndarray or torch.Tensor, the shape should be (H, W, C) or (H, W).

        Returns:
            torch.Tensor: The poisoned image.
        """

        def add_trigger(img):
            if img.dim() == 2:
                img = img.unsqueeze(0)
                img = self.add_trigger(img)
                img = img.squeeze()
            else:
                img = self.add_trigger(img)
            return img

        if type(img) == PIL.Image.Image:
            img = F.pil_to_tensor(img)
            img = add_trigger(img)
            # 1 x H x W
            if img.size(0) == 1:
                img = Image.fromarray(img.squeeze().numpy(), mode='L')
            # 3 x H x W
            elif img.size(0) == 3:
                img = Image.fromarray(img.permute(1, 2, 0).numpy())
            else:
                raise ValueError("Unsupportable image shape.")
            return img
        elif type(img) == np.ndarray:
            # H x W
            if len(img.shape) == 2:
                img = torch.from_numpy(img)
                img = add_trigger(img)
                img = img.numpy()
            # H x W x C
            else:
                img = torch.from_numpy(img).permute(2, 0, 1)
                img = add_trigger(img)
                img = img.permute(1, 2, 0).numpy()
            return img
        elif type(img) == torch.Tensor:
            # H x W
            if img.dim() == 2:
                img = add_trigger(img)
            # H x W x C
            else:
                img = img.permute(2, 0, 1)
                img = add_trigger(img)
                img = img.permute(1, 2, 0)
            return img
        else:
            raise TypeError('img should be PIL.Image.Image or numpy.ndarray or torch.Tensor. Got {}'.format(type(img)))


class AddVisionDatasetTrigger():
    """
    Add watermarked trigger to VisionDataset.
    Args:
        pattern (None | torch.Tensor): shape (1, 28, 28) or (28, 28).
        mask (None | torch.Tensor): shape (1, 28, 28) or (28, 28).
    """
    def __init__(self, pattern, mask, alpha):
        assert pattern is not None, "pattern is None, its shape must be (1, H, w) or (H, W)"
        assert mask is not None, "mask is None, its shape must be  (1, W, H) or(W, H)"
        assert alpha is not None, "alpha is None"
        self.pattern = pattern
        if self.pattern.dim() == 2:
            self.pattern = self.pattern.unsqueeze(0)
        self.mask = mask
        self.alpha = alpha

        if self.mask.dim() == 2:
            self.mask = self.mask.unsqueeze(0)
        # Accelerated calculation
        self.trigger = self.mask * self.pattern
    def add_trigger(self, img):
        """Add watermarked trigger to image.

        Args:
            img (torch.Tensor): shape (C, H, W).

        Returns:
            torch.Tensor: Poisoned image, shape (C, H, W).
        """
        # img = (1 - self.mask) * img + (1- self.alpha) * self.mask * img + self.alpha * self.mask * self.pattern
        #     = img - self.alpha * self.mask * img + self.alpha * self.mask * self.pattern
        #     = img +  self.alpha * self.mask * (self.pattern - img)
        img = img + self.alpha * self.mask * (self.pattern - img)
        return  img
        


    #输入和输出：img 数据类型，形状
    def __call__(self, img):
        img = F.pil_to_tensor(img)
        img = self.add_trigger(img)
        img = img.squeeze()
        img = Image.fromarray(img.numpy(), mode='L')
        return img
    

class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, y_target):
        return self.y_target
    
class PoisonedDatasetFolder(DatasetFolder): 
    """
    A generic poisoning data loader inherting torchvision.datasets.DatasetFolder, like "torchvision.datasets.ImageFolder".
    Its main logic is almost as same as  "PoisonedMNIST" except that "benign_dataset" must define some attributes such as
    "loader" and "extensions" etc. The definition of loder can refer to the implementation of torchvision.datasets.ImageFolder.

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
    Attributes:
        poison_indices(frozenset):Save the index of the poisoning sample in the dataset.
        poisoned_transform(Optional[Callable]):Compose which contains operation which can transform 
        samples into poisoning samples. 
        poisoned_target_transform(Optional[Callable]):Compose which contains operation which can transform 
        label into poisoning label. 

    """

    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoning_rate,
                 patterns, masks,
                 alphas,
                 combined, cover_rate,
                 poisoned_transform_index,
                 poisoned_target_transform_index):
        super(PoisonedDatasetFolder, self).__init__(
            benign_dataset.root,
            benign_dataset.loader,
            benign_dataset.extensions,
            benign_dataset.transform,
            benign_dataset.target_transform,
            None)
        
        self.dataset = benign_dataset
        self.y_target =  y_target
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoning_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poison_indices = list(tmp_list[0:poisoned_num])

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddDatasetFolderTrigger(patterns, masks,alphas))

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.dataset[index]
        sample = self.loader(path)

        if index in self.poison_indices:
            sample = self.poisoned_transform(sample)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
        return torch.tensor(sample), torch.tensor(target)
    
    def get_poison_indices(self):
        return self.poison_indices
    
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
                 num_compose,cover_rate):
        super(PoisonedVisionDataset, self).__init__(benign_dataset.root,transform = benign_dataset.transform,\
                                                    target_transform = benign_dataset.target_transform)
        
        self.dataset = benign_dataset
        self.y_target = y_target
        self.compose = compose

        # load patterns and transform torch.tensor,the Normalize the value of pattern.
        self.trigger_class_num = len(patterns)
        self.trigger_dir = trigger_dir
        self.patterns = [F.pil_to_tensor(Image.open(trigger_dir+pattern).convert("RGB")).float() / 255.0  for pattern in patterns]
        self.masks = [self.get_trigger_mask(self.patterns[i], trigger_dir+masks[i]) for i in range(len(patterns))]
        self.alphas = alphas
        self.compose = compose
        self.num_compose = num_compose
       

        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoning_rate)
        cover_num = int(total_num * cover_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)

        self.poison_indices = sorted(list(tmp_list[:poisoned_num]))
        self.cover_indices = sorted(list(tmp_list[poisoned_num:poisoned_num + cover_num]))



    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
       
        img, target = self.dataset[index]
        # img = torch.ToTensor(img)
        '''
        Add watermarked trigger to VisionDataset. 
        the objects  'img' and 'pattern' must has same type(torch.tensor), shape(C,W,H) and data type(float:0-1)
        '''
        num_pattern= len(self.patterns)
        if index in self.poison_indices:
            if self.compose:
                indexes = random.choices(range(len(self.patterns)), k=self.num_compose)
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
    def modify_target(self,target):
        return self.y_target
    
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
  
    def create_poisoned_dataset(self,dataset):

        benign_dataset = dataset
        
    
        assert 'y_target' in self.attack_schedule, "Attack_config must contain 'y_target' configuration! "
        y_target = self.attack_schedule['y_target']
        assert 'poisoning_rate' in self.attack_schedule, "Attack_config must contain 'poisoning_rate' configuration! "
        poisoning_rate = self.attack_schedule['poisoning_rate']
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
        assert 'num_compose' in self.attack_schedule, "Attack_config must contain 'num_compose' configuration! "
        num_compose = self.attack_schedule['num_compose']
    
        assert 'cover_rate' in self.attack_schedule, "Attack_config must contain 'cover_rate' configuration! "
        cover_rate = self.attack_schedule['cover_rate']
    
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

        if isinstance(benign_dataset,DatasetFolder):
            return PoisonedDatasetFolder(benign_dataset, y_target, poisoning_rate, trigger_dir, patterns, masks, alphas, compose, num_compose,\
                                          cover_rate)
        elif isinstance(benign_dataset,VisionDataset): 
            return PoisonedVisionDataset(benign_dataset, y_target, poisoning_rate, trigger_dir, patterns, masks, alphas, compose,num_compose, \
                                         cover_rate)
        else:
            log("Dataset must be the instance of 'DatasetFolder' or  'VisionDataset'")
        
    def interact_in_training():
        pass


