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
from torchvision import transforms 
from .Attack import Attack
from ..base.Base import *
from utils import save_img
import math
support_list = (
    DatasetFolder,
    MNIST,
    CIFAR10
)
#统一的trigger的添加逻辑（√)
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
    image and pattern must has same type(torch.tensor), shape(C,W,H) and data type(float:0-1)
    # input(torch.tensor): image; its type is VisionDataset
    # output(torch.tensor): poisioned image; It has the same shape as the input. 
    Args:
        image(torch.Tensor): shape (3, 32, 32) or (32, 32)
        pattern (None | torch.Tensor): shape (3, 32, 32) or (32, 32).
        mask (None | torch.Tensor): shape (3, 32, 32) or (32, 32).
        
    """
    def __init__(self, pattern, mask, alpha):
        assert pattern is not None, "pattern is None, its shape must be (C, H, w) or (H, W)"
        assert mask is not None, "mask is None, its shape must be  (C, W, H) or(W, H)"
        assert alpha is not None, "alpha is None"
        self.pattern = pattern
        self.mask = mask
        if self.pattern.dim() == 3:
            self.mask = self.mask.unsqueeze(0)
        self.alpha = alpha
        
        # Accelerated calculation
    def add_trigger(self, img):
        """Add watermarked trigger to image.
        intput(img, pattern) and output(Poisoned image) must be the same shape. 
        Args:
            img (torch.Tensor): shape (C, H, W).
            mask (torch.Tensor): shape (1, 28, 28) or (28, 28).
            pattern(torch.Tensor): shape (C, H, W).

        Returns:
            Poisoned image(torch.Tensor):shape (C, H, W).
        """
        # img = (img + self.alpha * self.mask * (float_pattern - img)).type(torch.uint8)
        img = img + self.alpha * self.mask * (self.pattern - img)
        return img


    def __call__(self, img):
        # img = F.pil_to_tensor(img)
        img = self.add_trigger(img)
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
                 pattern,
                 mask,
                 poisoned_transform_index,
                 poisoned_target_transform_index):
        super(PoisonedDatasetFolder, self).__init__(
            benign_dataset.root,
            benign_dataset.loader,
            benign_dataset.extensions,
            benign_dataset.transform,
            benign_dataset.target_transform,
            None)
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoning_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poison_indices = sorted(list(tmp_list[:poisoned_num]))

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddDatasetFolderTrigger(pattern, mask))

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
        path, target = self.samples[index]
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
        train_alpha(float):The alpha of the poisoning sample's trigger in test datasets.
        test_alpha(float):The alpha of the poisoning sample's trigger in train datasets.
        cover_rate(float):The proportion of regularization samples in the dataset.
        poisoned_transform_index(int):the index of function which can transform samples into poisoning samples
        in self.poisoned_transform.transforms(list) 
        poisoned_target_transform_index(int):the index of function in which can transform label into target label
        self.poisoned_target_transform(list) 

    Attributes:
    self.poison_indices(list):The index of the poisoning samples in the dataset.
    self.cover_indices(list):The index of the regularization samples in the clean dataset.
    self.poisoned_transform(Optional[Callable]):Compose which contains operation which can transform 
    samples into poisoning samples. 
    self.poisoned_target_transform(Optional[Callable]):Compose which contains operation which can transform 
    label into poisoning label. 

    """

    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoning_rate,
                 pattern_path,
                 pieces, mask_rate, 
                 alpha,
                 cover_rate,
                 poisoned_transform_index,
                 poisoned_target_transform_index):
        super(PoisonedVisionDataset, self).__init__(benign_dataset.root,transform = benign_dataset.transform,\
                                                    target_transform = benign_dataset.target_transform)
        
        self.dataset = benign_dataset
        self.y_target = y_target
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoning_rate)
        cover_num = int(total_num * cover_rate)

        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poison_indices = sorted(list(tmp_list[:poisoned_num]))
        if cover_num > 0:
            self.cover_indices = list(tmp_list[poisoned_num:poisoned_num + cover_num])
        else:
            self.cover_indices =list()
        self.alpha = alpha
        # Read image by the method "PIL.Image.open()"  and convert it into torch.Tensor.
        # It must be ensured that the value of each item of the ndarray "pattern" is a float which is in [0-1].
        trigger = Image.open(pattern_path).convert("RGB")
        pattern = F.pil_to_tensor(trigger)
        self.pattern =  pattern.float() / 255.0
        # print(self.pattern)
        size = trigger.size
        image_size = size[0]
        self.mask = self.get_trigger_mask(image_size, pieces, int(pieces*mask_rate))
   
        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
            self.cover_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
            self.cover_transform = copy.deepcopy(self.transform)
        
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddVisionDatasetTrigger(self.pattern, self.mask, self.alpha))
        self.cover_transform.transforms.insert(poisoned_transform_index, AddVisionDatasetTrigger(self.pattern, self.mask, self.alpha))
        # print(type(self.transform))
        # print(type(self.target_transform))
        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):

        img, target = self.dataset[index]
       
        # For transform, input is always PIL.Image, doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img.numpy(), mode='RGB')
        img = transforms.ToPILImage()(img)
        if index in self.poison_indices:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        elif index in self.cover_indices:
            img = self.poisoned_transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
        return img, target
    
    def get_trigger_mask(self, img_size, total_pieces, masked_pieces):
        '''
        Return mask(torch.tensor), which shape is (img_size,img_size) and the each item of mask is 0 or 1.
        mask is split into total_pieces in which randomly select masked_pieces and set 0.
        '''
        div_num = int(math.sqrt(total_pieces))
        step = int(img_size // div_num)
        candidate_idx = random.sample(list(range(total_pieces)), k=masked_pieces)
        mask = torch.zeros((img_size, img_size))
        for i in candidate_idx:
            x = int(i // div_num)  # column
            y = int(i % div_num)  # row
            mask[x * step: (x + 1) * step, y * step: (y + 1) * step] = 1
        return mask
    def get_poison_indices(self):
        return self.poison_indices
    def get_cover_indices(self):
        return self.cover_indices
    


class AdaptiveBlend(Base, Attack):
    """
    According to the specific attack strategy, override the create_poisoned_dataset() function of the parent class 
    to realize the algorithmic logic of generating the poisoned dataset.

    Args:
        task(dict):The attack strategy is used for the task, including datasets, model, Optimizer algorithm 
            and loss function.
        attack_schedule(dict): Parameters are needed according to attack strategy and model training
        
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
    

    def create_poisoned_dataset(self,dataset, alpha = 1.0):

        benign_dataset = dataset
        assert 'y_target' in self.attack_schedule, "Attack_config must contain 'y_target' configuration! "
        y_target = self.attack_schedule['y_target']
        assert 'poisoning_rate' in self.attack_schedule, "Attack_config must contain 'poisoning_rate' configuration! "
        poisoning_rate = self.attack_schedule['poisoning_rate']
        assert 'pattern' in self.attack_schedule, "Attack_config must contain 'pattern' configuration! "
        pattern = self.attack_schedule['pattern']

        assert 'pieces' in self.attack_schedule, "Attack_config must contain 'pieces' configuration! "
        pieces = self.attack_schedule['pieces']

        assert 'train_mask_rate' in self.attack_schedule, "Attack_config must contain 'train_mask_rate' configuration! "
        train_mask_rate = self.attack_schedule['train_mask_rate']

        assert 'train_alpha' in self.attack_schedule, "Attack_config must contain 'train_alpha' configuration! "
        train_alpha = self.attack_schedule['train_alpha']
        
        assert 'cover_rate' in self.attack_schedule, "Attack_config must contain 'cover_rate' configuration! "
        cover_rate = self.attack_schedule['cover_rate']
        
        if 'test_mask_rate' in self.attack_schedule:
            test_mask_rate = self.attack_schedule['test_mask_rate']
        else:
            test_mask_rate = 1.0 
        
        assert 'test_alpha' in self.attack_schedule, "Attack_config must contain 'test_alpha' configuration! "
        test_alpha = self.attack_schedule['test_alpha']

 
        # Set different attack config for the train and test dataset, so the object of "dataset" must has a attribute "train" which represents 
        # it will be used for training or test.
        if dataset.train:
            alpha = train_alpha
            mask_rate = train_mask_rate
            cover_rate = cover_rate
        else:
            alpha = test_alpha
            mask_rate = test_mask_rate
            cover_rate = 0

        assert 'poisoned_transform_index' in self.attack_schedule, "Attack_config must contain 'poisoned_transform_index' configuration! "
        poisoned_transform_index = self.attack_schedule['poisoned_transform_index']
        assert 'poisoned_target_transform_index' in self.attack_schedule, "Attack_config must contain 'poisoned_target_transform_index' configuration! "
        poisoned_target_transform_index = self.attack_schedule['poisoned_target_transform_index']
        
        dataset_type = type(benign_dataset)
        assert dataset_type in support_list, 'train_dataset is an unsupported dataset type, train_dataset should be a subclass of our support list.'
        work_dir = self.attack_schedule['work_dir']
        
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))
        msg = "\n\n\n==========Start creating poisoned_dataset==========\n"
        log(msg)
        msg = f"Total samples: {len(benign_dataset)},Among the poisoned samples:{int(len(benign_dataset) * poisoning_rate)}\n"
        log(msg)
        if isinstance(benign_dataset, DatasetFolder):
            return PoisonedDatasetFolder(benign_dataset, y_target, poisoning_rate, pattern, pieces, mask_rate, alpha, cover_rate,\
                                         poisoned_transform_index, poisoned_target_transform_index)
        elif isinstance(benign_dataset, VisionDataset):
            return PoisonedVisionDataset(benign_dataset, y_target, poisoning_rate, pattern, pieces, mask_rate, alpha, cover_rate,\
                                         poisoned_transform_index, poisoned_target_transform_index)
        else:
            log("Dataset must be the instance of 'DatasetFolder' or  'VisionDataset'")
        
    def interact_in_training():
        pass


