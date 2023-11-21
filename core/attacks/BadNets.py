# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/08/21 10:24:42
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : tmp.py
# @Description :This is the implement of BadNets [1].
#               Reference:[1] Badnets: Evaluating Backdooring Attacks on Deep Neural Networks. IEEE Access 2019.
import copy
import random
import numpy as np
import PIL
from PIL import Image
import torchvision
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import functional as F
from torchvision.transforms import Compose
from ..base.Base import *
from torchvision.datasets import DatasetFolder, MNIST, CIFAR10, CIFAR100
from .Attack import Attack
from ..base.Base import *
import torch
support_list = (
    DatasetFolder,
    MNIST,
    CIFAR10,
    CIFAR100
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

        return (self.weight * img + self.res).type(torch.uint8)


class AddDatasetFolderTrigger(AddTrigger):
    """Add watermarked trigger to DatasetFolder images.

    Args:
        pattern (torch.Tensor): shape (C, H, W) or (H, W).
        weight (torch.Tensor): shape (C, H, W) or (H, W).
    """

    def __init__(self, pattern, weight):
        super(AddDatasetFolderTrigger, self).__init__()

        if pattern is None:
            raise ValueError("Pattern can not be None.")
        else:
            self.pattern = pattern
            if self.pattern.dim() == 2:
                self.pattern = self.pattern.unsqueeze(0)

        if weight is None:
            raise ValueError("Weight can not be None.")
        else:
            self.weight = weight
            if self.weight.dim() == 2:
                self.weight = self.weight.unsqueeze(0)

        # Accelerated calculation
        self.res = self.weight * self.pattern
        self.weight = 1.0 - self.weight

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
        weight (None | torch.Tensor): shape (1, 28, 28) or (28, 28).
    """
    def __init__(self, pattern, weight):
        assert pattern is not None, "pattern is None, its shape must be (1, H, w) or (H, W)"
        assert weight is not None, "weight is None, its shape must be  (1, W, H) or(W, H)"
        self.pattern = pattern
        if self.pattern.dim() == 2:
            self.pattern = self.pattern.unsqueeze(0)

        self.weight = weight
        if self.weight.dim() == 2:
            self.weight = self.weight.unsqueeze(0)

        # Accelerated calculation
        self.res = self.weight * self.pattern
        self.weight = 1.0 - self.weight

    def add_trigger(self, img):
        """Add watermarked trigger to image.

        Args:
            img (torch.Tensor): shape (C, H, W).

        Returns:
            torch.Tensor: Poisoned image, shape (C, H, W).
        """
        return (self.weight * img + self.res).type(torch.uint8)

    def __call__(self, img):
        img = F.pil_to_tensor(img)
        img = self.add_trigger(img)
        img = img.squeeze()
        if img.dim() == 2:
            img = Image.fromarray(img.numpy(), mode='L')
        elif img.dim() == 3:
            img = Image.fromarray(img.permute(1, 2, 0).numpy())
        return img
    
    
class AddMNISTTrigger(AddTrigger):
    """
    Add watermarked trigger to MNIST image.
    Args:
        pattern (None | torch.Tensor): shape (1, 28, 28) or (28, 28).
        weight (None | torch.Tensor): shape (1, 28, 28) or (28, 28).
    """

    def __init__(self, pattern, weight):
        super(AddMNISTTrigger, self).__init__()

        if pattern is None:
            self.pattern = torch.zeros((1, 28, 28), dtype=torch.uint8)
            self.pattern[0, -2, -2] = 255
        else:
            self.pattern = pattern
            if self.pattern.dim() == 2:
                self.pattern = self.pattern.unsqueeze(0)

        if weight is None:
            self.weight = torch.zeros((1, 28, 28), dtype=torch.float32)
            self.weight[0, -2, -2] = 1.0
        else:
            self.weight = weight
            if self.weight.dim() == 2:
                self.weight = self.weight.unsqueeze(0)

        # Accelerated calculation
        self.res = self.weight * self.pattern
        self.weight = 1.0 - self.weight

    def __call__(self, img):
        img = F.pil_to_tensor(img)
        img = self.add_trigger(img)
        img = img.squeeze()
        img = Image.fromarray(img.numpy(), mode='L')
        return img


class AddCIFAR10Trigger(AddTrigger):
    """Add watermarked trigger to CIFAR10 image.

    Args:
        pattern (None | torch.Tensor): shape (3, 32, 32) or (32, 32).
        weight (None | torch.Tensor): shape (3, 32, 32) or (32, 32).
    """

    def __init__(self, pattern, weight):
        super(AddCIFAR10Trigger, self).__init__()
        if pattern is None:
            self.pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
            self.pattern[0, -3:, -3:] = 255
        else:
            self.pattern = pattern
            if self.pattern.dim() == 2:
                self.pattern = self.pattern.unsqueeze(0)

        if weight is None:
            self.weight = torch.zeros((1, 32, 32), dtype=torch.float32)
            self.weight[0, -3:, -3:] = 1.0
        else:
            self.weight = weight
            if self.weight.dim() == 2:
                self.weight = self.weight.unsqueeze(0)

        # Accelerated calculation
        self.res = self.weight * self.pattern
        self.weight = 1.0 - self.weight

    def __call__(self, img):
        img = F.pil_to_tensor(img)
        img = self.add_trigger(img)
        img = Image.fromarray(img.permute(1, 2, 0).numpy())
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
        poisoned_transform(Optional[Callable])：Compose which contains operation which can transform 
        samples into poisoning samples. 
        poisoned_target_transform(Optional[Callable]):Compose which contains operation which can transform 
        label into poisoning label. 

    """
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoning_rate,
                 pattern,
                 weight,
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
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddDatasetFolderTrigger(pattern, weight))

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
        weight (None | torch.Tensor): shape (1, W, H) or (W, H).
        poisoned_transform_index(int):the index of function which can transform samples into poisoning samples
        in self.poisoned_transform.transforms(list) 
        poisoned_target_transform_index(int):the index of function in which can transform label into target label
        self.poisoned_target_transform(list) 

    Attributes:
    self.poison_indices(frozenset):Save the index of the poisoning sample in the dataset.
    self.poisoned_transform(Optional[Callable])：Compose which contains operation which can transform 
    samples into poisoning samples. 
    self.poisoned_target_transform(Optional[Callable]):Compose which contains operation which can transform 
    label into poisoning label. 

    """
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoning_rate,
                 pattern,
                 weight,
                 poisoned_transform_index):
        super(PoisonedVisionDataset, self).__init__(
            benign_dataset.root,
            transform = benign_dataset.transform,
            target_transform = benign_dataset.target_transform)
        
       
        # The data types of data and targets here are best unified: for data and targets, 
        # MNIST returns the "torch.Tensor" type, and cifar10 returns the  "numpy" type and the "list" type respectively.
        
        if isinstance(benign_dataset.data,torch.Tensor):
            self.data =benign_dataset.data.numpy()
        elif isinstance(benign_dataset.data,list):
            self.data = np.array(benign_dataset.data)
        else:
            self.data = benign_dataset.data

        if isinstance(benign_dataset.targets,torch.Tensor):
            self.targets = benign_dataset.targets.numpy()
        elif isinstance(benign_dataset.targets,list):
            self.targets = np.array(benign_dataset.targets)
        else:
            self.targets = benign_dataset.targets
        
        # self.data = benign_dataset.data
        # self.targets = benign_dataset.targets

        self.classes = benign_dataset.classes

        self.y_target = y_target
        self.poisoning_rate = poisoning_rate
        self.poison_indices = None
        self.modified_targets = None
        self.set_poisoned_subdatasets(y_target = self.y_target, poisoning_rate = self.poisoning_rate)

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddVisionDatasetTrigger(pattern, weight))

    def __len__(self):
        return len(self.data)
    # def __getitem__(self, index):
    #     """
    #     Override the parent class's the _getitem__(self, index) function. In addition to returning
    #     sample and label also return index.
    #     """
    #     img, target = self.data[index], int(self.targets[index])

    #     # doing this so that it is consistent with all other datasets
    #     # to return a PIL Image
    #     img = Image.fromarray(img.numpy(), mode='L')
        
    #     if index in self.poison_indices: 
    #         img = self.poisoned_transform(img)
    #         target = self.poisoned_target_transform(target)
    #     else:
    #         if self.transform is not None:
    #             img = self.transform(img)
    #         if self.target_transform is not None:
    #             target = self.target_transform(target)
    #     return img, target, index
    def __getitem__(self, index):
        img, target = self.data[index], int(self.modified_targets[index])
        # doing this so that it is consistent with all other datasets to return a PIL Image
        # if isinstance(img,torch.Tensor):
        #     img = Image.fromarray(img.numpy(), mode='L')
        # else:
            # img = Image.fromarray(img)
        if img.shape[0] == 1:
            img = Image.fromarray(img.squeeze(), mode='L')
        # 3 x H x W
        elif img.shape[0] == 3:
            img = Image.fromarray(np.moveaxis(img, [0, 1, 2], [1, 2, 0]))
            
        # img = Image.fromarray(img)
        if index in self.poison_indices:
            img = self.poisoned_transform(img)
        elif self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index
    
    def set_poisoned_subdatasets(self, y_target = None, poisoning_rate = 0.0):

        poisoned_num = int(len(self.data) * poisoning_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        # self.targets == y_target
        tmp_list = np.arange(len(self.data))[~np.array(self.targets == y_target)]
        random.shuffle(tmp_list)
        self.poison_indices = sorted(list(tmp_list[:poisoned_num]))
        self.modified_targets = np.array(deepcopy(self.targets))
        self.modified_targets[self.poison_indices] = y_target
   
    def modify_targets(self, indces, labels):
        self.modified_targets[indces] = labels
        
    def get_real_targets(self):
        return self.targets
    def get_classes(self):
        return self.classes
    def get_y_target(self):
        return self.y_target
    def get_poisoning_rate(self):
        return self.poisoning_rate
    def get_poison_indices(self):
        return self.poison_indices
    def get_modified_targets(self):
        return self.modified_targets
   

class PoisonedMNIST(MNIST):
    """
    Poisoned MNIST datasets : inherit torchvision.datasets.MNIST and add the trigger generation logic 
    to the transform logic of the sample in function:__getitem__().   
    Args:
        pattern (None | torch.Tensor): shape (1, W, H) or (W, H).
        weight (None | torch.Tensor): shape (1, W, H) or (W, H).
        poisoned_transform_index(int):the index of function which can transform samples into poisoning samples
        in self.poisoned_transform.transforms(list) 
        poisoned_target_transform_index(int):the index of function in which can transform label into target label
        self.poisoned_target_transform(list) 

    Attributes:
    self.poison_indices(frozenset):Save the index of the poisoning sample in the dataset.
    self.poisoned_transform(Optional[Callable])：Compose which contains operation which can transform 
    samples into poisoning samples. 
    self.poisoned_target_transform(Optional[Callable]):Compose which contains operation which can transform 
    label into poisoning label. 

    """
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoning_rate,
                 pattern,
                 weight,
                 poisoned_transform_index,
                 poisoned_target_transform_index):
        super(PoisonedMNIST, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=True)
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
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddMNISTTrigger(pattern, weight))

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if index in self.poison_indices:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return torch.tensor(img), torch.tensor(target)

    def get_poison_indices(self):
        return self.poison_indices

#这与PoisonedMNIST(MNIST)逻辑上没有区别
#这里有毒数据集的定义可以通过继承torchvision.datasets.vision.VisionDataset来实现
class PoisonedCIFAR10(CIFAR10):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoning_rate,
                 pattern,
                 weight,
                 poisoned_transform_index,
                 poisoned_target_transform_index):
        super(PoisonedCIFAR10, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=True)    
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoning_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        # self.targets which cifar-10 returns is  "list", So convert it to np.ndarray
        tmp_list = np.arange(total_num)[~(np.array(self.targets) == y_target)]
        random.shuffle(tmp_list)
        self.poison_indices = np.array(sorted(tmp_list[:poisoned_num]))
        self.modified_targets = np.array(deepcopy(self.targets))
        self.modified_targets[self.poison_indices] = y_target

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddCIFAR10Trigger(pattern, weight))

        # Modify labels
        # if self.target_transform is None:
        #     self.poisoned_target_transform = Compose([])
        # else:
        #     self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        # self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    # def __getitem__(self, index):
    #     img, target = self.data[index], int(self.targets[index])

    #     # doing this so that it is consistent with all other datasets
    #     # to return a PIL Image
    #     img = Image.fromarray(img)

    #     if index in self.poison_indices:
    #         img = self.poisoned_transform(img)
    #         target = self.poisoned_target_transform(target)
    #     else:
    #         if self.transform is not None:
    #             img = self.transform(img)

    #         if self.target_transform is not None:
    #             target = self.target_transform(target)
    #     return img, target, index
    def __getitem__(self, index):
        img, target = self.data[index], int(self.modified_targets[index])
        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img)
        if index in self.poison_indices:
            img = self.poisoned_transform(img)
        elif self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index
    
    def get_poison_indices(self):
        return self.poison_indices
    def get_real_targets(self):
        return self.targets
    def modify_targets(self, indces, labels):
        self.modified_targets[indces] = labels

#这与PoisonedMNIST(MNIST)逻辑上没有区别
#这里有毒数据集的定义可以通过继承torchvision.datasets.vision.VisionDataset来实现
class PoisonedCIFAR100(CIFAR100):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoning_rate,
                 pattern,
                 weight,
                 poisoned_transform_index,
                 poisoned_target_transform_index):
        super(PoisonedCIFAR100, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=True)    
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoning_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        # self.targets which cifar-10 returns is  "list", So convert it to np.ndarray
        tmp_list = np.arange(total_num)[~(np.array(self.targets) == y_target)]
        random.shuffle(tmp_list)
        self.poison_indices = np.array(sorted(tmp_list[:poisoned_num]))

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddCIFAR10Trigger(pattern, weight))

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if index in self.poison_indices:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
        return img, target, index
    
    def get_poison_indices(self):
        return self.poison_indices
    def get_real_targets(self):
        return self.targets

class BadNets(Base, Attack):
    """
    According to the specific attack strategy, override the create_poisoned_dataset() function of the parent class 
        to realize the algorithmic logic of generating the poisoned dataset

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
    
    def create_poisoned_dataset(self, dataset, y_target=None, poisoning_rate = None):
        benign_dataset = dataset

        if y_target is None:
            assert 'y_target' in self.attack_schedule, "Attack_config must contain 'y_target' configuration! "
            y_target = self.attack_schedule['y_target']

        if poisoning_rate is None:
            assert 'poisoning_rate' in self.attack_schedule, "Attack_config must contain 'poisoning_rate' configuration! "
            poisoning_rate = self.attack_schedule['poisoning_rate']

        assert 'pattern' in self.attack_schedule, "Attack_config must contain 'pattern' configuration! "
        pattern = self.attack_schedule['pattern']
        assert 'weight' in self.attack_schedule, "Attack_config must contain 'weight' configuration! "
        weight = self.attack_schedule['weight']
        assert 'poisoned_transform_index' in self.attack_schedule, "Attack_config must contain 'poisoned_transform_index' configuration! "
        poisoned_transform_index = self.attack_schedule['poisoned_transform_index']
        
        if 'poisoned_target_transform_index' in self.attack_schedule:
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
        if dataset_type == DatasetFolder:
            return PoisonedDatasetFolder(benign_dataset, y_target, poisoning_rate, pattern, weight, poisoned_transform_index, poisoned_target_transform_index)
        elif dataset_type == MNIST:
            return PoisonedVisionDataset(benign_dataset, y_target, poisoning_rate, pattern, weight, poisoned_transform_index)
            # return PoisonedMNIST(benign_dataset, y_target, poisoning_rate, pattern, weight, poisoned_transform_index, poisoned_target_transform_index)
        elif dataset_type == CIFAR10:
            return PoisonedVisionDataset(benign_dataset, y_target, poisoning_rate, pattern, weight, poisoned_transform_index)
        # elif dataset_type == CIFAR10:
        #     return PoisonedCIFAR10(benign_dataset, y_target, poisoning_rate, pattern, weight, poisoned_transform_index, poisoned_target_transform_index)
        elif dataset_type == CIFAR100:
            return PoisonedCIFAR100(benign_dataset, y_target, poisoning_rate, pattern, weight, poisoned_transform_index, poisoned_target_transform_index)
        else:
            raise NotImplementedError
