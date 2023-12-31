# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/08/21 10:24:42
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : tmp.py
# @Description :This is the implement of BadNets [1].
#               Reference:[1] Badnets: Evaluating Backdooring Attacks on Deep Neural Networks. IEEE Access 2019.
from .DataPoisoningAttack import DataPoisoningAttack
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
from torchvision.datasets import DatasetFolder, MNIST, CIFAR10
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
        img = Image.fromarray(img.numpy(), mode='L')
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
        poisoned_set(frozenset):Save the index of the poisoning sample in the dataset.
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
        self.poisoned_set = sorted(list(tmp_list[:poisoned_num]))

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

        if index in self.poisoned_set:
            sample = self.poisoned_transform(sample)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return sample, target
    
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
    self.poisoned_set(frozenset):Save the index of the poisoning sample in the dataset.
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
        super(PoisonedVisionDataset, self).__init__(
            benign_dataset.root,
            transform = benign_dataset.transform,
            target_transform = benign_dataset.target_transform)
        self.data = benign_dataset.data
        self.targets = benign_dataset.targets
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoning_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = sorted(list(tmp_list[:poisoned_num]))

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddVisionDatasetTrigger(pattern, weight))

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target
    

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
    self.poisoned_set(frozenset):Save the index of the poisoning sample in the dataset.
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
        self.poisoned_set = sorted(list(tmp_list[:poisoned_num]))

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

        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target

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
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = sorted(list(tmp_list[:poisoned_num]))

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

        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target

class BadNets(DataPoisoningAttack):
    """
    According to the specific attack strategy, override the create_poisoned_dataset() function of the parent class 
        to realize the algorithmic logic of generating the poisoned dataset

    Args:
        task(dict):The attack strategy is used for the task, including datasets, model, Optimizer algorithm 
            and loss function.
        attack_config(dict): Parameters are needed according to attack strategy
        schedule=None(dict): Config related to model training
 
    Attributes:
        self.attack_config(dict): Initialized by the incoming  parameter "attack_config".
        self.attack_strategy(string): The name of attack_strategy.
    """
    def __init__(self, task, attack_config, schedule=None):
        super(BadNets,self).__init__(task, schedule)
        self.attack_config = attack_config
        assert 'attack_strategy' in self.attack_config, "Attack_config must contain 'attack_strategy' configuration! "
        self.attack_strategy = attack_config['attack_strategy']
        
    def create_poisoned_dataset(self,dataset):

        benign_dataset = dataset
        assert 'y_target' in self.attack_config, "Attack_config must contain 'y_target' configuration! "
        y_target = self.attack_config['y_target']
        assert 'poisoning_rate' in self.attack_config, "Attack_config must contain 'poisoning_rate' configuration! "
        poisoning_rate = self.attack_config['poisoning_rate']
        assert 'pattern' in self.attack_config, "Attack_config must contain 'pattern' configuration! "
        pattern = self.attack_config['pattern']
        assert 'weight' in self.attack_config, "Attack_config must contain 'weight' configuration! "
        weight = self.attack_config['weight']
        assert 'poisoned_transform_index' in self.attack_config, "Attack_config must contain 'poisoned_transform_index' configuration! "
        poisoned_transform_index = self.attack_config['poisoned_transform_index']
        assert 'poisoned_target_transform_index' in self.attack_config, "Attack_config must contain 'poisoned_target_transform_index' configuration! "
        poisoned_target_transform_index = self.attack_config['poisoned_target_transform_index']
        
        dataset_type = type(benign_dataset)
        assert dataset_type in support_list, 'train_dataset is an unsupported dataset type, train_dataset should be a subclass of our support list.'
    
        if dataset_type == DatasetFolder:
            return PoisonedDatasetFolder(benign_dataset, y_target, poisoning_rate, pattern, weight, poisoned_transform_index, poisoned_target_transform_index)
        elif dataset_type == MNIST:
            return PoisonedVisionDataset(benign_dataset, y_target, poisoning_rate, pattern, weight, poisoned_transform_index, poisoned_target_transform_index)
            # return PoisonedMNIST(benign_dataset, y_target, poisoning_rate, pattern, weight, poisoned_transform_index, poisoned_target_transform_index)
        elif dataset_type == CIFAR10:
            return PoisonedCIFAR10(benign_dataset, y_target, poisoning_rate, pattern, weight, poisoned_transform_index, poisoned_target_transform_index)
        else:
            raise NotImplementedError

