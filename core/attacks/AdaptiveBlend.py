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
support_list = [MNIST, CIFAR10]

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
    def __init__(self, benign_dataset, y_target, poisoning_rate, cover_rate, pattern_path, pieces, mask_rate, alpha):
        super(PoisonedVisionDataset, self).__init__(benign_dataset.root,transform = benign_dataset.transform,\
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
        self.poisoning_rate =  poisoning_rate
        self.cover_rate = cover_rate

        self.pattern_path = pattern_path
        self.pieces = pieces

        self.mask_rate = mask_rate
        self.alpha = alpha

        # Read image by the method "PIL.Image.open()"  and convert it into torch.Tensor.
        # It must be ensured that the value of each item of the ndarray "pattern" is a float which is in [0-1].
        trigger = Image.open(pattern_path).convert("RGB")
        self.pattern = F.pil_to_tensor(trigger).float() / 255.0

        self.mask = self.get_trigger_mask(trigger.size[0], self.pieces, int(self. pieces * self.mask_rate))

        self.poison_indices = None
        self.cover_indices = None
        self.modified_targets = None
        self.set_poisoned_subdatasets(y_target=self.y_target, poisoning_rate=self.poisoning_rate, cover_rate=self.cover_rate)


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        img, target = self.get_sample_by_index(index)
       
        # For transform, input is always PIL.Image, doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # print(f"img.size():{img.size()},img.size(0):{img.size(0)}")

        # if img.size(0)== 1:
        #     img = Image.fromarray(img.squeeze().numpy(), mode='L')
        # # 3 x H x W
        # elif img.size(0) == 3:
        #     img = img.permute(1, 2, 0).numpy()
        #     img = Image.fromarray((img * 255).astype('uint8'))

        # img = Image.fromarray(img.numpy(), mode='RGB')

        img = transforms.ToPILImage()(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    
    def get_sample_by_index(self,index):
        '''
        Add watermarked trigger to VisionDataset. 
        the objects  'img' and 'pattern' must has same type(torch.tensor), shape(C,W,H) and data type(float:0-1)
        intput(img, pattern) and output(Poisoned image) must be the same shape. 
        Args:
            img (torch.Tensor): shape (C, H, W).
            mask (torch.Tensor): shape (1, 28, 28) or (28, 28).
            pattern(torch.Tensor): shape (C, H, W).

        Returns:
            Poisoned image(torch.Tensor):shape (C, H, W).
        '''
        assert self.pattern is not None, "pattern is None, its shape must be (C, H, w) or (H, W)"
        assert self.mask is not None, "mask is None, its shape must be  (C, W, H) or(W, H)"
        assert self.alpha is not None, "alpha is None"
        pattern = self.pattern
        mask = self.mask
        alpha = self.alpha 

        if pattern.dim() == 3:
            mask = mask.unsqueeze(0)

        img, target = self.data[index], int(self.modified_targets[index])
        img = transforms.ToTensor()(img) 
        
        if index in self.poison_indices:
            # img = (img + self.alpha * self.mask * (float_pattern - img)).type(torch.uint8)
            img = img + alpha * mask * (pattern - img)
            target = self.y_target
        elif index in self.cover_indices:
            img = img + alpha * mask * (pattern - img)
        return img, target

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
    
    def create_poisoned_dataset(self, dataset, y_target=None, poisoning_rate=None):
        benign_dataset = dataset
        dataset_type = type(benign_dataset)
        assert dataset_type in support_list, 'train_dataset is an unsupported dataset type, train_dataset should be a subclass of our support list.'
        assert 'y_target' in self.attack_schedule, "Attack_config must contain 'y_target' configuration! "
        y_target = self.attack_schedule['y_target']
        assert 'poisoning_rate' in self.attack_schedule, "Attack_config must contain 'poisoning_rate' configuration! "
        poisoning_rate = self.attack_schedule['poisoning_rate']
        assert 'cover_rate' in self.attack_schedule, "Attack_config must contain 'cover_rate' configuration! "
        cover_rate = self.attack_schedule['cover_rate']

        assert 'pattern' in self.attack_schedule, "Attack_config must contain 'pattern' configuration! "
        pattern_path = self.attack_schedule['pattern']
        assert 'pieces' in self.attack_schedule, "Attack_config must contain 'pieces' configuration! "
        pieces = self.attack_schedule['pieces']

        assert 'train_mask_rate' in self.attack_schedule, "Attack_config must contain 'train_mask_rate' configuration! "
        train_mask_rate = self.attack_schedule['train_mask_rate']
        assert 'train_alpha' in self.attack_schedule, "Attack_config must contain 'train_alpha' configuration! "
        train_alpha = self.attack_schedule['train_alpha']

        assert 'test_mask_rate' in self.attack_schedule, "Attack_config must contain 'test_mask_rate' configuration! "
        test_mask_rate = self.attack_schedule['test_mask_rate']
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
  
        work_dir = self.attack_schedule['work_dir']
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))
        msg = "\n\n\n==========Start creating poisoned_dataset==========\n"
        log(msg)
        msg = f"Total samples: {len(benign_dataset)},Among the poisoned samples:{int(len(benign_dataset) * poisoning_rate)}\n"
        log(msg)

        if isinstance(benign_dataset, VisionDataset):
            return PoisonedVisionDataset(benign_dataset, y_target, poisoning_rate, cover_rate, pattern_path, pieces, mask_rate, alpha)
        else:
            log("Dataset must be the instance of 'VisionDataset'")
        

