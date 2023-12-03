# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/10/10 16:45:07
# @Author      : Zhenqian Zhu
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : tmp.py
# @Description  : xxxx
# #Alreadly exsiting trained model
import torch
import torch.nn as nn
import os 
import numpy as np
from models import BaselineMNISTNetwork, _ResNet
import threading
from torch.utils.data import DataLoader
from tqdm import trange
from tqdm import tqdm
from itertools import islice


def get_latent_rep(model, layer, datasets, device = torch.device("cpu")):

    def layer_hook(module, inp, out):
        # print(torch.reshape(inp[0].detach().cpu(), (-1,)).numpy().shape)
        # print(torch.reshape(out.data.detach().cpu(), (-1,)).numpy().shape)
        # outs.append(torch.reshape(inp[0].detach().cpu(), (-1,)).numpy())

        if isinstance(model, BaselineMNISTNetwork):
            size = out.data.detach().size(0)
            outs.append(torch.reshape(out.data.detach().cpu(), (size,-1)))
            # outs.append(torch.reshape(out.data.detach().cpu(), (-1,)).numpy())
        elif isinstance(model, _ResNet):
            size = inp[0].data.detach().size(0)
            outs.append(torch.reshape(inp[0].data.detach().cpu(), (size,-1)))
            # outs.append(torch.reshape(inp[0].data.detach().cpu(), (-1,)).numpy())
            
    if isinstance(model, nn.DataParallel):
        # hook = model.linear.register_forward_hook(layer_hook)
        hook = dict(model.module.named_children())[layer].register_forward_hook(layer_hook)
    else:
        # hook = model.linear.register_forward_hook(layer_hook)
        hook = dict(model.named_children())[layer].register_forward_hook(layer_hook)

    inps,outs = [],[]
    y_labels = []
    # for index in trange(len(datasets)):
    #     sample = datasets[index][0].unsqueeze(0).to(device)
    #     y_label = datasets[index][1]
    #     _ = model(sample)
    #     y_labels.append(y_label)
    # hook.remove()
    # # latents = np.array(outs)
    # # y_labels = np.array(y_labels)

    data_loader = DataLoader(
            datasets,
            batch_size=128,
            shuffle=False,
            num_workers=2,
            drop_last=False,
            pin_memory=True
        )
    # for batch in tqdm(islice(data_loader, 2), desc="Processing data", unit="batch"):
    for batch in tqdm(data_loader, desc="Processing data", unit="batch"):
        batch_img, batch_label = batch[0],batch[1]
        batch_img = batch_img.to(device)
        _ = model(batch_img)
        y_labels.append(batch_label)

    hook.remove()
    latents = torch.cat(outs, dim=0).numpy()
    y_labels = torch.cat(y_labels, dim=0).numpy()
    return latents, y_labels

# def get_latent_rep(model, layer, datasets, device = torch.device("cpu")):
#     """
#     ouuput:(torch.tensor,torch.tensor)
#     Returns the output of the specified layer of the model, but does not detach it from the compution graph.
#     Notice: "latents" are not the final output of the model, so when using nn.DataParallel() for multi-GPU training, 
#     In order to merge the intermediate output of the model on multiple GPUs, there are 2 methods:
#     1. In the function "forward" of model , extract the intermediate output ,then return it and the final output of the model.
#     2. Use dict types to prevent multiple GPUs from affecting each other and finally merge the intermediate output of each GPU.
#        This method must ensure that the intermediate output to the same device.
#     """
    
#     latent_dict = {}
#     def layer_hook(module, inp, out):
#         # Use the global keyword to declare that declare global variables to be modified
#         nonlocal latent_dict
#         if isinstance(model, BaselineMNISTNetwork) or isinstance(model.module, BaselineMNISTNetwork):
#             latent_dict[inp[0].device] = out
#         elif isinstance(model, _ResNet) or isinstance(model.module, _ResNet):
#             latent_dict[inp[0].device] = inp[0]
            
    
#          # outs.append(torch.reshape(inp[0].detach().cpu(), (-1,)).numpy())
#     # if isinstance(model, nn.DataParallel):
#     #     hook = model.module.layer4.register_forward_hook(layer_hook)
#     #     hook = model.linear.register_forward_hook(layer_hook)
#     # else:
#     #     hook = model.layer4.register_forward_hook(layer_hook)
#     #     hook = model.linear.register_forward_hook(layer_hook)
#     if isinstance(model, nn.DataParallel):
#         hook = dict(model.module.named_children())[layer].register_forward_hook(layer_hook)
#     else:
#         hook = dict(model.named_children())[layer].register_forward_hook(layer_hook)

#     inps,outs = [],[]
#     y_labels = []
#     samples = torch.tensor([])
#     for index in trange(len(datasets)):
#         sample = datasets[index][0].unsqueeze(0).to(device)
#         y_label = datasets[index][1]
#         print(samples.shape)
#         samples = torch.vstack((samples, sample))
#         y_labels.append(y_label)

#     _ = model(samples)
#     hook.remove()
#     latents = None
#     for key in latent_dict.keys():
#         if latents is None:
#             latents = latent_dict[key].to(device)
#         else:
#             latents = torch.cat((latents,latent_dict[key].to(device)))
#     latents = np.array(outs)
#     y_labels = np.array(y_labels)
#     return latents, y_labels

def get_latent_rep_without_detach(model, layer, samples, device = torch.device("cpu")):
    """
    ouuput:(torch.tensor,torch.tensor)
    Returns the output of the specified layer of the model, but does not detach it from the compution graph.
    Notice: "latents" are not the final output of the model, so when using nn.DataParallel() for multi-GPU training, 
    In order to merge the intermediate output of the model on multiple GPUs, there are 2 methods:
    1. In the function "forward" of model , extract the intermediate output ,then return it and the final output of the model.
    2. Use dict types to prevent multiple GPUs from affecting each other and finally merge the intermediate output of each GPU.
       This method must ensure that the intermediate output to the same device.
    """
    
    latent_dict = {}
    def layer_hook(module, inp, out):
        # Use the global keyword to declare that declare global variables to be modified
        nonlocal latent_dict
        if isinstance(model, BaselineMNISTNetwork) or isinstance(model.module, BaselineMNISTNetwork):
            latent_dict[inp[0].device] = out
        elif isinstance(model, _ResNet) or isinstance(model.module, _ResNet):
            latent_dict[inp[0].device] = inp[0]
            
    
         # outs.append(torch.reshape(inp[0].detach().cpu(), (-1,)).numpy())
    # if isinstance(model, nn.DataParallel):
    #     hook = model.module.layer4.register_forward_hook(layer_hook)
    #     hook = model.linear.register_forward_hook(layer_hook)
    # else:
    #     hook = model.layer4.register_forward_hook(layer_hook)
    #     hook = model.linear.register_forward_hook(layer_hook)
    if isinstance(model, nn.DataParallel):
        hook = dict(model.module.named_children())[layer].register_forward_hook(layer_hook)
    else:
        hook = dict(model.named_children())[layer].register_forward_hook(layer_hook)
    predicts = model(samples)
    hook.remove()
    latents = None
    for key in latent_dict.keys():
        if latents is None:
            latents = latent_dict[key].to(device)
        else:
            latents = torch.cat((latents,latent_dict[key].to(device)))
    return latents, predicts
