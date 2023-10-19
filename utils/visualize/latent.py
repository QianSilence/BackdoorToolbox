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
from models import ResNet
from tqdm import trange
def get_latent_rep(model, layer, datasets, device = None):
    if device is None:
        device = torch.device("cpu")
    def layer_hook(module, inp, out):
        # print(torch.reshape(inp[0].detach().cpu(), (-1,)).numpy().shape)
        # print(torch.reshape(out.data.detach().cpu(), (-1,)).numpy().shape)
        # outs.append(torch.reshape(inp[0].detach().cpu(), (-1,)).numpy())
        outs.append(torch.reshape(out.data.detach().cpu(), (-1,)).numpy())
    # if isinstance(model, nn.DataParallel):
    #     hook = model.module.layer4.register_forward_hook(layer_hook)
    # else:
    #     hook = model.layer4.register_forward_hook(layer_hook)
    if isinstance(model, nn.DataParallel):
        # hook = model.linear.register_forward_hook(layer_hook)
        hook = dict(model.module.named_children())[layer].register_forward_hook(layer_hook)
       
    else:
        # hook = model.linear.register_forward_hook(layer_hook)
        hook = dict(model.named_children())[layer].register_forward_hook(layer_hook)

    inps,outs = [],[]
    y_labels = []
    for index in trange(len(datasets)):
        sample = datasets[index][0].unsqueeze(0).to(device)
        y_label = datasets[index][1]
        _ = model(sample)
        y_labels.append(y_label)
    hook.remove()
    latents = np.array(outs)
    y_labels = np.array(y_labels)
    return latents,y_labels

def get_latent_rep_without_detach(model, layer, samples):
    """
    ouuput:(torch.tensor,torch.tensor)
    Returns the output of the specified layer of the model, but does not detach it from the compution graph.
    """
    latents = None 
    def layer_hook(module, inp, out):
        # Use the global keyword to declare that declare global variables to be modified
        nonlocal latents
        latents = out
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
    return latents, predicts

    
