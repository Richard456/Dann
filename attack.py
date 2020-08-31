from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


def fgsm_attack(model, alpha, criterion, images, labels, epsilon,device):
    
    model.train()
    """
    for param in model.parameters():
        param.requires_grad = False
    """
    images.requires_grad = True
    outputs,_ = model(input_data=images,alpha=alpha)
    
    model.zero_grad()
    loss = criterion(outputs, labels)
    loss.backward()
    
    attack_images = images + epsilon*images.grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1)
    
    """
    for param in model.parameters():
        param.requires_grad = True
    """

    return attack_images