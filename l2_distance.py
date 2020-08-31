import os
import sys
import torch
sys.path.append('../')
from models.model import *
from core.train import train_dann
from utils.utils import *
import numpy as np

def l2_distance(net_one,net_two):
    # print parameters

    print("printing parameters for first net: ")
   
    params_one=np.array([])
    params_two=np.array([])

    for name, param in net_one.named_parameters():
        if param.requires_grad:
            params_one=np.append(params_one,param.cpu().detach().numpy())
            print(name,": ",param)

    
    print("printing parameters for second net: ")
    
    for name, param in net_two.named_parameters():
        if param.requires_grad:
            params_two=np.append(params_two,param.cpu().detach().numpy())
            print(name,": ",param)
    
    print("Calculating l2 distance")

    distance = np.linalg.norm(params_one-params_two)
    
    return distance

# --------------------------------add nets here-----------------------------------------
# cifar10 source only
net_A= init_model(net=MNISTmodel(), restore="/nobackup/richard/pytorch-dann/runs/cifar10-source-only/0827_135956/cifar10_ training-cifar10-final.pt")

# cifar10_cifar10_c fog org dann
net_B =init_model(net=MNISTmodel(), restore="/nobackup/richard/pytorch-dann/attack_runs/cifar10-cifar10_c_fog/cifar10-cifar10_c-fog-dann-final.pt")


#----------------------------------------------------------------------------------------


print("l2 distance between these two nets is: ",l2_distance(net_A,net_B))
