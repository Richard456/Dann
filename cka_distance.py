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

    distance = np.linalg.norm(params_one-params_two)/np.linalg.norm(params_one)
    
    return distance

# --------------------------------add nets here-----------------------------------------
# cifar10 source only
# Pretrained CIFAR10-source
# net_S= init_model(net=MNISTmodel(), restore="/nobackup/richard/pytorch-dann/runs/cifar10-source-only/0827_135956/cifar10_ training-cifar10-final.pt")
# Pretrained CIFAR10c-source
# net_T= init_model(net=MNISTmodel(), restore="/nobackup/yguo/Dann/runs/cifar10c-source-only/0902_223539/cifar10_c_training-cifar10_c-100.pt")

# cifar10_cifar10_c fog org dann
# net_B =init_model(net=MNISTmodel(), restore="/nobackup/richard/pytorch-dann/attack_runs/cifar10-cifar10_c_fog/cifar10-cifar10_c-fog-dann-final.pt")

# Pretrained MNIST source
net_S = init_model(net=MNISTmodel(), restore="/nobackup/yguo/Dann/new_runs/mnist/mnist-mnistm-dann-final.pt")
# Pretrained MNIST-m source 
net_T = init_model(net=MNISTmodel(), restore="/nobackup/yguo/Dann/new_runs/mnistm/mnistm-mnist-dann-final.pt") 
net_0 = init_model(net=MNISTmodel(), restore=None) 
net_1 = init_model(net=MNISTmodel(), restore=None) 
# MNIST --> MNIST m 

# net_DANN = init_model(net=MNISTmodel(), restore="/nobackup/yguo/Dann/new_runs/mnist-mnistm_fixcls/mnist-mnistm-0-dann-final.pt")

# loader = get_cifar10('/nobackup/richard/dataset', 5000, True)
#----------------------------------------------------------------------------------------

sims = [] 
import torch.nn.functional as F 

for epoch in range(1,101,5):
# for epoch in [100]:
    # net_DANN = init_model(net=MNISTmodel(), restore="/nobackup/yguo/Dann/new_runs/mnist-mnistm_fixcls/mnist-mnistm-0-dann-{}.pt".format(epoch))
    net_DANN = init_model(net=MNISTmodel(), restore="/nobackup/yguo/Dann/runs/mnist-mnistm/mnist-mnistm-dann-{}.pt".format(epoch))
    # net_DANN = init_model(net=MNISTmodel(), restore="/nobackup/yguo/Dann/attack_runs/cifar10-cifar10_c_fog/cifar10-cifar10_c-dann-{}.pt".format(epoch))
    loader= get_data_loader(
    'mnistm', '/nobackup/yguo/datasets', 10000, 
    train=True)
    feature_Ss = [] 
    feature_Ts = []
    feature_DANNs = []
    feature_0s = []
    feature_1s = []
    idx = 0
    #for image, _ in loader: 
    image, _  = next(iter(loader)) 
    image = image.expand(image.data.shape[0], 3, 28, 28)
    feature_Ss.append(F.normalize(net_S.feature(image.cuda()).view(-1, 48 * 4 * 4)).detach().cpu().numpy())
    feature_Ts.append(F.normalize(net_T.feature(image.cuda()).view(-1, 48 * 4 * 4)).detach().cpu().numpy())
    feature_DANNs.append(F.normalize(net_DANN.feature(image.cuda()).view(-1, 48 * 4 * 4)).detach().cpu().numpy())
    feature_0s.append(F.normalize(net_0.feature(image.cuda()).view(-1, 48 * 4 * 4)).detach().cpu().numpy())
    feature_1s.append(F.normalize(net_1.feature(image.cuda()).view(-1, 48 * 4 * 4)).detach().cpu().numpy())
    idx += 1
    feature_S = np.vstack(feature_Ss)
    feature_T = np.vstack(feature_Ts)
    feature_0 = np.vstack(feature_0s)
    feature_1 = np.vstack(feature_1s)
    feature_DANN = np.vstack(feature_DANNs)
    sim = np.array([])
    from cka import cka, gram_rbf
    rbf_cka = cka(gram_rbf(feature_S, 0.5), gram_rbf(feature_T, 0.5))
    sim = np.append(sim, [rbf_cka])
    print("S-T similarity{}".format(rbf_cka))
    rbf_cka = cka(gram_rbf(feature_S, 0.5), gram_rbf(feature_DANN, 0.5))
    sim = np.append(sim, [rbf_cka])
    print("S-DANN similarity{}".format(rbf_cka))
    rbf_cka = cka(gram_rbf(feature_T, 0.5), gram_rbf(feature_DANN, 0.5))
    sim = np.append(sim, [rbf_cka])
    print("T-DANN similarity{}".format(rbf_cka))
        
    rbf_cka = cka(gram_rbf(feature_S, 0.5), gram_rbf(feature_0, 0.5), debiased = True)
    sim = np.append(sim, [rbf_cka])
    print("S-RI similarity{}".format(rbf_cka))
    rbf_cka = cka(gram_rbf(feature_T, 0.5), gram_rbf(feature_0, 0.5))
    sim = np.append(sim, [rbf_cka])
    print("T-RI similarity{}".format(rbf_cka))
    rbf_cka = cka(gram_rbf(feature_1, 0.5), gram_rbf(feature_0, 0.5), debiased = True)
    sim = np.append(sim, [rbf_cka])
    print("RI-RI similarity{}".format(rbf_cka))
    sims.append(sim)
    print(sim)
    result = np.vstack(sims)
    np.save("new_runs/sim_mnistm_original.npy", result)
    # np.save("new_runs/sim_cifar10c_original.npy", result)

    #print("l2 distance between these two nets is: ",l2_distance(net_A,net_B))



