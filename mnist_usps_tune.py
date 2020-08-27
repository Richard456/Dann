import os
import sys
import datetime

import torch
sys.path.append('../')
from models.model import *
from core.tune_labels import tune_labels
from utils.utils import get_data_loader, get_data_loader_weight, init_model, init_random_seed, get_dataset_root, get_model_root, get_data
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import shutil
import random
import numpy as np
from contextlib import redirect_stdout

data_mode_verbosity={
    0: "MNIST->USPS",
    1: "MNIST->USPS[0-4]",
    2: "MNIST->USPS[5-9]",
    3: "MNIST->USPS with mild shift",
    4: "MNIST->USPS with strong shift"
}

for data_mode in [0,1,2]:
    model_name = "mnist-usps-tune-source-only"
    dataset_root = get_dataset_root()
    model_root = os.path.expanduser(os.path.join('runs', model_name, data_mode_verbosity[data_mode]))
    model_root = os.path.join(model_root, datetime.datetime.now().strftime('%m%d_%H%M%S'))
    os.makedirs(model_root, exist_ok=True)
    logname = model_root + '/log.txt'
    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open(logname, "a")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)  

        def flush(self):
            #this flush method is needed for python 3 compatibility.
            #this handles the flush command by doing nothing.
            #you might want to specify some extra behavior here.
            pass    
    sys.stdout = Logger()
    class Config(object):
        # params for path
        dataset_root = dataset_root
        model_root = model_root
        config = os.path.join(model_root, 'config.txt')
        finetune_flag = False
        optimal = False
        target_train_subsample_size = 50
        data_mode=data_mode
        # params for datasets and data loader
        batch_size = 20

        # params for source dataset
        src_dataset = "mnist"
        src_classifier_restore = os.path.join(model_root, src_dataset + '-source-classifier-final.pt')
        class_num_src = 31

        # params for target dataset
        restore_root=os.path.expanduser(os.path.join('runs', 'mnist-source-only'))
        tgt_dataset = "usps"
        dann_restore = os.path.join(restore_root,'0807_201604/mnist_ training-mnist-final.pt')
        
        # params for pretrain
        num_epochs_src = 100
        log_step_src = 10
        save_step_src = 50
        eval_step_src = 20

        # params for training dann
        gpu_id = '0'

        ## for digit
        num_epochs = 560
        log_step = 1
        save_step = 25
        eval_step = 1

        ## for office
        # num_epochs = 1000
        # log_step = 10  # iters
        # save_step = 500
        # eval_step = 5  # epochs
        lr_adjust_flag = 'simple'
        src_only_flag = False

        manual_seed = 8888
        alpha = 0

        # params for optimizing models
        lr = 1e-4
        momentum = 0
        weight_decay = 0
        
        def __init__(self):
            public_props = (name for name in dir(self) if not name.startswith('_'))
            with open(self.config, 'w') as f:
                for name in public_props:
                    f.write(name + ': ' + str(getattr(self, name)) + '\n')

    params = Config()

    logger = SummaryWriter(params.model_root)

    # init random seed
    init_random_seed(params.manual_seed)

    # init device
    device = torch.device("cuda:" + params.gpu_id)

    
    source_weight, target_weight = get_data(params.data_mode)

    src_data_loader, num_src_train = get_data_loader_weight(
        params.src_dataset, params.dataset_root, params.batch_size, train=True, weights = source_weight)
    src_data_loader_eval, _ = get_data_loader_weight(
        params.src_dataset, params.dataset_root, params.batch_size, train=False, weights = source_weight)

    tgt_data_loader, num_tgt_train = get_data_loader_weight(
        params.tgt_dataset, params.dataset_root, params.batch_size, 
        train=True, subsample_size = params.target_train_subsample_size, weights = target_weight)
    tgt_data_loader_eval, _ = get_data_loader_weight(
        params.tgt_dataset, params.dataset_root, params.batch_size,
            train=False, weights = target_weight)
    # Cannot use the same sampler for both training and testing dataset 

    # load dann model
    dann = init_model(net=MNISTmodel(), restore=params.dann_restore).to(device)

    # freeze model but last layer
    for param in dann.parameters():
        param.requires_grad = False

    dann.classifier[6] = nn.Linear(100, 10)
    dann=dann.to(device)


    # train dann model
    print("Tuning mnist-source-only model")
    dann = tune_labels(dann, params,src_data_loader,src_data_loader_eval, tgt_data_loader, tgt_data_loader_eval, num_tgt_train, device, logger)
