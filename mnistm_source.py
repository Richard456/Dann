import os
import sys
import datetime

import torch
sys.path.append('../')
from models.model import *
from core.train_source import train_source
from utils.utils import get_data_loader, get_data_loader_weight, init_model, init_random_seed, get_dataset_root, get_model_root, get_data
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import shutil
import random
import numpy as np
from contextlib import redirect_stdout


model_name = "mnistm-source-only"
dataset_root = '/nobackup/yguo/datasets'
model_root = os.path.expanduser(os.path.join('runs', model_name))
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
    target_train_subsample_size = 0
    # params for datasets and data loader
    batch_size = 512

    # params for source dataset
    src_dataset = "mnistm"
    #src_classifier_restore = os.path.join(model_root, src_dataset + '-source-classifier-final.pt')
    class_num_src = 31

    # params for pretrain
    num_epochs_src = 100
    log_step_src = 10
    save_step_src = 50
    eval_step_src = 25

    # params for training dann
    gpu_id = '3'

    ## for digit
    num_epochs = 1
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

source_weight,_ = get_data(0)

src_data_loader = get_data_loader(
    params.src_dataset, params.dataset_root, params.batch_size, 
    train=True)

src_data_loader_eval = get_data_loader(
    params.src_dataset, params.dataset_root, params.batch_size,
        train=False)
# Cannot use the same sampler for both training and testing dataset 

# load dann model
dann = init_model(net=MNISTmodel(), restore=None).to(device)

"""
# freeze model but last layer
for param in dann.parameters():
    param.requires_grad = False

dann.classifier[6] = nn.Linear(100, 10)
dann=dann.to(device)
"""

# train dann model
print("Training MNIST")
dann = train_source(dann, params,  src_data_loader, src_data_loader_eval, device, logger)