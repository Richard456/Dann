import os
import sys
import datetime
from tensorboardX import SummaryWriter

import torch
sys.path.append('../')
from models.model import SVHNmodel
from core.train_weight import train_dann
from utils.utils import get_data_loader, get_data_loader_weight, init_model, init_random_seed
import numpy as np

class Config(object):
    # params for path
    model_name = "svhn-mnist-weight"
    model_base = '/nobackup/yguo/pytorch-dann'
    model_root = os.path.expanduser(os.path.join('~', 'Models', 'pytorch-DANN', model_name))
    note = 'paper-structure'
    model_root = os.path.join(model_base, model_name, note + '_' + datetime.datetime.now().strftime('%m%d_%H%M%S'))
    os.makedirs(model_root)
    config = os.path.join(model_root, 'config.txt')
    finetune_flag = False
    lr_adjust_flag = 'simple'
    src_only_flag = False

    # params for datasets and data loader
    batch_size = 128

    # params for source dataset
    src_dataset = "svhn"
    src_image_root = os.path.join('/nobackup/yguo/dataset', 'svhn')
    src_model_trained = True
    src_classifier_restore = os.path.join(model_root, src_dataset + '-source-classifier-final.pt')

    # params for target dataset
    tgt_dataset = "mnist"
    tgt_image_root = os.path.join('/nobackup/yguo/dataset', 'mnist')
    tgt_model_trained = True
    dann_restore = os.path.join(model_root, src_dataset + '-' + tgt_dataset + '-dann-final.pt')

    # params for training dann
    gpu_id = '0'

    ## for digit
    num_epochs = 200
    log_step = 50
    save_step = 100
    eval_step = 1

    ## for office
    # num_epochs = 1000
    # log_step = 10  # iters
    # save_step = 500
    # eval_step = 5  # epochs

    manual_seed = None
    alpha = 0

    # params for optimizing models
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-6

    def __init__(self):
        public_props = (name for name in dir(self) if not name.startswith('_'))
        with open(self.config, 'w') as f:
            for name in public_props:
                f.write(name + ': ' + str(getattr(self, name)) + '\n')

params = Config()
logger = SummaryWriter(params.model_root)
device = torch.device("cuda:" + params.gpu_id if torch.cuda.is_available() else "cpu")

# init random seed
init_random_seed(params.manual_seed)

# Create custom target sampler 
value = 0.0625
# WEIGHTS = torch.tensor(np.concatenate(
#     [[value], np.random.uniform(value, 1-value, 8), [1-value]]))


WEIGHTS = torch.ones(10)
target_sampler = torch.utils.data.sampler.WeightedRandomSampler(
    WEIGHTS, 1) # The number of samples will be updated automatically


# load dataset
src_data_loader, num_src_train = get_data_loader_weight(params.src_dataset, params.src_image_root, params.batch_size, train=True)
src_data_loader_eval = get_data_loader(params.src_dataset, params.src_image_root, params.batch_size, train=False)
tgt_data_loader, num_tgt_train = get_data_loader_weight(
    params.tgt_dataset, params.tgt_image_root, params.batch_size, train=True, sampler=target_sampler)
tgt_data_loader_eval, _ = get_data_loader_weight(
    params.tgt_dataset, params.tgt_image_root, params.batch_size, train=False, sampler=target_sampler)

print(num_src_train, num_tgt_train)
print(len(src_data_loader), len(tgt_data_loader))
# load dann model
dann = init_model(net=SVHNmodel(), restore=None)

# train dann model
print("Training dann model")
if not (dann.restored and params.dann_restore):
    dann = train_dann(dann, params, src_data_loader, tgt_data_loader, 
                      tgt_data_loader_eval, num_src_train, num_tgt_train, device, logger)
