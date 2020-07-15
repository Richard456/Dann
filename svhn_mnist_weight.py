import os
import sys
import datetime
from torch.utils.tensorboard import SummaryWriter

import torch
sys.path.append('../')
from models.model import SVHNmodel
from core.train_weight import train_dann
from utils.utils import get_data_loader, get_data_loader_weight, init_model, \
init_random_seed, get_dataset_root, get_model_root, get_data
import numpy as np

for data_mode, run_mode in zip([3,5,0], [0,1,2,3]):  
    class Config(object):
        # params for path
        model_name = "svhn-mnist-weight"
        model_root = get_model_root(model_name, data_mode, run_mode)
        if not os.path.exists(model_root):
            os.makedirs(model_root)
        config = os.path.join(model_root, 'config.txt')
        finetune_flag = False
        lr_adjust_flag = 'simple'
        src_only_flag = False
        data_mode = data_mode 
        run_mode = run_mode

        # params for datasets and data loader
        batch_size = 128

        # params for source dataset
        src_dataset = "svhn"
        src_image_root = get_dataset_root()
        src_model_trained = True
        src_classifier_restore = os.path.join(model_root, src_dataset + '-source-classifier-final.pt')

        # params for target dataset
        tgt_dataset = "mnist"
        tgt_image_root = get_dataset_root()
        tgt_model_trained = True
        dann_restore = os.path.join(model_root, src_dataset + '-' + tgt_dataset + '-dann-final.pt')

        # params for training dann
        gpu_id = '0'

        ## for digit
        num_epochs = 100
        log_step = 50
        save_step = 100
        eval_step = 1

        ## for office
        # num_epochs = 1000
        # log_step = 10  # iters
        # save_step = 500
        # eval_step = 5  # epochs

        manual_seed = 0
        alpha = 0

        # params for optimizing models
        lr = 0.001
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
    source_weight, target_weight = get_data(params.data_mode)


    # load dataset
    if params.data_mode == 3:
        src_data_loader, num_src_train = get_data_loader_weight(
            params.src_dataset, params.src_image_root, params.batch_size, train=True, weights = source_weight)
        src_data_loader_eval, _ = get_data_loader_weight(
            params.src_dataset, params.src_image_root, params.batch_size, train=False, weights = source_weight)
    else:
        src_data_loader, num_src_train = get_data_loader_weight(params.src_dataset, params.src_image_root, params.batch_size, train=True)
        src_data_loader_eval = get_data_loader(params.src_dataset, params.src_image_root, params.batch_size, train=False)
    tgt_data_loader, num_tgt_train = get_data_loader_weight(
        params.tgt_dataset, params.tgt_image_root, params.batch_size, train=True, weights = target_weight)
    tgt_data_loader_eval, _ = get_data_loader_weight(
        params.tgt_dataset, params.tgt_image_root, params.batch_size, train=False, weights = target_weight)
    # Cannot use the same sampler for both training and testing dataset 


    # load dann model
    dann = init_model(net=SVHNmodel(), restore=None)

    # train dann model
    print("Training dann model")
    if not (dann.restored and params.dann_restore):
        dann = train_dann(dann, params, src_data_loader, tgt_data_loader, 
                        tgt_data_loader_eval, num_src_train, num_tgt_train, device, logger)
