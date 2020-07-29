import os
import sys
import datetime

import torch
sys.path.append('../')
from models.model import *
from core.train_weight import train_dann
from utils.utils import get_data_loader, get_data_loader_weight, init_model, init_random_seed, get_dataset_root, get_model_root, get_data
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import shutil
from contextlib import redirect_stdout

for data_mode in [1]: 
    for run_mode in [1]: 
        for T in [0.6]: 
                model_name = "mnist-usps-weight"
                dataset_root = get_dataset_root()
                model_root = get_model_root(model_name, data_mode, run_mode)
                model_root = os.path.join(model_root, datetime.datetime.now().strftime('%m%d_%H%M%S'))
                os.makedirs(model_root, exist_ok=True)
                logname = model_root + '/log.txt'
                import sys
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
                    model_name = "mnist-usps-weight"
                    dataset_root = dataset_root
                    model_root = model_root
                    config = os.path.join(model_root, 'config.txt')
                    finetune_flag = False
                    data_mode = data_mode 
                    run_mode = run_mode
                    threshold = (T,T)
                    soft = False
                    quantile = False
                    optimal = False
                    source_train_subsample_size = 2000
                    target_train_subsample_size = 1800


                    # params for datasets and data loader
                    batch_size = 64

                    # params for source dataset
                    src_dataset = "mnist"
                    src_model_trained = True
                    src_classifier_restore = os.path.join(model_root, src_dataset + '-source-classifier-final.pt')
                    class_num_src = 31

                    # params for target dataset
                    tgt_dataset = "usps"
                    tgt_model_trained = True
                    dann_restore = os.path.join(model_root, src_dataset + '-' + tgt_dataset + '-dann-final.pt')

                    # params for pretrain
                    num_epochs_src = 100
                    log_step_src = 10
                    save_step_src = 50
                    eval_step_src = 20

                    # params for training dann
                    gpu_id = '2'

                    ## for digit
                    num_epochs = 100
                    log_step = 20
                    save_step = 50
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
                    lr = 1e-3
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
                device = torch.device("cuda:" + params.gpu_id if torch.cuda.is_available() else "cpu")


                print(data_mode, run_mode)
                source_weight, target_weight = get_data(params.data_mode)
                if params.optimal: 
                    source_weight = target_weight
                    src_data_loader, num_src_train = get_data_loader_weight(
                        params.src_dataset, params.dataset_root, params.batch_size, train=True, subsample_size = params.source_train_subsample_size, weights = source_weight)
                    src_data_loader_eval, _ = get_data_loader_weight(
                        params.src_dataset, params.dataset_root, params.batch_size, train=False, weights = source_weight)
                if params.data_mode in [3,6]: 
                    src_data_loader, num_src_train = get_data_loader_weight(
                        params.src_dataset, params.dataset_root, params.batch_size, train=True, subsample_size = params.source_train_subsample_size, weights = source_weight)
                    src_data_loader_eval, _ = get_data_loader_weight(
                        params.src_dataset, params.dataset_root, params.batch_size, train=False, weights = source_weight)
                else: 
                    src_data_loader, num_src_train = get_data_loader_weight(params.src_dataset, params.dataset_root, params.batch_size, train=True, subsample_size = params.source_train_subsample_size)
                    src_data_loader_eval = get_data_loader(params.src_dataset, params.dataset_root, params.batch_size, train=False)
                tgt_data_loader, num_tgt_train = get_data_loader_weight(
                    params.tgt_dataset, params.dataset_root, params.batch_size, train=True, subsample_size = params.target_train_subsample_size, weights = target_weight)
                tgt_data_loader_eval, _ = get_data_loader_weight(
                    params.tgt_dataset, params.dataset_root, params.batch_size, train=False, weights = target_weight)
                # Cannot use the same sampler for both training and testing dataset 
                print(source_weight, target_weight)


                # load dann model
                dann = init_model(net=LiptonCNNModel(), restore=None).to(device)

                # train dann model
                print("Training dann model")
                if not (dann.restored and params.dann_restore):
                    dann = train_dann(dann, params, src_data_loader, tgt_data_loader, 
                                    tgt_data_loader_eval, num_src_train, num_tgt_train, device, logger)
