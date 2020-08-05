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
import shutil

for data_mode in [1,2]: 
    for run_mode in [0,1]: 
        class Config(object):
            # params for path
            model_name = "svhn-mnist-weight"
            model_root = get_model_root(model_name, data_mode, run_mode)
            model_root = os.path.join(model_root, datetime.datetime.now().strftime('%m%d_%H%M%S'))
            os.makedirs(model_root, exist_ok=True)
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
            src_model_trained = False
            src_classifier_restore = os.path.join(model_root, src_dataset + '-source-classifier-final.pt')

            # params for target dataset
            tgt_dataset = "mnist"
            tgt_image_root = get_dataset_root()
            tgt_model_trained = False
            dann_restore = os.path.join(model_root, src_dataset + '-' + tgt_dataset + '-dann-final.pt')

            # params for training dann
            gpu_id = '2'

            ## for digit
            num_epochs = 20
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
        src_data_loader, num_src_train = get_data_loader_weight(
            params.src_dataset, params.src_image_root, params.batch_size, train=True, weights = source_weight)
        src_data_loader_eval, _ = get_data_loader_weight(
            params.src_dataset, params.src_image_root, params.batch_size, train=False, weights = source_weight)
   
        tgt_data_loader, num_tgt_train = get_data_loader_weight(
            params.tgt_dataset, params.tgt_image_root, params.batch_size, train=True, weights = target_weight)
        tgt_data_loader_eval, _ = get_data_loader_weight(
            params.tgt_dataset, params.tgt_image_root, params.batch_size, train=False, weights = target_weight)
        # Cannot use the same sampler for both training and testing dataset 

        #---------------------------Distribution Estimation---------------------------------
        # source distribtion
        source_dataset=src_data_loader.dataset
        digits_source,counts_source = np.unique(source_dataset.targets.numpy(), return_counts=True)
        
        distribution_source=counts_source/len(source_dataset.targets.numpy())

        # target distribution estimation
        tgt_data_loader_est, _ = get_data_loader_weight(
            params.tgt_dataset, params.dataset_root, 1000, train=False, weights = target_weight)
        target_dataset=enumerate(tgt_data_loader_est)
        sample=np.array([])
        for step, (_,labels,_) in target_dataset:
            sample=labels.numpy()
            break
        digits_target,counts_target = np.unique(sample, return_counts=True)
        print(len(sample))
        print(sample)
        print(digits_target,counts_target)

        # distribution embeddiing
        embedding=np.zeros(10)
        k=0
        for i in digits_target:
            embedding[i]=counts_target[k]
            k+=1
            
        distribution_target=embedding/len(sample)
        print(distribution_target)

        # the weight applied to loss
        weights_offset = [distribution_target[i]/distribution_source[i] for i in range(len(distribution_target))] 
        norm = np.linalg.norm(weights_offset)
        weights_offset = weights_offset/norm

        #-----------------------------------------------------------------------------------
        # load dann model
        dann = init_model(net=SVHNmodel(), restore=None)

        # train dann model
        print("Training dann model")
        if not (dann.restored and params.dann_restore):
            dann = train_dann(dann, params, src_data_loader, tgt_data_loader, src_data_loader_eval,
                            tgt_data_loader_eval, num_src_train, num_tgt_train, device, logger)
