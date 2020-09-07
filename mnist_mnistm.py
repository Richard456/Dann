import os
import sys

import torch
sys.path.append('../')
from models.model import MNISTmodel, MNISTmodel_plain
from core.train import train_dann
from utils.utils import get_data_loader, init_model, init_random_seed, get_dataset_root
from torch.utils.tensorboard import SummaryWriter


class Config(object):
    # params for path
    model_name = "mnist-mnistm"
    dataset_root = get_dataset_root()
    model_root = os.path.expanduser(os.path.join('runs', model_name))
    finetune_flag = False

    # params for datasets and data loader
    batch_size = 64

    # params for source dataset
    src_dataset = "mnist"
    src_model_trained = True
    src_classifier_restore = os.path.join(model_root, src_dataset + '-source-classifier-final.pt')
    class_num_src = 31

    # params for target dataset
    tgt_dataset = "mnistm"
    tgt_model_trained = True
    dann_restore = os.path.join(model_root, src_dataset + '-' + tgt_dataset + '-dann-final.pt')

    # params for pretrain
    num_epochs_src = 100
    log_step_src = 10
    save_step_src = 50
    eval_step_src = 20

    # params for training dann
    gpu_id = '0'

    ## for digit
    num_epochs = 100
    log_step = 20
    save_step = 1
    eval_step = 1

    ## for office
    # num_epochs = 1000
    # log_step = 10  # iters
    # save_step = 500
    # eval_step = 5  # epochs
    lr_adjust_flag = 'simple'
    src_only_flag = False

    manual_seed = 0
    alpha = 0

    # params for optimizing models
    lr = 2e-4
    momentum = 0
    weight_decay = 0

params = Config()

logger = SummaryWriter(params.model_root)

# init random seed
init_random_seed(params.manual_seed)

# init device
device = torch.device("cuda:" + params.gpu_id if torch.cuda.is_available() else "cpu")

# load dataset
src_data_loader = get_data_loader(params.src_dataset, params.dataset_root, params.batch_size, train=True)
src_data_loader_eval = get_data_loader(params.src_dataset, params.dataset_root, params.batch_size, train=False)
tgt_data_loader = get_data_loader(params.tgt_dataset, params.dataset_root, params.batch_size, train=True)
tgt_data_loader_eval = get_data_loader(params.tgt_dataset, params.dataset_root, params.batch_size, train=False)

# load dann model
dann = init_model(net=MNISTmodel(), restore=None)

# train dann model
print("Training dann model")
if not (dann.restored and params.dann_restore):
    dann = train_dann(dann, params, src_data_loader, tgt_data_loader, tgt_data_loader_eval, device,logger)
