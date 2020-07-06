import os
import sys
import datetime
from tensorboardX import SummaryWriter

import torch
sys.path.append('../')
from models.model import SVHNmodel
from core.train import train_dann
from utils.utils import get_data_loader, init_model, init_random_seed


class Config(object):
    # params for path
    model_name = "syndigits-svhn"
    model_base = '/home/wogong/models/pytorch-dann'
    note = 'default'
    model_root = os.path.join(model_base, model_name, note + '_' + datetime.datetime.now().strftime('%m%d_%H%M%S'))
    os.makedirs(model_root)
    config = os.path.join(model_root, 'config.txt')
    finetune_flag = False
    lr_adjust_flag = 'simple'
    src_only_flag = False

    # params for datasets and data loader
    batch_size = 128

    # params for source dataset
    src_dataset = "syndigits"
    src_image_root = os.path.join('/home/wogong/datasets', 'syndigits')
    src_model_trained = True
    src_classifier_restore = os.path.join(model_root, src_dataset + '-source-classifier-final.pt')

    # params for target dataset
    tgt_dataset = "svhn"
    tgt_image_root = os.path.join('/home/wogong/datasets', 'svhn')
    tgt_model_trained = True
    dann_restore = os.path.join(model_root, src_dataset + '-' + tgt_dataset + '-dann-final.pt')

    # params for GPU device
    gpu_id = '0'

    ## for digit
    num_epochs = 200
    log_step = 200
    save_step = 100
    eval_step = 1

    manual_seed = 42
    alpha = 0

    # params for SGD optimizer
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

# load dataset
src_data_loader = get_data_loader(params.src_dataset, params.src_image_root, params.batch_size, train=True)
src_data_loader_eval = get_data_loader(params.src_dataset, params.src_image_root, params.batch_size, train=False)
tgt_data_loader = get_data_loader(params.tgt_dataset, params.tgt_image_root, params.batch_size, train=True)
tgt_data_loader_eval = get_data_loader(params.tgt_dataset, params.tgt_image_root, params.batch_size, train=False)

# load dann model
dann = init_model(net=SVHNmodel(), restore=None)

# train dann model
print("Training dann model")
if not (dann.restored and params.dann_restore):
    dann = train_dann(dann, params, src_data_loader, tgt_data_loader, tgt_data_loader_eval, device, logger)
