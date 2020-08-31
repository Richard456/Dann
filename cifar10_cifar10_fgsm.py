import os
import sys

import torch
sys.path.append('../')
from models.model import *
from core.train import train_dann
from utils.utils import get_data_loader, init_model, init_random_seed, get_dataset_root
from torch.utils.tensorboard import SummaryWriter


class Config(object):
    # params for path
    model_name = "cifar10-cifar10_fgsm_fixcls"
    dataset_root = get_dataset_root()
    model_root = os.path.expanduser(os.path.join('new_runs', model_name))
    finetune_flag = False

    # params for datasets and data loader
    batch_size = 64
    noise_type="fog"

    # params for source dataset
    src_dataset = "cifar10"
    src_model_trained = True
    src_classifier_restore = os.path.join(model_root, src_dataset + '-source-classifier-final.pt')
    class_num_src = 31

    # params for target dataset
    tgt_dataset = "cifar10"
    tgt_model_trained = True
    dann_restore = "/nobackup/richard/pytorch-dann/runs/cifar10-source-only/0827_135956/cifar10_ training-cifar10-final.pt"

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
    save_step = 50
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
device = torch.device("cuda:" + params.gpu_id)

# load dataset
src_data_loader = get_data_loader(params.src_dataset, params.dataset_root, params.batch_size, train=True)
src_data_loader_eval = get_data_loader(params.src_dataset, params.dataset_root, params.batch_size, train=False)

tgt_data_loader = get_data_loader(params.tgt_dataset, params.dataset_root, params.batch_size, train=True, noise_type=params.noise_type)
tgt_data_loader_eval = get_data_loader(params.tgt_dataset, params.dataset_root, params.batch_size, train=False, noise_type=params.noise_type)

print(len(tgt_data_loader)," ",len(tgt_data_loader_eval))
# load dann model
dann = init_model(net=MNISTmodel(), restore=params.dann_restore)

# freeze classifier
for param in dann.parameters():
    param.requires_grad = False

dann.feature=nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=(5, 5)),  # 3 28 28, 32 24 24
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 32 12 12
            nn.Conv2d(in_channels=32, out_channels=48,
                      kernel_size=(5, 5)),  # 48 8 8
            nn.BatchNorm2d(48),
            nn.Dropout2d(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 48 4 4
        )

dann.discriminator=nn.Sequential(
            nn.Linear(48*4*4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 2),
        )

dann=dann.to(device)

# train dann model
print("Training dann model")
dann = train_dann(dann, params, src_data_loader, tgt_data_loader, tgt_data_loader_eval, device, logger, attack="fgsm")