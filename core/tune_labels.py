"""Train dann."""
import sys
sys.path.append('/nobackup/richard/pytorch-dann')
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from core.test import test
from core.test_weight import test_weight
from utils.utils import save_model
import torch.backends.cudnn as cudnn
import math
cudnn.benchmark = True

def tune_labels(model, params, tgt_data_loader, tgt_data_loader_eval, num_tgt, device, logger):
    """Train dann."""
    ####################
    # 1. setup network #
    ####################

    # setup criterion and optimizer

    if not params.finetune_flag:
        print("training non-office task")
        # optimizer = optim.SGD(model.parameters(), lr=params.lr, momentum=params.momentum, weight_decay=params.weight_decay)
        optimizer = optim.Adam(model.parameters(), lr=params.lr)
    else:
        print("training office task")
        parameter_list = [{
            "params": model.features.parameters(),
            "lr": 0.001
        }, {
            "params": model.fc.parameters(),
            "lr": 0.001
        }, {
            "params": model.bottleneck.parameters()
        }, {
            "params": model.classifier.parameters()
        }, {
            "params": model.discriminator.parameters()
        }]
        optimizer = optim.SGD(parameter_list, lr=0.01, momentum=0.9)

    criterion0 = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################
    global_step = 0
    for epoch in range(params.num_epochs):
        # set train state for Dropout and BN layers
        model.train()
        # zip source and target data pair
        len_dataloader = len(tgt_data_loader)
        dataset = enumerate(tgt_data_loader)
        for step, (images_tgt, class_tgt, _) in dataset:

            p = float(step + epoch * len_dataloader) / \
                params.num_epochs / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # if params.lr_adjust_flag == 'simple':
            #     lr = adjust_learning_rate(optimizer, p)
            # else:
            #     lr = adjust_learning_rate_office(optimizer, p)
            # logger.add_scalar('lr', lr, global_step)


            # make images variable
            class_tgt = class_tgt.to(device)
            images_tgt = images_tgt.to(device)

            # zero gradients for optimizer
            optimizer.zero_grad()
            
            # tune on target domain
            tgt_class_output,_= model(input_data=images_tgt, alpha=alpha)

            # classification loss
            tgt_loss_class = criterion0(tgt_class_output, class_tgt)
            
            # optimize dann
            tgt_loss_class.backward()
            optimizer.step()
            
            global_step += 1

            # print step info
            logger.add_scalar('tgt_loss_class', tgt_loss_class.item(), global_step)

            if ((step + 1) % params.log_step == 0):
                print(
                    "Epoch [{:4d}/{}] Step [{:2d}/{}]: tgt_loss_class={:.6f}"
                    .format(epoch + 1, params.num_epochs, step + 1, len_dataloader, tgt_loss_class.data.item()))

        # eval model
        if ((epoch + 1) % params.eval_step == 0):
            tgt_test_loss, tgt_acc, _ = test_weight(model, tgt_data_loader_eval, device, flag='target')
            logger.add_scalar('tgt_test_loss', tgt_test_loss, global_step)
            logger.add_scalar('tgt_acc', tgt_acc, global_step)


        # save model parameters
        if ((epoch + 1) % params.save_step == 0):
            save_model(model, params.model_root,
                       params.src_dataset + '-' + params.tgt_dataset +"tuning"+ "-dann-{}.pt".format(epoch + 1))

    # save final model
    save_model(model, params.model_root, params.src_dataset + '-' + params.tgt_dataset + " tuning" + "-dann-final.pt")

    return model

def adjust_learning_rate(optimizer, p):
    lr_0 = 0.01
    alpha = 10
    beta = 0.75
    lr = lr_0 / (1 + alpha * p)**beta
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_learning_rate_office(optimizer, p):
    lr_0 = 0.001
    alpha = 10
    beta = 0.75
    lr = lr_0 / (1 + alpha * p)**beta
    for param_group in optimizer.param_groups[:2]:
        param_group['lr'] = lr
    for param_group in optimizer.param_groups[2:]:
        param_group['lr'] = 10 * lr
    return lr
