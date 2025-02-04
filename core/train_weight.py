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

def train_dann(model, params, src_data_loader, tgt_data_loader,src_data_loader_eval, tgt_data_loader_eval, weights_offset, num_src, num_tgt, device, logger):
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

    # perform weighing
    class_weights = torch.FloatTensor(weights_offset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion0 = nn.CrossEntropyLoss()
    criterion1 = nn.CrossEntropyLoss(reduction = 'none')

    ####################
    # 2. train network #
    ####################
    global_step = 0
    for epoch in range(params.num_epochs):
        # set train state for Dropout and BN layers
        model.train()
        # zip source and target data pair
        len_dataloader = min(len(src_data_loader), len(tgt_data_loader))
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((images_src, class_src, idx_src), (images_tgt, _, idx_tgt)) in data_zip:

            p = float(step + epoch * len_dataloader) / \
                params.num_epochs / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # if params.lr_adjust_flag == 'simple':
            #     lr = adjust_learning_rate(optimizer, p)
            # else:
            #     lr = adjust_learning_rate_office(optimizer, p)
            # logger.add_scalar('lr', lr, global_step)

            # prepare domain label
            size_src = len(images_src)
            size_tgt = len(images_tgt)
            label_src = torch.zeros(size_src).long().to(device)  # source 0
            label_tgt = torch.ones(size_tgt).long().to(device)  # target 1

            # make images variable
            class_src = class_src.to(device)
            images_src = images_src.to(device)
            images_tgt = images_tgt.to(device)

            # zero gradients for optimizer
            optimizer.zero_grad()
            
            # train on source domain
            src_class_output, src_domain_output = model(input_data=images_src, alpha=alpha)

            # classification loss
            if params.run_mode == 0:
                src_loss_class = criterion0(src_class_output, class_src)
            else:
                src_loss_class = criterion(src_class_output, class_src)
            
            # source domain loss
            src_loss_domain = torch.dot(criterion1(src_domain_output, label_src),
            torch.tensor([weights_offset[label] for label in class_src]).cuda().to(device).float())
            #train on target domain
            _, tgt_domain_output = model(input_data=images_tgt, alpha=alpha) 
            # target domain loss        
            tgt_loss_domain = criterion0(tgt_domain_output, label_tgt)

            # total loss
            loss = src_loss_class + src_loss_domain + tgt_loss_domain

            # optimize dann
            loss.backward()
            optimizer.step()

            global_step += 1

            # print step info
            logger.add_scalar('src_loss_class', src_loss_class.item(), global_step)
            logger.add_scalar('src_loss_domain', src_loss_domain.item(), global_step)
            logger.add_scalar('tgt_loss_domain', tgt_loss_domain.item(), global_step)
            logger.add_scalar('loss', loss.item(), global_step)

            if ((step + 1) % params.log_step == 0):
                print(
                    "Epoch [{:4d}/{}] Step [{:2d}/{}]: src_loss_class={:.6f}, src_loss_domain={:.6f}, tgt_loss_domain={:.6f}, loss={:.6f}"
                    .format(epoch + 1, params.num_epochs, step + 1, len_dataloader, src_loss_class.data.item(),
                            src_loss_domain.data.item(), tgt_loss_domain.data.item(), loss.data.item()))

        # eval model
        if ((epoch + 1) % params.eval_step == 0):
            src_test_loss, src_acc, src_acc_domain = test_weight(model, src_data_loader_eval, device, flag='source')
            tgt_test_loss, tgt_acc, tgt_acc_domain = test_weight(model, tgt_data_loader_eval, device, flag='target')
            logger.add_scalar('src_test_loss', src_test_loss, global_step)
            logger.add_scalar('src_acc', src_acc, global_step)
            logger.add_scalar('src_acc_domain', src_acc_domain, global_step)
            logger.add_scalar('tgt_test_loss', tgt_test_loss, global_step)
            logger.add_scalar('tgt_acc', tgt_acc, global_step)
            logger.add_scalar('tgt_acc_domain', tgt_acc_domain, global_step)


        # save model parameters
        if ((epoch + 1) % params.save_step == 0):
            save_model(model, params.model_root,
                       params.src_dataset + '-' + params.tgt_dataset + "-dann-{}.pt".format(epoch + 1))

    # save final model
    save_model(model, params.model_root, params.src_dataset + '-' + params.tgt_dataset + "-dann-final.pt")

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
