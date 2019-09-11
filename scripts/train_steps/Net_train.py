#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import logging

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
import torch.nn as nn

from etc import extensions
from scripts.feature_fitting_layers.Net_and import Net_and
from scripts.utils import logUtils

__author__ = 'Lou Zehua'
__time__ = '2019/9/6 14:18'

# init log default config
logUtils.initLogDefaultConfig()
logger = logging.getLogger(__name__)

# Hyper-parameters 定义迭代次数， 学习率以及模型形状的超参数
input_size = 2
output_size = 1
num_epochs = 10000
learning_rate = 0.0001
act_threshold = 0
threshold = 1.0
total_try = 100
UPDATE_MODEL = False

def train(x, y, net, criterion, optimizer, num_epochs=num_epochs, act_threshold=act_threshold, threshold=threshold):
    temp_net = net
    loss_value = -1
    # Training: forward, loss, backward, step
    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        y_pred = net(x)
        # Compute loss
        loss = criterion(y_pred, y)
        mask = y_pred.ge(act_threshold).float()
        correct = (mask.numpy().flatten() == y.numpy().flatten()).sum()
        acc = correct.item() / x.size(0)
        # Zero gradients
        optimizer.zero_grad()
        # perform backward pass
        loss.backward()
        # update weights
        optimizer.step()

        cur_loss_value = loss.data.item()
        if loss_value == -1 or loss_value < cur_loss_value:
            loss_value = cur_loss_value
            temp_net = net
        if acc >= threshold:
            temp_net = net
            break

    return temp_net


if __name__ == '__main__':
    # input
    N = 100
    x_input_array = np.array(torch.rand(N, 2) > 0.5)
    x_input = Variable(torch.from_numpy(x_input_array)).float()

    # target
    label = []
    for x in x_input:
        label.append((sum(x) > 1).numpy())
    y_target = Variable(torch.Tensor(np.array(label))).float()
    y_target = y_target.view(x_input.shape[0], -1)
    y_target_array = np.array(y_target.numpy())

    times_try = 0
    temp_net = Net_and()


    for i in range(total_try):
        times_try = i
        # net
        temp_net = Net_and()
        # loss function
        criterion = nn.MSELoss()
        # optimizer
        optimizer = optim.Adam(temp_net.parameters(), lr=learning_rate)
        # train
        temp_net = train(x_input, y_target, temp_net, criterion, optimizer)
        # predict and test
        y_pred = temp_net.forward(x_input) > act_threshold
        y_pred_array = np.array(y_pred.detach().numpy())
        acc_float = sum(y_pred_array == y_target_array)/x_input.size(0)
        if acc_float == 1.0:
            break
    if UPDATE_MODEL:
        if times_try < total_try:
            temp_net.set_save_mode(extensions.EXT_MODELS__WHOLE_NET_PARAMS)
            temp_net.save_model()
            temp_net.set_save_mode(extensions.EXT_MODELS__STATE_DICT)
            temp_net.save_model()
    # accuracy
    y_pred = temp_net.forward(x_input) > act_threshold
    y_pred_array = np.array(y_pred.detach().numpy())
    logger.info(sum(y_pred_array == y_target_array)/x_input.size(0))
    logger.info(temp_net.state_dict())
    # reload model
    reload_model = Net_and()
    reload_model.set_save_mode(extensions.EXT_MODELS__STATE_DICT)
    reload_model.load_model()
    y_pred = reload_model.forward(x_input) > act_threshold
    y_pred_array = np.array(y_pred.detach().numpy())
    logger.info(sum(y_pred_array == y_target_array)/x_input.size(0))
    logger.info(reload_model.state_dict())
    logger.info(reload_model.__dict__)
