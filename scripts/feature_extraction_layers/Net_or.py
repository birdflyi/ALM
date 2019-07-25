#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import os

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
import torch.nn as nn

from etc import filePathConf
from scripts.feature_extraction_layers import training_purpose
from scripts.structure.Net_template import Net_template

__author__ = 'Lou Zehua'
__time__ = '2019/7/17 20:22'

# Hyper-parameters 定义迭代次数， 学习率以及模型形状的超参数
input_size = 2
output_size = 1
num_epochs = 10000
learning_rate = 0.0001
threshold = 0


class Net_or(Net_template):
    def __init__(self, alias=False):
        super(Net_or, self).__init__(alias)
        self.net_sequence = nn.Sequential(
            nn.Linear(input_size, output_size),
        )
        self.summary()


def train(x, y, net, criterion, optimizer, num_epochs=num_epochs, threshold=threshold):
    # Training: forward, loss, backward, step
    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        y_pred = net(x)
        # Compute loss
        loss = criterion(y_pred, y)
        mask = y_pred.ge(threshold).float()
        correct = (mask.numpy().flatten() == y.numpy()).sum()
        acc = correct.item() / x.size(0)
        # Zero gradients
        optimizer.zero_grad()
        # perform backward pass
        loss.backward()
        # update weights
        optimizer.step()

        # 每隔20轮打印一下当前的误差和精度
        if (epoch + 1) % 20 == 0:
            print('*' * 10)
            print('epoch {}'.format(epoch + 1))  # 训练轮数
            print('loss is {:.4f}'.format(loss.data.item()))  # 误差
            print('accuracy is {:.4f}'.format(acc))  # 精度
        if acc >= 1.0:
            break
    return net


if __name__ == '__main__':
    # input
    N = 100
    x_input_array = np.array(torch.rand(N, 2) > 0.5)
    x_input = Variable(torch.from_numpy(x_input_array)).float()
    # output
    net = Net_or()
    output = net(x_input)
    # target
    label = []
    for x in x_input:
        label.append(sum(x) > 0)
    y_target = Variable(torch.Tensor(label)).float()

    # loss function
    criterion = nn.MSELoss()
    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # train
    net = train(x_input, y_target, net, criterion, optimizer)

    # save model
    whole_save_path = os.path.join(filePathConf.absPathDict[filePathConf.MODELS_WHOLE_NET_PARAMS_DIR], training_purpose, 'Net_or.model')
    state_dict_save_path = os.path.join(filePathConf.absPathDict[filePathConf.MODELS_STATE_DICT_DIR], training_purpose, 'Net_or.state_dict')
    # net.save_whole_model(path=whole_save_path)
    # net.save_state_dict_model(path=state_dict_save_path)
    # load model
    model_whole = net.load_whole_model(path=whole_save_path)
    # model_whole = net.load_state_dict_model(path=state_dict_save_path)

    # predict test
    y_pred = model_whole.forward(x_input) > threshold
    y_pred_array = np.array(y_pred.detach().float().numpy().flatten())
    y_target_array = np.array(y_target.numpy())
    print(sum(y_pred_array == y_target_array))
    print(model_whole.state_dict())
    print(model_whole.__dict__)
