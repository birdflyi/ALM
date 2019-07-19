#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

from etc import filePathConf
from scripts import Step
from scripts.classifier_layers import training_purpose

__author__ = 'Lou Zehua'
__time__ = '2019/7/17 20:22'

threshold = 0


class Net_step(nn.Module):
    def __init__(self):
        super(Net_step, self).__init__()
        self.class_col = nn.Sequential(
            Step()
        )

    def forward(self, input):
        out = self.class_col(input)
        return out

    def save_state_dict_model(self, path):
        torch.save(net.state_dict(), path)

    def save_whole_model(self, path):
        torch.save(net, path)

    def load_state_dict_model(self, path):
        model = Net_step()
        model.load_state_dict(torch.load(path))
        return model

    def load_whole_model(self, path):
        model = torch.load(path)
        model.eval()
        return model


if __name__ == '__main__':
    # input
    N = 100
    x_input_array = np.array(torch.rand(N, 1) > 0.5)
    x_input = Variable(torch.from_numpy(x_input_array)).float()
    # output
    net = Net_step()
    output = net(x_input)
    # target
    label = []
    for x in x_input:
        label.append(x > 0)
    y_target = Variable(torch.Tensor(label)).float()

    # save model
    whole_save_path = os.path.join(filePathConf.absPathDict[filePathConf.MODELS_WHOLE_NET_PARAMS_DIR], training_purpose, 'Net_step.model')
    state_dict_save_path = os.path.join(filePathConf.absPathDict[filePathConf.MODELS_STATE_DICT_DIR], training_purpose, 'Net_step.state_dict')
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
