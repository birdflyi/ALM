#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

from etc import filePathConf
from etc.training_purposes import training_purposes, L_DIGITAL
from scripts.digital_layers import training_purpose
from scripts.primary_funcs.basic_nn_units import Step
from scripts.structure.Net_template import Net_template

__author__ = 'Lou Zehua'
__time__ = '2019/7/17 20:22'

threshold = 0


class Net_step(Net_template):
    def __init__(self, in_features=1, out_features=1, class_alias=None):
        super().__init__(in_features, out_features, class_alias)
        self.is_atomic = True
        self.net_sequence = nn.Sequential(
            Step(in_features, out_features)
        )
        self.set_caller_pyfile_path(os.path.abspath(__file__))
        self.set_purpose(training_purposes[L_DIGITAL])  # Set purpose manually if model is atomic
        self.summary()


if __name__ == '__main__':
    # input
    N = 100
    x_input_array = np.array(2 * torch.rand(N, 1) - 1)
    x_input = Variable(torch.from_numpy(x_input_array)).float()
    # output
    net = Net_step()
    output = net(x_input)
    # target
    label = []
    for x in x_input:
        label.append((x > 0).numpy())
    y_target = Variable(torch.Tensor(np.array(label))).float()
    y_target = y_target.view(x_input.shape[0], -1)

    # save model
    whole_save_path = os.path.join(filePathConf.absPathDict[filePathConf.MODELS_WHOLE_NET_PARAMS_DIR], training_purpose, 'Net_step.model')
    state_dict_save_path = os.path.join(filePathConf.absPathDict[filePathConf.MODELS_STATE_DICT_DIR], training_purpose, 'Net_step.state_dict')
    net.save_whole_model(path=whole_save_path)
    net.save_state_dict_model(path=state_dict_save_path)
    # load model
    model_whole = net.load_whole_model(path=whole_save_path)
    # model_whole = net.load_state_dict_model(path=state_dict_save_path)

    # predict test
    y_pred = model_whole.forward(x_input)
    y_pred_array = np.array(y_pred.detach().numpy())
    y_target_array = np.array(y_target.numpy())
    print(sum(y_pred_array == y_target_array))
    print(model_whole.state_dict())
    print(model_whole.__dict__)
