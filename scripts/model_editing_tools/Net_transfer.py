#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

from etc import filePathConf
from etc.training_purposes import training_purposes, R_REGRESSION, L_CLASSIFY
from scripts.classifier_layers.Net_step import Net_step
from scripts.feature_extraction_layers.Net_and import Net_and
from scripts.structure.Net_template import Net_template

__author__ = 'Lou Zehua'
__time__ = '2019/7/19 10:20'

threshold = 0

# todo: fit model
class Net_transfer(Net_template):
    def __init__(self, features_model, classifier_model, alias=None):
        super(Net_transfer, self).__init__(alias)
        self.features_model = features_model
        self.classifier_model = classifier_model
        self.net_sequence = nn.Sequential(
            features_model,
            classifier_model
        )
        for param in self.net_sequence.parameters():
            param.requires_grad = False
        for param in self.net_sequence[-1].parameters():
            param.requires_grad = True


if __name__ == '__main__':
    # input
    N = 100
    x_input_array = np.array(torch.rand(N, 2) > 0.5)
    x_input = Variable(torch.from_numpy(x_input_array)).float()
    # output
    and_whole_save_path = os.path.join(filePathConf.absPathDict[filePathConf.MODELS_WHOLE_NET_PARAMS_DIR], training_purposes[R_REGRESSION], 'Net_and.model')
    and_state_dict_save_path = os.path.join(filePathConf.absPathDict[filePathConf.MODELS_STATE_DICT_DIR], training_purposes[R_REGRESSION], 'Net_and.state_dict')
    step_whole_save_path = os.path.join(filePathConf.absPathDict[filePathConf.MODELS_WHOLE_NET_PARAMS_DIR], training_purposes[L_CLASSIFY], 'Net_step.model')
    step_state_dict_save_path = os.path.join(filePathConf.absPathDict[filePathConf.MODELS_STATE_DICT_DIR], training_purposes[L_CLASSIFY], 'Net_step.state_dict')
    model_and = Net_and()
    model_step = Net_step()
    model_and = model_and.load_whole_model(and_whole_save_path)
    model_step = model_step.load_whole_model(step_whole_save_path)
    net = Net_transfer(model_and, model_step)
    output = net(x_input)
    # target
    label = []
    for x in x_input:
        label.append(sum(x) > 1)
    y_target = Variable(torch.Tensor(label)).float()

    # save model
    transfer_whole_save_path = os.path.join(filePathConf.absPathDict[filePathConf.MODELS_WHOLE_NET_PARAMS_DIR], training_purposes[L_CLASSIFY], 'Net_and.model')
    transfer_state_dict_save_path = os.path.join(filePathConf.absPathDict[filePathConf.MODELS_STATE_DICT_DIR], training_purposes[L_CLASSIFY], 'Net_and.state_dict')
    net.save_whole_model(transfer_whole_save_path)
    net.save_state_dict_model(transfer_state_dict_save_path)
    # load model
    model_whole = net.load_whole_model(transfer_whole_save_path)
    # model_whole = net.load_state_dict_model(transfer_state_dict_save_path)

    # predict test
    y_pred = model_whole.forward(x_input)
    y_pred_array = np.array(y_pred.detach().float().numpy().flatten())
    y_target_array = np.array(y_target.numpy())
    print(sum(y_pred_array == y_target_array))
    print(model_whole.state_dict())
