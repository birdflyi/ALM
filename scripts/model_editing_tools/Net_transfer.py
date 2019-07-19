#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

from etc import filePathConf
from scripts.classifier_layers.Net_step import Net_step
from scripts.feature_extraction_layers.Net_and import Net_and

__author__ = 'Lou Zehua'
__time__ = '2019/7/19 10:20'

threshold = 0


class Net_transfer(nn.Module):
    def __init__(self, features_model, classifier_model):
        super(Net_transfer, self).__init__()
        self.features_model = features_model
        self.classifier_model = classifier_model
        self.class_col = nn.Sequential(
            features_model,
            classifier_model
        )
        for param in self.class_col.parameters():
            param.requires_grad = False
        for param in self.class_col[-1].parameters():
            param.requires_grad = True

    def forward(self, input):
        out = self.class_col(input)
        return out

    def save_state_dict_model(self, path):
        torch.save(net.state_dict(), path)

    def save_whole_model(self, path):
        torch.save(net, path)

    def load_state_dict_model(self, path):
        model = Net_transfer(self.features_model, self.classifier_model)
        model.load_state_dict(torch.load(path))
        return model

    def load_whole_model(self, path):
        model = torch.load(path)
        model.eval()
        return model


if __name__ == '__main__':
    # input
    N = 100
    x_input_array = np.array(torch.rand(N, 2) > 0.5)
    x_input = Variable(torch.from_numpy(x_input_array)).float()
    # output
    and_whole_save_path = os.path.join(filePathConf.absPathDict[filePathConf.MODELS_WHOLE_NET_PARAMS_DIR], 'Net_and.model')
    and_state_dict_save_path = os.path.join(filePathConf.absPathDict[filePathConf.MODELS_STATE_DICT_DIR], 'Net_and.state_dict')
    step_whole_save_path = os.path.join(filePathConf.absPathDict[filePathConf.MODELS_WHOLE_NET_PARAMS_DIR], 'Net_step.model')
    step_state_dict_save_path = os.path.join(filePathConf.absPathDict[filePathConf.MODELS_STATE_DICT_DIR], 'Net_step.state_dict')
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
    transfer_whole_save_path = os.path.join(filePathConf.absPathDict[filePathConf.MODELS_WHOLE_NET_PARAMS_DIR], 'Net_transfer.model')
    transfer_state_dict_save_path = os.path.join(filePathConf.absPathDict[filePathConf.MODELS_STATE_DICT_DIR], 'Net_transfer.state_dict')
    # net.save_whole_model(transfer_whole_save_path)
    # net.save_state_dict_model(transfer_state_dict_save_path)
    # load model
    model_whole = net.load_whole_model(transfer_whole_save_path)
    # model_whole = net.load_state_dict_model(transfer_state_dict_save_path)

    # predict test
    y_pred = model_whole.forward(x_input)
    y_pred_array = np.array(y_pred.detach().float().numpy().flatten())
    y_target_array = np.array(y_target.numpy())
    print(sum(y_pred_array == y_target_array))
    print(model_whole.state_dict())
