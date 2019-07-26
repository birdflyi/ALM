#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

from etc import filePathConf, extensions
from scripts.digital_layers.Net_step import Net_step
from scripts.feature_fitting_layers.Net_and import Net_and
from scripts.feature_fitting_layers.Net_not import Net_not
from scripts.feature_fitting_layers.Net_or import Net_or
from scripts.structure.Net_template import Net_template

__author__ = 'Lou Zehua'
__time__ = '2019/7/19 10:20'

threshold = 0

class Net_transfer(Net_template):
    def __init__(self, features_model, classifier_model, class_alias):
        super(Net_transfer, self).__init__(features_model.in_features, classifier_model.out_features, class_alias)
        # self.set_purposes(purpose)
        self.features_model, self.classifier_model = features_model, classifier_model
        self.auto_set_purpose(self.classifier_model)
        self.name = features_model.name.replace(features_model.name.split('(')[0], class_alias)
        # 'class_alias' is used for the definition of a new model in a python file.
        self.class_alias = class_alias
        self._save_pyfile_name = self.class_alias
        self.save_model_name = self.class_alias
        # Transfer structure
        self.net_sequence = nn.Sequential(
            features_model,
            classifier_model
        )
        for param in self.net_sequence.parameters():
            param.requires_grad = False
        for param in self.net_sequence[-1].parameters():
            param.requires_grad = True
        self.summary()
        # self.serialize_seq_atomic()

    def auto_set_purpose(self, classifier_model):
        self.set_purpose(classifier_model._purpose)

if __name__ == '__main__':
    # Transfer settings
    Net_features = Net_not()
    Net_classifier = Net_step()
    transfer_model_class_alias = 'Net_not_AD'

    # Const extension strings
    STATE_DICT_EXT = extensions.ext_models[extensions.EXT_MODELS__STATE_DICT]
    WHOLE_NET_PARAMS_EXT = extensions.ext_models[extensions.EXT_MODELS__WHOLE_NET_PARAMS]
    # input
    N = 100
    # x_input_array = np.array(torch.rand(N, 2) > 0.5)
    x_input_array = np.array(torch.rand(N, 1) > 0.5)
    x_input = Variable(torch.from_numpy(x_input_array)).float()
    features_whole_save_path = os.path.join(filePathConf.absPathDict[filePathConf.MODELS_WHOLE_NET_PARAMS_DIR], Net_features.get_purpose(), Net_features.class_alias + WHOLE_NET_PARAMS_EXT)
    features_state_dict_save_path = os.path.join(filePathConf.absPathDict[filePathConf.MODELS_STATE_DICT_DIR], Net_features.get_purpose(), Net_features.class_alias + STATE_DICT_EXT)
    classifier_whole_save_path = os.path.join(filePathConf.absPathDict[filePathConf.MODELS_WHOLE_NET_PARAMS_DIR], Net_classifier.get_purpose(), Net_classifier.class_alias + WHOLE_NET_PARAMS_EXT)
    classifier_state_dict_save_path = os.path.join(filePathConf.absPathDict[filePathConf.MODELS_STATE_DICT_DIR], Net_classifier.get_purpose(), Net_classifier.class_alias + STATE_DICT_EXT)
    Net_features = Net_features.load_whole_model(features_whole_save_path)
    Net_classifier = Net_classifier.load_whole_model(classifier_whole_save_path)

    # output
    net = Net_transfer(Net_features, Net_classifier, transfer_model_class_alias)
    output = net(x_input)
    # target
    label = []
    for x in x_input:
        # label.append(sum(x) > 1)  # Net_and
        # label.append(sum(x) > 0)  # Net_or
        label.append(1 - x > 0)  # Net_not
    y_target = Variable(torch.Tensor(label)).float()

    # save and load model pairs
    net.save_whole_model()
    model_whole = net.load_whole_model()
    net.save_state_dict_model()
    model_whole = net.load_state_dict_model()

    # predict test
    y_pred = model_whole.forward(x_input)
    y_pred_array = np.array(y_pred.detach().float().numpy().flatten())
    y_target_array = np.array(y_target.numpy())
    print(sum(y_pred_array == y_target_array))
    print(model_whole.state_dict())
    print(model_whole.__dict__)
