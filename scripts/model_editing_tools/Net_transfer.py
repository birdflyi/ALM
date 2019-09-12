#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import datetime
import importlib
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

from etc import filePathConf, extensions, training_purposes, code_style
from etc.extensions import ext_models_key2str
from etc.profiles import encoding
from scripts.structure.Net_template import Net_template
from scripts.utils.commons.dict_update import dict_update_left_merge_recursive
from scripts.utils.commons.transfer_modulePath_filePath import path_File2Module
from scripts.utils.fileUtils.move import movefile

__author__ = 'Lou Zehua'
__time__ = '2019/7/19 10:20'

threshold = 0


class Net_transfer(Net_template):
    def __init__(self, features_model, classifier_model, class_alias):
        super(Net_transfer, self).__init__(features_model.in_features, classifier_model.out_features, class_alias)
        # self.set_purposes(purpose)
        self.features_model, self.classifier_model = features_model, classifier_model
        self.auto_set_purpose(self.classifier_model)
        self.save_model_mode = extensions.EXT_MODELS__STATE_DICT  # default save mode: state dict model must exist.
        self.name = self.name.replace(self.name.split('(')[0], class_alias)
        # 'class_alias' is used for the definition of a new model in a python file.
        self.class_alias = class_alias
        self._save_pyfile_name = self.class_alias
        self.reset_save_model_name()
        self.save_pyfile_path = os.path.join(filePathConf.absPathDict[filePathConf.PY_NET_TEMPLATE_CODE_DIR],
                                             self.class_alias + extensions.ext_codes[extensions.EXT_CODES__PY])
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

    def rebuild_transfer_net_py(self):
        net_seq_models = list(self.get_net_dependent().values())
        net_seq_names = [net_model.name for net_model in net_seq_models]
        self.caller_pyfile_abspath.replace('\\', '/')
        net_seq_import = [net_model.get_net_import_str() for net_model in net_seq_models]
        param_time = datetime.datetime.now()
        param_class_name = self.class_alias
        param_net_sequence = ',\n{}'.join(net_seq_names).format(code_style.DEFAULT_INDENTATION__PY * 3)
        param_net_seq_import = '\n'.join(net_seq_import)
        param_input_size = self.net_sequence[0].in_features
        param_output_size = self.net_sequence[-1].in_features
        param_save_model_mode = ext_models_key2str[self.save_model_mode]
        args_pytmpl = [
            param_net_seq_import,
            param_time,
            param_class_name,
            param_input_size,
            param_output_size,
            param_class_name,
            param_net_sequence,
            param_class_name,
            param_save_model_mode
        ]
        with open(filePathConf.absPathDict[filePathConf.PY_NET_TEMPLATE_PATH], 'r', encoding=encoding) as f_temp:
            net_code_rebuild = f_temp.read()
        net_code_rebuild = net_code_rebuild.format(*args_pytmpl)
        with open(self.save_pyfile_path, 'w', encoding=encoding) as f_net:
            f_net.write(net_code_rebuild)


if __name__ == '__main__':
    from scripts.digital_layers.Net_step import Net_step
    from scripts.feature_fitting_layers.Net_not import Net_not
    # 1. Transfer settings
    BUILD_NEW_MODEL = True
    Net_features = Net_not()
    Net_classifier = Net_step()
    transfer_model_class_alias = 'Net_not_AD'
    # Extra settings for test
    # input
    N = 100
    # x_input_array = np.array(torch.rand(N, 2) > 0.5)
    x_input_array = np.array(torch.rand(N, 1) > 0.5)
    x_input = Variable(torch.from_numpy(x_input_array)).float()
    # target
    label = []
    for x in x_input:
        # label.append((sum(x) > 1).numpy())  # Net_and
        # label.append((sum(x) > 0).numpy())  # Net_or
        label.append((1 - x > 0).numpy())  # Net_not
    y_target = Variable(torch.Tensor(label)).float()

    # Init models
    # Const extension strings
    STATE_DICT_EXT = extensions.ext_models[extensions.EXT_MODELS__STATE_DICT]
    WHOLE_NET_PARAMS_EXT = extensions.ext_models[extensions.EXT_MODELS__WHOLE_NET_PARAMS]
    features_whole_save_path = os.path.join(filePathConf.absPathDict[filePathConf.MODELS_WHOLE_NET_PARAMS_DIR],
                                            Net_features.get_purpose(),
                                            Net_features.save_model_name + WHOLE_NET_PARAMS_EXT)
    features_state_dict_save_path = os.path.join(filePathConf.absPathDict[filePathConf.MODELS_STATE_DICT_DIR],
                                                 Net_features.get_purpose(),
                                                 Net_features.save_model_name + STATE_DICT_EXT)
    classifier_whole_save_path = os.path.join(filePathConf.absPathDict[filePathConf.MODELS_WHOLE_NET_PARAMS_DIR],
                                              Net_classifier.get_purpose(),
                                              Net_classifier.save_model_name + WHOLE_NET_PARAMS_EXT)
    classifier_state_dict_save_path = os.path.join(filePathConf.absPathDict[filePathConf.MODELS_STATE_DICT_DIR],
                                                   Net_classifier.get_purpose(),
                                                   Net_classifier.save_model_name + STATE_DICT_EXT)
    # Load Trained models
    Net_features = Net_features.load_whole_model(features_whole_save_path)
    Net_classifier = Net_classifier.load_whole_model(classifier_whole_save_path)

    # output
    # 2. Create a Net_transfer object
    net = Net_transfer(Net_features, Net_classifier, transfer_model_class_alias)
    output = net(x_input)

    # 3. Rebuild the result model
    net.rebuild_transfer_net_py()
    Net_trans_module_path = path_File2Module(net.save_pyfile_path, absfilepath=True)
    Net_trans_module = importlib.import_module(Net_trans_module_path)
    model_reload = getattr(Net_trans_module, transfer_model_class_alias)(
        net.in_features, net.out_features, transfer_model_class_alias)
    model_reload.__dict__.update(dict_update_left_merge_recursive(model_reload.__dict__, net.__dict__))
    model_reload._save_state_dict_model()
    cur_save_model_path = model_reload.save_model_path
    model_reload.load_state_dict_model(cur_save_model_path)

    # predict test
    y_pred = model_reload.forward(x_input)
    y_pred_array = np.array(y_pred.detach().float().numpy())
    y_target_array = np.array(y_target.numpy())
    print(sum(y_pred_array == y_target_array))
    print(model_reload.state_dict())
    print(model_reload.__dict__)

    # 3. Rebuild the result model
    net.rebuild_transfer_net_py()
    if BUILD_NEW_MODEL:
        cmd_str_python_run = lambda abs_path: 'python {}'.format(abs_path)
        cmd_str = cmd_str_python_run(net.save_pyfile_path)
        os.system(cmd_str)
        print('The model has been successfully built.')
        # Move validated file
        print('Move validated file to common nn scripts directory...')
        src_path = net.save_pyfile_path
        dest_path = os.path.join(training_purposes.project_purposes_scriptsDir[net.get_purpose()],
                                 net.class_alias + extensions.ext_codes[extensions.EXT_CODES__PY])
        if os.path.exists(dest_path):
            print('Target path exists a same name file. Moving file operation is canceled.')
        else:
            movefile(src_path, dest_path)
            print('Move file successfully.')
            # Rebuild model to update caller_pyfile_path
            cmd_str = cmd_str_python_run(dest_path)
            os.system(cmd_str)
            print('rebuilt model successfully.')
