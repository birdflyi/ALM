#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import importlib
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable

from etc import training_purposes, filePathConf, extensions
from etc.model_registry_settings import df_registry_col_names, CLASS_ALIAS, REL_CALLER_PYFILE_PATH
from etc.profiles import BASE_DIR
from scripts import df_registry
from scripts.model_editing_tools.Net_transfer import Net_transfer
from scripts.utils import logUtils
from scripts.utils.commons.dict_update import dict_update_left_join_recursive
from scripts.utils.commons.transfer_modulePath_filePath import path_File2Module
from scripts.utils.fileUtils.move import movefile

__author__ = 'Lou Zehua'
__time__ = '2019/7/29 11:09'

# init log default config
logUtils.initLogDefaultConfig()
logger = logging.getLogger(__name__)


def validate_name_with_modulePath(module_path, net_name):
    if module_path and net_name:
        if module_path.split('.')[-1] == net_name:
            return True
    return False


def get_net_caller_pyfile_abspath(net_class_alias):
    net_caller_pyfile_abspath = ''
    col_name__class_alias = df_registry_col_names[CLASS_ALIAS]
    col_name__rel_caller_pyfile_path = df_registry_col_names[REL_CALLER_PYFILE_PATH]
    registered_ids = df_registry[df_registry[col_name__class_alias] == net_class_alias].index.tolist()
    if len(registered_ids):
        registered_id = predict_registered_id(registered_ids)
        net_caller_pyfile_relpath = df_registry[col_name__rel_caller_pyfile_path][registered_id]
        net_caller_pyfile_abspath = os.path.join(BASE_DIR, net_caller_pyfile_relpath)
    return net_caller_pyfile_abspath


# todo: predict use model
def predict_registered_id(registered_ids):
    # use the latest registered model
    return registered_ids[-1]


def transfer_models(features_net_name, classifier_net_name, transfer_model_class_alias,
                    features_net_abspath=None, classifier_net_abspath=None, build_new_model=True):
    features_net_abspath = features_net_abspath or get_net_caller_pyfile_abspath(features_net_name)
    classifier_net_abspath = classifier_net_abspath or get_net_caller_pyfile_abspath(classifier_net_name)
    # 1. Auto load source nn models
    # Auto import python modules
    features_module_path = path_File2Module(features_net_abspath, absfilepath=True)
    classifier_module_path = path_File2Module(classifier_net_abspath, absfilepath=True)
    if not (validate_name_with_modulePath(features_module_path, features_net_name) and
            validate_name_with_modulePath(classifier_module_path, classifier_net_name)):
        raise Warning('The class alias is different from module name. Your nn node may be exist, but it will '
                      'cause an orphan nn node which will never be accessed. Please make sure your class alias '
                      'is the same with module name.')
    features_pymodule = importlib.import_module(features_module_path)
    classifier_pymodule = importlib.import_module(classifier_module_path)
    # Init models
    Net_features = getattr(features_pymodule, features_net_name)()
    Net_classifier = getattr(classifier_pymodule, classifier_net_name)()
    # Const extension strings
    STATE_DICT_EXT = extensions.ext_models[extensions.EXT_MODELS__STATE_DICT]
    features_state_dict_save_path = os.path.join(filePathConf.absPathDict[filePathConf.MODELS_STATE_DICT_DIR],
                                                 Net_features.get_purpose(),
                                                 Net_features.save_model_name + STATE_DICT_EXT)
    classifier_state_dict_save_path = os.path.join(filePathConf.absPathDict[filePathConf.MODELS_STATE_DICT_DIR],
                                                   Net_classifier.get_purpose(),
                                                   Net_classifier.save_model_name + STATE_DICT_EXT)
    # Load Trained models
    Net_features = Net_features.load_state_dict_model(features_state_dict_save_path)
    Net_classifier = Net_classifier.load_state_dict_model(classifier_state_dict_save_path)
    # 2. Create a Net_transfer object
    net = Net_transfer(Net_features, Net_classifier, transfer_model_class_alias)
    # 3. Rebuild the result model
    net.rebuild_transfer_net_py()
    Net_trans_module_path = path_File2Module(net.save_pyfile_path, absfilepath=True)
    Net_trans_module = importlib.import_module(Net_trans_module_path)
    model_reload = getattr(Net_trans_module, transfer_model_class_alias)(
        net.in_features, net.out_features, transfer_model_class_alias)
    model_reload.__dict__.update(dict_update_left_join_recursive(model_reload.__dict__, net.__dict__))
    model_reload.save_state_dict_model()
    cur_save_model_path = model_reload.save_model_path
    model_reload.load_state_dict_model(cur_save_model_path)
    if build_new_model:
        cmd_str_python_run = lambda abs_path: 'python {}'.format(abs_path)
        # cmd: python run
        cmd_str = cmd_str_python_run(net.save_pyfile_path)
        os.system(cmd_str)  # rebuild and validate
        logger.info('The model has been successfully built.')
        # Move the validated python file
        logger.info('Move the validated python file to common nn scripts directory...')
        src_path = net.save_pyfile_path
        dest_path = os.path.join(training_purposes.project_purposes_scriptsDir[net.get_purpose()],
                                 net.class_alias + extensions.ext_codes[extensions.EXT_CODES__PY])
        model_reload.set_caller_pyfile_path(src_path)
        if os.path.exists(dest_path):
            logger.warning('Target path exists a same name file. Moving file operation is canceled.')
        else:
            try:
                model_reload.unregister_net()
                movefile(src_path, dest_path)  # May effect on caller_pyfile_path and save_pyfile_path
            except BaseException as e:
                logger.error(e)
            logger.info('Move file successfully.')
            model_reload.set_caller_pyfile_path(dest_path)
            # Rebuild model to update caller_pyfile_path
            cmd_str = cmd_str_python_run(dest_path)
            os.system(cmd_str)
            logger.info('rebuilt model successfully.')
        # update model_reload
        Net_trans_module_path = path_File2Module(model_reload.caller_pyfile_abspath, absfilepath=True)
        Net_trans_module = importlib.import_module(Net_trans_module_path)
        model_reload = getattr(Net_trans_module, transfer_model_class_alias)(
            net.in_features, net.out_features, transfer_model_class_alias)
        model_reload.load_state_dict_model(cur_save_model_path)
    return model_reload


if __name__ == '__main__':
    # 1. Transfer settings
    features_net_name = 'Net_not'
    classifier_net_name = 'Net_step'
    transfer_model_class_alias = 'Net_not_AD'
    # features_net_abspath = 'E:\\Users\\zhlou\\PycharmProjects\\ALM\\scripts\\feature_fitting_layers\\Net_not.py'
    # classifier_net_abspath = 'E:\\Users\\zhlou\\PycharmProjects\\ALM\\scripts\\digital_layers\\Net_step.py'
    build_new_model = True
    # 2. Transfer models
    transfered_model = transfer_models(features_net_name, classifier_net_name, transfer_model_class_alias,
                                       # features_net_abspath, classifier_net_abspath,
                                       build_new_model=build_new_model)
    # test for transfered_model
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
    y_target = Variable(torch.Tensor(np.array(label))).float()
    y_target = y_target.view(x_input.shape[0], -1)
    # predict test
    y_pred = transfered_model.forward(x_input)
    y_pred_array = np.array(y_pred.detach().float().numpy())
    y_target_array = np.array(y_target.numpy())
    print(sum(y_pred_array == y_target_array))
    print(transfered_model.state_dict())
    print(transfered_model.__dict__)
