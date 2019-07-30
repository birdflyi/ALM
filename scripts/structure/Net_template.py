#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import copy
import os

import torch
import torch.nn as nn

from etc import filePathConf, extensions
from etc.training_purposes import training_purposes, R_ANALOG
from scripts.utils.commons.transfer_modulePath_filePath import path_File2Module

__author__ = 'Lou Zehua'
__time__ = '2019/7/25 16:39'


class Net_template(nn.Module):
    def __init__(self, in_features, out_features, class_alias):
        super(Net_template, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_atomic = False  # Only some specific nets build manually should be set to True.
        self._purpose = ''  # It is important for Net_transfer and atomic models.
        self.name = self.__str__()
        self.class_alias = class_alias or self.name.split('(')[0]
        self._save_model_path = ''  # There's only 1 param to store path, thus save and load should be called in pairs.
        self._save_pyfile_name = self.class_alias
        self.caller_pyfile_path = os.path.abspath(__file__)
        self.save_model_name = self.class_alias
        self.net_sequence = nn.Sequential()
        # summary for module: to describe the structure of net.
        self._net_flow_process = []
        self._net_dependent = {}
        self._old_net_flow_process = []
        self._old_net_dependent = {}
        self.summary()

    def get_net_flow_process(self):
        return copy.deepcopy(self._net_flow_process)

    def set_net_flow_process(self, net_flow_process):
        self._old_net_flow_process = copy.deepcopy(self._net_flow_process)
        self._net_flow_process = net_flow_process

    def get_net_dependent(self):
        return copy.deepcopy(self._net_dependent)

    def set_net_dependent(self, net_dependent):
        self._old_net_dependent = copy.deepcopy(self._net_dependent)
        self._net_dependent = net_dependent

    def get_purpose(self):
        return self._purpose

    def set_purpose(self, purpose):
        if not purpose in training_purposes.values():
            msg = 'Set self.purposes to any one of %s. Training for regression (with a continuous output) ' \
                  'or classification (with a discrete output)' % training_purposes
            raise Warning(msg)
        else:
            self._purpose = purpose

    def get_caller_pyfile_path(self):
        return self.caller_pyfile_path

    def set_caller_pyfile_path(self, path):
        self.caller_pyfile_path = path

    def check_purpose(self):
        purpose = training_purposes[R_ANALOG]  # Default set as R_ANALOG
        if self.net_sequence.__len__():
            if hasattr(self.net_sequence[-1], '_purpose'):
                purpose = self.net_sequence[-1].get_purpose()
        self.set_purpose(purpose)

    # Recall this function after updating net_sequence.
    def summary(self):
        net_flow_process = self.get_net_flow_process()
        net_dependent = self.get_net_dependent()
        for item in self.net_sequence:
            item_class_name = item.__str__()
            net_flow_process.append(item_class_name)
            net_dependent[item_class_name] = item
        self.set_net_flow_process(net_flow_process)
        self.set_net_dependent(net_dependent)

    def rebuild_seq_from_summary(self):
        net_sequential = []
        self.net_sequence = nn.Sequential()
        for net_name in self._net_flow_process:
            net_sequential.append(self._net_dependent[net_name])
        self.net_sequence = nn.Sequential(*net_sequential)

    # 1 top layer will be precipitated once.
    def serialize_seq_atomic(self):
        net_flow_process = []
        net_dependent = {}
        for module in self.net_sequence:
            if module is None:
                pass
            elif isinstance(module, Net_template):
                net_flow_process += module._net_flow_process
                net_dependent.update(module._net_dependent)
            elif isinstance(module, nn.Module):
                item_class_name = module.__str__()
                net_flow_process.append(item_class_name)
                net_dependent[item_class_name] = module
            else:
                raise TypeError("{} is not a Module subclass".format(
                    torch.typename(module)))
        self.set_net_flow_process(net_flow_process)
        self.set_net_dependent(net_dependent)
        self.rebuild_seq_from_summary()

    def rollback_seq(self, seq_backup=True):
        '''
        Rollback nn.Sequential() based on old summary of structure.
        :param seq_backup: back up nn.Sequential() or not. Change seq_backup=True if you want to roll back and save the structure of nn.Sequential() to old summary,
            or it will save the current structure to old summary.
        :return:
        '''
        net_flow_process = []
        net_dependent = {}
        if seq_backup:
            for item in self.net_sequence:
                item_class_name = item.__str__()
                net_flow_process.append(item_class_name)
                net_dependent[item_class_name] = item
        else:
            net_flow_process = self.get_net_flow_process()
            net_dependent = self.get_net_dependent()
        # Summary properties will be updated from old summary.
        self._net_flow_process = copy.deepcopy(self._old_net_flow_process)
        self._net_dependent = copy.deepcopy(self._old_net_dependent)
        # Back up current summary to old summary.
        self._old_net_flow_process = net_flow_process
        self._old_net_dependent = net_dependent
        # Rebuild nn.Sequential() from updated summary.
        self.rebuild_seq_from_summary()

    def forward(self, input):
        out = self.net_sequence(input)
        return out

    def save_state_dict_model(self, path=None):
        if path:
            self.save_model_name = path.replace('\\', '/').split('/')[-1].split('.')[0]
            self._save_model_path = path
        else:
            save_dir = filePathConf.absPathDict[filePathConf.MODELS_STATE_DICT_DIR]
            self.reset_save_model_path(save_dir, extensions.EXT_MODELS__STATE_DICT)
        torch.save(self.state_dict(), self._save_model_path)

    def save_whole_model(self, path=None):
        if path:
            self.save_model_name = path.replace('\\', '/').split('/')[-1].split('.')[0]
            self._save_model_path = path
        else:
            save_dir = filePathConf.absPathDict[filePathConf.MODELS_WHOLE_NET_PARAMS_DIR]
            self.reset_save_model_path(save_dir, extensions.EXT_MODELS__WHOLE_NET_PARAMS)
        torch.save(self, self._save_model_path)

    def load_state_dict_model(self, path=None):
        model = self
        self._save_model_path = path or self._save_model_path
        model.load_state_dict(torch.load(self._save_model_path))
        return model

    def load_whole_model(self, path=None):
        self._save_model_path = path or self._save_model_path
        model = torch.load(self._save_model_path)
        model.eval()
        return model

    def reset_save_model_path(self, save_dir, save_mode):
        if self._purpose:
            self._save_model_path = os.path.join(save_dir, self._purpose,
                self.save_model_name + extensions.ext_models[save_mode])
        else:
            msg = 'You must decide whether the model is training for regression (with a continuous output) ' \
                  'or classification (with a discrete output). Set purposes to any one of that with set_purpose.'
            raise Warning(msg)

    def reset_save_model_name(self):
        self.save_model_name = self.class_alias

    def get_net_import_str(self):
        import_module_path = path_File2Module(self.get_caller_pyfile_path(), absfilepath=True)
        import_fmt_str = 'from {} import {}'.format(import_module_path, self.class_alias)
        return import_fmt_str

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
