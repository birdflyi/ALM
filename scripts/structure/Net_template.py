#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import copy
import os

import torch
import torch.nn as nn

from etc import filePathConf, extensions
from etc.model_registry_settings import df_registry_col_names, CLASS_ALIAS, REL_CALLER_PYFILE_PATH
from etc.profiles import BASE_DIR
from etc.training_purposes import training_purposes, R_ANALOG
from scripts import df_registry, model_registry_path
from scripts.utils.commons.transfer_modulePath_filePath import path_File2Module
from scripts.utils.stringUtils.replace_chars import replace_chars

__author__ = 'Lou Zehua'
__time__ = '2019/7/25 16:39'


class Net_template(nn.Module):

    models_save_dir = {
        extensions.EXT_MODELS__STATE_DICT: filePathConf.absPathDict[filePathConf.MODELS_STATE_DICT_DIR],
        extensions.EXT_MODELS__WHOLE_NET_PARAMS: filePathConf.absPathDict[filePathConf.MODELS_WHOLE_NET_PARAMS_DIR]
    }

    def __init__(self, in_features, out_features, class_alias):
        super(Net_template, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_atomic = False  # Only some specific nets build manually should be set to True.
        self._purpose = ''  # It is important for Net_transfer and atomic models.
        self.name = self.__str__()
        self.class_alias = class_alias or self.name.split('(')[0]
        self._save_mode = extensions.EXT_MODELS__STATE_DICT
        self.save_model_path = ''  # There's only 1 param to store path, thus save and load should be called in pairs.
        self._save_pyfile_name = self.class_alias
        self.caller_pyfile_abspath = os.path.abspath(__file__)
        self.caller_pyfile_relpath = ''
        self.set_caller_pyfile_path(self.caller_pyfile_abspath)
        self.save_model_name = ''
        self.reset_save_model_name()  # init self.save_model_name
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
        return self.caller_pyfile_abspath

    def set_caller_pyfile_path(self, path):
        self.caller_pyfile_abspath = path
        # set caller_pyfile_relpath based on caller_pyfile_abspath
        caller_pyfile_relpath = self.caller_pyfile_abspath.replace('\\', '/')
        caller_pyfile_relpath = caller_pyfile_relpath.replace(BASE_DIR.replace('\\', '/'), '')
        caller_pyfile_relpath = caller_pyfile_relpath.strip('/')
        self.caller_pyfile_relpath = caller_pyfile_relpath

    def check_purpose(self):
        purpose = training_purposes[R_ANALOG]  # Default set as R_ANALOG
        if self.net_sequence.__len__():
            if hasattr(self.net_sequence[-1], '_purpose'):
                purpose = self.net_sequence[-1].get_purpose()
        self.set_purpose(purpose)

    # Recall this function after updating net_sequence.
    def summary(self):
        net_flow_process = []
        net_dependent = {}
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

    def set_save_mode(self, save_mode=None):
        """
        Set save_mode=None if you want to keep save_mode the default value: extensions.EXT_MODELS__STATE_DICT
        :param save_mode: A value belongs to list extensions.ext_models.keys()
        :return: None
        """
        if save_mode and save_mode not in extensions.ext_models.keys():
            raise Warning('The save_mode of the model must be a key of {}.'.format(extensions.ext_models.keys()))
        self._save_mode = save_mode or self._save_mode

    def save_model(self, save_mode=None):
        self.auto_set_save_model_path(save_mode)
        if self._save_mode == extensions.EXT_MODELS__STATE_DICT:
            self._save_state_dict_model()
        elif self._save_mode == extensions.EXT_MODELS__WHOLE_NET_PARAMS:
            self._save_whole_model()

    def load_model(self):
        self.auto_set_save_model_path()
        model = self
        if self._save_mode == extensions.EXT_MODELS__STATE_DICT:
            model = self.load_state_dict_model()
        elif self._save_mode == extensions.EXT_MODELS__WHOLE_NET_PARAMS:
            model = self.load_whole_model()
        return model

    def _save_state_dict_model(self, path=None):
        if path:
            self.save_model_name = path.replace('\\', '/').split('/')[-1].split('.')[0]
            self.save_model_path = path
        else:
            save_dir = filePathConf.absPathDict[filePathConf.MODELS_STATE_DICT_DIR]
            self.set_save_model_path(save_dir, extensions.EXT_MODELS__STATE_DICT)
        torch.save(self.state_dict(), self.save_model_path)

    def _save_whole_model(self, path=None):
        if path:
            self.save_model_name = path.replace('\\', '/').split('/')[-1].split('.')[0]
            self.save_model_path = path
        else:
            save_dir = filePathConf.absPathDict[filePathConf.MODELS_WHOLE_NET_PARAMS_DIR]
            self.set_save_model_path(save_dir, extensions.EXT_MODELS__WHOLE_NET_PARAMS)
        torch.save(self, self.save_model_path)

    def load_state_dict_model(self, path=None):
        model = self
        self.save_model_path = path or self.save_model_path
        model.load_state_dict(torch.load(self.save_model_path))
        return model

    def load_whole_model(self, path=None):
        self.save_model_path = path or self.save_model_path
        model = torch.load(self.save_model_path)
        model.eval()
        return model

    def auto_set_save_model_path(self, save_mode=None):
        self.set_save_mode(save_mode)
        if self._save_mode in self.models_save_dir.keys():
            self.set_save_model_path(save_dir=self.models_save_dir[self._save_mode], save_mode=self._save_mode)
        else:
            msg = 'Unknown save mode! Please set the save_model_path manually later.'
            raise Warning(msg)

    def set_save_model_path(self, save_dir, save_mode):
        if self._purpose:
            self.save_model_path = os.path.join(save_dir, self._purpose,
                                                self.save_model_name + extensions.ext_models[save_mode])
        else:
            msg = 'You must decide whether the model is training for regression (with a continuous output) ' \
                  'or classification (with a discrete output). Set purposes to any one of that with set_purpose.'
            raise Warning(msg)

    def reset_save_model_name(self):
        self.save_model_name = replace_chars(self.name, [' ', '=', ')', '\n', '_features'], '')
        self.save_model_name = self.save_model_name.replace('(', '__').replace(',', '_')

    def get_net_import_str(self):
        import_module_path = path_File2Module(self.get_caller_pyfile_path(), absfilepath=True)
        import_fmt_str = 'from {} import {}'.format(import_module_path, self.class_alias)
        return import_fmt_str

    # register net base on the relative path.
    def register_net(self):
        col_name__class_alias = df_registry_col_names[CLASS_ALIAS]
        col_name__rel_caller_pyfile_path = df_registry_col_names[REL_CALLER_PYFILE_PATH]
        temp_record = {
            col_name__class_alias: self.class_alias,
            col_name__rel_caller_pyfile_path: self.caller_pyfile_relpath
        }
        # delete the obsoleted records: each py file should have a unique class alias.
        if self.is_registered():
            registered_ids = df_registry[df_registry[col_name__rel_caller_pyfile_path] ==
                                         self.caller_pyfile_relpath].index.tolist()
            df_registry_del_conflicts = df_registry.drop(list(registered_ids))
        else:
            df_registry_del_conflicts = df_registry
        # add new records for current py file
        df_registered = df_registry_del_conflicts.append(temp_record, ignore_index=True)
        df_registered.to_csv(model_registry_path, sep=',', header=None, index=False)

    def unregister_net(self):
        col_name__rel_caller_pyfile_path = df_registry_col_names[REL_CALLER_PYFILE_PATH]
        if self.is_registered():
            registered_ids = df_registry[df_registry[col_name__rel_caller_pyfile_path] ==
                                         self.caller_pyfile_relpath].index.tolist()
            df_registry_unregistered = df_registry.drop(list(registered_ids))
            df_registry_unregistered.to_csv(model_registry_path, sep=',', header=None, index=False)

    def is_registered(self):
        # the identifier is field "rel_caller_pyfile_path"
        registered_rel_caller_pyfile_path = list(df_registry[df_registry_col_names[REL_CALLER_PYFILE_PATH]])
        return self.caller_pyfile_relpath in registered_rel_caller_pyfile_path

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
