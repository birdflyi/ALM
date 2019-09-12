#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import importlib
import logging

from etc import extensions
from etc.model_registry_settings import df_registry_col_names, CLASS_ALIAS, REL_CALLER_PYFILE_PATH
from scripts import df_registry
from scripts.structure.Net_template import Net_template
from scripts.utils import logUtils
from scripts.utils.commons.dict_update import dict_update_left_merge_recursive
from scripts.utils.commons.transfer_modulePath_filePath import path_File2Module

__author__ = 'Lou Zehua'
__time__ = '2019/7/28 11:55'

# init log default config
logUtils.initLogDefaultConfig()
logger = logging.getLogger(__name__)

class Net_update_class_dict(Net_template):
    """
    Update a Net_class when the common class Net_template got modified.
    """

    def __init__(self, src_model, default_save_mode_ref=None):
        if not isinstance(src_model, Net_template):
            raise TypeError('Parameter type error: Type {} is not a subclass of {}'.format(
                src_model.__class__, Net_template.__class__))
        super(Net_update_class_dict, self).__init__(src_model.in_features, src_model.out_features, None)
        self.default_save_mode_ref = default_save_mode_ref or extensions.EXT_MODELS__STATE_DICT
        self.src_model = src_model
        self._init_src_model()
        self.model = self.src_model.__class__(self.src_model.in_features, self.src_model.out_features,
                                              self.src_model.class_alias)
        self._transfer_updated_model()

    def _init_src_model(self):
        self.src_model.set_save_mode(self.default_save_mode_ref)
        self.src_model = self.src_model.load_model()

    def _transfer_updated_model(self):
        dest_model_updated_dict = dict_update_left_merge_recursive(self.model.__dict__, self.src_model.__dict__)
        self.model.__dict__.update(dest_model_updated_dict)

    def update_models(self):
        for temp_save_mode in extensions.ext_models.keys():
            self.model.set_save_mode(temp_save_mode)
            self.model.save_model()


if __name__ == '__main__':
    col_name__class_alias = df_registry_col_names[CLASS_ALIAS]
    col_name__rel_caller_pyfile_path = df_registry_col_names[REL_CALLER_PYFILE_PATH]
    # Update net registered in model_registry.csv when the common class Net_template got modified.
    for index, row in df_registry.iterrows():
        net_class_alias = row[col_name__class_alias]
        net_rel_caller_pyfile_path = row[col_name__rel_caller_pyfile_path]
        # Auto import python modules
        module_path = path_File2Module(net_rel_caller_pyfile_path, absfilepath=False)
        pymodule = importlib.import_module(module_path)
        # Init models
        cur_model = getattr(pymodule, net_class_alias)()
        net_update_tool = Net_update_class_dict(cur_model)
        net_update_tool.update_models()
        model_updated = net_update_tool.model
        logger.info(model_updated.state_dict())
        logger.info(model_updated.__dict__)
