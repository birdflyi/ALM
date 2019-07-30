#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import os

from etc import filePathConf, extensions
from scripts.digital_layers.Net_step import Net_step
from scripts.feature_fitting_layers.Net_add import Net_add
from scripts.feature_fitting_layers.Net_and import Net_and
from scripts.feature_fitting_layers.Net_multiply import Net_multiply
from scripts.feature_fitting_layers.Net_not import Net_not
from scripts.feature_fitting_layers.Net_or import Net_or
from scripts.feature_fitting_layers.Net_signal import Net_signal
from scripts.structure.Net_template import Net_template
from scripts.utils.commons.dict_update import dict_update_left_join_recursive

__author__ = 'Lou Zehua'
__time__ = '2019/7/28 11:55'


threshold = 0


# Must be called when update Net_template with option 'dest_model=None'.
class Net_update_class_dict(Net_template):
    def __init__(self, src_model, dest_model=None, class_alias=None, strict=True):
        if strict:  # Recommend strict=True to avoid type error in the future.
            if not isinstance(src_model, Net_template):
                raise TypeError('Parameter type error: Type {} is not a subclass of {}'.format(
                    src_model.__class__, Net_template.__class__))
            elif dest_model:
                if not src_model.in_features == dest_model.in_features or \
                        not src_model.out_features == dest_model.out_features:
                    raise BaseException('Unmatched shape between input model pair.')
        super(Net_update_class_dict, self).__init__(src_model.in_features, src_model.out_features, class_alias)
        self.src_model = src_model
        self.dest_model = dest_model or src_model.__class__(src_model.in_features, src_model.out_features, class_alias)
        self.model = self.get_updated_model(self.src_model, self.dest_model)

    def get_updated_model(self, src_model, dest_model):
        dest_model.__dict__.update(dict_update_left_join_recursive(dest_model.__dict__, src_model.__dict__))
        return dest_model


def update_models(src_model, model_save_path, state_dict_mode=True):
    # 1. Load trained src_model
    if state_dict_mode:
        src_model = src_model.load_state_dict_model(model_save_path)
    else:
        src_model = src_model.load_whole_model(model_save_path)

    # 2. Create a Net_update_class_dict object based on src_model
    net = Net_update_class_dict(src_model).model

    # 3. Save and load model pairs
    net.save_whole_model()
    model_reload = net.load_whole_model()
    net.save_state_dict_model()
    model_reload = net.load_state_dict_model()
    return model_reload


if __name__ == '__main__':
    # 1. Load trained src_model
    src_model = Net_step()
    STATE_DICT_EXT = extensions.ext_models[extensions.EXT_MODELS__STATE_DICT]
    WHOLE_NET_PARAMS_EXT = extensions.ext_models[extensions.EXT_MODELS__WHOLE_NET_PARAMS]
    features_whole_save_path = os.path.join(filePathConf.absPathDict[filePathConf.MODELS_WHOLE_NET_PARAMS_DIR], src_model.get_purpose(), src_model.class_alias + WHOLE_NET_PARAMS_EXT)
    model_reload = update_models(src_model, features_whole_save_path, state_dict_mode=False)
    print(model_reload.state_dict())
    print(model_reload.__dict__)
