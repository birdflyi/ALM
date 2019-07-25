#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import copy

import torch
import torch.nn as nn

__author__ = 'Lou Zehua'
__time__ = '2019/7/25 16:39'


class Net_template(nn.Module):
    def __init__(self, alias):
        super(Net_template, self).__init__()
        self.is_atomic = False
        self.name = self.__str__()
        self.alias = alias or self.name
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

    def save_state_dict_model(self, path):
        torch.save(self.state_dict(), path)

    def save_whole_model(self, path):
        torch.save(self, path)

    def load_state_dict_model(self, path):
        model = self()
        model.load_state_dict(torch.load(path))
        return model

    def load_whole_model(self, path):
        model = torch.load(path)
        model.eval()
        return model
