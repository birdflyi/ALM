#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import copy

__author__ = 'Lou Zehua'
__time__ = '2019/7/30 11:44'


def dict_update_union(base_dict, additional_dict):
    temp_base_dict = copy.deepcopy(base_dict)
    temp_base_dict.update(additional_dict)
    return temp_base_dict


def dict_update_left_merge(base_dict, additional_dict):
    reload_part_dict = {k: v for k, v in additional_dict.items()
                         if k in base_dict.keys()}
    temp_base_dict = copy.deepcopy(base_dict)
    temp_base_dict.update(reload_part_dict)
    return temp_base_dict


def dict_update_left_merge_recursive(base_dict, additional_dict):
    temp_base_dict = copy.deepcopy(base_dict)
    for k, v in additional_dict.items():
        if k in base_dict.keys():
            if isinstance(v, dict):  # use isinstance(v, dict) to check class type: OrderedDict etc.
                temp_base_dict[k] = dict_update_left_merge_recursive(temp_base_dict[k], v)
            else:
                temp_base_dict[k] = v
    return temp_base_dict


if __name__ == '__main__':
    temp_base_dict = {
        'x': 0,
    }
    base_dict = {
        'a': 1,
        'b': 2,
        'c': 3,
        'params': temp_base_dict
    }
    temp_dict = {
        'x': 11,
        'y': 12
    }
    additional_dict = {
        'a': 100,
        'aa': 11,
        'bb': 22,
        'cc': 33,
        'params': temp_dict
    }
    dict_result = dict_update_left_merge_recursive(base_dict, additional_dict)
    print(dict_result)
