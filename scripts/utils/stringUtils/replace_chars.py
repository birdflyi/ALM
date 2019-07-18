#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6

__author__ = 'Lou Zehua'
__time__ = '2019/4/9 17:25'


def replace_chars(src_data, src_chars, dst_char, mode_recursive=False):
    dst_data = src_data
    for src_char in src_chars:
        dst_data = replace_recursive(dst_data, src_char, dst_char) if mode_recursive else dst_data.replace(src_char, dst_char)
    return dst_data


def replace_recursive(src_data, src_char, dst_char):
    dst_data = src_data.replace(src_char, dst_char)
    if src_data.find(src_char) == -1:
        return dst_data
    else:
        return replace_recursive(dst_data, src_char, dst_char)
