#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import re

__author__ = 'Lou Zehua'
__time__ = '2019/4/9 17:24'


def re_recursive(src_data, re_str, dst_str):
    dst_data = re.sub(re_str, dst_str, src_data)
    cnt = re.subn(re_str, dst_str, src_data)[-1]
    if cnt == 0:
        return dst_data
    else:
        return re_recursive(dst_data, re_str, dst_str)