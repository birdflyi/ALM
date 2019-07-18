#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import threading

__author__ = 'Lou Zehua'
__time__ = '2018/10/16 14:05'


# 单例模式
class SingletonType(type):
    _instance_lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            with SingletonType._instance_lock:
                if not hasattr(cls, "_instance"):
                    cls._instance = super(SingletonType,cls).__call__(*args, **kwargs)
        return cls._instance