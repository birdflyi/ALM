#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6

__author__ = 'Lou Zehua'
__time__ = '2019/7/25 15:39'


class Net_register():
    def __init__(self, obj, re_registration=False):
        self.dict = self.format_dict(obj.__dict__)
        if re_registration or not self.is_registered():
            self.register()

    def is_registered(self):
        return False

    def register(self):
        print('register')

    def format_dict(self, obj_dict):
        return {}

    def get_registered_dict(self):
        self.dict = {}
        return self.dict
