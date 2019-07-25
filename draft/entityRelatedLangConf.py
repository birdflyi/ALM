#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
from scripts.utils.commons import SingletonType

__author__ = 'Lou Zehua'
__time__ = '2018/9/30 16:11'


# Default settings of languages related by entities
# 获取变量名列表方法 LangSet().__VARS__
class LangSet(metaclass=SingletonType):
    ZH = 'zh-cn'
    EN = 'en-us'
    __VARS__ = []

    def __init__(self):
        items = LangSet.__dict__.items()
        for varName, varValue in items:
            if not varName.startswith('__'):
                self.__VARS__.append(varValue)
