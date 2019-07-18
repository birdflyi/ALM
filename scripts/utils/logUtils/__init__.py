#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
from scripts.utils.logUtils import loadLogConfig

__all__ = ['initLogDefaultConfig']
__author__ = 'Lou Zehua'
__time__ = '2018/9/21 20:18'


def initLogDefaultConfig():
    loadLogConfig.setup_logging()
