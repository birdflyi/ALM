#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6

__author__ = 'Lou Zehua'
__time__ = '2018/10/12 17:49'


# Return True if any word in words is not empty and is a substring of sentString
def isAnyWordWithinStr(words, sentString):
    # return any(word and word in sentString for word in words)
    for word in words:
        if word and word in sentString:
            return True
    return False