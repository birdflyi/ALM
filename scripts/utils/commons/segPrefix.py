#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6

__author__ = ''
__time__ = ''


# split表示分隔符, 返回值为(prefix, subfix), pos确定从第几个查找到的分隔符分割, 默认pos=1，按第1个分隔符分割
def segPrefix(s, split='-', pos=1):
    pos = pos if pos >= 0 else 0
    seg_list = s.split(sep=split, maxsplit=-1)  # s:'a-b-c', seg_list: ['a', 'b', 'c']
    prefix = split.join(seg_list[0:pos])  # prefix: 'a'
    subfix = split.join(seg_list[pos:len(seg_list)])  # subfix: 'b-c'
    return prefix, subfix


def main():
    for i in range(4):
        p, s = segPrefix('a-b-c', pos=i)
        print(p, 'x', s)
    for i in range(4):
        p, s = segPrefix('a-b-c', pos=-i)
        print(p, 'x', s)


if __name__ == '__main__':
    main()
