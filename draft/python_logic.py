#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
from random import random

__author__ = 'Lou Zehua'
__time__ = '2019/7/29 18:17'


def sum_loc(xs):
    return sum(xs)

def step(x):
    return x > 0

def merge_func(f, g, xs):
    return g(f(xs))

x = 1.1
y = -2

print(sum_loc([x, y]))
print(step(sum_loc([x, y])))
print(merge_func(sum_loc, step, [x, y]))
print('-----------------------------------')
def env_diff(xs_2):
    diff = xs_2[1] - xs_2[0]
    return diff

def env_diff_digital(env_diff, step, xs_2):
    return step(env_diff(xs_2))

def execute(f):
    return f()

def init_env():
    return 0.5

def new_env():
    return random()

def conbine(x, y):
    return [x, y]

def diff_threshold():
    return 0.47

def abs_diff(diff):
    return abs(diff)


def init_cnt():
    return 0

def cnt_thrshold():
    return 5

def one():
    return 1

def zero():
    return 0



def pulse(x):
    return int(not x)

def reverse_count(x):
    pass


def save(x):
    global temp   # file or register
    temp = x

def count_step(x):
    return sum_loc(conbine(x, execute(one)))

def duration_threshold():
    return 5

def score(times_to_learn):
    return 1/times_to_learn

save(execute(init_cnt))
print(temp)
def need_to_learn():
    if env_diff_digital(env_diff, step, conbine(diff_threshold(), abs_diff(env_diff(conbine(execute(init_env), execute(new_env)))))):
        print('$', env_diff(conbine(diff_threshold(), abs_diff(env_diff(conbine(execute(init_env), execute(new_env)))))))
        return execute(one)
    elif temp > execute(duration_threshold):  # try something new. always in one same behavior will be punished.
        print('$', env_diff(conbine(diff_threshold(), abs_diff(env_diff(conbine(execute(init_env), execute(new_env)))))))
        return execute(zero)
    save(count_step(temp))
    print('do the same thing...')
    print(temp, env_diff(conbine(diff_threshold(), abs_diff(env_diff(conbine(execute(init_env), execute(new_env)))))))
    return need_to_learn()

print(step(need_to_learn()))
print(score(temp))

def net():
    next(add(way1(step(net_1()))), (way2(not(step(net_1())))))
