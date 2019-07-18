#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import itertools
import jellyfish

__author__ = 'Lou Zehua'
__time__ = '2018/9/29 9:46'

# 当匹配成功时，先按dist1判定哪个更相似，再按dist2判定哪个更相似
def _loc_char_match_dist(s1, s2):
    len1 = len(s1)
    len2 = len(s2)
    res1 = []
    res2 = []

    # length differs by 9 or more, no result
    if abs(len1-len2) >= 9:
        return None

    dist1 = jellyfish.levenshtein_distance(''.join(s1), ''.join(s2))

    # get minimum rating based on sums of codexes
    lensum = len1 + len2
    if lensum <= 4:
        min_rating = 5
    elif lensum <= 7:
        min_rating = 4
    elif lensum <= 11:
        min_rating = 3
    else:
        min_rating = 2

    # strip off common prefixes
    for c1, c2 in itertools.zip_longest(s1, s2):
        if c1 != c2:
            if c1:
                res1.append(c1)
            if c2:
                res2.append(c2)
    unmatched_count1 = unmatched_count2 = 0
    for c1, c2 in itertools.zip_longest(reversed(res1), reversed(res2)):
        if c1 != c2:
            if c1:
                unmatched_count1 += 1
            if c2:
                unmatched_count2 += 1
    dist2 = max(unmatched_count1, unmatched_count2)
    return (6 - dist1) >= min_rating, dist1, dist2


def loc_char_match_prob(word1, word2):
    if word1 is word2:
        return 1.0
    lenMax = max(len(word1), len(word2))
    lenMin = min(len(word1), len(word2))
    if lenMin == 0:
        return 0.0

    if _loc_char_match_dist(word1, word2) is None:
        list_1 = [False, lenMax, lenMax]
    else:
        list_1 = list(_loc_char_match_dist(word1, word2))
    # print('list0',list_1)
    if list_1[1] > lenMax:
        list_1[1] = lenMax
    if list_1[2] > lenMax:
        list_1[2] = lenMax
    if list_1[1] > list_1[2]:
        tempNum = list_1[1]
        list_1[1] = list_1[2]
        list_1[2] = tempNum

    alpha = 0.9
    # print('match_rating_comparison:', list_1)
    # print(1 - list_1[2]/lenMax)
    # print(((1 - list_1[2]/lenMax) + (list_1[2] - list_1[1])/lenMax * alpha) + list_1[0] * (1 - alpha))
    return ((1 - list_1[2]/lenMax) + alpha * (list_1[2] - list_1[1])/lenMax) + list_1[0] * (1 - alpha) * (list_1[2] - list_1[1])/lenMax


def main():
    list1 = [
        ["右臀部处","臀部"],
        ["右臀部","瘢痕"],
        ["右臀部","体表"],
        ["左手","左"],
        ["左手","伤情部位"],
        ["左手","手部"],
        ["左手","手"],
        ["左手","右手"],
    ]
    # for i in range(len(list1)):
    #     print('jaro_winkler:',jellyfish.jaro_winkler(list1[i][0],list1[i][1]))
    # for i in range(len(list1)):
    #     print('levenshtein_distance:',jellyfish.levenshtein_distance(list1[i][0],list1[i][1]))
    # for i in range(len(list1)):
    #     print('damerau_levenshtein_distance',jellyfish.damerau_levenshtein_distance(list1[i][0],list1[i][1]))
    # for i in range(len(list1)):
    #     print('jaro_distance:', jellyfish.jaro_distance(list1[i][0], list1[i][1]))
    # for i in range(len(list1)):
    #     print('hamming_distance:', jellyfish.hamming_distance(list1[i][0], list1[i][1]))
    # for i in range(len(list1)):
    #     print('match_rating_comparison:', jellyfish.match_rating_comparison(list1[i][0], list1[i][1]))

    # print('================================================================')
    # for i in range(len(list1)):
    #     print(loc_char_match_prob(list1[i][0], list1[i][1]))
    print(loc_char_match_prob("左侧横突骨折","左前臂7.0cm×5.0cm瘢痕"))
    print(loc_char_match_prob("左前臂","左前臂"))


if __name__ == '__main__':
    main()