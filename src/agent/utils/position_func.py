# -*- coding: utf-8 -*-
# @File Name:     position_func
# @Author :       Jun
# @Date:          2025/1/23

import numpy as np


def exp_func(x, a, b):
    return np.fabs(b * np.power(2 * (x - 0.5), a))


def position_loss(doc_id: int, length: int, a, b) -> float:
    return exp_func(doc_id / length, a, b)
