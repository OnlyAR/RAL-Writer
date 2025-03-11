# -*- coding: utf-8 -*-
# @File Name:     base
# @Author :       Jun
# @Date:          2025/1/23

class BaseAgent:
    def __init__(self, tqdm: bool = False):
        self._tqdm = tqdm

    def write(self, instruction: str, model: str, task_id=None, log: bool = False):
        raise NotImplementedError
