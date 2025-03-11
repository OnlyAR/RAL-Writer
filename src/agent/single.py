# -*- coding: utf-8 -*-
# @File Name:     single
# @Author :       Jun
# @Date:          2025/1/23
# @Description :
from datetime import datetime

from loguru import logger

import engine
from agent.base import BaseAgent
from agent_logger import MarkdownLogger


class SingleAgent(BaseAgent):

    def write(self, instruction: str, model: str, task_id=None, log: bool = False):
        if task_id is None:
            task_id = datetime.now().strftime("%Y%m%d-%H%M%S")

        md_logger = MarkdownLogger(name=f"{task_id}.md") if log else None
        logger.info("Instruction: {}", instruction)
        if log:
            md_logger.add_instruction(instruction)
        completion = engine.generate(prompt=instruction, model=model, stream=False, max_tokens=32768)
        # completion = engine.generate(prompt=instruction, model=model, stream=False, max_tokens=8192)
        return completion
