# -*- coding: utf-8 -*-
# @File Name:     naive
# @Author :       Jun
# @Date:          2025/1/16
from datetime import datetime

from loguru import logger
from openai.types.chat import ChatCompletion
from tqdm import tqdm

from agent.base import BaseAgent
from agent_logger import MarkdownLogger
from engine import generate, compose_chat_completion
from template import build_prompt


class ConcatAgent(BaseAgent):

    def write(self, instruction: str, model: str, task_id=None, log: bool = False) -> ChatCompletion:
        if task_id is None:
            task_id = datetime.now().strftime("%Y%m%d-%H%M%S")

        md_logger = MarkdownLogger(name=f"{task_id}.md") if log else None

        logger.info("Instruction: {}", instruction)

        plan_prompt = build_prompt(template="plan", instruction=instruction)
        plan = generate(prompt=plan_prompt, model=model, stream=False).choices[0].message.content

        logger.info("Plan: {}", plan)

        if log:
            md_logger.add_instruction(instruction)
            md_logger.add_plan(plan)

        steps = plan.split("\n")
        steps = [step.strip() for step in steps if step.strip()]
        written = ""
        completions = []

        if self._tqdm:
            loop = tqdm(enumerate(steps), total=len(steps), desc="Writing", leave=False)
        else:
            loop = enumerate(steps)

        for idx, step in loop:
            if idx == 0:
                prompt = build_prompt(template="write_begin", instruction=instruction, plan=plan, step=step)
            else:
                prompt = build_prompt(template="write_step", instruction=instruction, plan=plan, step=step, text=written)

            completion = generate(prompt=prompt, model=model, stream=False)
            completions.append(completion)
            written += "\n" + completion.choices[0].message.content

            if log:
                md_logger.add_qa(prompt=prompt, assistant=completion.choices[0].message.content)

        completion = compose_chat_completion(completions)
        if log:
            md_logger.add_result(completion.choices[0].message.content)
            md_logger.dump()
        return completion
