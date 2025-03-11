# -*- coding: utf-8 -*-
# @File Name:     agent_logger
# @Author :       Jun
# @Date:          2025/1/9
# @Description :
from pathlib import Path


class MarkdownLogger:
    def __init__(self, name: str):
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)

        self.instruction = None
        self.path = log_dir / name
        self.plan = None
        self.writing_steps = []
        self.result = None

    def add_instruction(self, instruction: str):
        self.instruction = instruction

    def add_plan(self, plan: str):
        self.plan = plan

    def add_qa(self, prompt, assistant):
        self.writing_steps.append({"prompt": prompt, "assistant": assistant})

    def add_result(self, result):
        self.result = result

    @staticmethod
    def _code_area(text):
        text = text.replace("```", "---")
        return f"```text\n{text}\n```"

    def dump(self):
        lines = ["## Instruction", self._code_area(self.instruction)]

        if self.plan:
            lines.extend(["## Plan", self._code_area(self.plan)])

        if self.writing_steps:
            lines.append("## Writing Steps")
            for idx, step in enumerate(self.writing_steps, start=1):
                lines.extend([f"### Prompt {idx}", self._code_area(step["prompt"]), f"### Assistant {idx}", self._code_area(step["assistant"])])

        lines.extend(
            ["## Result", self._code_area(self.result)]
        )

        self.path.write_text("\n\n".join([l.strip() for l in lines]))
