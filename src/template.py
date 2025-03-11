# -*- coding: utf-8 -*-
# @File Name:     template
# @Author :       Jun
# @date:          2024/12/10
# @Description :  build prompt from template
import re
from pathlib import Path
from jinja2 import Template

def _init_template_dict():
    template_dir_path = Path(__file__).parent / "template"
    template_dict = {}
    for file_path in template_dir_path.iterdir():
        if file_path.is_file():
            content = file_path.read_text(encoding="utf-8")
            template_dict[file_path.stem] = Template(content)
    return template_dict


def build_prompt(template: str, **kwargs):
    prompt = _template_dict[template]
    prompt = prompt.render(**kwargs)
    return prompt


_template_dict = _init_template_dict()
