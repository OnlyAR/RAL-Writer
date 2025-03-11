# -*- coding: utf-8 -*-
# @File Name:     pred
# @Author :       Jun
# @Date:          2025/1/17
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from agent.base import BaseAgent

load_dotenv()

from agent.concat import ConcatAgent
from agent.restate import RestateAgent
from agent.single import SingleAgent
from jinja2 import Template
import requests

logger.remove(0)

root_dir = Path(__file__).parent.parent
result_dir = root_dir / "results" / "pred"
instruction_template = Template((root_dir / "dataset/prompt/survey_gen.jinja").read_text(encoding="utf-8"))


def thread_write(agent: BaseAgent, param_dict, agent_name, backbone, result_dir_path):
    try:
        if agent_name == "restate":
            assert isinstance(agent, RestateAgent)
            result_path = result_dir_path / f"{agent_name}-a_{agent._a}-b_{agent._b}-k_{agent._top_k}"
        else:
            result_path = result_dir_path / agent_name
        result_path.mkdir(parents=True, exist_ok=True)
        result_dict = {k: v for k, v in param_dict.items()}
        result_path = result_path / f"{result_dict['group_id']}.json"
        if result_path.exists():
            print(f"Result file {result_path} already exists, skipping...")
            return
        response = agent.write(instruction=param_dict["instruction"], model=backbone, log=False).choices[0].message.content
        result_dict["response"] = response
        result_path.write_text(json.dumps(result_dict, ensure_ascii=False, indent=4), encoding="utf-8")
        print(f"Writing completed, result saved to {result_path}.")
    except Exception as e:
        print(f"Error occurred: {e}")


def get_compressed_papers(papers, rate):
    compressed_papers = []
    for paper in papers:
        try:
            response = requests.post(os.getenv("COMPRESS_URL"), data=json.dumps({
                "prompt": paper,
                "rate": rate
            }), headers={"Content-Type": "application/json"})
            response = response.json()
            compressed_papers.append(response['compressed_prompt'])
        except Exception as e:
            compressed_papers.append(paper)
            print(f"Can't compress papers. Error occurred: {e}.")
    return compressed_papers


def run(agent_name, backbone, dirname, chunk_size=1024, overlap=256, a=5, b=0.25, top_k=10):
    print(f"Agent Name: {agent_name}")
    print(f"Backbone: {backbone}")
    print(f"Dirname: {dirname}")
    print(f"Workers: {workers}")

    dataset_path = root_dir / "dataset/papers/cleaned"
    data_list = [f for f in dataset_path.iterdir()]

    result_dir_path = result_dir / dirname
    length = int(result_dir_path.name.split("-")[-1][:-1]) * 1000
    assert length in [4000, 8000, 16000], f"Invalid length: {length}"

    print(f"Total {len(data_list)} datasets to be processed.")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for data in data_list:
            paper_filenames = [f for f in data.iterdir()]
            papers = [file.read_text(encoding="utf-8") for file in paper_filenames]
            if agent_name == "compress":
                papers = get_compressed_papers(papers, 0.5)

            instruction = instruction_template.render(papers=papers, length=length)
            assert agent_name in ["concat", "restate", "single", "compress"], f"Invalid agent: {agent_name}"

            if agent_name == "concat" or agent_name == "compress":
                write_agent = ConcatAgent(tqdm=False)
            elif agent_name == "restate":
                write_agent = RestateAgent(tqdm=False, chunk_size=chunk_size, overlap=overlap, a=a, b=b, top_k=top_k)
            else:
                write_agent = SingleAgent(tqdm=False)

            param_dict = {
                "instruction": instruction,
                "paper": [paper.name for paper in paper_filenames],
                "group_id": data.name,
            }

            executor.submit(thread_write, write_agent, param_dict, agent_name, backbone, result_dir_path)


if __name__ == '__main__':
    # workers = 12
    workers = 1
    run("single", dirname="qwen2.5-14b-4k", backbone="qwen2.5-14b-instruct")
    # run("concat", dirname="qwen2.5-14b-4k", backbone="qwen2.5-14b-instruct")
    # run("restate", a=60, b=0.3, top_k=12, dirname="qwen2.5-14b-4k", backbone="qwen2.5-14b-instruct")
    # run("compress", dirname="qwen2.5-14b-4k", backbone="qwen2.5-14b-instruct")
    print("All tasks completed.")
