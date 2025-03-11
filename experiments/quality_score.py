# -*- coding: utf-8 -*-
# @File Name:     evaluate
# @Author :       Ye
# @Date:          2025/01/22
# @Description :
import json
import os
import sys
import time
from pathlib import Path

from jinja2 import Template

import dotenv
import openai
from tqdm import tqdm

dotenv.load_dotenv()

JUDGE_MODEL_URL = os.getenv("JUDGE_MODEL_URL")
JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME")
JUDGE_MODEL_API_KEY = os.getenv("JUDGE_MODEL_API_KEY")


def load_checklist(checklist_path: str) -> dict[str, list]:
    assert os.path.exists(checklist_path), f"{checklist_path} does not exist"
    try:
        with open(checklist_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(e)
        sys.exit(1)


def prepare_prompt(prompt_path: str, data_path: str, checklist: list[dict], response) -> str:
    assert os.path.exists(prompt_path), f"{prompt_path} does not exist"
    papers = []
    for file in os.listdir(data_path):
        with open(os.path.join(data_path, file), "r", encoding="utf-8") as f:
            papers.append(f.read())

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()

    return Template(prompt).render(checklists=json.dumps(checklist), response=response)


def load_responses(responses_path: str):
    assert os.path.exists(responses_path), f"{responses_path} does not exist"
    responses = {}
    if os.path.isdir(responses_path):
        for file in os.listdir(responses_path):
            with open(os.path.join(responses_path, file), "r", encoding="utf-8") as f:
                responses[file] = json.load(f)["response"]
    elif os.path.isfile(responses_path):
        with open(responses_path, "r", encoding="utf-8") as f:
            responses[os.path.basename(responses_path)] = json.load(f)["response"]

    return responses


def llm_as_judge_api(data_path: str, prompt_path: str, checklist_path: str, response_path: str, result_path: str):
    client = openai.OpenAI(api_key=JUDGE_MODEL_API_KEY, base_url=JUDGE_MODEL_URL)

    checklist = load_checklist(checklist_path)
    response_dict = load_responses(response_path)

    for response_file in tqdm(response_dict.keys(), desc=f"Evaluating for survey_gen"):
        output_path = f"{result_path}/{response_file}_results.jsonl"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        response_data = response_dict[response_file]
        result_dict = {}
        for key in checklist.keys():
            print("key:", key)
            checks = checklist[key]
            checklist_questions = [{"checklist_id": i, "checklist_content": q} for i, q in enumerate(checks)]
            prompt = prepare_prompt(prompt_path=prompt_path, data_path=data_path, checklist=checklist_questions, response=response_data)

            messages = [
                {
                    "role": "system",
                    "content": """You are a helpful evaluator. Your task is to evaluate the checklists of the responses given by the Large Language Models (LLMs) based on user instructions. These checklists consist of yes or no questions."""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            llm_judge_response_list = []
            for _ in range(3):
                try:
                    completion = client.chat.completions.create(
                        model=JUDGE_MODEL_NAME,
                        messages=messages,
                        temperature=0.8
                    )

                    llm_judge_response = completion.choices[0].message.content
                    llm_judge_response = (
                        llm_judge_response
                        .replace("```json", "")
                        .replace("```python", "")
                        .replace("```", "")
                        .replace("\n", "")
                        .replace("\\", "")
                    )
                    llm_judge_response_list = json.loads(llm_judge_response)

                    # Ensure the number of checklist items matches the model response
                    assert len(llm_judge_response_list) == len(checklist_questions)
                    break
                except Exception as e:
                    print(e)
                    time.sleep(3)
                    continue
            result_dict[key] = llm_judge_response_list

        # Save the evaluation results to the output file
        with open(output_path, "a", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=4)
            f.write("\n")


def main(data_path: str, prompt_path: str, checklist_path: str, responses_path: str, result_path: str):
    llm_as_judge_api(data_path=data_path, prompt_path=prompt_path, checklist_path=checklist_path,
                     response_path=responses_path, result_path=result_path)


if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent
    groups = [d.name for d in (root_dir / "dataset/papers/cleaned").iterdir()]
    exp_name = "qwen2.5-14b-4k"
    agent_name = "single"

    for group in groups:
        main(
            data_path=str(root_dir / f"dataset/papers/cleaned/{group}"),
            prompt_path=str(root_dir / "dataset/prompt/checklist.jinja"),
            checklist_path=str(root_dir / "dataset/checklist/checklist.json"),
            responses_path=str(root_dir / f"results/pred/{exp_name}/{agent_name}"),
            result_path=str(root_dir / f"results/quality/{exp_name}/{agent_name}/{group}.jsonl")
        )
