# -*- coding: utf-8 -*-
# @File Name:     consistency_score
# @Author :       Jun
# @Date:          2025/2/4
import asyncio
import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from jinja2 import Template
from openai import AsyncOpenAI
from pydantic import BaseModel

root_path = Path(__file__).parent.parent
gpt_qa_template_path = root_path / "dataset" / "prompt" / "gpt_qa.jinja"
judge_qa_template_path = root_path / "dataset" / "prompt" / "judge_qa.jinja"

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"EvalLog/logfile_{time.strftime('%Y-%m-%d_%H_%M_%S')}.log", encoding="utf-8")
    ]
)

client = AsyncOpenAI(
    base_url=os.environ["JUDGE_MODEL_BASE_URL"],
    api_key=os.environ["JUDGE_MODEL_API_KEY"],
    max_retries=100,
    timeout=100.0
)


class JUDGE(BaseModel):
    score: float


exp_name = "qwen2.5-14b-4k"
task = "combine"  # single or combine

assert task in ["single", "combine"], "task must be single or combine"

gpt_qa_template = Template(gpt_qa_template_path.read_text(encoding="utf-8"))
judge_qa_template = Template(judge_qa_template_path.read_text(encoding="utf-8"))

pred_dir = root_path / "results" / "pred" / exp_name
agent_names = ['compress', 'single', 'concat', 'restate-a_60-b_0.3-k_12']

result_dir = root_path / "results" / "qa" / "judge" / exp_name / task
papers_root_dir = root_path / "dataset" / "papers" / "cleaned"
question_dir = root_path / "dataset" / "questions" / task


async def async_chat(messages: List[List[Dict[str, str]]], **kwargs) -> List[str]:
    response = await asyncio.gather(*(async_get_api_reply(message, **kwargs) for message in messages))
    return response


total_prompt_tokens = 0
total_completion_tokens = 0


async def async_get_api_reply(messages, is_json=False, is_structure_output=False, response_format=None) -> str:
    global total_prompt_tokens, total_completion_tokens
    if is_json:
        result = await client.chat.completions.create(
            messages=messages,
            model=os.getenv("JUDGE_MODEL_NAME"),
            temperature=0.00,
            response_format={
                'type': 'json_object'
            }
        )
        logging.info(result)
        logging.error(f"return element is:{result}")
        logging.error(f"usage messages is {result.usage}")
        total_completion_tokens += result.usage.completion_tokens
        total_prompt_tokens += result.usage.prompt_tokens

        return result.choices[0].message.content
    elif is_structure_output:
        result = await client.beta.chat.completions.parse(
            messages=messages,
            temperature=0.00,
            model=os.getenv("JUDGE_MODEL_NAME"),
            response_format=response_format
        )
        logging.error(f"return element is:{result}")
        logging.error(f"usage messages is {result.usage}")
        total_completion_tokens += result.usage.completion_tokens
        total_prompt_tokens += result.usage.prompt_tokens

        return result.choices[0].message.parsed
    else:
        result = await client.chat.completions.create(
            messages=messages,
            temperature=0.00,
            model=os.getenv("JUDGE_MODEL_NAME"),
        )
        logging.error(f"return element is:{result}")
        logging.error(f"usage messages is {result.usage}")
        total_completion_tokens += result.usage.completion_tokens
        total_prompt_tokens += result.usage.prompt_tokens
        return result.choices[0].message.content


def get_paths(agent_name):
    pred_result_dir = Path(f"{pred_dir}/{agent_name}")
    judge_result_dir = Path(f"{result_dir}/{agent_name}")
    judge_result_dir.mkdir(parents=True, exist_ok=True)
    return pred_result_dir, judge_result_dir


def build_paper_question_dict(pred_result_dir):
    logging.info(f"pred result dir: {pred_result_dir}")
    group_names = os.listdir(pred_result_dir)
    logging.info(f"Group names: {group_names}")

    group_names = [group_name.split('.')[0] for group_name in group_names]
    all_question_list = []
    for group_name in group_names:
        # paper_id_dict = {}
        logging.info(f"Group name: {group_name}")
        logging.info(f"Paper root dir: {papers_root_dir}")
        paper_list = os.listdir(Path(papers_root_dir) / Path(group_name))

        logging.info(f"paper list for group {group_name} :{paper_list}")
        question_files = [f"{question_dir}/{group_name}.json"]

        logging.info(f"question files: {question_files}")
        group_questions = []
        for question_file in question_files:
            logging.info(f"question file is {question_dir}/{question_file}")
            with open(question_file, "r", encoding="utf-8") as f:
                questions = json.load(f)
                group_questions.extend(questions)
        logging.info(f"group questions number of {group_name} is {len(group_questions)}")
        all_question_list.append((group_name, group_questions))
    return all_question_list


async def paper_qa_judge(paper_item, question_item) -> dict:
    paper = paper_item["response"]
    question = question_item["question"]
    answer = question_item["answer"]
    question_paper = question_item["paper"]
    topic = question_item["topic"]
    subtopic = question_item["subtopic"]
    while True:
        try:
            qa_instruction = gpt_qa_template.render(paper=paper, question=question)
            msg = []
            msg.append([{"role": "user", "content": qa_instruction + "If You can't find the answer, please respond with 'I don't know"}])

            logging.error(f"processing group: {paper_item['group_id']}")
            preds = await async_chat(msg)
            pred = preds[0]
            logging.error(f"pred is {pred}")
            judge_instruction = judge_qa_template.render(
                question=question,
                answer=answer,
                predict=pred
            )

            msg = []
            msg.append([{"role": "user", "content": judge_instruction}])
            model_judges = await async_chat(msg, is_json=False, is_structure_output=True, response_format=JUDGE)
            logging.error(f"judges is {model_judges}")
            model_judge = model_judges[0]
            logging.error(f"judge is {model_judge}")
            judge = {}
            judge["score"] = float(model_judge.score)
            judge.update({
                "paper": question_paper,
                "topic": topic,
                "subtopic": subtopic,
                "question": question,
                "answer": answer,
                "pred": pred
            })
            return judge
        except Exception:
            traceback.logging.info_exc()


async def paper_task(paper_item, question_list, task_name, judge_result_dir):
    try:
        # paper_item["judge"] = []
        score_list = []
        judges = []
        # for question in tqdm(question_list, desc=f"{task_name}"):
        batch_size = 1
        for i in range(0, len(question_list), batch_size):
            batch_questions = question_list[i:i + batch_size]
            tasks = [paper_qa_judge(paper_item, question) for question in batch_questions]
            batch_judges = await asyncio.gather(*tasks)

            for judge in batch_judges:
                if judge:
                    # judge = await paper_qa_judge(paper_item, question)
                    # paper_item["judge"].append(judge)
                    score_list.append(judge["score"])
                    judges.append(judge)

            # paper_item["average_score"] = sum(score_list) / len(score_list)

            with open(judge_result_dir / f"{task_name}.json", "w", encoding="utf-8") as f:
                json.dump(judges, f, indent=2, ensure_ascii=False)

        # logging.info(f"Saved: {judge_result_dir / f'{task_name}.json'}")
    except Exception:
        traceback.logging.info_exc()


async def judge_workflow():
    max_task_num = 5
    for i in range(0, len(agent_names), max_task_num):
        batch_agent_names = agent_names[i:i + max_task_num]
        batch_tasks = []
        for agent_name in batch_agent_names:
            pred_result_dir, judge_result_dir = get_paths(agent_name)
            all_questions_list = build_paper_question_dict(pred_result_dir)

            for group_name, group_questions in all_questions_list:
                paper_item_path = pred_result_dir / f"{group_name}.json"
                paper_item = json.load(paper_item_path.open("r", encoding="utf-8"))
                batch_tasks.append(paper_task(paper_item, group_questions, group_name, judge_result_dir))
        await asyncio.gather(*batch_tasks)


if __name__ == "__main__":
    asyncio.run(judge_workflow())
    logging.error(f"total prompt tokens is {total_prompt_tokens}")
    logging.error(f"total completion tokens is {total_completion_tokens}")
