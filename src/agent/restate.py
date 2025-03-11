# -*- coding: utf-8 -*-
# @File Name:     naive
# @Author :       Jun
# @Date:          2025/1/16
# @Description :
from datetime import datetime

import numpy as np
import tiktoken
from loguru import logger
from openai.types.chat import ChatCompletion
from tqdm import tqdm

import engine
from agent.base import BaseAgent
from agent.utils.splitter import LaTeXSplitter
from agent_logger import MarkdownLogger
from template import build_prompt

encoder = tiktoken.encoding_for_model("gpt-4o")


class RestateAgent(BaseAgent):
    def __init__(self, chunk_size, overlap, a, b, top_k, tqdm=False):
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._a = a
        self._b = b
        self._top_k = top_k
        super().__init__(tqdm=tqdm)

    @staticmethod
    def _similarity_score(doc: np.ndarray, query: np.ndarray) -> float:
        return np.dot(doc, query) / (np.linalg.norm(doc) * np.linalg.norm(query))

    @staticmethod
    def _exp_func(x, a, b):
        return np.fabs(b * np.power(2 * (x - 0.5), a))

    @staticmethod
    def _position_score(doc_id: int, length: int, a, b) -> float:
        return RestateAgent._exp_func(doc_id / length, a, b)

    def write(self, instruction: str, model: str, task_id=None, log: bool = False) -> ChatCompletion:
        keys = instruction.split("<keys>")[1].split("</keys>")[0].strip()
        if task_id is None:
            task_id = f"a_{self._a}-b_{self._b}-k_{self._top_k}-" + datetime.now().strftime("%Y%m%d-%H%M%S")

        logger.info("Splitting docs...")
        split_docs = LaTeXSplitter(chunk_size=self._chunk_size, chunk_overlap=self._overlap).split(instruction)
        logger.info(f"Splitting done. {len(split_docs)} docs in total.")
        logger.info(f"Example doc: {split_docs[0]}")

        logger.info("Building doc Embedding...")
        doc_items = engine.get_embeddings(split_docs)
        docs_embeddings_pairs = list(zip(split_docs, doc_items))
        logger.info("Embedding done.")

        md_logger = MarkdownLogger(name=f"{task_id}.md") if log else None

        logger.info("Instruction: {}", instruction)

        plan_prompt = build_prompt(template="plan", instruction=instruction)
        while True:
            plan = engine.generate(prompt=plan_prompt, model=model, stream=False).choices[0].message.content

            logger.info("Plan: {}", plan)

            if log:
                md_logger.add_instruction(instruction)
                md_logger.add_plan(plan)

            steps = plan.split("\n")
            steps = [step.strip() for step in steps if step.strip()]
            if len(steps) < 20:
                break

        step_embeddings = engine.get_embeddings(steps)
        steps_embeddings_pairs = list(zip(steps, step_embeddings))

        written = ""
        completions = []

        if self._tqdm:
            loop = tqdm(enumerate(steps_embeddings_pairs), total=len(steps_embeddings_pairs), desc="Writing", leave=False)
        else:
            loop = enumerate(steps_embeddings_pairs)

        for idx, step_pair in loop:
            step = step_pair[0]
            step_embedding = step_pair[1]

            doc_items = [{"id": doc_idx, "embedding": doc_pair[1]} for doc_idx, doc_pair in enumerate(docs_embeddings_pairs)]
            for doc_item in doc_items:
                doc_item["score"] = self._similarity_score(doc_item["embedding"], step_embedding) - \
                                    self._position_score(doc_item["id"], len(doc_items), self._a, self._b)

            doc_items = sorted(doc_items, key=lambda x: x["score"], reverse=True)

            for k in range(self._top_k, 0, -1):
                if len(doc_items) > self._top_k:
                    picked_doc_ids = [doc_item["id"] for doc_item in doc_items[:k]]
                else:
                    picked_doc_ids = [doc_item["id"] for doc_item in doc_items]

                picked_doc_ids.reverse()

                all_scores = [doc_item["score"] for doc_item in doc_items]
                all_scores.sort(reverse=True)

                picked_docs = [docs_embeddings_pairs[doc_id][0] for doc_id in picked_doc_ids]

                if idx == 0 or k < self._top_k:
                    prompt = build_prompt(template="write_begin_with_restate", instruction=instruction, plan=plan, step=step,
                                          restatement="\n\n---\n\n".join(picked_docs))
                else:
                    prompt = build_prompt(template="write_step_with_restate", instruction=instruction, plan=plan, step=step, written=written,
                                          restatement="\n\n---\n\n".join(picked_docs), keys=keys)
                if len(encoder.encode(prompt, disallowed_special=())) < 110000:
                    break
                print(f"Prompt length: {len(encoder.encode(prompt, disallowed_special=()))} > 110000 when k={k}, try to reduce k.")

            completion = engine.generate(prompt=prompt, model=model, stream=False)
            completions.append(completion)
            written += "\n" + completion.choices[0].message.content

            if log:
                md_logger.add_qa(prompt=prompt, assistant=completion.choices[0].message.content)

        completion = engine.compose_chat_completion(completions)
        if log:
            md_logger.add_result(completion.choices[0].message.content)
            md_logger.dump()
        return completion
