# -*- coding: utf-8 -*-
# @File Name:     engine
# @Author :       Jun
# @date:          2024/12/10
# @Description :  Chatbot engine
from typing import Union, List

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

load_dotenv()

client = OpenAI()


def generate(
        prompt: str, model: str, stream: bool = False, temperature: float = 0.3, max_tokens: int = 8192
) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
    while True:
        try:
            return client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=temperature,
                stream=stream,
                # max_tokens=32768
                max_tokens=max_tokens
            )
        except Exception as e:
            print(e)


def compose_chat_completion(
        completions: List[ChatCompletion]
) -> ChatCompletion:
    contents = [completion.choices[0].message.content for completion in completions]
    result = completions[-1]
    result.choices[0].message.content = "\n\n".join(contents)
    return result


def get_embeddings(texts: List[str], model: str = "bge-base-en-v1.5") -> List[np.ndarray]:
    embeddings = client.embeddings.create(input=texts, model=model).data
    embeddings = [np.array(embedding.embedding) for embedding in embeddings]
    return embeddings


if __name__ == '__main__':
    emb1, emb2 = get_embeddings(["你好", "我是谁"])
    print(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))