# -*- coding: utf-8 -*-
# @File Name:     length_score
# @Author :       Jun
# @Date:          2025/2/9
import json
import re
from pathlib import Path

from tqdm import tqdm

name = "qwen2.5-14b"
save_path = f"length_score-{name}.tsv"
file_dirs = [f"{name}-{l}k" for l in [4, 8, 16]]
methods = ["single", "contact", "compress", "restate-a_60-b_0.3-k_12"]

root_dir = Path(__file__).parent.parent
root_result_dir = root_dir / "results" / "pred"

def count_words(text):
    pattern = r"\b(?!\d+\b)[a-zA-Z]+(?:['’-][a-zA-Z]+)*\b"
    words = re.findall(pattern, text)
    return len(words)

def length_score(target, x):
    if x >= target:
        return 1
    else:
        return max(0, 1 - (target / x - 1) / 2)

if __name__ == '__main__':
    table = ["Method\tLength\tMean\tStd\tScore"]
    for method in methods:
        for file_dir in file_dirs:
            length = int(file_dir.split("-")[-1][:-1]) * 1000
            scores = []
            output_lengths = []
            result_path = root_result_dir / file_dir / method
            dirs = list(result_path.iterdir())
            for file in tqdm(dirs, desc="Calculating scores"):
                jd = json.loads(file.read_text(encoding='utf-8'))
                response = jd["response"]
                output_length = count_words(response)
                scores.append(length_score(length, output_length))
                output_lengths.append(output_length)
            print(f"Average score: {sum(scores) / len(scores)}")

            mean = sum(output_lengths) / len(output_lengths)
            std = (sum([(x - mean) ** 2 for x in output_lengths]) / len(output_lengths)) ** 0.5
            table.append(f"{method}\t{length}\t{mean}\t{std}\t{100 * sum(scores) / len(scores)}")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(table))