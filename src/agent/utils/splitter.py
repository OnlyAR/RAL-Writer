# -*- coding: utf-8 -*-
# @File Name:     splitter
# @Author :       Jun
# @Date:          2025/1/23
from typing import List, Optional

import nltk


class TextSplitter:
    r"""
    Modified from:
    https://github.com/langchain-ai/langchain/blob/v0.1.5/libs/langchain/langchain/text_splitter.py
    """

    def __init__(self, chunk_size: int, chunk_overlap: int,
                 separators: Optional[List[str]] = None,
                 length_func=lambda t: len(nltk.word_tokenize(t))):
        self._separators = ["\n\n", "\n", ". ", ", ", " ", ""] + ([] if separators is None else separators)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        assert self._chunk_overlap < self._chunk_size, "chunk overlap must be smaller than chunk size"
        self._length_func = length_func

    @staticmethod
    def _split_text(text: str, separator: str) -> List[str]:
        splits = text.split(separator) if separator else list(text)
        return [split for split in splits if split]

    @staticmethod
    def _join_docs(docs: List[str], separator: str) -> str:
        return separator.join(docs).strip()

    def _count(self, text: str) -> int:
        return self._length_func(text)

    def _merge(self, splits: List[str], separator: str) -> List[str]:
        merged_docs = []
        in_process_docs = []
        for split in splits:
            text = self._join_docs(in_process_docs, separator)
            if self._count(text + split) > self._chunk_size:
                if len(in_process_docs) > 0:
                    merged_docs.append(text)

                    if self._chunk_overlap == 0:
                        in_process_docs = []
                    else:
                        while self._count(text) > self._chunk_overlap:
                            in_process_docs.pop(0)
                            text = self._join_docs(in_process_docs, separator)

            in_process_docs.append(split)

        if len(in_process_docs) > 0:
            text = self._join_docs(in_process_docs, separator)
            merged_docs.append(text)

        return merged_docs

    def split(self, text: str) -> List[str]:
        return self._split(text, self._separators)

    def _split(self, text: str, separators: List[str]) -> List[str]:
        separators = separators[:]
        separator = separators.pop(0)

        splits = self._split_text(text, separator)
        final_chunks: List = []
        good_splits: List = []
        for split in splits:
            if self._count(split) < self._chunk_size:
                good_splits.append(split)
            else:
                if good_splits:
                    merged_text = self._merge(good_splits, separator)
                    final_chunks.extend(merged_text)
                    good_splits = []
                if not separators:
                    final_chunks.append(split)
                else:
                    extra_chunks = self._split(split, separators)
                    final_chunks.extend(extra_chunks)

        if good_splits:
            merged_text = self._merge(good_splits, separator)
            final_chunks.extend(merged_text)
        return final_chunks


class LaTeXSplitter(TextSplitter):
    separators = [
        # First, try to split along Latex sections
        "\n\\chapter{",
        "\n\\section{",
        "\n\\subsection{",
        "\n\\subsubsection{",
        # Now split by environments
        "\n\\begin{tabel}",
        "\n\\begin{tabel*}",
        "\n\\begin{enumerate}",
        "\n\\begin{itemize}",
        "\n\\begin{description}",
        "\n\\begin{list}",
        "\n\\begin{quote}",
        "\n\\begin{quotation}",
        "\n\\begin{verse}",
        "\n\\begin{verbatim}",
        # Now split by math environments
        "\n\\begin{align}",
        "$$",
        "$",
    ]

    def __init__(self, chunk_size: int, chunk_overlap: int, length_function=lambda t: len(nltk.word_tokenize(t))):
        super().__init__(chunk_size, chunk_overlap, self.separators, length_function)
