"""Token ids. A write (a, v) is the token pair  @a  =v .

  [0, 2**A)              address tokens  "@a"
  [2**A, 2**A + 2**V)    value tokens    "=v"
  then BOS, EOS
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

from .isa import Arch

Write = tuple[int, int]


@dataclass(frozen=True)
class Vocab:
    arch: Arch

    @property
    def value_base(self) -> int:
        return self.arch.n_cells

    @property
    def bos(self) -> int:
        return self.arch.n_cells + self.arch.n_values

    @property
    def eos(self) -> int:
        return self.bos + 1

    @property
    def size(self) -> int:
        return self.eos + 1

    def addr_token(self, a: int) -> int:
        return a

    def value_token(self, v: int) -> int:
        return self.value_base + v

    def encode(self, writes: Iterable[Write]) -> tuple[int, ...]:
        """BOS followed by the token pairs of the given writes."""
        return (self.bos,) + tuple(t for a, v in writes for t in (self.addr_token(a), self.value_token(v)))

    def decode(self, ids: Iterable[int]) -> tuple[Write, ...]:
        """Token pairs -> writes. Stops at EOS; ignores BOS and an unpaired trailing address."""
        pending: int | None = None
        out: list[Write] = []
        for t in ids:
            if t == self.eos:
                break
            if t == self.bos:
                continue
            if t < self.value_base:
                pending = t
            elif pending is not None:
                out.append((pending, t - self.value_base))
                pending = None
        return tuple(out)

    def token_text(self, t: int) -> str:
        if t == self.bos:
            return "<bos>"
        if t == self.eos:
            return "<eos>"
        return f"@{t}" if t < self.value_base else f"={t - self.value_base}"

    def tokenizer_json(self) -> dict:
        """A HuggingFace `tokenizers` WordLevel tokenizer so text-generation pipelines work on text."""
        vocab = {self.token_text(t): t for t in range(self.size)}
        return {
            "version": "1.0",
            "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<eos>"},
            "pre_tokenizer": {"type": "WhitespaceSplit"},
            "normalizer": None,
            "decoder": None,
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
                "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 0}}],
                "special_tokens": {"<bos>": {"id": "<bos>", "ids": [self.bos], "tokens": ["<bos>"]}},
            },
            "added_tokens": [
                {"id": self.bos, "content": "<bos>", "special": True, "single_word": False,
                 "lstrip": False, "rstrip": False, "normalized": False},
                {"id": self.eos, "content": "<eos>", "special": True, "single_word": False,
                 "lstrip": False, "rstrip": False, "normalized": False},
            ],
        }

    def tokenizer_config_json(self) -> dict:
        return {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "unk_token": "<eos>",
            "clean_up_tokenization_spaces": False,
            "model_max_length": self.arch.max_positions,
        }


def dumps(obj: dict) -> str:
    return json.dumps(obj, indent=1)
