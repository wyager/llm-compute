"""Run a compiled program with off-the-shelf HuggingFace tooling and check it against the interpreter."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

from .asm import parse_memory
from .interp import Write, run as interp_run, start_writes
from .isa import Arch
from .tokens import Vocab


@dataclass(frozen=True)
class Model:
    arch: Arch
    symbols: dict[str, int]
    pc_addr: int
    n_instructions: int
    model: object  # transformers LlamaForCausalLM


def load(model_dir: str, attn: str = "sdpa") -> Model:
    import torch
    from transformers import LlamaForCausalLM

    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    meta = cfg["llmc"]
    arch = Arch(meta["addr_bits"], meta["val_bits"], cfg["max_position_embeddings"])
    model = LlamaForCausalLM.from_pretrained(model_dir, dtype=torch.float32, attn_implementation=attn).eval()
    return Model(arch, meta["symbols"], meta["pc_addr"], meta["n_instructions"], model)


def generate(m: Model, init: tuple[Write, ...], max_writes: int | None = None) -> tuple[Write, ...]:
    """Feed BOS + the initial writes and let the model run the program. Returns the writes it makes."""
    import torch

    voc = Vocab(m.arch)
    prompt = voc.encode(init)
    budget = m.arch.max_positions - len(prompt) - 1
    max_new = budget if max_writes is None else min(budget, 2 * max_writes)
    with torch.no_grad():
        out = m.model.generate(
            torch.tensor([prompt]), max_new_tokens=max_new, do_sample=False,
            eos_token_id=voc.eos, pad_token_id=voc.eos,
        )
    return voc.decode(out[0, len(prompt):].tolist())


def format_writes(writes: tuple[Write, ...], symbols: dict[str, int], pc_addr: int) -> str:
    names = {a: n for n, a in symbols.items()}
    return " ".join(f"{names.get(a, f'[{a}]')}={v}" for a, v in writes)


def final_memory(init: tuple[Write, ...], trace: tuple[Write, ...]) -> dict[int, int]:
    return dict(init + trace)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="run a compiled program on the transformer")
    ap.add_argument("model_dir")
    ap.add_argument("--mem", default="", help='initial memory, e.g. "n=10,x=3"')
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--trace", action="store_true", help="print every write")
    ap.add_argument("--no-check", action="store_true", help="skip the interpreter comparison")
    args = ap.parse_args(argv)

    m = load(args.model_dir, args.attn)
    init = start_writes(parse_memory(args.mem, m.symbols), m.arch)
    t0 = time.time()
    trace = generate(m, init)
    dt = time.time() - t0
    names = {a: n for n, a in m.symbols.items()}
    print(f"{len(trace)} writes ({2 * len(trace)} tokens) in {dt:.1f}s")
    if args.trace:
        print(format_writes(trace, m.symbols, m.pc_addr))
    mem = final_memory(init, trace)
    print("final memory:", {names.get(a, a): v for a, v in sorted(mem.items()) if a != m.pc_addr})
    if args.no_check:
        return 0
    prog = _load_program(args.model_dir)
    expected = tuple(interp_run(prog, init, m.arch, len(trace) + 1))
    if expected == trace:
        print("matches the reference interpreter")
        return 0
    n = next((i for i, (x, y) in enumerate(zip(expected, trace)) if x != y), min(len(expected), len(trace)))
    print(f"MISMATCH at write {n}: expected {expected[n:n + 3]} got {trace[n:n + 3]}", file=sys.stderr)
    return 1


def _load_program(model_dir: str):
    from .asm import assemble

    with open(os.path.join(model_dir, "program.asm")) as f:
        return assemble(f.read()).instrs


if __name__ == "__main__":
    sys.exit(main())
