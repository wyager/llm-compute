"""Inspect the residual stream of a compiled model, slot by slot.

    uv run python -m llmc.debug out/sum --mem n=5 --steps 3
"""
from __future__ import annotations

import argparse

import torch

from .asm import parse_memory
from .compile import Layout, make_layout
from .interp import start_writes
from .run import load
from .tokens import Vocab


def slot_values(lay: Layout, h: torch.Tensor, only: tuple[str, ...] = ()) -> str:
    parts = []
    for name, w in lay.slots:
        if name in ("one", "pad") or (only and name not in only):
            continue
        vals = h[lay.off(name): lay.off(name) + w].tolist()
        if w == 1:
            parts.append(f"{name}={vals[0]:.3g}")
        else:
            as_bits = all(abs(v - round(v)) < 1e-3 and round(v) in (0, 1, -1) for v in vals)
            if as_bits and all(round(v) >= 0 for v in vals):
                parts.append(f"{name}={sum(round(v) << j for j, v in enumerate(vals))}")
            else:
                parts.append(f"{name}=[{' '.join(f'{v:.2g}' for v in vals)}]")
    return "  ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--mem", default="")
    ap.add_argument("--steps", type=int, default=2, help="how many generated tokens to trace")
    ap.add_argument("--attn", default="eager")
    args = ap.parse_args()
    m = load(args.model_dir, args.attn)
    lay = make_layout(m.arch, m.model.config.num_attention_heads)
    voc = Vocab(m.arch)
    ids = list(voc.encode(start_writes(parse_memory(args.mem, m.symbols), m.arch)))
    for _ in range(args.steps):
        with torch.no_grad():
            out = m.model(torch.tensor([ids]), output_hidden_states=True)
        print(f"--- position {len(ids) - 1}: input {voc.token_text(ids[-1])}")
        for li, h in enumerate(out.hidden_states):
            print(f"  after layer {li}: {slot_values(lay, h[0, -1])}")
        nxt = int(out.logits[0, -1].argmax())
        top = out.logits[0, -1].topk(3)
        print(f"  -> {voc.token_text(nxt)}   top3: {[(voc.token_text(int(i)), round(float(v), 1)) for v, i in zip(top.values, top.indices)]}")
        ids.append(nxt)
        if nxt == voc.eos:
            break


if __name__ == "__main__":
    main()
