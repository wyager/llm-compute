"""End-to-end: compile -> load with transformers -> generate -> compare with the interpreter."""
from __future__ import annotations

import os
import random

import pytest

from llmc.asm import assemble, parse_memory
from llmc.compile import compile_program, save
from llmc.interp import run as interp_run, start_writes
from llmc.isa import Arch
from llmc.run import generate, load
from llmc.tokens import Vocab

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")
ARCH = Arch()


def build(text: str, out_dir: str, arch: Arch = ARCH):
    prog = assemble(text, arch)
    save(compile_program(prog, arch), out_dir, arch)
    with open(os.path.join(out_dir, "program.asm"), "w") as f:
        f.write(text)
    return prog


def check(text: str, mem: str, tmp_path, attn: str = "sdpa", arch: Arch = ARCH) -> tuple:
    prog = build(text, str(tmp_path), arch)
    m = load(str(tmp_path), attn)
    init = start_writes(parse_memory(mem, prog.symbols), arch)
    got = generate(m, init)
    want = tuple(interp_run(prog.instrs, init, arch, len(got) + 2))
    if len(want) > len(got):  # the run hit the context limit; compare what the model had room for
        assert 2 * len(got) >= arch.max_positions - 2 * len(init) - 4
        want = want[: len(got)]
    assert got == want
    return dict(init + got)


@pytest.mark.parametrize("name,mem,expect", [
    ("sum", "n=5", {2: 15}),
    ("fib", "n=10", {3: 55}),
    ("gcd", "a=84,b=36", {1: 12}),
    ("sieve", "limit=12,base=100", {104: 1, 106: 1, 108: 1, 109: 1, 110: 1}),
])
def test_examples(name, mem, expect, tmp_path):
    with open(os.path.join(EXAMPLES, f"{name}.asm")) as f:
        final = check(f.read(), mem, tmp_path)
    for addr, value in expect.items():
        assert final[addr] == value


def test_eager_attention(tmp_path):
    with open(os.path.join(EXAMPLES, "sum.asm")) as f:
        assert check(f.read(), "n=7", tmp_path, attn="eager")[2] == 28


def test_every_alu_op(tmp_path):
    """Random straight-line ALU programs over random immediates and cells, including wraparound."""
    rng = random.Random(0)
    ops = ["add", "sub", "and", "or", "xor", "lt", "eq"]
    lines = []
    for i in range(24):
        op = rng.choice(ops)
        a = f"#{rng.randrange(1024)}" if rng.random() < 0.5 else str(rng.randrange(8))
        b = f"#{rng.randrange(1024)}" if rng.random() < 0.5 else str(rng.randrange(8))
        lines.append(f"{op} {10 + i}, {a}, {b}")
    lines.append("halt")
    mem = ",".join(f"{c}={rng.randrange(1024)}" for c in range(8))
    check("\n".join(lines), mem, tmp_path)


def test_indirect_and_computed_jump(tmp_path):
    text = """
    let p = 1
    let v = 2
      store p, #77        ; mem[mem[p]] = 77
      load  v, p          ; v = mem[mem[p]]
      add   pc, #0, #5    ; computed jump: skip the next instruction
      halt
      halt
      add   v, v, #1
      halt
    """
    final = check(text, "p=200", tmp_path)
    assert final[200] == 77 and final[2] == 78


def test_tokenizer_round_trip():
    voc = Vocab(ARCH)
    writes = ((3, 7), (1023, 0), (0, 1023))
    assert voc.decode(voc.encode(writes)) == writes
    assert [voc.token_text(t) for t in voc.encode(writes)[:3]] == ["<bos>", "@3", "=7"]
