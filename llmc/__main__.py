"""`python -m llmc compile prog.asm out_dir` / `python -m llmc run out_dir --mem n=10`"""
import argparse
import os
import shutil
import sys

from .asm import assemble
from .compile import compile_program, save
from .isa import Arch


def main() -> int:
    ap = argparse.ArgumentParser(prog="llmc")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compile", help="compile an .asm file into a Llama checkpoint")
    c.add_argument("source")
    c.add_argument("out_dir")
    c.add_argument("--addr-bits", type=int, default=10)
    c.add_argument("--val-bits", type=int, default=10)
    c.add_argument("--max-positions", type=int, default=4096)
    r = sub.add_parser("run", help="run a compiled checkpoint (see llmc.run for options)")
    r.add_argument("rest", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if args.cmd == "run":
        from .run import main as run_main

        return run_main(args.rest)
    arch = Arch(args.addr_bits, args.val_bits, args.max_positions)
    with open(args.source) as f:
        text = f.read()
    prog = assemble(text, arch)
    compiled = compile_program(prog, arch)
    save(compiled, args.out_dir, arch)
    shutil.copyfile(args.source, os.path.join(args.out_dir, "program.asm"))
    cfg = compiled.config
    print(f"{len(prog.instrs)} instructions -> {args.out_dir}: {cfg['num_hidden_layers']} layers, "
          f"d={cfg['hidden_size']}, {cfg['num_attention_heads']} heads, mlp={cfg['intermediate_size']}, "
          f"vocab={cfg['vocab_size']}, {compiled.n_params / 1e6:.2f}M params")
    return 0


if __name__ == "__main__":
    sys.exit(main())
