"""A tiny assembler: text -> tuple[Instr, ...]. Pure functions, no state.

Syntax (one instruction per line, `;` starts a comment):

    let x = 3            ; name cell 3 "x"  (pc is predefined)
    loop:                ; label
      add  dst, a, b     ; dst <- a + b       (also sub, and, or, xor, lt, eq)
      mov  dst, a        ; dst <- a           (sugar for add dst, a, #0)
      load dst, a        ; dst <- mem[a]
      store a, b         ; mem[a] <- b
      jmp  L
      jz   a, L          ; jump if a == 0
      jnz  a, L          ; jump if a != 0
      halt

Operands: `#n` is an immediate, anything else is a cell (a name or a number).
Destinations are always cells.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .isa import Alu, Arch, Cell, Halt, Imm, Instr, Jump, Load, Op, Operand, Store


@dataclass(frozen=True)
class Program:
    instrs: tuple[Instr, ...]
    symbols: Mapping[str, int]  # named cells
    labels: Mapping[str, int]  # label -> instruction index


class AsmError(ValueError):
    pass


def _strip(line: str) -> str:
    return line.split(";", 1)[0].strip()


def _parse_int(text: str, what: str) -> int:
    try:
        return int(text, 0)
    except ValueError:
        raise AsmError(f"bad {what}: {text!r}") from None


def _cell(text: str, symbols: Mapping[str, int]) -> int:
    if text in symbols:
        return symbols[text]
    if text.startswith("#"):
        raise AsmError(f"expected a cell, got immediate {text!r}")
    return _parse_int(text, "cell")


def _operand(text: str, symbols: Mapping[str, int]) -> Operand:
    if text.startswith("#"):
        return Imm(_parse_int(text[1:], "immediate"))
    return Cell(_cell(text, symbols))


def _label(text: str, labels: Mapping[str, int]) -> int:
    if text in labels:
        return labels[text]
    return _parse_int(text, "jump target")


def _split_line(line: str) -> tuple[str, tuple[str, ...]]:
    head, _, rest = line.partition(" ")
    args = tuple(a.strip() for a in rest.split(",")) if rest.strip() else ()
    return head.strip().lower(), args


def _collect(lines: tuple[str, ...], arch: Arch) -> tuple[tuple[str, ...], dict[str, int], dict[str, int]]:
    """First pass: separate declarations and labels from instruction lines."""
    symbols: dict[str, int] = {"pc": arch.pc_addr}
    labels: dict[str, int] = {}
    body: list[str] = []
    for raw in lines:
        line = _strip(raw)
        if not line:
            continue
        if line.startswith("let "):
            name, _, addr = line[4:].partition("=")
            symbols[name.strip()] = _parse_int(addr.strip(), "cell address")
        elif line.endswith(":"):
            labels[line[:-1].strip()] = len(body)
        else:
            body.append(line)
    return tuple(body), symbols, labels


def _instr(line: str, symbols: Mapping[str, int], labels: Mapping[str, int]) -> Instr:
    head, args = _split_line(line)

    def need(n: int) -> None:
        if len(args) != n:
            raise AsmError(f"{head} takes {n} operands: {line!r}")

    match head:
        case "add" | "sub" | "and" | "or" | "xor" | "lt" | "eq":
            need(3)
            return Alu(Op(head), _cell(args[0], symbols), _operand(args[1], symbols), _operand(args[2], symbols))
        case "mov":
            need(2)
            return Alu(Op.ADD, _cell(args[0], symbols), _operand(args[1], symbols), Imm(0))
        case "load":
            need(2)
            return Load(_cell(args[0], symbols), _operand(args[1], symbols))
        case "store":
            need(2)
            return Store(_operand(args[0], symbols), _operand(args[1], symbols))
        case "jmp":
            need(1)
            return Jump(Op.JMP, None, _label(args[0], labels))
        case "jz" | "jnz":
            need(2)
            return Jump(Op(head), _operand(args[0], symbols), _label(args[1], labels))
        case "halt":
            need(0)
            return Halt()
    raise AsmError(f"unknown instruction: {line!r}")


def _check(p: Program, arch: Arch) -> Program:
    if len(p.instrs) > arch.n_values:
        raise AsmError(f"program has {len(p.instrs)} instructions; the PC only holds {arch.n_values}")

    def ok_cell(a: int) -> None:
        if not 0 <= a < arch.n_cells:
            raise AsmError(f"cell {a} out of range")

    def ok_operand(o: Operand | None) -> None:
        match o:
            case Cell(addr=a):
                ok_cell(a)
            case Imm(value=v):
                if not 0 <= v < arch.n_values:
                    raise AsmError(f"immediate {v} out of range")

    for i in p.instrs:
        match i:
            case Alu(dst=d, a=a, b=b):
                ok_cell(d), ok_operand(a), ok_operand(b)
            case Load(dst=d, a=a):
                ok_cell(d), ok_operand(a)
            case Store(a=a, b=b):
                ok_operand(a), ok_operand(b)
            case Jump(a=a, target=t):
                ok_operand(a)
                if not 0 <= t < arch.n_values:
                    raise AsmError(f"jump target {t} out of range")
    return p


def assemble(text: str, arch: Arch = Arch()) -> Program:
    body, symbols, labels = _collect(tuple(text.splitlines()), arch)
    instrs = tuple(_instr(line, symbols, labels) for line in body)
    return _check(Program(instrs, symbols, labels), arch)


def parse_memory(spec: str, symbols: Mapping[str, int]) -> tuple[tuple[int, int], ...]:
    """Parse "x=5,7=9" into ((3, 5), (7, 9)) using the program's symbols."""
    if not spec.strip():
        return ()
    pairs = []
    for item in spec.split(","):
        name, _, value = item.partition("=")
        pairs.append((_cell(name.strip(), symbols), _parse_int(value.strip(), "value")))
    return tuple(pairs)


def instruction_lines(text: str) -> tuple[int, ...]:
    """Source line index of each instruction, in program order (for highlighting a listing)."""
    return tuple(
        n for n, raw in enumerate(text.splitlines())
        for line in [_strip(raw)]
        if line and not line.startswith("let ") and not line.endswith(":")
    )
