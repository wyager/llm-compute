"""The instruction set, as immutable data.

The machine has 2**A memory cells of V-bit values. All state lives in memory,
including the program counter, which is the cell at address PC_ADDR.

Execution is a stream of memory writes (address, value). Each write becomes two
tokens in the transformer: an address token followed by a value token.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union


@dataclass(frozen=True)
class Arch:
    """Bit widths of the machine. A <= V so any value can serve as an address."""

    addr_bits: int = 10
    val_bits: int = 10
    max_positions: int = 4096

    @property
    def n_cells(self) -> int:
        return 1 << self.addr_bits

    @property
    def n_values(self) -> int:
        return 1 << self.val_bits

    @property
    def pc_addr(self) -> int:
        return self.n_cells - 1


@dataclass(frozen=True)
class Cell:
    """An operand that names a memory cell."""

    addr: int


@dataclass(frozen=True)
class Imm:
    """An immediate operand."""

    value: int


Operand = Union[Cell, Imm]


class Op(Enum):
    ADD = "add"
    SUB = "sub"
    AND = "and"
    OR = "or"
    XOR = "xor"
    LT = "lt"
    EQ = "eq"
    LOAD = "load"
    STORE = "store"
    JMP = "jmp"
    JZ = "jz"
    JNZ = "jnz"
    HALT = "halt"


ALU_OPS = frozenset({Op.ADD, Op.SUB, Op.AND, Op.OR, Op.XOR, Op.LT, Op.EQ})
DATA_OPS = ALU_OPS | {Op.LOAD}  # instructions that write `dst`
JUMP_OPS = frozenset({Op.JMP, Op.JZ, Op.JNZ})


@dataclass(frozen=True)
class Alu:
    """dst <- a op b"""

    op: Op
    dst: int
    a: Operand
    b: Operand


@dataclass(frozen=True)
class Load:
    """dst <- mem[a]"""

    dst: int
    a: Operand


@dataclass(frozen=True)
class Store:
    """mem[a] <- b"""

    a: Operand
    b: Operand


@dataclass(frozen=True)
class Jump:
    """pc <- target if cond(a) else pc + 1. `a` is None for unconditional jumps."""

    op: Op
    a: Operand | None
    target: int


@dataclass(frozen=True)
class Halt:
    pass


Instr = Union[Alu, Load, Store, Jump, Halt]


def instr_op(i: Instr) -> Op:
    match i:
        case Alu(op=op) | Jump(op=op):
            return op
        case Load():
            return Op.LOAD
        case Store():
            return Op.STORE
        case Halt():
            return Op.HALT
    raise TypeError(i)


def operand_a(i: Instr) -> Operand | None:
    match i:
        case Alu(a=a) | Load(a=a) | Store(a=a) | Jump(a=a):
            return a
        case _:
            return None


def operand_b(i: Instr) -> Operand | None:
    match i:
        case Alu(b=b) | Store(b=b):
            return b
        case _:
            return None
