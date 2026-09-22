"""Reference interpreter. Produces exactly the write trace the transformer must emit.

Semantics of one step, given the trace so far:
  pc     = value of the most recent write to PC_ADDR, or 0 if there is none
  exec   = the most recent write was to PC_ADDR, or there is no write at all

So a run is started by writing the entry point to PC_ADDR (see `start_writes`);
initial memory contents go before that write.
  if exec:   run prog[pc]:
      data op   -> write (dst, result)
      store     -> write (mem[a], b)
      jump      -> write (PC_ADDR, target or pc + 1)
      halt      -> stop  (as does a pc outside the program)
  else:      write (PC_ADDR, pc + 1)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .isa import Alu, Arch, Cell, Halt, Imm, Instr, Jump, Load, Op, Operand, Store

Write = tuple[int, int]


@dataclass(frozen=True)
class Machine:
    arch: Arch
    trace: tuple[Write, ...]

    def read(self, addr: int) -> int:
        return next((v for a, v in reversed(self.trace) if a == addr), 0)

    @property
    def memory(self) -> dict[int, int]:
        return dict(self.trace)  # later writes win

    @property
    def pc(self) -> int:
        return self.read(self.arch.pc_addr)

    @property
    def exec_phase(self) -> bool:
        return not self.trace or self.trace[-1][0] == self.arch.pc_addr

    def with_write(self, w: Write) -> "Machine":
        return Machine(self.arch, self.trace + (w,))


def _operand(m: Machine, o: Operand) -> int:
    match o:
        case Imm(value=v):
            return v
        case Cell(addr=a):
            return m.read(a)
    raise TypeError(o)


def _alu(op: Op, x: int, y: int, arch: Arch) -> int:
    mask = arch.n_values - 1
    match op:
        case Op.ADD:
            return (x + y) & mask
        case Op.SUB:
            return (x - y) & mask
        case Op.AND:
            return x & y
        case Op.OR:
            return x | y
        case Op.XOR:
            return x ^ y
        case Op.LT:
            return int(x < y)
        case Op.EQ:
            return int(x == y)
    raise ValueError(op)


def _as_addr(v: int, arch: Arch) -> int:
    return v & (arch.n_cells - 1)


def step(prog: tuple[Instr, ...], m: Machine) -> Write | None:
    """The next write, or None to halt."""
    arch = m.arch
    pc = m.pc
    if not m.exec_phase:
        return (arch.pc_addr, (pc + 1) & (arch.n_values - 1))
    if pc >= len(prog):
        return None
    i = prog[pc]
    match i:
        case Alu(op=op, dst=dst, a=a, b=b):
            return (dst, _alu(op, _operand(m, a), _operand(m, b), arch))
        case Load(dst=dst, a=a):
            return (dst, m.read(_as_addr(_operand(m, a), arch)))
        case Store(a=a, b=b):
            return (_as_addr(_operand(m, a), arch), _operand(m, b))
        case Jump(op=op, a=a, target=t):
            x = 0 if a is None else _operand(m, a)
            taken = {Op.JMP: True, Op.JZ: x == 0, Op.JNZ: x != 0}[op]
            return (arch.pc_addr, t if taken else (pc + 1) & (arch.n_values - 1))
        case Halt():
            return None
    raise TypeError(i)


def start_writes(init: tuple[Write, ...], arch: Arch, entry: int = 0) -> tuple[Write, ...]:
    """The prefix that starts a run: initial memory, then a jump to `entry`."""
    return init + ((arch.pc_addr, entry),)


def run(prog: tuple[Instr, ...], init: tuple[Write, ...], arch: Arch, max_writes: int) -> Iterator[Write]:
    """Yield the writes the machine makes after the initial ones. Stops on halt or max_writes."""
    m = Machine(arch, init)
    for _ in range(max_writes):
        w = step(prog, m)
        if w is None:
            return
        yield w
        m = m.with_write(w)
