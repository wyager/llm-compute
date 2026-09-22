"""Compile a program into the exact weights of a Llama transformer.

No training. The residual stream is a fixed set of named coordinates, each
holding a 0/1 feature (plus one big constant coordinate that makes RMSNorm act
as a known linear map). Attention heads are exact key/value lookups over the
token stream: "the latest value token whose address matches this query", with
recency supplied by a low-frequency RoPE pair. MLP units are exact step
functions of integer linear forms, so SiLU behaves as ReLU to float32
precision.

How the machine runs, per token position (see interp.py for the semantics):

  layer 1 attn   cur_addr <- the address token this position belongs to
  layer 2 attn   last_*   <- the most recent completed write (or BOS)
                 pc       <- the most recent write to PC_ADDR (or 0)
  layer 2 mlp    exec/upd phase flag; decode prog[pc] into op flags, operand
                 keys, immediates, dst, jump target and pc+1 (one unit per
                 instruction: the program *is* the weights)
  layer 3 attn   x, y     <- memory[operand a], memory[operand b]
  layer 3 mlp    carries, comparisons, x==0, indirect-address key
  layer 4 attn   ind      <- memory[x]           (for load)
  layer 4 mlp    out      <- the address or value to emit, by phase and opcode
  lm_head        the token whose bits agree with `out`
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
from safetensors.numpy import save_file

from .asm import Program
from .isa import Alu, Arch, Cell, Halt, Imm, Instr, Jump, Load, Op, Operand, Store, instr_op
from .tokens import Vocab, dumps

Row = Mapping[int, float]  # residual coordinate -> weight

HEAD_DIM = 64
RECENCY_PAIR = 8  # RoPE pair whose frequency supplies recency
CONTENT_PAIRS = tuple(range(16, 32))  # pairs rotated by < 1e-3 rad over the context
CONTENT_DIMS = tuple(p for p in CONTENT_PAIRS) + tuple(p + HEAD_DIM // 2 for p in CONTENT_PAIRS)

BIG_CONST = 1.0e5  # value of the `one` coordinate in the residual stream
STEP_K = 80.0  # gate pre-activation scale; SiLU(±20) is exactly ReLU in float32
SOFTMAX_GAP = 30.0  # minimum score gap between the winner and the runner-up
LOGIT_SCALE = 20.0

OP_FLAGS = ("add", "sub", "and", "or", "xor", "lt", "eq", "load", "store", "jmp", "jz", "jnz", "data", "running")


# ---------------------------------------------------------------------------
# residual stream layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    slots: tuple[tuple[str, int], ...]

    @property
    def dim(self) -> int:
        return sum(w for _, w in self.slots)

    def off(self, name: str) -> int:
        o = 0
        for n, w in self.slots:
            if n == name:
                return o
            o += w
        raise KeyError(name)

    def width(self, name: str) -> int:
        return dict(self.slots)[name]

    def c(self, name: str, i: int = 0) -> int:
        assert 0 <= i < self.width(name), (name, i)
        return self.off(name) + i

    def coords(self, name: str) -> tuple[int, ...]:
        return tuple(range(self.off(name), self.off(name) + self.width(name)))


def make_layout(arch: Arch, n_heads: int) -> Layout:
    a, v = arch.addr_bits, arch.val_bits
    slots = (
        ("one", 1),
        ("is_addr", 1), ("is_val", 1), ("is_bos", 1),
        ("tok", v),  # this token's own bits
        ("cur_addr", a),  # layer 1
        ("last_addr", a), ("last_val", v), ("last_bos", 1), ("pc", v),  # layer 2 attn
        ("exec", 1), ("upd", 1), ("op", len(OP_FLAGS)),  # layer 2 mlp
        ("key1", a), ("key2", a), ("x", v), ("y", v), ("dst", a), ("target", v), ("pcnext", v),
        ("carry_add", v), ("carry_sub", v), ("x_nz", 1), ("x_z", 1), ("lt", 1), ("eq", 1), ("ind_key", a), ("emit_addr", 1),  # l3 mlp
        ("ind", v),  # layer 4 attn
        ("kind", 3), ("out", v),  # layer 4 mlp: kind = (addr, val, eos)
    )
    used = sum(w for _, w in slots)
    dim = -(-used // n_heads) * n_heads  # HF only needs hidden_size % num_heads == 0
    return Layout(slots + (("pad", dim - used),) if dim > used else slots)


def bits(value: int, n: int) -> tuple[int, ...]:
    return tuple((value >> j) & 1 for j in range(n))


def bit_row(lay: Layout, slot: str, value: int, n: int, scale: float = 1.0) -> dict[int, float]:
    """Coordinates of `slot` holding the 0/1 bits of `value`, as a row."""
    return {lay.c(slot, j): scale * b for j, b in enumerate(bits(value, n)) if b}


def merge(*rows: Row) -> dict[int, float]:
    out: dict[int, float] = {}
    for r in rows:
        for k, val in r.items():
            out[k] = out.get(k, 0.0) + val
    return {k: val for k, val in out.items() if val != 0.0}


# ---------------------------------------------------------------------------
# attention: exact lookups
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Head:
    q: Mapping[int, Row]  # head dim -> row over residual coords (includes the 1/sqrt(64) rescale)
    k: Mapping[int, Row]
    v: Mapping[int, Row]
    o: Mapping[int, Row]  # residual coord -> {head dim: coeff}


@dataclass(frozen=True)
class Scales:
    """Attention score constants, derived from the context length."""

    theta: float  # RoPE angle per position of the recency pair
    recency: float  # ab in score = -ab * sin(delta * theta)
    big: float  # weight of one matching address bit

    @staticmethod
    def for_arch(arch: Arch) -> "Scales":
        n = arch.max_positions
        theta = 0.8 / n  # monotone (< pi/2 over the window) with a healthy per-step slope
        slope = theta * math.cos(0.8)  # smallest per-step score gap per unit of ab
        recency = SOFTMAX_GAP / slope
        big = 2.0 ** math.ceil(math.log2(2 * recency))
        return Scales(theta, recency, big)

    @property
    def rope_theta(self) -> float:
        # inv_freq[pair] = rope_theta ** (-pair / 32); pin the recency pair to `theta`.
        return self.theta ** (-(HEAD_DIM // 2) / RECENCY_PAIR)


def lookup_head(lay: Layout, sc: Scales, arch: Arch, query: Mapping[int, Row], key_bits: Mapping[int, Row],
                key_gate: Row, bos_weight: float, values: Iterable[tuple[Row, int]]) -> Head:
    """A head that attends to the most recent key with the best content match.

    query[j]    row giving the query's j-th address bit as +1/-1 (0 = don't care)
    key_bits[j] row giving the key's j-th address bit as +1/-1 (BOS: 0)
    key_gate    row that is 1 for admissible keys and 0 otherwise, or -1 to veto
    bos_weight  extra score, in units of `big`, for the BOS token (the default/fallback key)
    values      (row read from the attended token, residual coord written)
    """
    qs = math.sqrt(HEAD_DIM)  # undo HF's 1/sqrt(head_dim)
    n_bits = len(query)
    d_bos, d_gate = CONTENT_DIMS[n_bits], CONTENT_DIMS[n_bits + 1]
    q = {CONTENT_DIMS[j]: {c: qs * sc.big * w for c, w in row.items()} for j, row in query.items()}
    k = {CONTENT_DIMS[j]: dict(row) for j, row in key_bits.items()}
    q[d_bos] = {lay.c("one"): qs * sc.big * bos_weight}
    k[d_bos] = {lay.c("is_bos"): 1.0}
    q[d_gate] = {lay.c("one"): qs * sc.big * 2 * (n_bits + 1)}
    k[d_gate] = dict(key_gate)
    # recency: q = (1, 0), k = (0, -ab) in the recency pair -> score -ab sin(delta theta)
    q[RECENCY_PAIR] = {lay.c("one"): qs}
    k[RECENCY_PAIR + HEAD_DIM // 2] = {lay.c("one"): -sc.recency}
    v = {i: dict(row) for i, (row, _) in enumerate(values)}
    o = {coord: {i: 1.0} for i, (_, coord) in enumerate(values)}
    return Head(q, k, v, o)


def pm_bits(lay: Layout, slot: str, n: int) -> dict[int, Row]:
    """Rows reading `slot`'s 0/1 bits as +1/-1 (BOS, whose bits are 0, reads as 0)."""
    return {j: {lay.c(slot, j): 2.0, lay.c("one"): -1.0, lay.c("is_bos"): 1.0} for j in range(n)}


def const_query(lay: Layout, value: int, n: int) -> dict[int, Row]:
    return {j: {lay.c("one"): 2.0 * b - 1.0} for j, b in enumerate(bits(value, n))}


def slot_query(lay: Layout, slot: str, n: int) -> dict[int, Row]:
    return {j: {lay.c(slot, j): 1.0} for j in range(n)}


def slot_values(lay: Layout, src: str, dst: str, n: int) -> tuple[tuple[Row, int], ...]:
    return tuple(({lay.c(src, j): 1.0}, lay.c(dst, j)) for j in range(n))


def memory_head(lay: Layout, sc: Scales, arch: Arch, query: Mapping[int, Row], dst: str) -> Head:
    """memory[query address] -> dst slot. Keys are completed writes (value tokens); BOS reads as 0."""
    a = arch.addr_bits
    return lookup_head(
        lay, sc, arch, query, pm_bits(lay, "cur_addr", a),
        key_gate={lay.c("is_val"): 1.0, lay.c("is_bos"): 1.0, lay.c("is_addr"): -1.0},
        bos_weight=a - 1,
        values=slot_values(lay, "tok", dst, arch.val_bits),
    )


def attention_layers(lay: Layout, sc: Scales, arch: Arch) -> tuple[tuple[Head, ...], ...]:
    a, v = arch.addr_bits, arch.val_bits
    prev_addr_token = lookup_head(  # layer 1: the address token this write belongs to
        lay, sc, arch, query={}, key_bits={},
        key_gate={lay.c("is_addr"): 1.0}, bos_weight=0.0,
        values=slot_values(lay, "tok", "cur_addr", a),
    )
    last_write = lookup_head(  # layer 2: the most recent completed write
        lay, sc, arch, query={}, key_bits={},
        key_gate={lay.c("is_val"): 1.0, lay.c("is_bos"): 1.0}, bos_weight=0.0,
        values=slot_values(lay, "cur_addr", "last_addr", a) + slot_values(lay, "tok", "last_val", v)
        + (({lay.c("is_bos"): 1.0}, lay.c("last_bos")),),
    )
    fetch_pc = memory_head(lay, sc, arch, const_query(lay, arch.pc_addr, a), "pc")
    fetch_x = memory_head(lay, sc, arch, slot_query(lay, "key1", a), "x")
    fetch_y = memory_head(lay, sc, arch, slot_query(lay, "key2", a), "y")
    fetch_ind = memory_head(lay, sc, arch, slot_query(lay, "ind_key", a), "ind")
    return ((prev_addr_token,), (last_write, fetch_pc), (fetch_x, fetch_y), (fetch_ind,))


# ---------------------------------------------------------------------------
# MLPs: exact step functions of integer linear forms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """Adds `out` to the residual stream when sum(weights * residual) >= threshold.

    The linear form must take integer values (up to jitter below 0.25).
    """

    weights: Row
    threshold: int
    out: Row


def gated(lay: Layout, gates: Iterable[str], weights: Row, threshold: int, out: Row, arch: Arch) -> Step:
    """A step that can only fire when every gate feature (each 0/1) is 1."""
    gates = tuple(gates)
    m = float(4 << arch.val_bits)  # larger than any linear form used here
    w = merge(weights, {_c(lay, g): m for g in gates})
    return Step(w, threshold + int(m) * len(gates), out)


def copy_bits(lay: Layout, arch: Arch, gates: Iterable[str], src: str, dst: str, n: int) -> tuple[Step, ...]:
    """dst[j] += src[j] when all gates hold."""
    gates = tuple(gates)
    return tuple(gated(lay, gates, {lay.c(src, j): 1.0}, 1, {lay.c(dst, j): 1.0}, arch) for j in range(n))


def const_bits(lay: Layout, arch: Arch, gates: Iterable[str], value: int, dst: str, n: int) -> tuple[Step, ...]:
    """dst[j] += bit j of value when all gates hold."""
    gates = tuple(gates)
    return tuple(gated(lay, gates, {}, 0, {lay.c(dst, j): 1.0}, arch) for j, b in enumerate(bits(value, n)) if b)


def op_gate(op: str) -> str:
    return f"op:{op}"


def _c(lay: Layout, name: str) -> int:
    """Coordinate of a 1-wide slot, or of an op flag written as 'op:add'."""
    if name.startswith("op:"):
        return lay.c("op", OP_FLAGS.index(name[3:]))
    return lay.c(name)


def decode_row(lay: Layout, arch: Arch, index: int, i: Instr) -> dict[int, float]:
    """What the decode table writes into the residual stream for instruction `index`."""
    a, v = arch.addr_bits, arch.val_bits
    if isinstance(i, Halt):
        return {}
    op = instr_op(i)
    flags = [op.value, "running"] + (["data"] if isinstance(i, (Alu, Load)) else [])
    row = {lay.c("op", OP_FLAGS.index(f)): 1.0 for f in flags}

    def operand(o: Operand | None, key: str, val: str) -> dict[int, float]:
        match o:
            case Cell(addr=addr):
                return {lay.c(key, j): 2.0 * b - 1.0 for j, b in enumerate(bits(addr, a))}
            case Imm(value=value):
                return bit_row(lay, val, value, v)
            case _:
                return {}

    match i:
        case Alu(dst=dst, a=oa, b=ob):
            extra = merge(operand(oa, "key1", "x"), operand(ob, "key2", "y"), bit_row(lay, "dst", dst, a))
        case Load(dst=dst, a=oa):
            extra = merge(operand(oa, "key1", "x"), bit_row(lay, "dst", dst, a))
        case Store(a=oa, b=ob):
            extra = merge(operand(oa, "key1", "x"), operand(ob, "key2", "y"))
        case Jump(a=oa, target=t):
            extra = merge(operand(oa, "key1", "x"), bit_row(lay, "target", t, v))
        case Halt():
            return {}
        case _:
            raise TypeError(i)
    return merge(row, extra, bit_row(lay, "pcnext", (index + 1) % arch.n_values, v))


def mlp_layer2(lay: Layout, arch: Arch, prog: Program) -> tuple[Step, ...]:
    a, v = arch.addr_bits, arch.val_bits
    table = tuple(
        Step({lay.c("pc", j): 2.0 * b - 1.0 for j, b in enumerate(bits(idx, v))}, sum(bits(idx, v)), row)
        for idx, i in enumerate(prog.instrs)
        for row in [decode_row(lay, arch, idx, i)]
        if row
    )
    # exec phase: last write went to the PC cell, or the last "write" is BOS
    exec_ = Step(merge({lay.c("last_addr", j): 2.0 * b - 1.0 for j, b in enumerate(bits(arch.pc_addr, a))},
                       {lay.c("last_bos"): float(a)}),
                 sum(bits(arch.pc_addr, a)), {lay.c("exec"): 1.0, lay.c("upd"): -1.0})
    one = Step({lay.c("one"): 1.0}, 1, {lay.c("upd"): 1.0})
    return table + (exec_, one)


def int_form(lay: Layout, slot: str, n: int, sign: float = 1.0) -> dict[int, float]:
    return {lay.c(slot, j): sign * (1 << j) for j in range(n)}


def mlp_layer3(lay: Layout, arch: Arch) -> tuple[Step, ...]:
    a, v = arch.addr_bits, arch.val_bits
    x, y = int_form(lay, "x", v), int_form(lay, "y", v)
    steps: list[Step] = []
    for j in range(1, v):  # carries into bit j
        lo_x, lo_y, neg_lo_y = int_form(lay, "x", j), int_form(lay, "y", j), int_form(lay, "y", j, -1.0)
        steps.append(Step(merge(lo_x, lo_y), 1 << j, {lay.c("carry_add", j): 1.0}))
        # x - y = x + ~y + 1, and x_lo + (2^j - 1 - y_lo) + 1 >= 2^j  <=>  x_lo - y_lo >= 0
        steps.append(Step(merge(lo_x, neg_lo_y), 0, {lay.c("carry_sub", j): 1.0}))
    steps += [
        Step(x, 1, {lay.c("x_nz"): 1.0}),
        Step({c: -w for c, w in x.items()}, 0, {lay.c("x_z"): 1.0}),
        Step(merge(y, {c: -w for c, w in x.items()}), 1, {lay.c("lt"): 1.0}),
        Step(merge(x, {c: -w for c, w in y.items()}), 0, {lay.c("eq"): 1.0}),
        Step(merge(x, {c: -w for c, w in y.items()}), 1, {lay.c("eq"): -1.0}),
        Step({_c(lay, "op:load"): 1.0}, 1, {lay.c("ind_key", j): -1.0 for j in range(a)}),
        Step({lay.c("is_val"): 1.0, lay.c("is_bos"): 1.0}, 1, {lay.c("emit_addr"): 1.0}),
    ]
    steps += [gated(lay, ["op:load"], {lay.c("x", j): 1.0}, 1, {lay.c("ind_key", j): 2.0}, arch) for j in range(a)]
    return tuple(steps)


def parity_steps(lay: Layout, arch: Arch, gates: tuple[str, ...], form: Row, bias: int, out: int) -> tuple[Step, ...]:
    """out += parity(form + bias) for form + bias in {0,1,2,3}, when the gates hold."""
    return tuple(gated(lay, gates, form, k - bias, {out: sign}, arch) for k, sign in ((1, 1.0), (2, -1.0), (3, 1.0)))


def mlp_layer4(lay: Layout, arch: Arch) -> tuple[Step, ...]:
    a, v = arch.addr_bits, arch.val_bits
    E = "exec"
    VAL, ADDR = "is_addr", "emit_addr"  # which half of the write this position emits
    steps: list[Step] = []
    # --- value candidates ---
    for j in range(v):
        xj, yj = lay.c("x", j), lay.c("y", j)
        add_form = merge({xj: 1.0, yj: 1.0}, {lay.c("carry_add", j): 1.0} if j else {})
        sub_form = merge({xj: 1.0, yj: -1.0}, {lay.c("carry_sub", j): 1.0} if j else {})
        steps += parity_steps(lay, arch, (VAL, E, "op:add"), add_form, 0, lay.c("out", j))
        steps += parity_steps(lay, arch, (VAL, E, "op:sub"), sub_form, 1 if j else 2, lay.c("out", j))  # x_j + ~y_j + carry
        both = {xj: 1.0, yj: 1.0}
        steps.append(gated(lay, (VAL, E, "op:and"), both, 2, {lay.c("out", j): 1.0}, arch))
        steps.append(gated(lay, (VAL, E, "op:or"), both, 1, {lay.c("out", j): 1.0}, arch))
        steps.append(gated(lay, (VAL, E, "op:xor"), both, 1, {lay.c("out", j): 1.0}, arch))
        steps.append(gated(lay, (VAL, E, "op:xor"), both, 2, {lay.c("out", j): -1.0}, arch))
    steps.append(gated(lay, (VAL, E, "op:lt"), {lay.c("lt"): 1.0}, 1, {lay.c("out", 0): 1.0}, arch))
    steps.append(gated(lay, (VAL, E, "op:eq"), {lay.c("eq"): 1.0}, 1, {lay.c("out", 0): 1.0}, arch))
    steps += copy_bits(lay, arch, (VAL, E, "op:load"), "ind", "out", v)
    steps += copy_bits(lay, arch, (VAL, E, "op:store"), "y", "out", v)
    steps += copy_bits(lay, arch, (VAL, E, "op:jmp"), "target", "out", v)
    steps += copy_bits(lay, arch, (VAL, E, "op:jz", "x_z"), "target", "out", v)
    steps += copy_bits(lay, arch, (VAL, E, "op:jz", "x_nz"), "pcnext", "out", v)
    steps += copy_bits(lay, arch, (VAL, E, "op:jnz", "x_nz"), "target", "out", v)
    steps += copy_bits(lay, arch, (VAL, E, "op:jnz", "x_z"), "pcnext", "out", v)
    steps += copy_bits(lay, arch, (VAL, "upd"), "pcnext", "out", v)
    # --- address candidates (share the low bits of `out`) ---
    steps += copy_bits(lay, arch, (ADDR, E, "op:data"), "dst", "out", a)
    steps += copy_bits(lay, arch, (ADDR, E, "op:store"), "x", "out", a)
    steps += const_bits(lay, arch, (ADDR, E, "op:jmp"), arch.pc_addr, "out", a)
    steps += const_bits(lay, arch, (ADDR, E, "op:jz"), arch.pc_addr, "out", a)
    steps += const_bits(lay, arch, (ADDR, E, "op:jnz"), arch.pc_addr, "out", a)
    steps += const_bits(lay, arch, (ADDR, "upd"), arch.pc_addr, "out", a)
    # --- which kind of token to emit ---
    emits_addr = {lay.c("is_val"): 1.0, lay.c("is_bos"): 1.0}
    kind_addr, kind_val, kind_eos = (lay.c("kind", i) for i in range(3))
    steps += [
        Step(emits_addr, 1, {kind_addr: 1.0}),
        Step(merge(emits_addr, {lay.c(E): 1.0, _c(lay, "op:running"): -1.0}), 2, {kind_eos: 1.0, kind_addr: -1.0}),
        Step({lay.c("is_addr"): 1.0}, 1, {kind_val: 1.0}),
    ]
    return tuple(steps)


# ---------------------------------------------------------------------------
# materialize
# ---------------------------------------------------------------------------


def dense(rows: Mapping[int, Row], shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, dtype=np.float32)
    for r, row in rows.items():
        for c, w in row.items():
            m[r, c] = w
    return m


def attn_weights(heads: tuple[Head, ...], n_heads: int, dim: int) -> dict[str, np.ndarray]:
    def proj(which: str) -> np.ndarray:
        rows = {h * HEAD_DIM + d: row for h, head in enumerate(heads) for d, row in getattr(head, which).items()}
        return dense(rows, (n_heads * HEAD_DIM, dim))

    o_rows: dict[int, dict[int, float]] = {}
    for h, head in enumerate(heads):
        for coord, row in head.o.items():
            o_rows[coord] = merge(o_rows.get(coord, {}), {h * HEAD_DIM + d: w for d, w in row.items()})
    return {"q_proj": proj("q"), "k_proj": proj("k"), "v_proj": proj("v"), "o_proj": dense(o_rows, (dim, n_heads * HEAD_DIM))}


def mlp_weights(lay: Layout, steps: tuple[Step, ...], inner: int) -> dict[str, np.ndarray]:
    """Each step is two SiLU units: step(z >= t) = (2/K) [silu(K(z - t + .75)) - silu(K(z - t + .25))]."""
    gate: dict[int, dict[int, float]] = {}
    up: dict[int, dict[int, float]] = {}
    down: dict[int, dict[int, float]] = {}
    one = lay.c("one")
    for i, s in enumerate(steps):
        for u, (offset, sign) in enumerate(((0.75, 1.0), (0.25, -1.0))):
            r = 2 * i + u
            gate[r] = merge({c: STEP_K * w for c, w in s.weights.items()}, {one: STEP_K * (offset - s.threshold)})
            up[r] = {one: 1.0}
            for c, w in s.out.items():
                down.setdefault(c, {})[r] = sign * 2.0 / STEP_K * w
    assert 2 * len(steps) <= inner
    return {
        "gate_proj": dense(gate, (inner, lay.dim)),
        "up_proj": dense(up, (inner, lay.dim)),
        "down_proj": dense(down, (lay.dim, inner)),
    }


def norm_weight(lay: Layout) -> np.ndarray:
    """RMSNorm scale that maps every coordinate to itself and `one` to 1, given `one` = BIG_CONST."""
    g = np.full(lay.dim, BIG_CONST / math.sqrt(lay.dim), dtype=np.float32)
    g[lay.c("one")] = 1.0 / math.sqrt(lay.dim)
    return g


def embeddings(lay: Layout, arch: Arch, voc: Vocab) -> np.ndarray:
    e = np.zeros((voc.size, lay.dim), dtype=np.float32)
    e[:, lay.c("one")] = BIG_CONST
    for t in range(voc.size):
        if t < voc.value_base:
            e[t, lay.c("is_addr")] = 1.0
            value = t
        elif t < voc.bos:
            e[t, lay.c("is_val")] = 1.0
            value = t - voc.value_base
        else:
            e[t, lay.c("is_bos")] = float(t == voc.bos)
            continue
        for j, b in enumerate(bits(value, arch.val_bits)):
            e[t, lay.c("tok", j)] = b
    return e


def unembedding(lay: Layout, arch: Arch, voc: Vocab) -> np.ndarray:
    """logit(token) = scale * (kind match * KK + agreement of the token's bits with `out`)."""
    w = arch.val_bits
    kk = 2.0 * w + 2.0
    m = np.zeros((voc.size, lay.dim), dtype=np.float32)
    kind_addr, kind_val, kind_eos = (lay.c("kind", i) for i in range(3))
    for t in range(voc.size):
        if t == voc.bos:
            m[t, lay.c("one")] = -LOGIT_SCALE * kk
            continue
        if t == voc.eos:
            m[t, kind_eos] = LOGIT_SCALE * kk
            continue
        is_addr = t < voc.value_base
        value, n = (t, arch.addr_bits) if is_addr else (t - voc.value_base, w)
        m[t, kind_addr if is_addr else kind_val] = LOGIT_SCALE * kk
        for j, b in enumerate(bits(value, n)):  # agreement = sum (2b-1)(2out-1)
            m[t, lay.c("out", j)] = LOGIT_SCALE * 2.0 * (2 * b - 1)
            m[t, lay.c("one")] += -LOGIT_SCALE * (2 * b - 1)
    return m


@dataclass(frozen=True)
class Compiled:
    config: dict
    weights: dict[str, np.ndarray]
    layout: Layout
    scales: Scales

    @property
    def n_params(self) -> int:
        return sum(int(w.size) for w in self.weights.values())


def compile_program(prog: Program, arch: Arch = Arch()) -> Compiled:
    assert arch.addr_bits <= arch.val_bits, "values must be wide enough to hold an address"
    n_heads = 3
    lay = make_layout(arch, n_heads)
    sc = Scales.for_arch(arch)
    voc = Vocab(arch)
    layers_attn = attention_layers(lay, sc, arch)
    layers_mlp = ((), mlp_layer2(lay, arch, prog), mlp_layer3(lay, arch), mlp_layer4(lay, arch))
    inner = -(-max(2 * len(s) for s in layers_mlp) // 64) * 64
    weights: dict[str, np.ndarray] = {"model.embed_tokens.weight": embeddings(lay, arch, voc)}
    for i, (heads, steps) in enumerate(zip(layers_attn, layers_mlp)):
        for name, w in attn_weights(heads, n_heads, lay.dim).items():
            weights[f"model.layers.{i}.self_attn.{name}.weight"] = w
        for name, w in mlp_weights(lay, steps, inner).items():
            weights[f"model.layers.{i}.mlp.{name}.weight"] = w
        weights[f"model.layers.{i}.input_layernorm.weight"] = norm_weight(lay)
        weights[f"model.layers.{i}.post_attention_layernorm.weight"] = norm_weight(lay)
    weights["model.norm.weight"] = norm_weight(lay)
    weights["lm_head.weight"] = unembedding(lay, arch, voc)
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": voc.size,
        "hidden_size": lay.dim,
        "intermediate_size": inner,
        "num_hidden_layers": len(layers_attn),
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_heads,
        "head_dim": HEAD_DIM,
        "hidden_act": "silu",
        "max_position_embeddings": arch.max_positions,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_type": "default", "rope_theta": sc.rope_theta},
        "attention_bias": False,
        "mlp_bias": False,
        "tie_word_embeddings": False,
        "bos_token_id": voc.bos,
        "eos_token_id": voc.eos,
        "dtype": "float32",
        "use_cache": True,
        "llmc": {"addr_bits": arch.addr_bits, "val_bits": arch.val_bits, "pc_addr": arch.pc_addr,
                 "symbols": dict(prog.symbols), "n_instructions": len(prog.instrs)},
    }
    return Compiled(config, weights, lay, sc)


def save(c: Compiled, out_dir: str, arch: Arch) -> None:
    os.makedirs(out_dir, exist_ok=True)
    voc = Vocab(arch)
    save_file(c.weights, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        f.write(dumps(c.config))
    with open(os.path.join(out_dir, "generation_config.json"), "w") as f:
        f.write(dumps({"bos_token_id": voc.bos, "eos_token_id": voc.eos, "do_sample": False,
                       "max_length": arch.max_positions}))
    with open(os.path.join(out_dir, "tokenizer.json"), "w") as f:
        f.write(dumps(voc.tokenizer_json()))
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w") as f:
        f.write(dumps(voc.tokenizer_config_json()))
