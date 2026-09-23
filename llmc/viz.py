"""Render a video of the compiled transformer running a program.

Every number on screen is read out of the real HF model's activations: the
residual stream after each layer, which past token each attention head picked,
and which MLP units fired. Nothing is simulated.

    uv run python -m llmc.viz out/fib --mem n=6 -o fib.mp4 [--gif fib.gif]
"""
from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path  # noqa: E402

from .asm import assemble, instruction_lines, parse_memory  # noqa: E402
from .compile import STEP_K, Layout, make_layout  # noqa: E402
from .interp import Write, start_writes  # noqa: E402
from .isa import Alu, Cell, Imm, Instr, Jump, Load, Operand, Store  # noqa: E402
from .run import Model, generate, load  # noqa: E402
from .tokens import Vocab  # noqa: E402

# --- palette (dark surface; see the dataviz reference palette) --------------
SURFACE = "#1a1a19"
TEXT = "#ffffff"
TEXT2 = "#c3c2b7"
MUTED = "#8a8980"
ZERO = "#383835"  # diverging midpoint: a residual coordinate holding 0
NEG, POS = "#e66767", "#3987e5"  # diverging arms: -1 red, +1 blue
FIRED, IDLE = "#c98500", "#2a2a28"  # MLP unit fired / idle
DATA_TOK, PC_TOK, FUTURE_TOK = "#199e70", "#5f5e58", "#242423"
HILITE = "#2e2e2b"

RESIDUAL_CMAP = LinearSegmentedColormap.from_list("resid", [NEG, ZERO, POS])
MLP_CMAP = ListedColormap([IDLE, FIRED])
MONO = "DejaVu Sans Mono"

HIDDEN_SLOTS = ("one", "pad")
HEAD_ROLES = {  # (layer, head) -> what the head looks up
    (0, 0): "address",
    (1, 0): "last write",
    (1, 1): "pc",
    (2, 0): "a",
    (2, 1): "b",
    (3, 0): "mem[a]",
}


# ---------------------------------------------------------------------------
# capture: one forward pass over the finished run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Capture:
    name: str
    source: str
    instrs: tuple[Instr, ...]
    symbols: dict[str, int]
    pc_addr: int
    n_params: int
    voc: Vocab
    layout: Layout
    ids: tuple[int, ...]  # BOS, prompt, generated tokens (without the final EOS)
    n_prompt: int  # tokens before the first generated one
    halted: bool
    hidden: np.ndarray  # [layers + 1, T, d]
    attn_argmax: np.ndarray  # [layers, heads, T] position each head attends to
    mlp_fired: tuple[np.ndarray, ...]  # per layer: [T, used steps] bool
    logits_top: np.ndarray  # [T] predicted next token


def _used_steps(gate: np.ndarray) -> np.ndarray:
    """Indices of MLP steps (unit pairs) that have any weights."""
    pairs = np.abs(gate).sum(axis=1).reshape(-1, 2).sum(axis=1)
    return np.nonzero(pairs)[0]


def capture(model_dir: str, mem: str) -> Capture:
    m: Model = load(model_dir, "eager")
    with open(os.path.join(model_dir, "program.asm")) as f:
        source = f.read()
    prog = assemble(source, m.arch)
    voc = Vocab(m.arch)
    init = start_writes(parse_memory(mem, m.symbols), m.arch)
    trace = generate(m, init)
    prompt = voc.encode(init)
    ids = voc.encode(init + trace)
    halted = len(ids) < m.arch.max_positions - 1

    layers = m.model.model.layers
    acts: dict[int, torch.Tensor] = {}
    hooks = [
        layer.mlp.down_proj.register_forward_hook(lambda _mod, inp, _out, i=i: acts.__setitem__(i, inp[0][0]))
        for i, layer in enumerate(layers)
    ]
    try:
        with torch.no_grad():
            out = m.model(torch.tensor([ids]), output_hidden_states=True, output_attentions=True)
    finally:
        for h in hooks:
            h.remove()

    def fired(i: int) -> np.ndarray:
        used = _used_steps(layers[i].mlp.gate_proj.weight.detach().numpy())
        a = acts[i].numpy()
        step = (2.0 / STEP_K) * (a[:, 2 * used] - a[:, 2 * used + 1])
        return step > 0.5

    return Capture(
        name=os.path.basename(os.path.normpath(model_dir)),
        source=source,
        instrs=prog.instrs,
        symbols=dict(m.symbols),
        pc_addr=m.pc_addr,
        n_params=sum(p.numel() for p in m.model.parameters()),
        voc=voc,
        layout=make_layout(m.arch, m.model.config.num_attention_heads),
        ids=tuple(ids),
        n_prompt=len(prompt),
        halted=halted,
        hidden=np.stack([h[0].numpy() for h in out.hidden_states]),
        attn_argmax=np.stack([a[0].argmax(-1).numpy() for a in out.attentions]),
        mlp_fired=tuple(fired(i) for i in range(len(layers))),
        logits_top=out.logits[0].argmax(-1).numpy(),
    )


# ---------------------------------------------------------------------------
# reading the residual stream
# ---------------------------------------------------------------------------


def slot(c: Capture, layer: int, pos: int, name: str) -> np.ndarray:
    lay = c.layout
    return c.hidden[layer, pos, lay.off(name): lay.off(name) + lay.width(name)]


def as_int(bits: np.ndarray) -> int:
    return int(sum(int(round(b)) << j for j, b in enumerate(bits) if round(b) > 0))


def flag(c: Capture, layer: int, pos: int, name: str) -> bool:
    return bool(slot(c, layer, pos, name)[0] > 0.5)


def cell_name(c: Capture, addr: int) -> str:
    names = {a: n for n, a in c.symbols.items()}
    return names.get(addr, f"[{addr}]")


def writes_upto(c: Capture, n_tokens: int) -> tuple[Write, ...]:
    return c.voc.decode(c.ids[:n_tokens])


def token_label(c: Capture, pos: int) -> str:
    t = c.ids[pos]
    if t < c.voc.value_base:
        return f"@{cell_name(c, t)}"
    return c.voc.token_text(t)


def operand_text(c: Capture, o: Operand | None, value: int) -> str:
    match o:
        case Cell(addr=a):
            return f"mem[{cell_name(c, a)}] = {value}"
        case Imm(value=v):
            return f"#{v}"
    return ""


def readout(c: Capture, pos: int) -> tuple[str, ...]:
    """One line per stage (input, layers 1-4, output), decoded from the activations at `pos`."""
    L = len(c.hidden) - 1
    is_addr = flag(c, 0, pos, "is_addr")
    exec_ = flag(c, L, pos, "exec")
    pc = as_int(slot(c, 2, pos, "pc"))
    instr = c.instrs[pc] if exec_ and pc < len(c.instrs) else None
    nxt = int(c.logits_top[pos])
    in_text = token_label(c, pos) if c.ids[pos] != c.voc.bos else "<bos>"
    half = "address" if is_addr else "value" if c.ids[pos] != c.voc.bos else "start"

    l1 = (f"attention: this value belongs to address @{cell_name(c, as_int(slot(c, 1, pos, 'cur_addr')))}"
          if not is_addr and c.ids[pos] != c.voc.bos else "attention: (nothing to do for this token)")
    last = "<bos>" if flag(c, 2, pos, "last_bos") else (
        f"{cell_name(c, as_int(slot(c, 2, pos, 'last_addr')))}={as_int(slot(c, 2, pos, 'last_val'))}")
    if instr is not None:
        src = c.source.splitlines()[instruction_lines(c.source)[pc]].split(";")[0].strip()
        l2 = f"attention: pc = {pc}, last write {last}   MLP decodes  {pc}: {src}"
    elif exec_:
        l2 = f"attention: pc = {pc}, last write {last}   MLP decodes  {pc}: halt"
    else:
        l2 = f"attention: pc = {pc}, last write {last}   MLP: data was written, so advance pc"
    x, y = as_int(slot(c, 3, pos, "x")), as_int(slot(c, 3, pos, "y"))
    a_op = getattr(instr, "a", None)
    b_op = getattr(instr, "b", None)
    reads = [operand_text(c, o, v) for o, v in ((a_op, x), (b_op, y)) if isinstance(o, Cell)]
    imms = [operand_text(c, o, v) for o, v in ((a_op, x), (b_op, y)) if isinstance(o, Imm)]
    l3 = ("attention reads " + "   ".join(reads) if reads else "attention: no memory reads") + (
        f"   (immediate {' '.join(imms)})" if imms else "")
    if isinstance(instr, Jump) and instr.a is not None:
        l3 += f"   MLP: a == 0 is {flag(c, 3, pos, 'x_z')}"
    out_val = as_int(slot(c, L, pos, "out"))
    if isinstance(instr, Load):
        l4 = f"attention reads mem[{x}] = {as_int(slot(c, 4, pos, 'ind'))}   "
    else:
        l4 = ""
    if nxt == c.voc.eos:
        l4 += "MLP: halt"
    elif is_addr:
        l4 += f"MLP selects the value to write: {out_val}"
    else:
        l4 += f"MLP selects the address to write: {cell_name(c, out_val)}"
    return (
        f"input     {in_text}   ({half} token)",
        f"layer 1   {l1}",
        f"layer 2   {l2}",
        f"layer 3   {l3}",
        f"layer 4   {l4}",
        f"output    {c.voc.token_text(nxt) if nxt >= c.voc.value_base else '@' + cell_name(c, nxt)}",
    )


def arcs(c: Capture, pos: int) -> tuple[tuple[int, int, str], ...]:
    """(layer, attended position, role) for the heads whose lookup matters at `pos`."""
    L = len(c.hidden) - 1
    is_addr = flag(c, 0, pos, "is_addr")
    exec_ = flag(c, L, pos, "exec")
    pc = as_int(slot(c, 2, pos, "pc"))
    instr = c.instrs[pc] if exec_ and pc < len(c.instrs) else None

    def wanted(layer: int, head: int) -> bool:
        match (layer, head):
            case (0, 0):
                return not is_addr and c.ids[pos] != c.voc.bos
            case (1, _):
                return True
            case (2, 0):
                return isinstance(getattr(instr, "a", None), Cell)
            case (2, 1):
                return isinstance(getattr(instr, "b", None), Cell)
            case (3, 0):
                return isinstance(instr, Load)
        return False

    def role(layer: int, head: int) -> str:
        match (layer, head):
            case (2, 0):
                return cell_name(c, instr.a.addr)  # type: ignore[union-attr]
            case (2, 1):
                return cell_name(c, instr.b.addr)  # type: ignore[union-attr]
            case (3, 0):
                return f"mem[{cell_name(c, instr.a.addr)}]" if isinstance(instr.a, Cell) else "mem[#]"  # type: ignore[union-attr]
        return HEAD_ROLES[(layer, head)]

    return tuple(
        (layer, int(c.attn_argmax[layer, head, pos]), role(layer, head))
        for (layer, head) in HEAD_ROLES
        if wanted(layer, head)
    )


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

W, H, DPI = 1920, 1080, 100
LEFT_X, NET_X0, NET_X1 = 0.025, 0.335, 0.975
TAPE_Y, TAPE_H = 0.875, 0.022


@dataclass
class Scene:
    fig: plt.Figure
    c: Capture
    resid_ims: list
    resid_axes: list
    mlp_ims: list
    mlp_axes: list
    tape: list
    dyn: list  # artists redrawn every frame


def _residual_view(c: Capture) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """Column indices of the visible residual coordinates, and (slot, start col, width)."""
    cols, spans = [], []
    for name, w in c.layout.slots:
        if name in HIDDEN_SLOTS:
            continue
        spans.append((name, len(cols), w))
        cols += list(range(c.layout.off(name), c.layout.off(name) + w))
    return np.array(cols), spans


def build_scene(c: Capture) -> Scene:
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=SURFACE)
    cols, spans = _residual_view(c)
    n_cols = len(cols)

    # left column: title
    fig.text(LEFT_X, 0.945, f"{c.name}.asm", color=TEXT, fontsize=30, weight="bold", va="center")
    fig.text(LEFT_X, 0.905, f"running on a {c.n_params / 1e6:.1f}M-parameter Llama", color=TEXT2, fontsize=15,
             va="center")
    fig.text(LEFT_X, 0.877, "weights compiled from the program, not trained", color=MUTED, fontsize=12, va="center")

    # network rows
    rows: list[tuple[str, str, int]] = [("resid", "embedding", 0)]
    for layer in range(1, len(c.hidden)):
        if c.mlp_fired[layer - 1].shape[1]:
            rows.append(("mlp", f"layer {layer} MLP", layer - 1))
        rows.append(("resid", f"after layer {layer}", layer))
    heights = {"resid": 0.062, "mlp": 0.026}
    gap = 0.011
    y = 0.785
    resid_ims, resid_axes, mlp_ims, mlp_axes = [], [], [], []
    for kind, label, idx in rows:
        h = heights[kind]
        y -= h
        ax = fig.add_axes((NET_X0, y, NET_X1 - NET_X0, h))
        ax.set_axis_off()
        if kind == "resid":
            im = ax.imshow(np.zeros((1, n_cols)), cmap=RESIDUAL_CMAP, vmin=-1, vmax=1, aspect="auto",
                           interpolation="nearest")
            ax.vlines(np.arange(n_cols - 1) + 0.5, -0.5, 0.5, color=SURFACE, lw=0.6)
            for _, start, _w in spans[1:]:
                ax.axvline(start - 0.5, color=SURFACE, lw=3)
            resid_ims.append(im)
            resid_axes.append(ax)
        else:
            n = c.mlp_fired[idx].shape[1]
            im = ax.imshow(np.zeros((1, n)), cmap=MLP_CMAP, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
            ax.set_xlim(-0.5, max(m.shape[1] for m in c.mlp_fired) - 0.5)
            mlp_ims.append(im)
            mlp_axes.append(ax)
        fig.text(NET_X0 - 0.008, y + h / 2, label, color=TEXT2 if kind == "resid" else MUTED,
                 fontsize=11 if kind == "resid" else 9.5, ha="right", va="center")
        y -= gap
    # slot names under the last residual row
    bottom_ax = resid_axes[-1]
    for name, start, w in spans:
        if w < 3:
            continue
        xf = NET_X0 + (NET_X1 - NET_X0) * (start + w / 2) / n_cols
        fig.text(xf, y + gap - 0.006, name, color=MUTED, fontsize=8, rotation=90, ha="center", va="top",
                 family=MONO)
    fig.text(NET_X0, 0.805, "residual stream: each column is one coordinate, grouped by what it holds.  ",
             color=MUTED, fontsize=10, va="center")
    _legend(fig)

    # token tape
    fig.text(NET_X0 - 0.008, TAPE_Y + TAPE_H / 2, "tokens", color=TEXT2, fontsize=11, ha="right", va="center")
    T = len(c.ids)
    wcell = (NET_X1 - NET_X0) / T
    tape = []
    for p in range(T):
        r = Rectangle((NET_X0 + p * wcell, TAPE_Y), wcell * 0.82, TAPE_H, transform=fig.transFigure,
                      facecolor=FUTURE_TOK, edgecolor="none")
        fig.add_artist(r)
        tape.append(r)
    return Scene(fig, c, resid_ims, resid_axes, mlp_ims, mlp_axes, tape, [])


def _legend(fig: plt.Figure) -> None:
    """Right-aligned swatch legend, measured so swatches sit next to their labels."""
    renderer = fig.canvas.get_renderer()
    items = [(POS, "+1"), (ZERO, "0"), (NEG, "\u22121"), (FIRED, "MLP unit fired"), (DATA_TOK, "data write"),
             (PC_TOK, "pc write")]
    x = NET_X1
    for color, text in reversed(items):
        t = fig.text(x, 0.805, text, color=TEXT2, fontsize=10, ha="right", va="center")
        x -= t.get_window_extent(renderer).width / W + 0.004
        fig.add_artist(Rectangle((x - 0.008, 0.799), 0.008, 0.012, transform=fig.transFigure, facecolor=color,
                                     edgecolor="none"))
        x -= 0.008 + 0.016


def _tape_color(c: Capture, pos: int) -> str:
    t = c.ids[pos]
    if t == c.voc.bos:
        return TEXT2
    addr = t if t < c.voc.value_base else c.ids[pos - 1]
    return PC_TOK if addr == c.pc_addr else DATA_TOK


def _arc(fig: plt.Figure, x0: float, x1: float, y: float, height: float, color: str, lw: float) -> PathPatch:
    verts = [(x0, y), (x0, y + height), (x1, y + height), (x1, y)]
    p = PathPatch(Path(verts, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]), transform=fig.transFigure,
                  facecolor="none", edgecolor=color, lw=lw, capstyle="round")
    fig.add_artist(p)
    return p


def draw_frame(s: Scene, pos: int, stage: int) -> None:
    """Show the forward pass at `pos` computed up to `stage` (0 = input only, 5 = output emitted)."""
    c, fig = s.c, s.fig
    for a in s.dyn:
        a.remove()
    s.dyn = []
    cols, _ = _residual_view(c)
    n_layers = len(c.hidden) - 1

    # network
    for i, im in enumerate(s.resid_ims):
        im.set_data(np.clip(c.hidden[i, pos, cols], -1, 1)[None, :])
        im.set_alpha(1.0 if i <= stage else 0.12)
    for j, im in enumerate(s.mlp_ims):
        layer = [k for k in range(n_layers) if c.mlp_fired[k].shape[1]][j]
        im.set_data(c.mlp_fired[layer][pos][None, :].astype(float))
        im.set_alpha(1.0 if layer + 1 <= stage else 0.12)
    if 1 <= stage <= n_layers:  # frame the layer being computed
        ax = s.resid_axes[stage]
        bb = ax.get_position()
        r = Rectangle((bb.x0 - 0.003, bb.y0 - 0.004), bb.width + 0.006, bb.height + 0.008, transform=fig.transFigure,
                      facecolor="none", edgecolor=TEXT, lw=1.2)
        fig.add_artist(r)
        s.dyn.append(r)

    # tape: tokens up to pos are known; pos + 1 appears at stage 5
    shown = pos + (2 if stage >= 5 else 1)
    for p, r in enumerate(s.tape):
        r.set_facecolor(_tape_color(c, p) if p < min(shown, len(c.ids)) else FUTURE_TOK)
    T = len(c.ids)
    wcell = (NET_X1 - NET_X0) / T
    xc = lambda p: NET_X0 + (p + 0.41) * wcell  # noqa: E731
    marker = fig.text(xc(pos), TAPE_Y - 0.004, "▲", color=TEXT, fontsize=10, ha="center", va="top")
    s.dyn.append(marker)
    visible = [(layer, target, role) for layer, target, role in arcs(c, pos) if layer + 1 <= stage and target != pos]
    for k, (layer, target, role) in enumerate(sorted(visible, key=lambda a: -a[1])):
        current = layer + 1 == stage or stage > n_layers
        color = TEXT if current else MUTED
        height = 0.018 + 0.024 * k
        s.dyn.append(_arc(fig, xc(pos), xc(target), TAPE_Y + TAPE_H, height, color, 1.6 if current else 0.9))
        label = role if c.ids[target] != c.voc.bos else f"{role} (unset: 0)"
        mid = max((xc(pos) + xc(target)) / 2, NET_X0 + 0.004 * len(label))
        s.dyn.append(fig.text(mid, TAPE_Y + TAPE_H + height * 0.75 + 0.003, label, color=color, fontsize=10,
                              ha="center", va="bottom", family=MONO,
                              bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0)))

    # recent writes
    writes = writes_upto(c, shown)
    budget, recent = 0, []
    for w in reversed(writes):
        budget += len(f"{cell_name(c, w[0])}={w[1]}") + 2
        if budget > 80:
            break
        recent.insert(0, w)
    x = NET_X0
    for k, (a, v) in enumerate(recent):
        latest = k == len(recent) - 1 and stage >= 5 and c.ids[min(pos + 1, T - 1)] >= c.voc.value_base
        txt = f"{cell_name(c, a)}={v}"
        t = fig.text(x, 0.838, txt, color=TEXT if latest else (MUTED if a == c.pc_addr else TEXT2), fontsize=12,
                     family=MONO, va="center", weight="bold" if latest else "normal")
        s.dyn.append(t)
        x += 0.0072 * (len(txt) + 2)
    s.dyn.append(fig.text(NET_X1, 0.838, f"token {pos + 1 - c.n_prompt + 1} / {T - c.n_prompt + 1}", color=MUTED,
                          fontsize=11, ha="right", va="center"))

    # readout
    lines = readout(c, pos)
    for k, line in enumerate(lines):
        active = k == stage
        s.dyn.append(fig.text(NET_X0 - 0.06, 0.225 - k * 0.034, line, color=TEXT if active else MUTED, fontsize=13,
                              family=MONO, va="center", weight="bold" if active else "normal"))

    _draw_program(s, pos, stage)
    _draw_memory(s, writes)


def _draw_program(s: Scene, pos: int, stage: int) -> None:
    c, fig = s.c, s.fig
    n_layers = len(c.hidden) - 1
    lines = [ln.rstrip() for ln in c.source.splitlines()]
    body = [(n, ln) for n, ln in enumerate(lines) if ln.strip() and not ln.strip().startswith(";")
            and not ln.strip().startswith("let ")]
    exec_ = flag(c, n_layers, pos, "exec")
    pc = as_int(slot(c, 2, pos, "pc"))
    hl_line = instruction_lines(c.source)[pc] if exec_ and pc < len(c.instrs) and stage >= 2 else None
    y0 = 0.815
    dy = min(0.026, 0.40 / max(1, len(body)))
    s.dyn.append(fig.text(LEFT_X, y0 + 0.012, "program", color=TEXT2, fontsize=13, weight="bold", va="bottom"))
    idx = {n: i for i, n in enumerate(instruction_lines(c.source))}
    for k, (n, ln) in enumerate(body):
        y = y0 - (k + 0.5) * dy
        code = ln.split(";")[0].rstrip()
        hot = n == hl_line
        if hot:
            r = FancyBboxPatch((LEFT_X - 0.006, y - dy * 0.45), 0.19, dy * 0.9, boxstyle="round,pad=0,rounding_size=0.004",
                               transform=fig.transFigure, facecolor=HILITE, edgecolor=DATA_TOK, lw=1.2)
            fig.add_artist(r)
            s.dyn.append(r)
        prefix = f"{idx[n]:>2}  " if n in idx else "    "
        s.dyn.append(fig.text(LEFT_X, y, prefix + code, color=TEXT if hot else TEXT2, fontsize=min(12, 460 * dy),
                              family=MONO,
                              va="center", weight="bold" if hot else "normal"))
    s.body_end = y0 - (len(body) + 0.5) * dy  # type: ignore[attr-defined]


def _draw_memory(s: Scene, writes: tuple[Write, ...]) -> None:
    c, fig = s.c, s.fig
    mem = dict(writes)
    last = writes[-1] if writes else None
    y = getattr(s, "body_end", 0.4) - 0.03
    s.dyn.append(fig.text(LEFT_X, y, "memory", color=TEXT2, fontsize=13, weight="bold", va="bottom"))
    y -= 0.014
    named = sorted(c.symbols.items(), key=lambda kv: (kv[1] == c.pc_addr, kv[1]))
    extra_rows = 2 if any(a not in c.symbols.values() for a in mem) else 0
    dy = min(0.026, (y - 0.03) / (len(named) + extra_rows + 1))
    for name, addr in named:
        y -= dy
        hot = last is not None and last[0] == addr
        val = mem.get(addr, 0)
        s.dyn.append(fig.text(LEFT_X, y, name, color=TEXT2, fontsize=12, family=MONO, va="center"))
        s.dyn.append(fig.text(LEFT_X + 0.05, y, f"@{addr}", color=MUTED, fontsize=10, family=MONO, va="center"))
        s.dyn.append(fig.text(LEFT_X + 0.13, y, f"{val}", color=TEXT if hot else TEXT2, fontsize=12, family=MONO,
                              va="center", ha="right", weight="bold" if hot else "normal"))
    extra = sorted((a, v) for a, v in mem.items() if a not in c.symbols.values())
    if extra:
        y -= 0.032
        chunks = [f"[{a}]={v}" for a, v in extra]
        line, lines = "", []
        for ch in chunks:
            if len(line) + len(ch) > 34:
                lines.append(line)
                line = ""
            line += ch + " "
        lines.append(line)
        for ln in lines[:3]:
            s.dyn.append(fig.text(LEFT_X, y, ln, color=TEXT2, fontsize=11, family=MONO, va="center"))
            y -= 0.024


# ---------------------------------------------------------------------------
# schedule and encoding
# ---------------------------------------------------------------------------


def schedule(c: Capture, slow_tokens: int, hold: int) -> tuple[tuple[int, int], ...]:
    """(position, stage) per frame: the first `slow_tokens` tokens step through every layer, the rest are one
    frame each."""
    positions = range(c.n_prompt - 1, len(c.ids))
    frames: list[tuple[int, int]] = []
    for k, p in enumerate(positions):
        if k < slow_tokens:
            frames += [(p, st) for st in range(6) for _ in range(hold)]
        else:
            frames.append((p, 5))
    last = frames[-1]
    return tuple(frames) + (last,) * 36  # linger on the final state


def render(c: Capture, out: str, fps: int, slow_tokens: int, hold: int, gif: str | None, gif_width: int) -> None:
    s = build_scene(c)
    frames = schedule(c, slow_tokens, hold)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r",
           str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow",
           "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for k, (pos, stage) in enumerate(frames):
        draw_frame(s, pos, stage)
        s.fig.canvas.draw()
        proc.stdin.write(bytes(s.fig.canvas.buffer_rgba()))
        if k % 50 == 0:
            print(f"\rframe {k}/{len(frames)}", end="", flush=True)
    proc.stdin.close()
    proc.wait()
    print(f"\r{len(frames)} frames -> {out} ({len(frames) / fps:.1f}s)")
    if gif:
        vf = f"fps={min(fps, 15)},scale={gif_width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];" \
             "[b][p]paletteuse=dither=none"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out, "-vf", vf, gif], check=True)
        print(f"-> {gif}")


def still(c: Capture, out: str, pos: int | None, stage: int) -> None:
    s = build_scene(c)
    draw_frame(s, c.n_prompt - 1 + (pos or 0), stage)
    s.fig.savefig(out, facecolor=SURFACE)
    print(f"-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="render the compiled model running a program")
    ap.add_argument("model_dir")
    ap.add_argument("--mem", default="")
    ap.add_argument("-o", "--out", default="run.mp4")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--slow", type=int, default=12, help="tokens shown layer by layer before speeding up")
    ap.add_argument("--hold", type=int, default=6, help="frames per layer while slow")
    ap.add_argument("--gif", help="also write a GIF here")
    ap.add_argument("--gif-width", type=int, default=960)
    ap.add_argument("--still", type=int, help="write one PNG of this generated token instead of a video")
    ap.add_argument("--stage", type=int, default=5)
    args = ap.parse_args()
    c = capture(args.model_dir, args.mem)
    if args.still is not None:
        still(c, args.out, args.still, args.stage)
    else:
        render(c, args.out, args.fps, args.slow, args.hold, args.gif, args.gif_width)


if __name__ == "__main__":
    main()
