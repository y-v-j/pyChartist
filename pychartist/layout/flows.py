"""Geometry for flow / relationship diagrams: sankey, chord, arc, layered DAG (flowchart)."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def bezier(p0, p1, p2, p3, n: int = 30) -> np.ndarray:
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, float) for p in (p0, p1, p2, p3))
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3


# ------------------------------------------------------------------------- sankey
def sankey_layout(nodes: list[str], links: list[tuple[int, int, float]], node_pad: float = 0.04,
                  node_width: float = 0.025, iterations: int = 24) -> dict:
    """Column assignment by longest path, heights by throughput, barycentric relaxation.

    Coordinates are in the unit square. Returns dict(nodes=[{x0,x1,y0,y1,value}], links=[...]).
    """
    n = len(nodes)
    out_l, in_l = defaultdict(list), defaultdict(list)
    for li, (s, t, v) in enumerate(links):
        out_l[s].append(li)
        in_l[t].append(li)
    value = [max(sum(links[i][2] for i in out_l[k]), sum(links[i][2] for i in in_l[k])) for k in range(n)]
    # depth = longest path from a source (graph assumed acyclic)
    depth = [0] * n
    order = _topo(n, links)
    for k in order:
        for li in out_l[k]:
            t = links[li][1]
            depth[t] = max(depth[t], depth[k] + 1)
    maxd = max(depth) if n else 0
    # push sinks to the last column
    for k in range(n):
        if not out_l[k]:
            depth[k] = maxd
    cols = defaultdict(list)
    for k in range(n):
        cols[depth[k]].append(k)
    ky = min((1 - node_pad * (len(c) - 1)) / max(sum(value[k] for k in c), 1e-12) for c in cols.values())
    h = [value[k] * ky for k in range(n)]
    y = [0.0] * n
    for c in cols.values():
        acc = 0.0
        for k in c:
            y[k] = acc
            acc += h[k] + node_pad
    xpos = [d / maxd * (1 - node_width) if maxd else 0.0 for d in depth]

    def resolve(col):
        col.sort(key=lambda k: y[k])
        acc = 0.0
        for k in col:
            y[k] = max(y[k], acc)
            acc = y[k] + h[k] + node_pad
        over = acc - node_pad - 1
        if over > 0:
            acc = 1.0
            for k in reversed(col):
                y[k] = min(y[k], acc - h[k])
                acc = y[k] - node_pad

    for it in range(iterations):
        alpha = 0.99 ** it
        for d in (range(1, maxd + 1) if it % 2 == 0 else range(maxd - 1, -1, -1)):
            for k in cols[d]:
                nb = in_l[k] if it % 2 == 0 else out_l[k]
                tot = sum(links[i][2] for i in nb)
                if tot:
                    other = [links[i][0] if it % 2 == 0 else links[i][1] for i in nb]
                    centre = sum((y[o] + h[o] / 2) * links[i][2] for o, i in zip(other, nb)) / tot
                    y[k] += (centre - h[k] / 2 - y[k]) * alpha
            resolve(cols[d])
    # link band positions
    so = {k: y[k] for k in range(n)}
    to = {k: y[k] for k in range(n)}
    link_geo = []
    for li in sorted(range(len(links)), key=lambda i: (y[links[i][1]], y[links[i][0]])):
        s, t, v = links[li]
        w = v * ky
        link_geo.append({"index": li, "source": s, "target": t, "value": v,
                         "sy0": so[s], "sy1": so[s] + w, "ty0": to[t], "ty1": to[t] + w,
                         "sx": xpos[s] + node_width, "tx": xpos[t]})
        so[s] += w
        to[t] += w
    link_geo.sort(key=lambda g: g["index"])
    return {"nodes": [{"name": nodes[k], "x0": xpos[k], "x1": xpos[k] + node_width, "y0": y[k],
                       "y1": y[k] + h[k], "value": value[k], "depth": depth[k]} for k in range(n)],
            "links": link_geo}


def sankey_band(g: dict, n: int = 40) -> np.ndarray:
    """Closed polygon for one sankey link (two cubic curves)."""
    mx = (g["sx"] + g["tx"]) / 2
    top = bezier((g["sx"], g["sy0"]), (mx, g["sy0"]), (mx, g["ty0"]), (g["tx"], g["ty0"]), n)
    bot = bezier((g["tx"], g["ty1"]), (mx, g["ty1"]), (mx, g["sy1"]), (g["sx"], g["sy1"]), n)
    return np.vstack([top, bot])


def _topo(n, links):
    indeg = [0] * n
    adj = defaultdict(list)
    for s, t, _ in links:
        adj[s].append(t)
        indeg[t] += 1
    stack = [k for k in range(n) if indeg[k] == 0]
    order = []
    while stack:
        k = stack.pop()
        order.append(k)
        for t in adj[k]:
            indeg[t] -= 1
            if indeg[t] == 0:
                stack.append(t)
    if len(order) < n:
        raise ValueError("sankey links contain a cycle")
    return order


# -------------------------------------------------------------------------- chord
def chord_layout(matrix, gap: float = 0.03) -> dict:
    """Group arcs and ribbons for a square flow matrix (as d3.chord).

    Angles in radians, clockwise from 12 o'clock is handled by the renderer.
    """
    m = np.asarray(matrix, float)
    n = m.shape[0]
    totals = m.sum(1)
    k = (2 * math.pi - gap * n) / totals.sum()
    groups, sub = [], {}
    a = 0.0
    for i in range(n):
        start = a
        for j in range(n):
            sub[(i, j)] = (a, a + m[i, j] * k)
            a += m[i, j] * k
        groups.append({"index": i, "start": start, "end": a, "value": totals[i]})
        a += gap
    chords = []
    for i in range(n):
        for j in range(i, n):
            if m[i, j] <= 0 and m[j, i] <= 0:
                continue
            chords.append({"source": i, "target": j, "s": sub[(i, j)], "t": sub[(j, i)],
                           "value": m[i, j] + (m[j, i] if i != j else 0)})
    return {"groups": groups, "chords": chords}


def arc_points(a0, a1, r, n: int = 30) -> np.ndarray:
    t = np.linspace(a0, a1, max(n, 2))
    return np.column_stack([r * np.sin(t), r * np.cos(t)])  # clockwise from 12 o'clock


def ribbon_polygon(s, t, r, n: int = 30) -> np.ndarray:
    """Closed outline of a chord ribbon from arc s=(a0,a1) to arc t=(b0,b1) through the centre."""
    p_s = arc_points(s[0], s[1], r, n)
    p_t = arc_points(t[0], t[1], r, n)
    c1 = bezier(p_s[-1], (0, 0), (0, 0), p_t[0], n)
    c2 = bezier(p_t[-1], (0, 0), (0, 0), p_s[0], n)
    return np.vstack([p_s, c1, p_t, c2])


def annulus_polygon(a0, a1, r0, r1, n: int = 40) -> np.ndarray:
    return np.vstack([arc_points(a0, a1, r1, n), arc_points(a1, a0, r0, n)])


# ---------------------------------------------------------------------------- arc
def semicircle(x0: float, x1: float, n: int = 40, up: bool = True) -> np.ndarray:
    c, r = (x0 + x1) / 2, abs(x1 - x0) / 2
    t = np.linspace(math.pi, 0, n)
    return np.column_stack([c + r * np.cos(t), (1 if up else -1) * r * np.sin(t)])


# ---------------------------------------------------------------------- flowchart
def layered_layout(nodes: list[str], edges: list[tuple[str, str]], direction: str = "TB",
                   sweeps: int = 8) -> dict[str, tuple[float, float]]:
    """Sugiyama-style layout for small DAGs: longest-path layers, barycentre ordering.

    Back-edges (cycles, e.g. 'retry' loops) are ignored for layering. Returns {node: (x, y)} with
    layers one unit apart (y decreasing downward for TB, x increasing for LR).
    """
    idx = {n: i for i, n in enumerate(nodes)}
    succ, pred = defaultdict(list), defaultdict(list)
    # DFS to find back edges
    state, back = {}, set()

    def dfs(u):
        state[u] = 1
        for (a, b) in edges:
            if a == u:
                if state.get(b) == 1:
                    back.add((a, b))
                elif b not in state:
                    dfs(b)
        state[u] = 2
    for n in nodes:
        if n not in state:
            dfs(n)
    for a, b in edges:
        if (a, b) not in back:
            succ[a].append(b)
            pred[b].append(a)
    layer = {n: 0 for n in nodes}
    changed = True
    while changed:
        changed = False
        for a in nodes:
            for b in succ[a]:
                if layer[b] < layer[a] + 1:
                    layer[b] = layer[a] + 1
                    changed = True
    nl = max(layer.values()) + 1 if nodes else 0
    rows = [[n for n in nodes if layer[n] == k] for k in range(nl)]
    pos = {n: float(i) for row in rows for i, n in enumerate(row)}
    for s in range(sweeps):
        rng = range(1, nl) if s % 2 == 0 else range(nl - 2, -1, -1)
        for k in rng:
            nb = pred if s % 2 == 0 else succ
            row = rows[k]
            bary = {n: (np.mean([pos[m] for m in nb[n]]) if nb[n] else pos[n]) for n in row}
            row.sort(key=lambda n: (bary[n], idx[n]))
            for i, n in enumerate(row):
                pos[n] = float(i)
    width = max(len(r) for r in rows) if rows else 1
    out = {}
    for k, row in enumerate(rows):
        off = (width - len(row)) / 2
        for i, n in enumerate(row):
            a, b = off + i, -float(k)
            out[n] = (a, b) if direction == "TB" else (-b, -a)
    return out
