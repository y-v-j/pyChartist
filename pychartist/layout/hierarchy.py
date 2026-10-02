"""Tree building plus rectangle (squarified), circle-packing and convex-polygon treemap layouts."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Node:
    label: str
    value: float = 0.0
    children: list["Node"] = field(default_factory=list)
    parent: "Node | None" = None
    depth: int = 0
    id: str = ""
    # layout outputs
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    r: float = 0.0
    polygon: np.ndarray | None = None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def walk(self):
        """Pre-order traversal."""
        yield self
        for c in self.children:
            yield from c.walk()

    def leaves(self):
        return [n for n in self.walk() if n.is_leaf]

    def top(self) -> "Node":
        """Ancestor at depth 1 (the top-level group); self for the root."""
        n = self
        while n.parent is not None and n.parent.parent is not None:
            n = n.parent
        return n


def build_tree(df: pd.DataFrame, path: list[str], value: str | None = None,
               root_label: str = "total") -> Node:
    """Hierarchy from tidy rows: ``path`` columns from outer to inner level, summed ``value``."""
    root = Node(root_label, id=root_label)
    index: dict[tuple, Node] = {(): root}
    vals = df[value].to_numpy(dtype=float) if value else np.ones(len(df))
    keys = df[path].astype(str).to_numpy()
    for row, v in zip(keys, vals):
        if not np.isfinite(v) or v <= 0:
            continue
        parent = root
        for d in range(len(path)):
            k = tuple(row[: d + 1])
            node = index.get(k)
            if node is None:
                node = Node(row[d], parent=parent, depth=d + 1, id="/".join(k))
                parent.children.append(node)
                index[k] = node
            parent = node
        parent.value += v
    _sum(root)
    return root


def _sum(node: Node) -> float:
    if node.children:
        node.value = sum(_sum(c) for c in node.children)
    return node.value


def sort_tree(node: Node) -> None:
    node.children.sort(key=lambda n: -n.value)
    for c in node.children:
        sort_tree(c)


# ------------------------------------------------------------- squarified treemap
def squarify(values, x: float, y: float, w: float, h: float) -> list[tuple[float, float, float, float]]:
    """Squarified treemap (Bruls, Huizing & van Wijk 2000). ``values`` sorted descending.

    Returns ``(x, y, w, h)`` per value, in input order.
    """
    values = [float(v) for v in values]
    total = sum(values)
    if total <= 0 or not values:
        return [(x, y, 0.0, 0.0)] * len(values)
    scale = w * h / total
    areas = [v * scale for v in values]
    rects: list = []
    i = 0
    while i < len(areas):
        short = min(w, h)
        row = [areas[i]]
        j = i + 1
        while j < len(areas) and _worst(row + [areas[j]], short) <= _worst(row, short):
            row.append(areas[j])
            j += 1
        s = sum(row)
        if w >= h:   # lay the row out as a column on the left
            cw = s / h if h else 0
            yy = y
            for a in row:
                rh = a / cw if cw else 0
                rects.append((x, yy, cw, rh))
                yy += rh
            x, w = x + cw, w - cw
        else:        # row along the top
            rh = s / w if w else 0
            xx = x
            for a in row:
                rw = a / rh if rh else 0
                rects.append((xx, y, rw, rh))
                xx += rw
            y, h = y + rh, h - rh
        i = j
    return rects


def _worst(row, short):
    s = sum(row)
    if s <= 0 or short <= 0:
        return math.inf
    return max(max(short * short * r / (s * s), (s * s) / (short * short * r)) for r in row if r > 0)


def treemap_layout(root: Node, x=0.0, y=0.0, w=1.0, h=1.0, pad: float = 0.0,
                   pad_top: float = 0.0) -> Node:
    """Assign x0..y1 to every node; ``pad`` insets children, ``pad_top`` reserves a label band."""
    sort_tree(root)
    root.x0, root.y0, root.x1, root.y1 = x, y, x + w, y + h
    _treemap(root, pad, pad_top)
    return root


def _treemap(node: Node, pad: float, pad_top: float):
    if not node.children:
        return
    top = pad_top if node.depth > 0 else 0
    x, y = node.x0 + pad, node.y0 + pad + top
    w, h = max(node.x1 - node.x0 - 2 * pad, 0), max(node.y1 - node.y0 - 2 * pad - top, 0)
    for c, (cx, cy, cw, ch) in zip(node.children, squarify([c.value for c in node.children], x, y, w, h)):
        c.x0, c.y0, c.x1, c.y1 = cx, cy, cx + cw, cy + ch
        _treemap(c, pad, pad_top)


# ------------------------------------------------------------------ circle packing
def _place(b, a, c):
    """Place circle c tangent to circles a and b (d3-hierarchy packSiblings)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    d2 = dx * dx + dy * dy
    if d2:
        a2 = (a[2] + c[2]) ** 2
        b2 = (b[2] + c[2]) ** 2
        if a2 > b2:
            x = (d2 + b2 - a2) / (2 * d2)
            yy = math.sqrt(max(0.0, b2 / d2 - x * x))
            c[0] = b[0] - x * dx - yy * dy
            c[1] = b[1] - x * dy + yy * dx
        else:
            x = (d2 + a2 - b2) / (2 * d2)
            yy = math.sqrt(max(0.0, a2 / d2 - x * x))
            c[0] = a[0] + x * dx - yy * dy
            c[1] = a[1] + x * dy + yy * dx
    else:
        c[0] = a[0] + c[2]
        c[1] = a[1]


def _intersects(a, b) -> bool:
    dr = a[2] + b[2] - 1e-6
    dx, dy = b[0] - a[0], b[1] - a[1]
    return dr > 0 and dr * dr > dx * dx + dy * dy


def pack_siblings(radii) -> tuple[np.ndarray, float]:
    """Front-chain circle packing of sibling circles (port of d3.packSiblings).

    Returns (n x 2 centres relative to the enclosing circle's centre, enclosing radius).
    """
    circles = [[0.0, 0.0, float(r)] for r in radii]
    n = len(circles)
    if n == 0:
        return np.zeros((0, 2)), 0.0
    a = circles[0]
    if n == 1:
        return np.array([[0.0, 0.0]]), a[2]
    b = circles[1]
    a[0], b[0], b[1] = -b[2], a[2], 0.0
    if n == 2:
        c = enclose(circles)
        return np.array([[ci[0] - c[0], ci[1] - c[1]] for ci in circles]), c[2]
    _place(b, a, circles[2])
    # doubly linked front chain: dicts of next / prev by circle index
    nxt, prv = {}, {}
    ia, ib, ic = 0, 1, 2
    nxt[ia], prv[ic] = ib, ib
    nxt[ib], prv[ia] = ic, ic
    nxt[ic], prv[ib] = ia, ia
    i = 3
    while i < n:
        _place(circles[ia], circles[ib], circles[i])
        ic = i
        j, k = nxt[ib], prv[ia]
        sj, sk = circles[ib][2], circles[ia][2]
        restart = False
        while True:
            if sj <= sk:
                if _intersects(circles[j], circles[ic]):
                    ib = j
                    nxt[ia], prv[ib] = ib, ia
                    restart = True
                    break
                sj += circles[j][2]
                j = nxt[j]
            else:
                if _intersects(circles[k], circles[ic]):
                    ia = k
                    nxt[ia], prv[ib] = ib, ia
                    restart = True
                    break
                sk += circles[k][2]
                k = prv[k]
            if j == nxt[k]:
                break
        if restart:
            continue
        prv[ic], nxt[ic] = ia, ib
        nxt[ia] = ic
        prv[ib] = ic
        ib = ic
        # choose the new closest-to-centre pair
        def score(node):
            ca, cb = circles[node], circles[nxt[node]]
            ab = ca[2] + cb[2]
            dx = (ca[0] * cb[2] + cb[0] * ca[2]) / ab
            dy = (ca[1] * cb[2] + cb[1] * ca[2]) / ab
            return dx * dx + dy * dy
        best, best_score = ia, score(ia)
        c = nxt[ic]
        while c != ib:
            s = score(c)
            if s < best_score:
                best, best_score = c, s
            c = nxt[c]
        ia = best
        ib = nxt[ia]
        i += 1
    chain = [ib]
    c = nxt[ib]
    while c != ib:
        chain.append(c)
        c = nxt[c]
    e = enclose([circles[q] for q in chain])
    centres = np.array([[ci[0] - e[0], ci[1] - e[1]] for ci in circles])
    return centres, e[2]


def enclose(circles) -> tuple[float, float, float]:
    """Smallest circle enclosing the given circles (convex minimisation; exact enough for layout)."""
    from scipy.optimize import minimize
    arr = np.asarray([[c[0], c[1], c[2]] for c in circles], dtype=float)
    if len(arr) == 1:
        return tuple(arr[0])

    def f(p):
        return np.max(np.hypot(arr[:, 0] - p[0], arr[:, 1] - p[1]) + arr[:, 2])

    w = arr[:, 2] ** 2
    x0 = (arr[:, :2] * w[:, None]).sum(0) / w.sum()
    res = minimize(f, x0, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 2000})
    return float(res.x[0]), float(res.x[1]), float(f(res.x)) + 1e-9


def pack_layout(root: Node, radius: float = 1.0, pad: float = 0.0) -> Node:
    """Hierarchical circle packing: leaf area proportional to value. Fills a circle of ``radius``."""
    sort_tree(root)
    _pack(root, pad)
    k = radius / root.r if root.r else 1.0
    _place_tree(root, 0.0, 0.0, k)
    return root


def _pack(node: Node, pad: float):
    if node.is_leaf:
        node.r = math.sqrt(max(node.value, 0.0))
        return
    for c in node.children:
        _pack(c, pad)
    centres, r = pack_siblings([c.r + pad for c in node.children])
    for c, (x, y) in zip(node.children, centres):
        c.cx, c.cy = x, y        # relative to parent centre (unscaled)
    node.r = r + pad


def _place_tree(node: Node, ox: float, oy: float, k: float):
    node.cx, node.cy = ox + node.cx * k if node.parent else ox, oy + node.cy * k if node.parent else oy
    node.r *= k
    for c in node.children:
        _place_tree(c, node.cx, node.cy, k)


# ---------------------------------------------------------------- convex treemap
def regular_polygon(n: int = 96, radius: float = 1.0, cx: float = 0.0, cy: float = 0.0,
                    rotation: float = math.pi / 2) -> np.ndarray:
    t = rotation + np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([cx + radius * np.cos(t), cy + radius * np.sin(t)])


def polygon_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_centroid(p: np.ndarray) -> tuple[float, float]:
    x, y = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a = cross.sum() / 2
    if abs(a) < 1e-15:
        return float(x.mean()), float(y.mean())
    return float(((x + x1) * cross).sum() / (6 * a)), float(((y + y1) * cross).sum() / (6 * a))


def clip_halfplane(p: np.ndarray, normal, offset: float) -> np.ndarray:
    """Keep the part of convex polygon p where dot(normal, pt) <= offset (Sutherland-Hodgman)."""
    out = []
    n = len(p)
    d = p @ np.asarray(normal) - offset
    for i in range(n):
        a, b = p[i], p[(i + 1) % n]
        da, db = d[i], d[(i + 1) % n]
        if da <= 0:
            out.append(a)
        if (da < 0 < db) or (db < 0 < da):
            t = da / (da - db)
            out.append(a + t * (b - a))
    return np.array(out) if out else np.zeros((0, 2))


def _split(poly: np.ndarray, frac: float, angles: int):
    """Cut ``poly`` by a straight line so the first piece has ``frac`` of its area.

    Tries ``angles`` orientations and keeps the cut whose pieces are the 'fattest'
    (largest min area / squared-diameter ratio)."""
    total = polygon_area(poly)
    best = None
    for k in range(angles):
        th = math.pi * k / angles
        nrm = np.array([math.cos(th), math.sin(th)])
        proj = poly @ nrm
        lo, hi = proj.min(), proj.max()
        for _ in range(48):  # bisection on the offset
            mid = (lo + hi) / 2
            if polygon_area(clip_halfplane(poly, nrm, mid)) < frac * total:
                lo = mid
            else:
                hi = mid
        off = (lo + hi) / 2
        a = clip_halfplane(poly, nrm, off)
        b = clip_halfplane(poly, -nrm, -off)
        if len(a) < 3 or len(b) < 3:
            continue
        fat = min(_fatness(a), _fatness(b))
        if best is None or fat > best[0]:
            best = (fat, a, b)
    return best[1], best[2]


def _fatness(p: np.ndarray) -> float:
    d = np.max(np.hypot(*(p[:, None, :] - p[None, :, :]).transpose(2, 0, 1)))
    return polygon_area(p) / (d * d) if d else 0.0


def _partition(values: list[float]) -> int:
    """Index splitting values (sorted descending) into two runs of nearly equal sum."""
    total = sum(values)
    acc, best_i, best_diff = 0.0, 1, math.inf
    for i in range(1, len(values)):
        acc += values[i - 1]
        diff = abs(2 * acc - total)
        if diff < best_diff:
            best_i, best_diff = i, diff
    return best_i


def convex_layout(root: Node, outline: np.ndarray | None = None, angles: int = 12) -> Node:
    """Convex treemap: recursively bisect a convex outline into convex polygons with area
    proportional to value (balanced binary splits, best-aspect cut orientation per split)."""
    sort_tree(root)
    root.polygon = regular_polygon() if outline is None else np.asarray(outline, float)
    _convex(root, angles)
    return root


def _convex(node: Node, angles: int):
    if not node.children:
        return
    _assign(node.children, node.polygon, angles)
    for c in node.children:
        _convex(c, angles)


def _assign(nodes: list[Node], poly: np.ndarray, angles: int):
    if len(nodes) == 1:
        nodes[0].polygon = poly
        return
    vals = [n.value for n in nodes]
    i = _partition(vals)
    left, right = nodes[:i], nodes[i:]
    frac = sum(vals[:i]) / sum(vals)
    a, b = _split(poly, frac, angles)
    _assign(left, a, angles)
    _assign(right, b, angles)
