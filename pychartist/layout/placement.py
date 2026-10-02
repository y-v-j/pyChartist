"""Point/label placement: beeswarm offsets, word-cloud spiral, Venn/Euler circle geometry."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np


# ------------------------------------------------------------------------ beeswarm
def beeswarm(values, radius: float, scale: float = 1.0, side: str = "both") -> np.ndarray:
    """Offsets perpendicular to the value axis so circles of ``radius`` never overlap.

    ``values`` are positions on the value axis, multiplied by ``scale`` to get the same units as
    ``radius`` (e.g. pixels per data unit). Greedy, in value order (as d3 beeswarm examples).
    Returns offsets in radius units' space (same units as ``radius``).
    """
    v = np.asarray(values, float) * scale
    order = np.argsort(v, kind="stable")
    placed_x: list[float] = []
    placed_y: list[float] = []
    out = np.zeros(len(v))
    d = 2 * radius
    lo = 0
    for idx in order:
        x = v[idx]
        while lo < len(placed_x) and placed_x[lo] < x - d:
            lo += 1
        # neighbours on the value axis that could collide
        nx = np.array(placed_x[lo:])
        ny = np.array(placed_y[lo:])
        candidates = [0.0]
        if len(nx):
            dy = np.sqrt(np.maximum(d * d - (nx - x) ** 2, 0))
            candidates += list(ny + dy + 1e-9) + list(ny - dy - 1e-9)
        if side == "positive":
            candidates = [c for c in candidates if c >= 0]
        elif side == "negative":
            candidates = [c for c in candidates if c <= 0]
        for y in sorted(candidates, key=abs):
            if not len(nx) or np.all((nx - x) ** 2 + (ny - y) ** 2 >= d * d - 1e-6):
                break
        out[idx] = y
        # keep placed lists sorted by x (x is non-decreasing since we iterate in order)
        placed_x.append(x)
        placed_y.append(y)
    return out


# ---------------------------------------------------------------------- word cloud
STOPWORDS = set("""a about above after again against all am an and any are as at be because been before
being below between both but by can could did do does doing down during each few for from further had has
have having he her here hers herself him himself his how i if in into is it its itself just me more most
my myself no nor not now of off on once only or other our ours ourselves out over own same she should so
some such than that the their theirs them themselves then there these they this those through to too
under until up very was we were what when where which while who whom why will with would you your yours
yourself yourselves upon shall may might must one said much every well say says also let us even ever
never though yet could would should mr mrs miss sir lady""".split())


def word_frequencies(text: str, stopwords=None, min_length: int = 3, top: int | None = 100) -> dict:
    """Lower-cased word counts with stop-words removed (largest first)."""
    sw = STOPWORDS if stopwords is None else set(stopwords)
    words = re.findall(r"[a-zA-Z][a-zA-Z'\-]+", text.lower())
    words = [w.strip("'-") for w in words]
    counts = Counter(w for w in words if len(w) >= min_length and w not in sw)
    return dict(counts.most_common(top))


def text_extent(word: str, fontsize: float, weight: str = "bold") -> tuple[float, float]:
    """Width/height (points) of rendered text using matplotlib's font metrics."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    bb = TextPath((0, 0), word, size=fontsize, prop=FontProperties(weight=weight)).get_extents()
    return bb.width, max(bb.height, fontsize * 0.75)


def wordcloud_layout(freqs: dict, width: float = 800, height: float = 450, max_font: float = 72,
                     min_font: float = 10, rotate_frac: float = 0.0, seed: int = 7,
                     pad: float = 2.0) -> list[dict]:
    """Archimedean-spiral word placement (as in Jason Davies' d3-cloud).

    Font size scales with sqrt(frequency). Returns dicts with word, x, y (centre, points),
    size, rotation and count; words that do not fit are dropped.
    """
    rng = np.random.default_rng(seed)
    items = sorted(freqs.items(), key=lambda kv: -kv[1])
    if not items:
        return []
    fmax, fmin = items[0][1], items[-1][1]
    placed: list[tuple[float, float, float, float]] = []
    out = []
    for word, f in items:
        t = 1.0 if fmax == fmin else (math.sqrt(f) - math.sqrt(fmin)) / (math.sqrt(fmax) - math.sqrt(fmin))
        size = min_font + t * (max_font - min_font)
        rot = 90 if rng.random() < rotate_frac else 0
        while size >= min_font * 0.8:
            w, h = text_extent(word, size)
            if rot:
                w, h = h, w
            pos = _spiral_fit(w + pad, h + pad, width, height, placed, rng)
            if pos is not None:
                x, y = pos
                placed.append((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
                out.append({"word": word, "x": x, "y": y, "size": size, "rotation": rot,
                            "count": f, "width": w, "height": h})
                break
            size *= 0.85
    return out


def _spiral_fit(w, h, W, H, placed, rng, step=2.0):
    cx, cy = W / 2, H / 2
    start = rng.uniform(0, 2 * math.pi)
    ratio = W / H
    for i in range(6000):
        t = i * 0.1
        x = cx + ratio * step * t * math.cos(t + start) / 2
        y = cy + step * t * math.sin(t + start) / 2
        x0, y0, x1, y1 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            if abs(x - cx) > W and abs(y - cy) > H:
                return None
            continue
        if all(x1 <= a0 or x0 >= a1 or y1 <= b0 or y0 >= b1 for a0, b0, a1, b1 in placed):
            return x, y
    return None


# ------------------------------------------------------------------ Venn / Euler
def lens_area(r1: float, r2: float, d: float) -> float:
    """Area of intersection of two circles with radii r1, r2 and centre distance d."""
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    a = r1 * r1 * math.acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
    b = r2 * r2 * math.acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
    c = 0.5 * math.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    return a + b - c


def distance_for_overlap(r1: float, r2: float, overlap: float) -> float:
    """Centre distance giving the requested intersection area (bisection)."""
    if overlap <= 0:
        return r1 + r2 + 0.05 * min(r1, r2)
    if overlap >= math.pi * min(r1, r2) ** 2:
        return abs(r1 - r2)
    lo, hi = abs(r1 - r2), r1 + r2
    for _ in range(80):
        mid = (lo + hi) / 2
        if lens_area(r1, r2, mid) > overlap:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def set_regions(sets: dict[str, set]) -> dict[tuple[str, ...], int]:
    """Exclusive region sizes for 2 or 3 sets, keyed by the tuple of member set names."""
    names = list(sets)
    universe = set().union(*sets.values())
    regions: dict[tuple, int] = {}
    for item in universe:
        key = tuple(n for n in names if item in sets[n])
        regions[key] = regions.get(key, 0) + 1
    return regions


def euler_circles(sizes: dict[str, float], overlaps: dict[tuple[str, str], float]):
    """Area-proportional circles for 2-3 sets: exact pairwise for 2, approximate for 3.

    ``sizes`` = total set sizes; ``overlaps`` = pairwise intersection sizes.
    Returns {name: (x, y, r)}.
    """
    names = list(sizes)
    r = {n: math.sqrt(sizes[n] / math.pi) for n in names}

    def dist(a, b):
        ov = overlaps.get((a, b), overlaps.get((b, a), 0.0))
        return distance_for_overlap(r[a], r[b], ov)

    if len(names) == 1:
        return {names[0]: (0.0, 0.0, r[names[0]])}
    a, b = names[0], names[1]
    dab = dist(a, b)
    pos = {a: (0.0, 0.0), b: (dab, 0.0)}
    if len(names) == 3:
        c = names[2]
        dac, dbc = dist(a, c), dist(b, c)
        # law of cosines; clamp when the three distances are not a valid triangle
        cos_t = (dab ** 2 + dac ** 2 - dbc ** 2) / (2 * dab * dac) if dab and dac else 0.0
        t = math.acos(max(-1.0, min(1.0, cos_t)))
        pos[c] = (dac * math.cos(t), -dac * math.sin(t))
    elif len(names) > 3:
        raise ValueError("euler supports 2 or 3 sets")
    cx = np.mean([p[0] for p in pos.values()])
    cy = np.mean([p[1] for p in pos.values()])
    return {n: (pos[n][0] - cx, pos[n][1] - cy, r[n]) for n in names}


def venn_circles(names: list[str]):
    """Fixed, symmetric (not area-proportional) Venn geometry for 2 or 3 sets."""
    if len(names) == 2:
        return {names[0]: (-0.5, 0.0, 1.0), names[1]: (0.5, 0.0, 1.0)}
    if len(names) == 3:
        d = 0.62
        return {names[0]: (-d * math.cos(math.pi / 6), d * 0.5, 1.0),
                names[1]: (d * math.cos(math.pi / 6), d * 0.5, 1.0),
                names[2]: (0.0, -d, 1.0)}
    raise ValueError("venn supports 2 or 3 sets")


def region_label_points(circles: dict, n: int = 160) -> dict[tuple[str, ...], tuple[float, float]]:
    """A label anchor inside every non-empty region (point farthest from region borders on a grid)."""
    names = list(circles)
    xs = [c[0] - c[2] for c in circles.values()] + [c[0] + c[2] for c in circles.values()]
    ys = [c[1] - c[2] for c in circles.values()] + [c[1] + c[2] for c in circles.values()]
    gx, gy = np.meshgrid(np.linspace(min(xs), max(xs), n), np.linspace(min(ys), max(ys), n))
    inside = {k: np.hypot(gx - c[0], gy - c[1]) - c[2] for k, c in circles.items()}  # <0 inside
    out = {}
    member = np.stack([inside[k] < 0 for k in names])
    codes = np.zeros(gx.shape, int)
    for i in range(len(names)):
        codes |= member[i].astype(int) << i
    for code in range(1, 2 ** len(names)):
        mask = codes == code
        if not mask.any():
            continue
        # margin = distance to the nearest circle boundary
        margin = np.min(np.stack([np.abs(inside[k]) for k in names]), axis=0)
        margin = np.where(mask, margin, -1)
        i, j = np.unravel_index(np.argmax(margin), margin.shape)
        key = tuple(names[b] for b in range(len(names)) if code >> b & 1)
        out[key] = (float(gx[i, j]), float(gy[i, j]))
    return out
