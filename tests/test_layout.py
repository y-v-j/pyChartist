"""Geometry invariants of the layout algorithms (independent of any rendering)."""
import math

import numpy as np
import pandas as pd
import pytest

from pychartist import layout as L
from pychartist.charts.part_to_whole import _largest_remainder
from pychartist.layout.hierarchy import polygon_area


def _pairwise(c):
    return np.hypot(*(c[:, None, :] - c[None, :, :]).transpose(2, 0, 1))


@pytest.mark.parametrize("n", [1, 2, 3, 7, 40, 150])
def test_pack_siblings_no_overlap_and_enclosed(n):
    rng = np.random.default_rng(n)
    r = rng.uniform(0.1, 3, n)
    c, R = L.pack_siblings(r)
    if n > 1:
        d = _pairwise(c)
        np.fill_diagonal(d, np.inf)
        assert (r[:, None] + r[None, :] - d).max() < 1e-6
    assert (np.hypot(c[:, 0], c[:, 1]) + r - R).max() < 1e-6
    # front-chain packing is dense: total circle area is a good fraction of the enclosure
    assert np.sum(r ** 2) / R ** 2 > (0.99 if n == 1 else 0.4)


def test_squarify_areas_and_tiling():
    vals = [50, 25, 12, 8, 3, 2]
    rects = L.squarify(vals, 0, 0, 10, 6)
    for (x, y, w, h), v in zip(rects, vals):
        assert w * h == pytest.approx(60 * v / sum(vals))
        assert 0 <= x and x + w <= 10 + 1e-9 and 0 <= y and y + h <= 6 + 1e-9
    assert sum(w * h for *_, w, h in rects) == pytest.approx(60)


def test_hierarchical_layouts_preserve_values():
    df = pd.DataFrame({"g": list("aaabbc"), "s": list("uvwxyz"), "v": [5, 3, 2, 6, 1, 4]})
    root = L.treemap_layout(L.build_tree(df, ["g", "s"], "v"), 0, 0, 3, 2)
    for leaf in root.leaves():
        assert (leaf.x1 - leaf.x0) * (leaf.y1 - leaf.y0) == pytest.approx(6 * leaf.value / 21)
    root = L.convex_layout(L.build_tree(df, ["g", "s"], "v"))
    tot = polygon_area(root.polygon)
    for n in root.walk():
        assert polygon_area(n.polygon) / tot == pytest.approx(n.value / 21, abs=1e-9)
    root = L.pack_layout(L.build_tree(df, ["g", "s"], "v"))
    leaves = root.leaves()
    areas = np.array([l.r ** 2 for l in leaves])
    assert areas / areas.sum() == pytest.approx(np.array([l.value for l in leaves]) / 21)


def test_beeswarm_min_distance():
    x = np.random.default_rng(1).normal(size=400)
    y = L.beeswarm(x, radius=3, scale=40)
    P = np.column_stack([x * 40, y])
    d = _pairwise(P)
    np.fill_diagonal(d, np.inf)
    assert d.min() >= 6 - 1e-6


def test_euler_two_sets_exact_overlap():
    c = L.euler_circles({"A": 120, "B": 45}, {("A", "B"): 30})
    (x1, y1, r1), (x2, y2, r2) = c.values()
    assert math.pi * r1 ** 2 == pytest.approx(120)
    assert L.lens_area(r1, r2, math.hypot(x2 - x1, y2 - y1)) == pytest.approx(30, rel=1e-6)


def test_set_regions_partition():
    sets = {"A": {1, 2, 3, 4}, "B": {3, 4, 5}, "C": {4, 5, 6, 7}}
    reg = L.set_regions(sets)
    assert reg == {("A",): 2, ("A", "B"): 1, ("A", "B", "C"): 1, ("B", "C"): 1, ("C",): 2}
    anchors = L.region_label_points(L.venn_circles(list(sets)))
    assert len(anchors) == 7      # every region of a 3-set Venn gets a label anchor


def test_largest_remainder_sums_exactly():
    for vals in ([1, 1, 1], [0.5, 0.3, 0.2], [123, 4567, 89, 1]):
        u = _largest_remainder(vals, 100)
        assert u.sum() == 100
        assert np.all(np.abs(u - np.asarray(vals) / sum(vals) * 100) < 1)


def test_sankey_flow_conservation():
    nodes = list("abcde")
    links = [(0, 2, 5), (1, 2, 3), (2, 3, 6), (2, 4, 2), (1, 4, 1)]
    lay = L.sankey_layout(nodes, links)
    for g in lay["links"]:
        assert g["sy1"] - g["sy0"] == pytest.approx(g["ty1"] - g["ty0"])
    n = lay["nodes"]
    assert n[2]["value"] == 8 and n[2]["x0"] > n[0]["x0"] and n[3]["x0"] > n[2]["x0"]
    for node in n:   # inside the unit square, no overlap within a column
        assert -1e-9 <= node["y0"] <= node["y1"] <= 1 + 1e-9


def test_chord_angles_cover_circle():
    m = np.array([[0, 5, 2], [3, 0, 1], [4, 4, 0]])
    lay = L.chord_layout(m, gap=0.05)
    span = sum(g["end"] - g["start"] for g in lay["groups"])
    assert span + 3 * 0.05 == pytest.approx(2 * math.pi)


def test_layered_layout_handles_cycles():
    pos = L.layered_layout(list("abcd"), [("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")])
    assert pos["a"][1] > pos["b"][1] > pos["c"][1] > pos["d"][1]


def test_us_tile_grid():
    from pychartist.layout.shapes import US_STATE_NAMES
    assert set(L.US_TILE_GRID) == set(US_STATE_NAMES)
    assert len(set(L.US_TILE_GRID.values())) == 51


def test_word_frequencies():
    f = L.word_frequencies("The cat and the hat. Cat! cat? hat", top=None)
    assert f == {"cat": 3, "hat": 2}
    words = L.wordcloud_layout(f, 300, 200)
    boxes = [(w["x"] - w["width"] / 2, w["y"] - w["height"] / 2, w["x"] + w["width"] / 2, w["y"] + w["height"] / 2)
             for w in words]
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]
