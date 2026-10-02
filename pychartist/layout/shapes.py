"""Icons drawn as plain polygons (no SVG assets) and the US state tile-map grid.

Every icon is a list of closed polygons inside the unit square, so the same geometry renders as a
matplotlib ``PathPatch`` and as a Plotly ``fill="toself"`` trace or layout shape.
"""
from __future__ import annotations

import math

import numpy as np


def _circle(cx, cy, r, n=40):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def _person():
    head = _circle(0.5, 0.83, 0.15)
    t = np.linspace(math.pi, 0, 16)
    shoulders = np.column_stack([0.5 + 0.27 * np.cos(t), 0.40 + 0.22 * np.sin(t)])
    body = np.vstack([[[0.23, 0.0]], shoulders, [[0.77, 0.0]]])
    return [head, body]


def _star(points=5, inner=0.42):
    ang = np.linspace(math.pi / 2, math.pi / 2 + 2 * math.pi, 2 * points, endpoint=False)
    rad = np.where(np.arange(2 * points) % 2 == 0, 0.5, 0.5 * inner)
    return [np.column_stack([0.5 + rad * np.cos(ang), 0.48 + rad * np.sin(ang)])]


def _heart(n=60):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    return [np.column_stack([0.5 + x / 34, 0.52 + y / 34])]


def _drop(n=50):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    x = 0.5 + 0.36 * np.sin(t) * np.sin(t / 2) ** 1.0
    y = 0.95 - 0.95 * (1 - np.cos(t)) / 2 * 0.98
    return [np.column_stack([x, 1 - y])]


def _house():
    return [np.array([[0.12, 0.0], [0.12, 0.55], [0.0, 0.55], [0.5, 1.0], [1.0, 0.55], [0.88, 0.55],
                      [0.88, 0.0], [0.6, 0.0], [0.6, 0.3], [0.4, 0.3], [0.4, 0.0]])]


ICONS = {
    "square": lambda: [np.array([[0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95]])],
    "circle": lambda: [_circle(0.5, 0.5, 0.45)],
    "person": _person,
    "star": _star,
    "heart": _heart,
    "drop": _drop,
    "house": _house,
}


def icon_polygons(icon: str, x: float, y: float, size: float) -> list[np.ndarray]:
    """Polygons for ``icon`` with its unit box placed at (x, y) and scaled to ``size``."""
    try:
        polys = ICONS[icon]()
    except KeyError:
        raise KeyError(f"unknown icon {icon!r}; choose from {sorted(ICONS)}") from None
    return [np.column_stack([x + p[:, 0] * size, y + p[:, 1] * size]) for p in polys]


# (column, row) of each state in a 12 x 8 tile grid, row 0 at the top.
US_TILE_GRID = {
    "AK": (0, 0), "ME": (10, 0),
    "VT": (9, 1), "NH": (10, 1),
    "WA": (0, 2), "ID": (1, 2), "MT": (2, 2), "ND": (3, 2), "MN": (4, 2), "IL": (5, 2),
    "WI": (6, 2), "MI": (7, 2), "NY": (8, 2), "RI": (9, 2), "MA": (10, 2),
    "OR": (0, 3), "NV": (1, 3), "WY": (2, 3), "SD": (3, 3), "IA": (4, 3), "IN": (5, 3),
    "OH": (6, 3), "PA": (7, 3), "NJ": (8, 3), "CT": (9, 3),
    "CA": (0, 4), "UT": (1, 4), "CO": (2, 4), "NE": (3, 4), "MO": (4, 4), "KY": (5, 4),
    "WV": (6, 4), "VA": (7, 4), "MD": (8, 4), "DE": (9, 4),
    "AZ": (1, 5), "NM": (2, 5), "KS": (3, 5), "AR": (4, 5), "TN": (5, 5), "NC": (6, 5),
    "SC": (7, 5), "DC": (8, 5),
    "OK": (3, 6), "LA": (4, 6), "MS": (5, 6), "AL": (6, 6), "GA": (7, 6),
    "HI": (0, 7), "TX": (3, 7), "FL": (8, 7),
}

US_STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}
