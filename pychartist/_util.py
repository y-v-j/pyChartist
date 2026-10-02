"""Small helpers shared by the chart modules."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb


def as_frame(data, columns=None) -> pd.DataFrame:
    """Accept a DataFrame, dict, Series, list of dicts or 2-D array."""
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, pd.Series):
        return data.rename_axis(data.index.name or "index").reset_index(
            name=data.name if data.name is not None else "value")
    if isinstance(data, dict):
        if all(np.isscalar(v) for v in data.values()):
            return pd.DataFrame({"category": list(data), "value": list(data.values())})
        return pd.DataFrame(data)
    return pd.DataFrame(data, columns=columns)


def need(df: pd.DataFrame, *cols) -> None:
    missing = [c for c in cols if c is not None and c not in df.columns]
    if missing:
        raise KeyError(f"column(s) {missing} not in data; available: {list(df.columns)}")


def ordered(values) -> list:
    """Unique values in Categorical order, else first-appearance order."""
    s = pd.Series(values)
    if isinstance(s.dtype, pd.CategoricalDtype):
        present = set(s.dropna())
        return [c for c in s.cat.categories if c in present]
    return list(pd.unique(s.dropna()))


def rgba(color: str, alpha: float) -> str:
    r, g, b = (int(round(255 * c)) for c in to_rgb(color))
    return f"rgba({r},{g},{b},{alpha})"


def luminance(color: str) -> float:
    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(c) for c in to_rgb(color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def text_on(fill: str, theme) -> str:
    """White or near-black ink, whichever contrasts more with ``fill``."""
    lum = luminance(fill)
    return "#ffffff" if (1.05 / (lum + 0.05)) >= ((lum + 0.05) / 0.05) else "#0b0b0b"


def compact(v: float, digits: int = 1) -> str:
    """1,284 / 12.9K / 4.2M style labels."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    a = abs(v)
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= 10 * div or (a >= div and suf != "K"):
            return f"{v / div:.{digits}f}{suf}"
    if float(v).is_integer():
        return f"{int(v):,}"
    return f"{v:,.{max(digits, 1)}f}" if a >= 1 else f"{v:.3g}"


def fold_other(data: pd.DataFrame, category: str, value: str | None = None, n: int = 7,
               label: str = "Other") -> pd.DataFrame:
    """Keep the ``n`` largest categories and fold the rest into ``label``.

    With ``value`` the categories are ranked by summed value, otherwise by row count.
    """
    df = data.copy()
    score = df.groupby(category, observed=True)[value].sum() if value else df[category].value_counts()
    keep = set(score.nlargest(n).index)
    df[category] = df[category].where(df[category].isin(keep), label)
    return df


def polygon_path(xs, ys, close: bool = True) -> str:
    pts = " L".join(f"{x:.4f},{y:.4f}" for x, y in zip(xs, ys))
    return f"M{pts}" + (" Z" if close else "")


def mpl_hide_axes(ax) -> None:
    ax.set_axis_off()


def aggregate(df: pd.DataFrame, by, value: str | None, agg: str = "sum") -> pd.DataFrame:
    """Aggregate ``value`` by ``by`` columns; with ``value=None`` count rows into 'count'."""
    by = [by] if isinstance(by, str) else list(by)
    if value is None:
        return df.groupby(by, observed=True, sort=False).size().reset_index(name="count")
    return df.groupby(by, observed=True, sort=False)[value].agg(agg).reset_index()


def blend(color: str, background: str, alpha: float) -> str:
    """Solid hex of ``color`` at ``alpha`` over ``background`` (works in every backend)."""
    from matplotlib.colors import to_hex
    a, b = to_rgb(color), to_rgb(background)
    return to_hex(tuple(alpha * x + (1 - alpha) * y for x, y in zip(a, b)))
