"""Chart registry and the ``@chart`` decorator shared by every chart function.

The decorator adds the common keyword arguments to each chart function's public signature:

=============  ===============================================================================
``backend``    ``"matplotlib"`` / ``"seaborn"`` (static) or ``"plotly"`` (interactive)
``theme``      theme name or :class:`~pychartist.themes.Theme`
``title``      chart title
``ax``         an existing matplotlib Axes to draw into (matplotlib backend only)
``figsize``    matplotlib figure size in inches
``width``      Plotly width in pixels (``None`` = responsive)
``height``     Plotly height in pixels
``palette``    list of colours, or ``{category: colour}`` mapping, overriding the theme
=============  ===============================================================================
"""
from __future__ import annotations

import functools
import inspect
import warnings
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Callable

import matplotlib as mpl
import pandas as pd

from ._config import get_option, normalize_backend
from .chart import Chart
from .themes import Theme, get_theme, plotly_template_name

CATEGORIES = {
    "comparison": "1. Comparison",
    "correlation": "2. Correlation (relational)",
    "part_to_whole": "3. Part-to-whole & hierarchical",
    "temporal": "4. Data over time (temporal)",
    "distribution": "5. Distribution",
    "geo_other": "6. Geospatial & other",
    "eda": "EDA (charts + statistical tests)",
}


@dataclass
class ChartSpec:
    name: str
    func: Callable
    category: str
    backends: tuple[str, ...]
    alt_names: tuple[str, ...]
    use: str
    avoid: str
    notes: str = ""


_REGISTRY: dict[str, ChartSpec] = {}
_ALIASES: dict[str, str] = {}


def _key(s: str) -> str:
    """Normalise a chart name: case, separators and any unicode whitespace (e.g. NBSP)."""
    s = s.lower().replace("-", " ").replace("_", " ").replace("/", " ")
    return " ".join(s.split())


@dataclass
class Render:
    """Per-call drawing context handed to every chart implementation as its first argument."""
    backend: str
    theme: Theme
    kind: str
    title: str | None = None
    ax: object = None
    figsize: tuple | None = None
    width: int | None = None
    height: int | None = None
    palette: object = None
    meta: dict = field(default_factory=dict)
    title_ax: object = None

    @property
    def mpl(self) -> bool:
        """True for every matplotlib-figure backend (matplotlib and seaborn)."""
        return self.backend in ("matplotlib", "seaborn")

    @property
    def sns(self) -> bool:
        """True when the caller asked for seaborn; charts without a seaborn version use matplotlib."""
        return self.backend == "seaborn"

    @property
    def plotly(self) -> bool:
        return self.backend == "plotly"

    # ----------------------------------------------------------- matplotlib
    def subplots(self, nrows: int = 1, ncols: int = 1, polar: bool = False, figsize=None,
                 **kwargs):
        """Return (fig, ax). Re-uses the caller's ``ax`` when one was given."""
        if self.ax is not None and nrows == ncols == 1:
            ax = self.ax
            if polar and getattr(ax, "name", "") != "polar":
                raise ValueError(f"{self.kind} needs a polar Axes: create it with "
                                 "plt.subplot(projection='polar')")
            self.title_ax = ax
            return ax.figure, ax
        from matplotlib.figure import Figure
        fig = Figure(figsize=figsize or self.figsize or get_option("figsize"), layout="constrained")
        subplot_kw = kwargs.pop("subplot_kw", {})
        if polar:
            subplot_kw["projection"] = "polar"
        axes = fig.subplots(nrows, ncols, subplot_kw=subplot_kw or None, squeeze=True, **kwargs)
        self.title_ax = axes if nrows == ncols == 1 else None
        return fig, axes

    # --------------------------------------------------------------- colours
    def color_map(self, categories) -> dict:
        """Stable {category: colour}. Order = pandas Categorical order or first appearance."""
        cats = _ordered_unique(categories)
        pal = self.palette
        if isinstance(pal, dict):
            missing = [c for c in cats if c not in pal]
            base = {c: pal[c] for c in cats if c in pal}
            for i, c in enumerate(missing):
                base[c] = self.theme.color(len(base) + i)
            return base
        colors = list(pal) if pal is not None else list(self.theme.categorical)
        if len(cats) > len(colors) and pal is None:
            warnings.warn(f"{self.kind}: {len(cats)} categories exceed the theme's {len(colors)} "
                          "distinguishable colours; colours repeat. Fold small categories into "
                          "'Other' (pychartist.fold_other) or use small_multiples.", stacklevel=4)
        return {c: colors[i % len(colors)] for i, c in enumerate(cats)}

    def colors(self, n: int) -> list[str]:
        if self.palette is not None and not isinstance(self.palette, dict):
            pal = list(self.palette)
            return [pal[i % len(pal)] for i in range(n)]
        return self.theme.colors(n)


def _ordered_unique(values) -> list:
    if isinstance(values, pd.Series) and isinstance(values.dtype, pd.CategoricalDtype):
        present = set(values.dropna().unique())
        return [c for c in values.cat.categories if c in present]
    if isinstance(values, pd.Index | pd.Series):
        return list(pd.unique(values.dropna()))
    return list(dict.fromkeys(v for v in values if v is not None))


_COMMON = ("backend", "theme", "title", "ax", "figsize", "width", "height", "palette")


def chart(name: str, category: str, *, alt_names=(), use: str = "", avoid: str = "",
          backends=("matplotlib", "plotly"), seaborn: bool = False, notes: str = ""):
    """Register a chart implementation ``impl(r: Render, *args, **kwargs)``.

    ``seaborn=True`` marks charts with a seaborn-native drawing path; every matplotlib chart also
    accepts ``backend="seaborn"`` (it then draws with matplotlib under the same theme).
    """
    if seaborn and "seaborn" not in backends:
        backends = tuple(b for b in backends if b != "plotly") + ("seaborn",) + \
            (("plotly",) if "plotly" in backends else ())
    if category not in CATEGORIES:
        raise ValueError(category)

    def deco(impl):
        sig = inspect.signature(impl)
        params = list(sig.parameters.values())[1:]  # drop `r`
        common = [inspect.Parameter(n, inspect.Parameter.KEYWORD_ONLY, default=None) for n in _COMMON]
        var_kw = [p for p in params if p.kind is inspect.Parameter.VAR_KEYWORD]
        params = [p for p in params if p.kind is not inspect.Parameter.VAR_KEYWORD]
        public_sig = sig.replace(parameters=params + common + var_kw, return_annotation=Chart)

        @functools.wraps(impl)
        def wrapper(*args, backend=None, theme=None, title=None, ax=None, figsize=None,
                    width=None, height=None, palette=None, **kwargs):
            b = normalize_backend(backend)
            if ax is not None and b == "plotly":
                b = "matplotlib"
            if b == "matplotlib" and "matplotlib" not in backends and "seaborn" in backends:
                b = "seaborn"          # seaborn-only (figure-level) charts are their matplotlib form
            allowed = set(backends) | ({"seaborn"} if "matplotlib" in backends else set())
            if b not in allowed:
                raise NotImplementedError(f"{name!r} is available for {backends}, not {b!r}")
            th = get_theme(theme, backend=b)
            r = Render(backend=b, theme=th, kind=name, title=title, ax=ax, figsize=figsize,
                       width=width, height=height, palette=palette)
            with ExitStack() as stack:
                if b in ("matplotlib", "seaborn") and not th.inherit:
                    stack.enter_context(mpl.rc_context(th.mpl_rc()))
                out = impl(r, *args, **kwargs)
                return _finish(r, out)

        wrapper.__signature__ = public_sig
        wrapper.spec = ChartSpec(name, wrapper, category, tuple(backends), tuple(alt_names),
                                 use, avoid, notes)
        doc = impl.__doc__ or ""
        wrapper.__doc__ = (f"{doc.rstrip()}\n\n    Common keyword arguments: "
                           f"{', '.join(_COMMON)} (see pychartist.registry).\n"
                           f"    Backends: {', '.join(backends)}.  Returns a pychartist.Chart.\n")
        _REGISTRY[name] = wrapper.spec
        for alias in (name, impl.__name__, *alt_names):
            _ALIASES[_key(alias)] = name
        return wrapper
    return deco


def _finish(r: Render, out) -> Chart:
    if isinstance(out, Chart):
        return out
    fig = out
    if r.plotly:
        layout = dict(template=plotly_template_name(r.theme))
        if r.title:
            layout["title"] = dict(text=r.title)
        if r.width:
            layout["width"] = r.width
        if r.height or fig.layout.height is None:
            layout["height"] = r.height or fig.layout.height or get_option("height")
        fig.update_layout(**layout)
    elif r.title:
        if r.title_ax is not None:
            r.title_ax.set_title(r.title)
        else:
            fig.suptitle(r.title, x=0.01, ha="left", fontsize=13, fontweight="bold",
                         color=r.theme.ink)
    return Chart(fig, r.backend, r.kind, r.theme, meta=r.meta, owns_figure=r.ax is None)


# ---------------------------------------------------------------- catalog API
def get_spec(name: str) -> ChartSpec:
    try:
        return _REGISTRY[_ALIASES[_key(name)]]
    except KeyError:
        raise KeyError(f"unknown chart {name!r}; see pychartist.catalog()") from None


def plot(kind: str, *args, **kwargs) -> Chart:
    """Draw any registered chart by name or alternative name, e.g. ``plot("spider chart", ...)``."""
    return get_spec(kind).func(*args, **kwargs)


def catalog(category: str | None = None) -> pd.DataFrame:
    """DataFrame of every chart: category, function, backends, alternative names, when to use."""
    rows = []
    for spec in _REGISTRY.values():
        if category and spec.category != category:
            continue
        rows.append({"chart": spec.name, "category": CATEGORIES[spec.category],
                     "function": f"pc.{spec.func.__name__}", "backends": ", ".join(spec.backends),
                     "alt_names": ", ".join(spec.alt_names), "use_when": spec.use,
                     "avoid_when": spec.avoid})
    df = pd.DataFrame(rows)
    if not df.empty:
        order = list(CATEGORIES.values())
        df["_o"] = df["category"].map(order.index)
        df = df.sort_values(["_o"], kind="stable").drop(columns="_o").reset_index(drop=True)
    return df
