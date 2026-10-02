"""One-call exploratory analyses: the right charts *and* the right statistical tests.

Follows the EDA decision table (see :mod:`pychartist.stats`)::

    >>> import pychartist as pc
    >>> pen = pc.datasets.load("penguins")
    >>> pc.eda.analyze(pen, "body_mass_g")                       # 1 numerical
    >>> pc.eda.analyze(pen, "species")                           # 1 categorical
    >>> pc.eda.analyze(pen, "flipper_length_mm", "body_mass_g")  # 2 numerical
    >>> pc.eda.analyze(pen, "species", "body_mass_g")            # categorical x numerical
    >>> pc.eda.analyze(pen, "species", "island")                 # 2 categorical

Each returns an :class:`EDAResult` holding ``charts``, a ``tests`` DataFrame and a ``summary``; it
renders as HTML in notebooks and ``to_dash()`` gives a Dash layout.
"""
from __future__ import annotations

import base64
import html
from dataclasses import dataclass, field

import pandas as pd

from . import stats
from .chart import Chart


@dataclass
class EDAResult:
    kind: str
    variables: tuple
    charts: list[Chart]
    tests: pd.DataFrame
    summary: pd.DataFrame | pd.Series | None = None
    notes: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (f"<EDAResult {self.kind} {self.variables}: {len(self.charts)} charts, "
                f"{len(self.tests)} tests>")

    def _tests_html(self) -> str:
        if self.tests.empty:
            return ""
        cols = [c for c in ["test", "statistic", "pvalue", "effect_name", "effect_size", "significant", "note"]
                if c in self.tests]
        return self.tests[cols].to_html(index=False, float_format=lambda v: f"{v:.4g}", border=0)

    def _repr_html_(self) -> str:
        parts = [f"<h4>EDA: {html.escape(self.kind)} — {html.escape(', '.join(map(str, self.variables)))}</h4>"]
        for n in self.notes:
            parts.append(f"<p>{html.escape(n)}</p>")
        parts.append("<div style='display:flex;flex-wrap:wrap;gap:12px'>")
        for c in self.charts:
            if c.is_plotly:
                parts.append(f"<div style='flex:1 1 460px'>{c.to_html(include_plotlyjs='cdn')}</div>")
            else:
                b64 = base64.b64encode(c.to_image("png")).decode()
                parts.append(f"<div style='flex:1 1 460px'><img style='max-width:100%' src='data:image/png;base64,{b64}'/></div>")
        parts.append("</div>")
        parts.append(self._tests_html())
        if self.summary is not None:
            s = self.summary if isinstance(self.summary, pd.DataFrame) else self.summary.to_frame("value")
            parts.append(s.to_html(float_format=lambda v: f"{v:.4g}", border=0))
        return "\n".join(parts)

    def _ipython_display_(self):
        """Notebook display: each chart through its own rich repr (Plotly stays interactive even in
        untrusted notebooks, where scripts inside an HTML blob would be blocked)."""
        from IPython.display import HTML, Markdown, display
        display(Markdown(f"#### EDA: {self.kind} — {', '.join(map(str, self.variables))}"))
        for n in self.notes:
            display(Markdown(n))
        for c in self.charts:
            display(c)
        if not self.tests.empty:
            display(HTML(self._tests_html()))
        if self.summary is not None:
            display(self.summary if isinstance(self.summary, pd.DataFrame) else self.summary.to_frame("value"))

    def show(self):
        self._ipython_display_()

    def to_dash(self, id_prefix: str = "eda"):
        """A Dash ``html.Div`` with every chart and the test table."""
        from dash import dash_table, html as dhtml
        header = dhtml.H4(f"{self.kind}: {', '.join(map(str, self.variables))}")
        children = [dhtml.Div(c.to_dash(id=f"{id_prefix}-{i}"), style={"flex": "1 1 420px"})
                     for i, c in enumerate(self.charts)]
        t = self.tests.copy()
        for c in t.select_dtypes("number"):
            t[c] = t[c].map(lambda v: f"{v:.4g}")
        t = t.astype(str)
        children.append(dash_table.DataTable(t.to_dict("records"), [{"name": c, "id": c} for c in t.columns],
                                             style_table={"overflowX": "auto"},
                                             # inherit page colours so the table follows light/dark mode
                                             style_header={"backgroundColor": "transparent", "color": "inherit",
                                                           "fontWeight": "600"},
                                             style_data={"backgroundColor": "transparent", "color": "inherit"},
                                             style_cell={"fontFamily": "system-ui", "fontSize": 13,
                                                         "border": "1px solid rgba(128,128,128,0.25)",
                                                         "textAlign": "left", "whiteSpace": "normal"}))
        return dhtml.Div([header, dhtml.Div(children, style={"display": "flex", "flexWrap": "wrap", "gap": "12px"})])


def _kw(backend, theme):
    return {k: v for k, v in dict(backend=backend, theme=theme).items() if v is not None}


def numeric(df, x, *, backend=None, theme=None, alpha=0.05) -> EDAResult:
    """1 numerical: histogram + KDE, box plot, Q-Q plot; Shapiro-Wilk and Kolmogorov-Smirnov."""
    from . import box, histogram, qq_plot
    kw = _kw(backend, theme)
    d = df.assign(_all="all")
    charts = [histogram(df, x, kde=True, title=f"Distribution of {x}", **kw),
              box(d, "_all", x, points="all", title=f"{x}: quartiles and outliers", **kw),
              qq_plot(df, x, title=f"Normal Q-Q plot of {x}", **kw)]
    return EDAResult("1 numerical", (x,), charts, stats.results_frame(stats.normality(df[x], alpha)),
                     stats.describe(df[x]))


def categorical(df, x, *, backend=None, theme=None, alpha=0.05, expected=None) -> EDAResult:
    """1 categorical: frequency bar chart + pie; chi-square goodness of fit (uniform by default)."""
    from . import bar, pie
    kw = _kw(backend, theme)
    counts = df[x].value_counts().rename_axis(x).reset_index(name="count")
    charts = [bar(counts, x, "count", title=f"Frequency of {x}", **kw),
              pie(counts, x, "count", title=f"Share of {x}", **kw)]
    gof = stats.chi2_goodness_of_fit(df[x].dropna(), expected=expected, alpha=alpha)
    summary = counts.assign(percent=lambda t: 100 * t["count"] / t["count"].sum()).set_index(x)
    return EDAResult("1 categorical", (x,), charts, stats.results_frame([gof]), summary)


def numeric_pair(df, x, y, *, backend=None, theme=None, alpha=0.05) -> EDAResult:
    """2 numerical: scatter with OLS fit, correlation heatmap of all numeric columns; Pearson, Spearman."""
    from . import heatmap, scatter
    kw = _kw(backend, theme)
    charts = [scatter(df, x, y, trendline="ols", show_stats=True, title=f"{y} vs {x}", **kw),
              heatmap(df, corr=True, method="spearman", title="Spearman correlation matrix", **kw)]
    res = [stats.pearson(df[x], df[y], alpha), stats.spearman(df[x], df[y], alpha)]
    return EDAResult("2 numerical", (x, y), charts, stats.results_frame(res))


def group_compare(df, cat, num, *, backend=None, theme=None, alpha=0.05) -> EDAResult:
    """Categorical x numerical: grouped box + violin; t-test & Mann-Whitney (2 groups) or ANOVA &
    Kruskal-Wallis (3+), with the recommended one chosen by per-group normality."""
    from . import box, violin
    kw = _kw(backend, theme)
    res = stats.compare_groups(df, cat, num, alpha)
    charts = [box(df, cat, num, points="all", show_stats=True, title=f"{num} by {cat}", **kw),
              violin(df, cat, num, title=f"Distribution of {num} by {cat}", **kw)]
    notes = [f"Within-group normality (Shapiro-Wilk): {'yes' if res['normal_within_groups'] else 'no'}; "
             f"Levene p = {res['levene_p']:.3g}. Recommended: {res['recommended'].test}."]
    summary = df.groupby(cat, observed=True)[num].describe()
    return EDAResult(f"categorical x numerical ({len(res['groups'])} groups)", (cat, num), charts,
                     stats.results_frame(res["tests"]), summary, notes)


def categorical_pair(df, a, b, *, backend=None, theme=None, alpha=0.05) -> EDAResult:
    """2 categorical: 100% stacked bar + mosaic (Pearson residuals); chi-square & Fisher's exact."""
    from . import mosaic, stacked_bar
    kw = _kw(backend, theme)
    counts = df.groupby([a, b], observed=True).size().reset_index(name="count")
    charts = [stacked_bar(counts, a, "count", b, normalize=True, title=f"{b} within {a}", **kw),
              mosaic(df, a, b, title=f"Mosaic of {a} x {b}", **kw)]
    return EDAResult("2 categorical", (a, b), charts, stats.results_frame(stats.association(df, a, b, alpha)),
                     pd.crosstab(df[a], df[b], margins=True))


def analyze(df: pd.DataFrame, x: str, y: str | None = None, **kw) -> EDAResult:
    """Pick the analysis from the variable kinds (numerical / categorical) of ``x`` and ``y``."""
    kx = stats.variable_kind(df[x])
    if y is None:
        return numeric(df, x, **kw) if kx == "numerical" else categorical(df, x, **kw)
    ky = stats.variable_kind(df[y])
    if kx == ky == "numerical":
        return numeric_pair(df, x, y, **kw)
    if kx == ky == "categorical":
        return categorical_pair(df, x, y, **kw)
    cat, num = (x, y) if kx == "categorical" else (y, x)
    return group_compare(df, cat, num, **kw)


def overview(df: pd.DataFrame, *, backend=None, theme=None) -> EDAResult:
    """Dataset overview: column kinds, missing values, correlation heatmap."""
    from . import bar, heatmap
    kw = _kw(backend, theme)
    kinds = pd.DataFrame({"dtype": df.dtypes.astype(str), "kind": [stats.variable_kind(df[c]) for c in df],
                          "missing": df.isna().sum(), "unique": df.nunique()})
    miss = kinds.reset_index(names="column")
    charts = [bar(miss, "column", "missing", sort="desc", title="Missing values per column", **kw)]
    if df.select_dtypes("number").shape[1] >= 2:
        charts.append(heatmap(df, corr=True, title="Pearson correlation", **kw))
    return EDAResult("overview", tuple(df.columns[:6]) + (("...",) if df.shape[1] > 6 else ()), charts,
                     pd.DataFrame(), kinds)
