"""Seaborn figure-level statistical charts (pair plot, joint plot, cluster map) with Plotly twins.

``backend="matplotlib"`` routes to seaborn for these, since seaborn *is* their matplotlib form.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .._util import as_frame, need, ordered
from ..registry import chart
from ._draw import go


def _detach(fig):
    """seaborn figure-level functions register with pyplot; detach so notebooks don't show twice."""
    import matplotlib.pyplot as plt
    plt.close(fig)
    return fig


def _numeric_cols(df, exclude=()):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]


@chart("pair_plot", "eda", backends=("seaborn", "plotly"),
       alt_names=("pairplot", "scatterplot matrix", "splom", "pairs plot"),
       use="First look at all pairwise relationships and marginal distributions of numeric columns.",
       avoid="More than ~8 variables (cells get tiny).")
def pair_plot(r, data, vars=None, hue=None, *, kind="scatter", diag="kde", corner=False, cell_size=1.9):
    """Grid of pairwise scatter plots (``kind`` = scatter | kde | reg | hist) with marginal
    ``diag`` plots (kde | hist)."""
    df = as_frame(data)
    cols = list(vars) if vars else _numeric_cols(df, exclude=[hue] if hue else [])
    d = df.dropna(subset=cols + ([hue] if hue else []))
    levels = ordered(d[hue]) if hue else [None]
    cmap = r.color_map(levels) if hue else {None: r.colors(1)[0]}
    if r.mpl:
        import seaborn as sns
        kw = dict(vars=cols, kind=kind, diag_kind=diag, corner=corner, height=cell_size,
                  plot_kws=dict(alpha=0.7, edgecolor=r.theme.surface, linewidth=0.4) if kind == "scatter" else {})
        if hue:
            kw.update(hue=hue, hue_order=levels, palette=cmap)
        else:
            kw["plot_kws"]["color"] = cmap[None] if kind == "scatter" else None
            kw["diag_kws"] = dict(color=cmap[None])
            if kw["plot_kws"].get("color") is None:
                kw["plot_kws"].pop("color")
        g = sns.pairplot(d, **kw)
        return _detach(g.figure)
    fig = go().Figure()
    for lv in levels:
        sub = d if lv is None else d[d[hue] == lv]
        fig.add_trace(go().Splom(dimensions=[dict(label=c, values=sub[c]) for c in cols], name=str(lv),
                                 showlegend=lv is not None, diagonal_visible=False, showupperhalf=not corner,
                                 marker=dict(size=4, color=cmap[lv], opacity=0.7, line=dict(width=0))))
    fig.update_layout(height=r.height or 160 * len(cols) + 80, width=r.width or 160 * len(cols) + 160,
                      legend_title_text=hue or "")
    return fig


@chart("joint_plot", "eda", backends=("seaborn", "plotly"),
       alt_names=("jointplot", "marginal plot", "scatter with marginals"),
       use="One numeric pair in detail: joint scatter/density plus each marginal distribution.",
       avoid="Many groups (colours overlap in the margins).")
def joint_plot(r, data, x, y, hue=None, *, kind="scatter", size=6):
    """Bivariate plot (``kind`` = scatter | kde | hist | hex | reg) with marginal histograms/KDEs."""
    df = as_frame(data).dropna(subset=[x, y])
    need(df, x, y, hue)
    levels = ordered(df[hue]) if hue else [None]
    cmap = r.color_map(levels) if hue else {None: r.colors(1)[0]}
    if r.mpl:
        import seaborn as sns
        kw = dict(data=df, x=x, y=y, kind=kind, height=size)
        if hue and kind not in ("hex", "reg"):
            kw.update(hue=hue, hue_order=levels, palette=cmap)
        else:
            kw["color"] = cmap[levels[0]]
        if kind == "hex":
            kw["joint_kws"] = dict(cmap=r.theme.seq_cmap(), color=None)
            kw.pop("color")
            kw["marginal_kws"] = dict(color=cmap[levels[0]])
        g = sns.jointplot(**kw)
        return _detach(g.figure)
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=2, column_widths=[0.82, 0.18], row_heights=[0.18, 0.82], shared_xaxes=True,
                        shared_yaxes=True, horizontal_spacing=0.01, vertical_spacing=0.01)
    for lv in levels:
        sub = df if lv is None else df[df[hue] == lv]
        fig.add_trace(go().Scatter(x=sub[x], y=sub[y], mode="markers", name=str(lv), legendgroup=str(lv),
                                   showlegend=lv is not None, marker=dict(size=7, color=cmap[lv], opacity=0.75,
                                                                          line=dict(color=r.theme.surface, width=0.5)),
                                   hovertemplate=f"{x}: %{{x:,.4~g}}<br>{y}: %{{y:,.4~g}}<extra></extra>"), row=2, col=1)
        fig.add_trace(go().Histogram(x=sub[x], marker_color=cmap[lv], opacity=0.6, showlegend=False,
                                     legendgroup=str(lv)), row=1, col=1)
        fig.add_trace(go().Histogram(y=sub[y], marker_color=cmap[lv], opacity=0.6, showlegend=False,
                                     legendgroup=str(lv)), row=2, col=2)
    fig.update_layout(barmode="overlay", height=r.height or 640, legend_title_text=hue or "")
    fig.update_xaxes(title_text=x, row=2, col=1)
    fig.update_yaxes(title_text=y, row=2, col=1)
    fig.update_xaxes(showticklabels=False, showgrid=False, row=2, col=2)
    fig.update_yaxes(showticklabels=False, showgrid=False, row=1, col=1)
    return fig


@chart("cluster_map", "eda", backends=("seaborn", "plotly"),
       alt_names=("clustermap", "clustered heatmap", "heatmap with dendrogram"),
       use="Matrix data (samples x features, expression tables) with rows/columns ordered by clustering.",
       avoid="Very large matrices without prior filtering.")
def cluster_map(r, data, labels=None, *, method="average", metric="euclidean", z_score=1, row_cluster=True,
                col_cluster=True):
    """Heatmap with rows/columns reordered by hierarchical clustering and their dendrograms.
    ``z_score``: 0 standardises rows, 1 columns, None keeps raw values."""
    df = as_frame(data)
    mat = df.set_index(labels) if labels else df
    mat = mat.select_dtypes("number").dropna()
    if r.mpl:
        import seaborn as sns
        diverging = z_score is not None
        g = sns.clustermap(mat, method=method, metric=metric, z_score=z_score, row_cluster=row_cluster,
                           col_cluster=col_cluster, cmap=r.theme.div_cmap() if diverging else r.theme.seq_cmap(),
                           center=0 if diverging else None, figsize=r.figsize or (8, 8),
                           linewidths=0.3, linecolor=r.theme.surface, dendrogram_ratio=0.15,
                           cbar_kws=dict(label="z-score" if diverging else "value"),
                           tree_kws=dict(colors=r.theme.ink_secondary, linewidths=0.8))
        r.meta["row_order"] = list(mat.index[g.dendrogram_row.reordered_ind]) if row_cluster else list(mat.index)
        r.meta["col_order"] = list(mat.columns[g.dendrogram_col.reordered_ind]) if col_cluster else list(mat.columns)
        return _detach(g.figure)
    from scipy.cluster import hierarchy as sch
    from scipy.stats import zscore
    z = mat.to_numpy(float)
    if z_score is not None:
        z = zscore(z, axis=1 - z_score if z_score in (0, 1) else 0, ddof=1)
        z = np.nan_to_num(z)
    ro = sch.leaves_list(sch.linkage(z, method=method, metric=metric)) if row_cluster else np.arange(z.shape[0])
    co = sch.leaves_list(sch.linkage(z.T, method=method, metric=metric)) if col_cluster else np.arange(z.shape[1])
    zz = z[np.ix_(ro, co)]
    m = np.abs(zz).max() if z_score is not None else None
    fig = go().Figure(go().Heatmap(
        z=zz, x=[str(c) for c in mat.columns[co]], y=[str(i) for i in mat.index[ro]],
        colorscale=r.theme.div_colorscale() if z_score is not None else r.theme.seq_colorscale(),
        zmin=-m if m else None, zmax=m, xgap=1, ygap=1, colorbar=dict(title=dict(text="z-score" if z_score is not None
                                                                                  else "value"), outlinewidth=0)))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    r.meta["row_order"] = list(mat.index[ro])
    r.meta["col_order"] = list(mat.columns[co])
    return fig
