"""5. Distribution charts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from .._util import as_frame, blend, compact, need, ordered, rgba
from ..layout import beeswarm as _swarm_offsets
from ..registry import chart
from ._draw import go, mpl_compact_axis, mpl_legend


def _groups(df, x, group):
    need(df, x, group)
    d = df.dropna(subset=[x])
    levels = ordered(d[group]) if group else [None]
    return d, levels, {g: (d if g is None else d[d[group] == g])[x].to_numpy(float) for g in levels}


def _kde(v, grid, bw=None):
    if len(v) < 2 or np.std(v) == 0:
        return np.zeros_like(grid)
    return sps.gaussian_kde(v, bw_method=bw)(grid)


def _grid(values, pad=0.08, n=256):
    allv = np.concatenate([v for v in values if len(v)])
    lo, hi = allv.min(), allv.max()
    span = (hi - lo) or 1
    return np.linspace(lo - pad * span, hi + pad * span, n)


def _stats_note(r, df, cat, num):
    from ..stats import compare_groups
    res = compare_groups(df, cat, num)
    r.meta["tests"] = res
    rec = res["recommended"]
    return f"{rec.test}: p = {rec.pvalue:.2g}" + (f", {rec.effect_name} = {rec.effect_size:.2f}" if rec.effect_name else "")


# ================================================================== histogram
@chart("histogram", "distribution",
       alt_names=("frequency distribution graph", "frequency distribution chart", "histogram chart"), seaborn=True,
       use="Shape, skew, modality and spread of one numeric variable.",
       avoid="Few observations, or comparing many groups (use box / violin / ridgeline).")
def histogram(r, data, x, group=None, *, bins="auto", stat="count", kde=False, cumulative=False):
    """Binned counts of ``x`` (``stat`` = count | density | percent); ``group`` overlays layers."""
    df, levels, vals = _groups(as_frame(data), x, group)
    edges = np.histogram_bin_edges(np.concatenate(list(vals.values())), bins=bins)
    cmap = r.color_map(levels) if group else {None: r.colors(1)[0]}
    grid = _grid(list(vals.values()))
    if r.sns:
        import seaborn as sns
        fig, ax = r.subplots()
        kw = dict(data=df, x=x, bins=edges, stat=stat, kde=kde, cumulative=cumulative, ax=ax,
                  edgecolor=r.theme.gap, linewidth=0.8, element="bars", alpha=0.75 if group else 1)
        kw.update(hue=group, hue_order=levels, palette=cmap) if group else kw.update(color=cmap[None])
        sns.histplot(**kw)
        if group and ax.get_legend():
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1.01, 1), frameon=False)
        ax.grid(False)
        ax.grid(True, axis="y")
        return fig
    if r.mpl:
        fig, ax = r.subplots()
        for g in levels:
            v = vals[g]
            w = np.full(len(v), 100 / len(v)) if stat == "percent" else None
            ax.hist(v, bins=edges, density=stat == "density", weights=w, cumulative=cumulative, color=cmap[g],
                    alpha=0.75 if group else 1, edgecolor=r.theme.gap, linewidth=0.8, label=None if g is None else str(g))
            if kde and not cumulative:
                scale = 1 if stat == "density" else len(v) * np.diff(edges).mean() * (100 / len(v) if stat == "percent" else 1)
                ax.plot(grid, _kde(v, grid) * scale, color=cmap[g] if group else r.theme.ink_secondary, lw=2)
        ax.set_xlabel(x)
        ax.set_ylabel(stat)
        ax.grid(False)
        ax.grid(True, axis="y")
        mpl_compact_axis(ax.yaxis)
        mpl_legend(ax, r, title=group)
        return fig
    fig = go().Figure()
    histnorm = {"count": None, "density": "probability density", "percent": "percent"}[stat]
    for g in levels:
        fig.add_trace(go().Histogram(
            x=vals[g], xbins=dict(start=edges[0], end=edges[-1], size=np.diff(edges).mean()), histnorm=histnorm,
            cumulative=dict(enabled=cumulative), name=str(g), showlegend=g is not None, opacity=0.75 if group else 1,
            marker=dict(color=cmap[g], line=dict(color=r.theme.gap, width=0.8))))
        if kde and not cumulative:
            v = vals[g]
            scale = 1 if stat == "density" else len(v) * np.diff(edges).mean() * (100 / len(v) if stat == "percent" else 1)
            fig.add_trace(go().Scatter(x=grid, y=_kde(v, grid) * scale, mode="lines", showlegend=False, hoverinfo="skip",
                                       line=dict(color=cmap[g] if group else r.theme.ink_secondary, width=2)))
    fig.update_layout(barmode="overlay", bargap=0, legend_title_text=group or "")
    fig.update_xaxes(title_text=x, showgrid=False)
    fig.update_yaxes(title_text=stat)
    return fig


# ==================================================================== density
@chart("density", "distribution", alt_names=("density plot", "kde plot", "kernel density plot", "density chart"),
       seaborn=True,
       use="Smooth estimate of a distribution's shape; compare a few groups.",
       avoid="Small samples or bounded/discrete data where the kernel smears past the limits.")
def density(r, data, x, group=None, *, fill=True, bw=None, common_norm=False):
    """Gaussian kernel density estimate of ``x`` per ``group``."""
    df, levels, vals = _groups(as_frame(data), x, group)
    cmap = r.color_map(levels) if group else {None: r.colors(1)[0]}
    grid = _grid(list(vals.values()))
    n_all = sum(len(v) for v in vals.values())
    if r.sns:
        import seaborn as sns
        fig, ax = r.subplots()
        kw = dict(data=df, x=x, fill=fill, bw_method=bw or "scott", common_norm=common_norm, ax=ax, linewidth=2,
                  alpha=0.15 if fill else 1)
        kw.update(hue=group, hue_order=levels, palette=cmap) if group else kw.update(color=cmap[None])
        sns.kdeplot(**kw)
        if group and ax.get_legend():
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1.01, 1), frameon=False)
        ax.grid(False)
        ax.grid(True, axis="y")
        return fig
    curves = {g: _kde(vals[g], grid, bw) * (len(vals[g]) / n_all if common_norm else 1) for g in levels}
    if r.mpl:
        fig, ax = r.subplots()
        for g in levels:
            ax.plot(grid, curves[g], color=cmap[g], lw=2, label=None if g is None else str(g))
            if fill:
                ax.fill_between(grid, curves[g], color=cmap[g], alpha=0.12, lw=0)
        ax.set_xlabel(x)
        ax.set_ylabel("density")
        ax.set_ylim(bottom=0)
        ax.grid(False)
        ax.grid(True, axis="y")
        mpl_legend(ax, r, title=group)
        return fig
    fig = go().Figure()
    for g in levels:
        fig.add_trace(go().Scatter(x=grid, y=curves[g], mode="lines", name=str(g), showlegend=g is not None,
                                   line=dict(color=cmap[g], width=2), fill="tozeroy" if fill else None,
                                   fillcolor=rgba(cmap[g], 0.12),
                                   hovertemplate=f"{x}: %{{x:,.4~g}}<br>density: %{{y:.3g}}<extra>{g or ''}</extra>"))
    fig.update_xaxes(title_text=x, showgrid=False)
    fig.update_yaxes(title_text="density", rangemode="tozero")
    fig.update_layout(legend_title_text=group or "", hovermode="x unified")
    return fig


# ================================================================== ridgeline
@chart("ridgeline", "distribution", alt_names=("ridgeline plot", "joyplot", "joy plot", "ridge plot"),
       use="Compare the distribution of one variable across many ordered groups.",
       avoid="Few groups (a density overlay is simpler) or precise values.")
def ridgeline(r, data, x, group, *, overlap=1.6, bw=None, sort=None, fill_alpha=0.85):
    """Stacked, overlapping density curves, one row per ``group`` level (first level at the top)."""
    df, levels, vals = _groups(as_frame(data), x, group)
    if sort == "median":
        levels = sorted(levels, key=lambda g: np.median(vals[g]))
    grid = _grid(list(vals.values()))
    curves = {g: _kde(vals[g], grid, bw) for g in levels}
    peak = max(c.max() for c in curves.values()) or 1
    n = len(levels)
    if n <= len(r.theme.categorical):
        cmap = r.color_map(levels)
    else:
        cmap = dict(zip(levels, r.theme.seq_steps(n)))
    if r.mpl:
        fig, ax = r.subplots(figsize=r.figsize or (7.5, 0.5 * n + 1.8))
        for i, g in enumerate(levels):
            base = (n - 1 - i)
            yv = curves[g] / peak * overlap + base
            ax.fill_between(grid, base, yv, color=cmap[g], alpha=fill_alpha, lw=0, zorder=2 * i)
            ax.plot(grid, yv, color=r.theme.surface, lw=1.5, zorder=2 * i + 1)
            ax.axhline(base, color=r.theme.grid, lw=0.8, zorder=0)
        ax.set_yticks([n - 1 - i for i in range(n)], [str(g) for g in levels])
        ax.set_xlabel(x)
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.1, n - 1 + overlap * 1.05)
        return fig
    fig = go().Figure()
    for i, g in enumerate(levels):
        base = (n - 1 - i)
        yv = curves[g] / peak * overlap + base
        fig.add_trace(go().Scatter(x=grid, y=np.full_like(grid, base), mode="lines", line=dict(width=0),
                                   hoverinfo="skip", showlegend=False))
        fig.add_trace(go().Scatter(x=grid, y=yv, mode="lines", fill="tonexty", fillcolor=rgba(cmap[g], fill_alpha),
                                   line=dict(color=r.theme.surface, width=1.5), name=str(g), showlegend=False,
                                   customdata=curves[g],
                                   hovertemplate=f"<b>{g}</b><br>{x}: %{{x:,.4~g}}<br>density: %{{customdata:.3g}}<extra></extra>"))
    fig.update_yaxes(tickvals=[n - 1 - i for i in range(n)], ticktext=[str(g) for g in levels], showgrid=True,
                     range=[-0.1, n - 1 + overlap * 1.05])
    fig.update_xaxes(title_text=x, showgrid=False)
    fig.update_layout(height=r.height or 40 * n + 200)
    return fig


# ==================================================================== horizon
@chart("horizon", "distribution", alt_names=("horizon chart", "horizon graph"),
       use="Many time series compared in very little vertical space.",
       avoid="Audiences unfamiliar with folded bands; when space allows a line chart.")
def horizon(r, data, x, y, group=None, *, bands=3, center=0.0, agg="mean"):
    """Each series folded into ``bands`` layered bands (darker = larger |deviation|) around
    ``center`` (number or 'mean'); negatives are mirrored up in the other diverging hue."""
    df = as_frame(data)
    need(df, x, y, group)
    levels = ordered(df[group]) if group else [None]
    series = {}
    for g in levels:
        sub = df if g is None else df[df[group] == g]
        s = sub.groupby(x)[y].agg(agg).sort_index()
        c = s.mean() if center == "mean" else center
        series[g] = (s.index.to_numpy(), s.to_numpy(float) - c)
    hmax = max(np.abs(v).max() for _, v in series.values()) or 1
    bh = hmax / bands
    neg_c, _, pos_c = r.theme.diverging
    pos_cols = [blend(pos_c, r.theme.surface, a) for a in np.linspace(0.35, 1.0, bands)]
    neg_cols = [blend(neg_c, r.theme.surface, a) for a in np.linspace(0.35, 1.0, bands)]
    n = len(levels)

    def layers(v):
        out = []
        for k in range(bands):
            out.append((np.clip(v - k * bh, 0, bh), pos_cols[k]))
            out.append((np.clip(-v - k * bh, 0, bh), neg_cols[k]))
        return out

    if r.mpl:
        fig, axes = r.subplots(n, 1, figsize=r.figsize or (8.5, 0.55 * n + 1.0), sharex=True,
                               gridspec_kw=dict(hspace=0.05))
        axes = np.atleast_1d(axes)
        for ax, g in zip(axes, levels):
            xv, v = series[g]
            for layer, col in layers(v):
                ax.fill_between(xv, 0, layer, color=col, lw=0, step=None)
            ax.set_ylim(0, bh)
            ax.set_yticks([])
            ax.grid(False)
            for s in ("left", "top", "right"):
                ax.spines[s].set_visible(False)
            ax.set_ylabel(str(g) if g is not None else y, rotation=0, ha="right", va="center", fontsize=8.5)
        axes[-1].set_xlabel(x)
        r.title_ax = None
        return fig
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.01)
    for i, g in enumerate(levels):
        xv, v = series[g]
        for layer, col in layers(v):
            fig.add_trace(go().Scatter(x=xv, y=layer, mode="lines", fill="tozeroy", fillcolor=col, line=dict(width=0),
                                       hoverinfo="skip", showlegend=False), row=i + 1, col=1)
        fig.add_trace(go().Scatter(x=xv, y=np.zeros_like(v), mode="markers", marker=dict(opacity=0), showlegend=False,
                                   customdata=v, hovertemplate=f"<b>{g}</b> %{{x}}: %{{customdata:+,.4~g}}<extra></extra>"),
                      row=i + 1, col=1)
        fig.update_yaxes(range=[0, bh], showticklabels=False, showgrid=False, showline=False, row=i + 1, col=1,
                         title=dict(text=str(g) if g is not None else y, font=dict(size=10)), title_standoff=4)
    fig.update_layout(height=r.height or 34 * n + 160, hovermode="x unified")
    return fig


# =========================================================== radial histogram
@chart("radial_histogram", "distribution", alt_names=("radial histogram", "circular histogram", "polar histogram",
                                                      "wind rose"),
       use="Cyclic variables (hour of day, wind direction, month) or an eye-catching histogram.",
       avoid="Accuracy-critical, non-cyclic data.")
def radial_histogram(r, data, x, *, bins=24, period=None, group=None, labels=None):
    """Histogram wrapped around a circle. ``period`` = full-circle span (e.g. 24 for hours,
    360 for degrees); default = data range. ``group`` stacks counts."""
    df = as_frame(data).dropna(subset=[x])
    need(df, x, group)
    v = df[x].to_numpy(float)
    lo = 0.0 if period else v.min()
    span = period or (v.max() - v.min()) or 1
    edges = np.linspace(lo, lo + span, bins + 1)
    levels = ordered(df[group]) if group else [None]
    cmap = r.color_map(levels) if group else {None: r.colors(1)[0]}
    counts = {}
    for g in levels:
        sub = v if g is None else df.loc[df[group] == g, x].to_numpy(float)
        counts[g] = np.histogram(np.mod(sub - lo, span) + lo if period else sub, bins=edges)[0]
    theta = (edges[:-1] - lo) / span * 2 * np.pi
    width = 2 * np.pi / bins
    if r.mpl:
        fig, ax = r.subplots(polar=True, figsize=r.figsize or (6.5, 6.5))
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        bottom = np.zeros(bins)
        for g in levels:
            ax.bar(theta, counts[g], width=width, bottom=bottom, align="edge", color=cmap[g], edgecolor=r.theme.gap,
                   lw=1, label=None if g is None else str(g))
            bottom += counts[g]
        ticks = labels or [compact(e) for e in edges[:-1:max(1, bins // 8)]]
        ax.set_xticks(theta[::max(1, bins // 8)], ticks)
        ax.spines["polar"].set_visible(False)
        mpl_legend(ax, r, title=group)
        return fig
    fig = go().Figure()
    for g in levels:
        fig.add_trace(go().Barpolar(r=counts[g], theta=np.degrees(theta + width / 2), width=np.degrees(width),
                                    name=str(g), showlegend=g is not None, marker_color=cmap[g],
                                    marker_line=dict(color=r.theme.gap, width=1),
                                    customdata=np.column_stack([edges[:-1], edges[1:]]),
                                    hovertemplate="[%{customdata[0]:.4~g}, %{customdata[1]:.4~g}): %{r}<extra></extra>"))
    step = max(1, bins // 8)
    fig.update_layout(polar=dict(angularaxis=dict(rotation=90, direction="clockwise",
                                                  tickvals=list(np.degrees(theta[::step])),
                                                  ticktext=labels or [compact(e) for e in edges[:-1:step]]),
                                 bargap=0), barmode="stack", legend_title_text=group or "")
    return fig


# ====================================================== strip / jitter / beeswarm
def _cat_points(r, data, x, y, color, mode, size, alpha, seed):
    df = as_frame(data)
    need(df, y, x, color)
    d = df.dropna(subset=[y])
    cats = ordered(d[x]) if x else [None]
    hue = color or x
    hue_levels = ordered(d[hue]) if hue else [None]
    cmap = r.color_map(hue_levels) if hue else {None: r.colors(1)[0]}
    rng = np.random.default_rng(seed)
    pos = {c: i for i, c in enumerate(cats)}
    base = d[x].map(pos).to_numpy(float) if x else np.zeros(len(d))
    vals = d[y].to_numpy(float)
    if mode == "jitter":
        off = rng.uniform(-0.28, 0.28, len(d))
    elif mode == "swarm":
        off = np.zeros(len(d))
        # lay the swarm out in typographic points using the real figure geometry, then convert the
        # sideways offsets back to category-axis units (1 unit = one category band)
        from .._config import get_option
        fw, fh = r.figsize or get_option("figsize")
        if r.plotly:
            fw, fh = (r.width or get_option("width")) * 0.75 / 72, (r.height or get_option("height")) * 0.75 / 72
        axis_h_pts, axis_w_pts = fh * 72 * 0.78, fw * 72 * 0.85
        span = (vals.max() - vals.min()) or 1
        rad = (np.sqrt(size) / 2 if r.mpl else np.sqrt(size) * 1.25 * 0.75 / 2) + 0.25
        band_pts = axis_w_pts / max(len(cats), 1)
        for c in cats:
            m = (d[x] == c).to_numpy() if x else np.ones(len(d), bool)
            o = _swarm_offsets(vals[m], radius=rad, scale=axis_h_pts / span)
            off[m] = np.clip(o / band_pts, -0.48, 0.48)
        r.meta["swarm_clipped"] = int(np.sum(np.abs(off) >= 0.48))
    else:
        off = np.zeros(len(d))
    xs = base + off
    return d, cats, hue, hue_levels, cmap, xs, vals


def _points_chart(r, data, x, y, color, mode, size, alpha, seed, sns_fn):
    d, cats, hue, hue_levels, cmap, xs, vals = _cat_points(r, data, x, y, color, mode, size, alpha, seed)
    if r.sns and x:
        import seaborn as sns
        fig, ax = r.subplots()
        kw = dict(data=d, x=x, y=y, order=cats, hue=hue, hue_order=hue_levels, palette=cmap, ax=ax, size=np.sqrt(size),
                  alpha=alpha, edgecolor=r.theme.surface, linewidth=0.5, legend=bool(color))
        if sns_fn == "stripplot":
            kw["jitter"] = mode == "jitter"
        getattr(sns, sns_fn)(**kw)
        if color and ax.get_legend():
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1.01, 1), frameon=False)
        ax.grid(False)
        ax.grid(True, axis="y")
        return fig
    hv = d[hue].to_numpy() if hue else np.array([None] * len(d))
    if r.mpl:
        fig, ax = r.subplots()
        for g in hue_levels:
            m = hv == g if hue else np.ones(len(d), bool)
            ax.scatter(xs[m], vals[m], s=size, color=cmap[g], alpha=alpha, edgecolor=r.theme.surface, lw=0.5,
                       label=str(g) if color else None)
        if x:
            ax.set_xticks(range(len(cats)), [str(c) for c in cats])
            ax.set_xlabel(x)
        else:
            ax.set_xticks([])
        ax.set_ylabel(y)
        ax.grid(False)
        ax.grid(True, axis="y")
        ax.set_xlim(-0.6, len(cats) - 0.4)
        mpl_legend(ax, r, title=color)
        return fig
    fig = go().Figure()
    for g in hue_levels:
        m = hv == g if hue else np.ones(len(d), bool)
        fig.add_trace(go().Scatter(
            x=xs[m], y=vals[m], mode="markers", name=str(g), showlegend=bool(color),
            marker=dict(size=np.sqrt(size) * 1.25, color=cmap[g], opacity=alpha, line=dict(color=r.theme.surface, width=0.5)),
            customdata=d[x].astype(str).to_numpy()[m] if x else None,
            hovertemplate=(f"{x}: %{{customdata}}<br>" if x else "") + f"{y}: %{{y:,.4~g}}<extra></extra>"))
    fig.update_xaxes(tickvals=list(range(len(cats))), ticktext=[str(c) for c in cats] if x else [""],
                     title_text=x, showgrid=False, range=[-0.6, len(cats) - 0.4])
    fig.update_yaxes(title_text=y)
    fig.update_layout(legend_title_text=color or "")
    return fig


@chart("strip", "distribution", alt_names=("strip plot", "individual value plot", "single-axis scatter plot",
                                           "strip chart", "dot strip"), seaborn=True,
       use="Every observation of a numeric variable, per category, when n is small.",
       avoid="Many points - they overlap; use jitter, beeswarm or a density.")
def strip(r, data, x, y, color=None, *, size=36, alpha=0.7):
    """Points on a single value axis per category, no jitter (overlaps are visible as darker)."""
    return _points_chart(r, data, x, y, color, "strip", size, alpha, 0, "stripplot")


@chart("jitter", "distribution", alt_names=("jitter plot", "jittered strip plot", "jittered dot plot"), seaborn=True,
       use="A strip plot with random horizontal offset so overlapping points separate.",
       avoid="When exact position across the category axis could be misread as data.")
def jitter(r, data, x, y, color=None, *, size=30, alpha=0.65, seed=0):
    """Strip plot with uniform random offsets within each category band."""
    return _points_chart(r, data, x, y, color, "jitter", size, alpha, seed, "stripplot")


@chart("beeswarm", "distribution", alt_names=("beeswarm chart", "swarm plot", "bee swarm", "swarmplot"), seaborn=True,
       use="Every point visible without overlap, revealing the distribution shape (n up to ~1-2k).",
       avoid="Very large n - the swarm overflows its band.")
def beeswarm(r, data, x, y, color=None, *, size=22, alpha=0.9):
    """Points nudged sideways just enough to never overlap (greedy swarm layout)."""
    return _points_chart(r, data, x, y, color, "swarm", size, alpha, 0, "swarmplot")


# ======================================================== one-dimensional heatmap
@chart("heatmap_1d", "distribution",
       alt_names=("one-dimensional heatmap", "1d heatmap", "warming stripes", "climate stripes", "stripe chart"),
       use="The evolution of one variable as coloured stripes (e.g. temperature anomalies).",
       avoid="Colour-blind-critical messages without labels; precise values.")
def heatmap_1d(r, data, x, value, *, diverging=True, center="mean", agg="mean", show_axis=True):
    """One thin coloured stripe per ``x`` (ordered); diverging around ``center`` by default."""
    df = as_frame(data)
    s = df.groupby(x)[value].agg(agg).sort_index()
    v = s.to_numpy(float)
    c = (np.nanmean(v) if center == "mean" else center) if diverging else None
    if diverging:
        m = np.nanmax(np.abs(v - c))
        zmin, zmax = c - m, c + m
    else:
        zmin, zmax = np.nanmin(v), np.nanmax(v)
    if r.mpl:
        fig, ax = r.subplots(figsize=r.figsize or (9, 2.6))
        im = ax.imshow(v[None, :], aspect="auto", cmap=r.theme.div_cmap() if diverging else r.theme.seq_cmap(),
                       vmin=zmin, vmax=zmax, extent=(-0.5, len(v) - 0.5, 0, 1))
        idx = np.linspace(0, len(v) - 1, min(8, len(v))).astype(int)
        ax.set_xticks(idx, [str(s.index[i])[:10] for i in idx]) if show_axis else ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.01, shrink=0.9, label=value)
        cb.outline.set_visible(False)
        return fig
    fig = go().Figure(go().Heatmap(
        z=[v], x=[str(i) for i in s.index], y=[value], zmin=zmin, zmax=zmax, xgap=0,
        colorscale=r.theme.div_colorscale() if diverging else r.theme.seq_colorscale(),
        hovertemplate=f"%{{x}}<br>{value}: %{{z:,.4~g}}<extra></extra>", colorbar=dict(outlinewidth=0)))
    fig.update_yaxes(visible=False)
    fig.update_xaxes(showgrid=False, visible=show_axis, nticks=10)
    fig.update_layout(height=r.height or 260)
    return fig


# ================================================================ box / violin
@chart("box", "distribution", alt_names=("box plot", "box-and-whisker plot", "box and whisker", "boxplot",
                                         "grouped box plot"), seaborn=True,
       use="Compare medians, quartiles and outliers across groups.",
       avoid="Multi-modal data (a box hides modes - use violin or strip).")
def box(r, data, x, y, color=None, *, points="outliers", show_stats=False, notch=False):
    """Box (quartiles) with 1.5 IQR whiskers per ``x`` category. ``points`` = 'outliers' | 'all' |
    False. ``show_stats`` runs t-test/Mann-Whitney (2 groups) or ANOVA/Kruskal-Wallis (3+)."""
    df = as_frame(data)
    need(df, x, y, color)
    d = df.dropna(subset=[y, x])
    cats = ordered(d[x])
    hue = color or x
    levels = ordered(d[hue])
    cmap = r.color_map(levels)
    note = _stats_note(r, d, x, y) if show_stats else None
    if r.sns:
        import seaborn as sns
        fig, ax = r.subplots()
        sns.boxplot(data=d, x=x, y=y, order=cats, hue=hue, hue_order=levels, palette=cmap, ax=ax, notch=notch,
                    showfliers=points == "outliers", width=0.55, linewidth=1.2, legend=bool(color), saturation=1,
                    linecolor=r.theme.ink_secondary, fill=True, boxprops=dict(alpha=0.75))
        if points == "all":
            sns.stripplot(data=d, x=x, y=y, order=cats, hue=hue, hue_order=levels, palette=cmap, dodge=bool(color),
                          ax=ax, size=3, alpha=0.5, legend=False, edgecolor=r.theme.surface, linewidth=0.3)
        if color and ax.get_legend():
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1.01, 1), frameon=False)
        ax.grid(False)
        ax.grid(True, axis="y")
        if note:
            ax.text(0, 1.01, note, transform=ax.transAxes, fontsize=8.5, color=r.theme.ink_secondary, va="bottom")
        return fig
    if r.mpl:
        fig, ax = r.subplots()
        sub_levels = ordered(d[color]) if color else [None]
        k = len(sub_levels)
        width = 0.55 / k
        for j, g in enumerate(sub_levels):
            data_ = [d[(d[x] == c) & ((d[color] == g) if color else True)][y].to_numpy(float) for c in cats]
            pos = np.arange(len(cats)) - 0.275 + width * (j + 0.5)
            bp = ax.boxplot(data_, positions=pos, widths=width * 0.85, patch_artist=True, notch=notch,
                            showfliers=points == "outliers", medianprops=dict(color=r.theme.ink, lw=1.8),
                            whiskerprops=dict(color=r.theme.ink_secondary), capprops=dict(color=r.theme.ink_secondary),
                            flierprops=dict(marker="o", markersize=4, markerfacecolor=r.theme.muted, markeredgecolor="none"))
            for patch, c in zip(bp["boxes"], cats):
                col = cmap[g] if color else cmap[c]
                patch.set(facecolor=blend(col, r.theme.surface, 0.75), edgecolor=r.theme.ink_secondary, lw=1)
            if points == "all":
                rng = np.random.default_rng(0)
                for p, arr, c in zip(pos, data_, cats):
                    ax.scatter(p + rng.uniform(-width * 0.25, width * 0.25, len(arr)), arr, s=8, alpha=0.5,
                               color=cmap[g] if color else cmap[c], lw=0, zorder=3)
            if color:
                ax.scatter([], [], marker="s", s=60, color=cmap[g], label=str(g))
        ax.set_xticks(range(len(cats)), [str(c) for c in cats])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(False)
        ax.grid(True, axis="y")
        mpl_legend(ax, r, title=color)
        if note:
            ax.text(0, 1.01, note, transform=ax.transAxes, fontsize=8.5, color=r.theme.ink_secondary, va="bottom")
        return fig
    fig = go().Figure()
    for g in (ordered(d[color]) if color else cats):
        sub = d[d[color] == g] if color else d[d[x] == g]
        fig.add_trace(go().Box(
            x=sub[x].astype(str), y=sub[y], name=str(g), marker_color=cmap[g], fillcolor=rgba(cmap[g], 0.75),
            line=dict(color=r.theme.ink_secondary, width=1.2), boxpoints="all" if points == "all" else
            ("outliers" if points == "outliers" else False), jitter=0.3, pointpos=0, notched=notch,
            marker=dict(size=4, opacity=0.6), showlegend=bool(color)))
    fig.update_layout(boxmode="group" if color else "overlay", legend_title_text=color or "")
    fig.update_xaxes(title_text=x, categoryorder="array", categoryarray=[str(c) for c in cats], showgrid=False)
    fig.update_yaxes(title_text=y)
    if note:
        fig.add_annotation(text=note, x=0, y=1.02, xref="paper", yref="paper", showarrow=False, xanchor="left",
                           yanchor="bottom", font=dict(size=11, color=r.theme.ink_secondary))
    return fig


@chart("violin", "distribution", alt_names=("violin plot", "grouped violin plot", "bean plot"), seaborn=True,
       use="Full distribution shape (incl. multi-modality) per group, with a box inside.",
       avoid="Small samples where the KDE invents shape.")
def violin(r, data, x, y, color=None, *, inner="box", show_stats=False, bw=None, split=False):
    """Mirrored KDE per category with an inner box (``inner`` = box | quartile | point | None)."""
    df = as_frame(data)
    need(df, x, y, color)
    d = df.dropna(subset=[y, x])
    cats = ordered(d[x])
    hue = color or x
    levels = ordered(d[hue])
    cmap = r.color_map(levels)
    note = _stats_note(r, d, x, y) if show_stats else None
    if r.sns:
        import seaborn as sns
        fig, ax = r.subplots()
        sns.violinplot(data=d, x=x, y=y, order=cats, hue=hue, hue_order=levels, palette=cmap, ax=ax, inner=inner,
                       split=split and bool(color), bw_method=bw or "scott", linewidth=1, saturation=1,
                       linecolor=r.theme.ink_secondary, legend=bool(color), cut=0, density_norm="width")
        for coll in ax.collections:
            coll.set_alpha(0.8)
        if color and ax.get_legend():
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1.01, 1), frameon=False)
        ax.grid(False)
        ax.grid(True, axis="y")
        if note:
            ax.text(0, 1.01, note, transform=ax.transAxes, fontsize=8.5, color=r.theme.ink_secondary, va="bottom")
        return fig
    if r.mpl:
        fig, ax = r.subplots()
        sub_levels = ordered(d[color]) if color else [None]
        k = len(sub_levels)
        width = 0.8 / k
        for j, g in enumerate(sub_levels):
            data_ = [d[(d[x] == c) & ((d[color] == g) if color else True)][y].to_numpy(float) for c in cats]
            pos = np.arange(len(cats)) - 0.4 + width * (j + 0.5)
            ok = [i for i, a in enumerate(data_) if len(a) > 1]
            parts = ax.violinplot([data_[i] for i in ok], positions=pos[ok], widths=width * 0.9, showextrema=False,
                                  bw_method=bw)
            for body, i in zip(parts["bodies"], ok):
                col = cmap[g] if color else cmap[cats[i]]
                body.set(facecolor=col, edgecolor=r.theme.surface, alpha=0.8, lw=1)
            if inner in ("box", "quartile"):
                for p, a in zip(pos[ok], [data_[i] for i in ok]):
                    q1, med, q3 = np.percentile(a, [25, 50, 75])
                    ax.vlines(p, q1, q3, color=r.theme.ink, lw=4 if inner == "box" else 1.2)
                    ax.scatter([p], [med], s=18, color=r.theme.surface, zorder=3)
            elif inner == "point":
                for p, a in zip(pos[ok], [data_[i] for i in ok]):
                    ax.scatter(np.full(len(a), p), a, s=4, color=r.theme.ink, alpha=0.4, lw=0)
            if color:
                ax.scatter([], [], marker="s", s=60, color=cmap[g], label=str(g))
        ax.set_xticks(range(len(cats)), [str(c) for c in cats])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(False)
        ax.grid(True, axis="y")
        mpl_legend(ax, r, title=color)
        if note:
            ax.text(0, 1.01, note, transform=ax.transAxes, fontsize=8.5, color=r.theme.ink_secondary, va="bottom")
        return fig
    fig = go().Figure()
    groups_ = ordered(d[color]) if color else cats
    for i, g in enumerate(groups_):
        sub = d[d[color] == g] if color else d[d[x] == g]
        side = ("negative" if i == 0 else "positive") if (split and color and len(groups_) == 2) else "both"
        fig.add_trace(go().Violin(
            x=sub[x].astype(str), y=sub[y], name=str(g), side=side, line_color=cmap[g], fillcolor=rgba(cmap[g], 0.8),
            box_visible=inner == "box", meanline_visible=False, points="all" if inner == "point" else False,
            showlegend=bool(color), spanmode="hard", box=dict(fillcolor=r.theme.ink, line=dict(color=r.theme.ink)),
            bandwidth=None))
    fig.update_layout(violinmode="overlay" if split else ("group" if color else "overlay"), legend_title_text=color or "")
    fig.update_xaxes(title_text=x, categoryorder="array", categoryarray=[str(c) for c in cats], showgrid=False)
    fig.update_yaxes(title_text=y)
    if note:
        fig.add_annotation(text=note, x=0, y=1.02, xref="paper", yref="paper", showarrow=False, xanchor="left",
                           yanchor="bottom", font=dict(size=11, color=r.theme.ink_secondary))
    return fig


# ===================================================================== QQ plot
@chart("qq_plot", "distribution", alt_names=("q-q plot", "quantile-quantile plot", "normal probability plot", "qq"),
       use="Check a sample against the normal distribution (companion to Shapiro-Wilk / KS).",
       avoid="Tiny samples where any pattern is noise.")
def qq_plot(r, data, x, *, show_stats=True):
    """Ordered sample vs theoretical normal quantiles with the least-squares reference line."""
    v = pd.to_numeric(as_frame(data)[x], errors="coerce").dropna().to_numpy(float)
    (osm, osr), (slope, intercept, _) = sps.probplot(v, dist="norm")
    note = None
    if show_stats:
        from ..stats import kolmogorov_smirnov, shapiro_wilk
        sw, ks = shapiro_wilk(v), kolmogorov_smirnov(v)
        r.meta["tests"] = [sw, ks]
        note = f"Shapiro-Wilk p = {sw.pvalue:.2g} · KS p = {ks.pvalue:.2g}"
    line_x = np.array([osm.min(), osm.max()])
    col = r.colors(1)[0]
    if r.mpl:
        fig, ax = r.subplots(figsize=r.figsize or (5.5, 5))
        ax.scatter(osm, osr, s=16, color=col, alpha=0.75, lw=0)
        ax.plot(line_x, intercept + slope * line_x, color=r.theme.ink_secondary, lw=1.5)
        ax.set_xlabel("theoretical normal quantile")
        ax.set_ylabel(f"{x} (ordered)")
        if note:
            ax.text(0.01, 0.99, note, transform=ax.transAxes, va="top", fontsize=8.5, color=r.theme.ink_secondary)
        return fig
    fig = go().Figure()
    fig.add_trace(go().Scatter(x=osm, y=osr, mode="markers", marker=dict(size=6, color=col, opacity=0.75), showlegend=False,
                               hovertemplate="theoretical %{x:.2f}<br>sample %{y:,.4~g}<extra></extra>"))
    fig.add_trace(go().Scatter(x=line_x, y=intercept + slope * line_x, mode="lines", hoverinfo="skip",
                               line=dict(color=r.theme.ink_secondary, width=1.5), showlegend=False))
    fig.update_xaxes(title_text="theoretical normal quantile")
    fig.update_yaxes(title_text=f"{x} (ordered)")
    if note:
        fig.add_annotation(text=note, x=0, y=1, xref="paper", yref="paper", xanchor="left", yanchor="top",
                           showarrow=False, font=dict(size=11, color=r.theme.ink_secondary))
    return fig
