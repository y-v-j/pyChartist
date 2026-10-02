"""1. Comparison charts."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .._util import aggregate, as_frame, blend, compact, need, ordered, rgba, text_on
from ..layout import icon_polygons, word_frequencies, wordcloud_layout
from ..registry import chart
from ._draw import (go, mpl_bar_labels, mpl_blank, mpl_compact_axis, mpl_legend, mpl_poly,
                    mpl_value_grid, pl_hide_axes, pl_poly, pl_value_grid)


# =================================================================== bar engine
def _bar_engine(r, data, cat, val, group=None, orient="h", mode="group", sort=None, labels=None,
                normalize=False, agg="sum", horizontal_first_on_top=True):
    df = as_frame(data)
    need(df, cat, val, group)
    keys = [cat] + ([group] if group else [])
    agg_df = df.groupby(keys, observed=True, sort=False)[val].agg(agg).reset_index()
    cats = ordered(df[cat])
    if sort in ("desc", "asc", True):
        tot = agg_df.groupby(cat, observed=True)[val].sum()
        cats = list(tot.sort_values(ascending=(sort == "asc")).index)
    groups = ordered(df[group]) if group else [val]
    if group:
        wide = agg_df.pivot_table(index=cat, columns=group, values=val, aggfunc="sum", observed=True)
    else:
        wide = agg_df.set_index(cat)[[val]]
    wide = wide.reindex(index=cats, columns=groups).fillna(0.0)
    if normalize:
        wide = wide.div(wide.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
    cmap = r.color_map(groups) if group else {val: r.colors(1)[0]}
    single = len(groups) == 1
    if labels is None:
        labels = single and len(cats) <= 25
    vlabel = f"{val} (%)" if normalize else val
    if r.sns and mode == "group" and not normalize:
        return _bar_sns(r, df, cat, val, group, cats, groups, cmap, orient, agg, labels, vlabel)
    if r.mpl:
        return _bar_mpl(r, wide, cats, groups, cmap, orient, mode, labels, vlabel, group)
    return _bar_plotly(r, wide, cats, groups, cmap, orient, mode, labels, vlabel, group, normalize)


def _bar_sns(r, df, cat, val, group, cats, groups, cmap, orient, agg, labels, vlabel):
    """seaborn.barplot: same layout, plus 95% bootstrap CI error bars for mean/median."""
    import seaborn as sns
    fig, ax = r.subplots()
    d = df.dropna(subset=[val]).copy()
    d[cat] = d[cat].astype(str)
    horiz = orient == "h"
    ci = agg in ("mean", "median")
    kw = dict(data=d, estimator=agg, errorbar=("ci", 95) if ci else None, order=[str(c) for c in cats],
              ax=ax, orient=orient, err_kws={"color": r.theme.ink_secondary, "linewidth": 1.2},
              capsize=0.15 if ci else 0, saturation=1, edgecolor=r.theme.gap, linewidth=1 if group else 0)
    kw.update(x=val, y=cat) if horiz else kw.update(x=cat, y=val)
    if group:
        kw.update(hue=group, hue_order=groups, palette=cmap)
    else:
        kw.update(color=cmap[val], width=0.6)
    sns.barplot(**kw)
    if ci:
        r.meta["error_bars"] = "95% bootstrap confidence interval of the " + agg
    if labels and not ci:      # with CI bars the error bar itself marks the estimate
        for cont in ax.containers:
            mpl_bar_labels(ax, r, cont)
    if horiz:
        ax.set_xlabel(vlabel)
        ax.set_ylabel("")
        mpl_value_grid(ax, "x")
        mpl_compact_axis(ax.xaxis)
    else:
        ax.set_ylabel(vlabel)
        ax.set_xlabel("")
        mpl_value_grid(ax, "y")
        mpl_compact_axis(ax.yaxis)
    if group:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.01, 1), frameon=False, title=group)
    return fig


def _bar_mpl(r, wide, cats, groups, cmap, orient, mode, labels, vlabel, group):
    fig, ax = r.subplots()
    n, k = len(cats), len(groups)
    pos = np.arange(n)
    band = 0.78 if k > 1 else 0.6
    horiz = orient == "h"
    pos_base = np.zeros(n)
    neg_base = np.zeros(n)
    for j, g in enumerate(groups):
        vals = wide[g].to_numpy(dtype=float)
        if mode == "group":
            w = band / k
            p = pos - band / 2 + w * (j + 0.5)
            base = None
        else:
            w = band * 0.8
            p = pos
            base = np.where(vals >= 0, pos_base, neg_base)
        kw = dict(color=cmap[g], label=str(g) if group else None, edgecolor=r.theme.gap,
                  linewidth=1.0 if (mode != "group" or k > 1) else 0)
        if horiz:
            cont = ax.barh(p, vals, height=w, left=base, **kw)
        else:
            cont = ax.bar(p, vals, width=w, bottom=base, **kw)
        if mode != "group":
            pos_base += np.where(vals >= 0, vals, 0)
            neg_base += np.where(vals < 0, vals, 0)
        if labels:
            mpl_bar_labels(ax, r, cont)
    tick_labels = [str(c) for c in cats]
    if horiz:
        ax.set_yticks(pos, tick_labels)
        ax.invert_yaxis()
        ax.set_xlabel(vlabel)
        mpl_value_grid(ax, "x")
        mpl_compact_axis(ax.xaxis)
        ax.axvline(0, color=r.theme.baseline, lw=1)
    else:
        rot = 0 if max((len(t) for t in tick_labels), default=0) * n < 70 else 45
        ax.set_xticks(pos, tick_labels, rotation=rot, ha="right" if rot else "center")
        ax.set_ylabel(vlabel)
        mpl_value_grid(ax, "y")
        mpl_compact_axis(ax.yaxis)
        ax.axhline(0, color=r.theme.baseline, lw=1)
    ax.margins(x=0.02 if not horiz else 0.08, y=0.08 if not horiz else 0.02)
    mpl_legend(ax, r, title=group)
    return fig


def _bar_plotly(r, wide, cats, groups, cmap, orient, mode, labels, vlabel, group, normalize):
    fig = go().Figure()
    horiz = orient == "h"
    for g in groups:
        vals = wide[g].to_numpy(dtype=float)
        xy = dict(x=vals, y=[str(c) for c in cats]) if horiz else dict(x=[str(c) for c in cats], y=vals)
        tip = "%{y}" if horiz else "%{x}"
        val_ref = "%{x:,.4~g}" if horiz else "%{y:,.4~g}"
        fig.add_trace(go().Bar(
            **xy, orientation=orient, name=str(g), marker_color=cmap[g],
            marker_line=dict(color=r.theme.gap, width=1 if mode != "group" else 0),
            text=[compact(v) for v in vals] if labels else None,
            textposition="outside" if labels else None, cliponaxis=False,
            textfont=dict(color=r.theme.ink_secondary),
            hovertemplate=f"<b>{tip}</b><br>{vlabel}: {val_ref}<extra>{g if group else ''}</extra>"))
    fig.update_layout(barmode="group" if mode == "group" else "relative", bargap=0.25,
                      showlegend=bool(group), legend_title_text=group or "")
    cat_axis = dict(categoryorder="array", categoryarray=[str(c) for c in cats])
    if horiz:
        fig.update_yaxes(**cat_axis, autorange="reversed")
        fig.update_xaxes(title_text=vlabel)
        pl_value_grid(fig, "x")
    else:
        fig.update_xaxes(**cat_axis)
        fig.update_yaxes(title_text=vlabel)
        pl_value_grid(fig, "y")
    return fig


@chart("bar", "comparison", alt_names=("bar chart", "bar graph", "horizontal bar chart"),
       use="Compare values across categories; long labels; values of similar magnitude.",
       avoid="More than ~30 bars, or a value axis that cannot start at zero.",
       seaborn=True)
def bar(r, data, x, y, color=None, *, sort=None, labels=None, agg="sum"):
    """Horizontal bar chart: one bar per category ``x`` with length ``y`` (aggregated by ``agg``).

    ``color`` adds grouped bars per level of that column. ``sort="desc"|"asc"`` orders bars.
    """
    return _bar_engine(r, data, x, y, color, "h", "group", sort, labels, agg=agg)


@chart("column", "comparison", alt_names=("column graph", "vertical bar chart", "column chart"),
       use="Like a bar chart for short labels, ordered categories or a few time points.",
       avoid="Long category labels (use bar) or many time points (use line).",
       seaborn=True)
def column(r, data, x, y, color=None, *, sort=None, labels=None, agg="sum"):
    """Vertical bars: category ``x`` on the horizontal axis, value ``y``."""
    return _bar_engine(r, data, x, y, color, "v", "group", sort, labels, agg=agg)


@chart("grouped_bar", "comparison",
       alt_names=("grouped bar chart", "grouped column chart", "clustered bar chart",
                  "paired bar chart", "grouped bar column chart"),
       use="Compare sub-group values within each category on a common baseline.",
       avoid="Too many groups/categories or poor colour separation (keep groups <= 4-5).",
       seaborn=True)
def grouped_bar(r, data, x, y, group, *, orient="v", sort=None, labels=False, agg="sum"):
    """Clustered bars: one cluster per ``x`` category, one bar per ``group`` level."""
    return _bar_engine(r, data, x, y, group, orient, "group", sort, labels, agg=agg)


# ===================================================================== lollipop
@chart("lollipop", "comparison", alt_names=("lollipop plot", "lollipop chart"),
       use="A lighter bar chart for many categories.",
       avoid="Values very close together - bars are easier to compare.")
def lollipop(r, data, x, y, *, orient="h", sort="desc", agg="sum", labels=True):
    """Stem + dot per category: ``x`` categories, ``y`` values."""
    df = aggregate(as_frame(data), x, y, agg)
    if sort:
        df = df.sort_values(y, ascending=(sort == "asc"))
    cats, vals = [str(c) for c in df[x]], df[y].to_numpy(float)
    col = r.colors(1)[0]
    if r.mpl:
        fig, ax = r.subplots()
        pos = np.arange(len(cats))
        if orient == "h":
            ax.hlines(pos, 0, vals, color=col, lw=2)
            ax.scatter(vals, pos, s=70, color=col, edgecolor=r.theme.surface, linewidth=2, zorder=3)
            ax.set_yticks(pos, cats)
            ax.invert_yaxis()
            ax.set_xlabel(y)
            mpl_value_grid(ax, "x")
            mpl_compact_axis(ax.xaxis)
            if labels:
                for p, v in zip(pos, vals):
                    ax.annotate(compact(v), (v, p), xytext=(8, 0), textcoords="offset points",
                                va="center", fontsize=8.5, color=r.theme.ink_secondary)
        else:
            ax.vlines(pos, 0, vals, color=col, lw=2)
            ax.scatter(pos, vals, s=70, color=col, edgecolor=r.theme.surface, linewidth=2, zorder=3)
            ax.set_xticks(pos, cats, rotation=45 if len(cats) > 8 else 0, ha="right" if len(cats) > 8 else "center")
            ax.set_ylabel(y)
            mpl_value_grid(ax, "y")
            mpl_compact_axis(ax.yaxis)
        return fig
    fig = go().Figure()
    xs, ys = (vals, cats) if orient == "h" else (cats, vals)
    stem_x, stem_y = [], []
    for c, v in zip(cats, vals):
        if orient == "h":
            stem_x += [0, v, None]
            stem_y += [c, c, None]
        else:
            stem_x += [c, c, None]
            stem_y += [0, v, None]
    fig.add_trace(go().Scatter(x=stem_x, y=stem_y, mode="lines", line=dict(color=col, width=2),
                               hoverinfo="skip", showlegend=False))
    fig.add_trace(go().Scatter(x=xs, y=ys, mode="markers+text" if labels and orient == "h" else "markers",
                               marker=dict(size=12, color=col, line=dict(color=r.theme.surface, width=2)),
                               text=[compact(v) for v in vals], textposition="middle right",
                               textfont=dict(color=r.theme.ink_secondary), showlegend=False,
                               hovertemplate="<b>%{y}</b>: %{x:,.4~g}<extra></extra>" if orient == "h"
                               else "<b>%{x}</b>: %{y:,.4~g}<extra></extra>"))
    if orient == "h":
        fig.update_yaxes(categoryorder="array", categoryarray=cats, autorange="reversed")
        fig.update_xaxes(title_text=y)
        pl_value_grid(fig, "x")
    else:
        fig.update_xaxes(categoryorder="array", categoryarray=cats)
        fig.update_yaxes(title_text=y)
        pl_value_grid(fig, "y")
    return fig


# ======================================================================= bullet
@chart("bullet", "comparison", alt_names=("bullet graph", "bullet chart"),
       use="Benchmark one actual value per metric against a target and qualitative ranges.",
       avoid="Anything that is not performance-vs-target data.")
def bullet(r, data, label, value, target=None, ranges=None):
    """One compact bullet per row: actual ``value`` bar, ``target`` tick, grey ``ranges`` bands.

    ``ranges`` = list of columns holding the *upper bounds* of the qualitative bands
    (e.g. ``["poor", "satisfactory", "good"]``), lowest first.
    """
    df = as_frame(data).reset_index(drop=True)
    need(df, label, value, target, *(ranges or []))
    n = len(df)
    band_cols = list(ranges or [])
    greys = [blend(r.theme.ink, r.theme.surface, a) for a in np.linspace(0.20, 0.07, max(len(band_cols), 1))]
    bar_col = r.colors(1)[0]
    if r.mpl:
        fig, axes = r.subplots(n, 1, figsize=r.figsize or (7.5, 0.75 * n + 0.6), sharex=False)
        axes = np.atleast_1d(axes)
        for ax, (_, row) in zip(axes, df.iterrows()):
            top = max([row[c] for c in band_cols] + [row[value]] + ([row[target]] if target else []))
            prev = 0
            for c, g in zip(band_cols, greys):
                ax.barh(0, row[c] - prev, left=prev, height=0.9, color=g, edgecolor="none")
                prev = row[c]
            ax.barh(0, row[value], height=0.32, color=bar_col, edgecolor="none")
            if target:
                ax.vlines(row[target], -0.32, 0.32, color=r.theme.ink, lw=2.5)
            ax.set_xlim(0, top * 1.02)
            ax.set_yticks([0], [str(row[label])])
            ax.tick_params(axis="y", length=0)
            ax.grid(False)
            for s in ("left", "top", "right"):
                ax.spines[s].set_visible(False)
            mpl_compact_axis(ax.xaxis)
        r.title_ax = None
        return fig
    fig = go().Figure()
    h = 1.0 / n
    for i, (_, row) in enumerate(df.iterrows()):
        top = max([row[c] for c in band_cols] + [row[value]] + ([row[target]] if target else []))
        steps, prev = [], 0
        for c, g in zip(band_cols, greys):
            steps.append(dict(range=[prev, row[c]], color=g))
            prev = row[c]
        fig.add_trace(go().Indicator(
            mode="number+gauge", value=row[value], number=dict(font=dict(size=16, color=r.theme.ink)),
            domain=dict(x=[0.22, 0.88], y=[1 - (i + 1) * h + 0.25 * h, 1 - i * h - 0.25 * h]),
            title=dict(text=str(row[label]), font=dict(size=12, color=r.theme.ink_secondary)),
            gauge=dict(shape="bullet", axis=dict(range=[0, top * 1.02], tickfont=dict(color=r.theme.muted)),
                       bar=dict(color=bar_col, thickness=0.35), steps=steps, bgcolor=r.theme.plotly_bg,
                       borderwidth=0,
                       threshold=dict(line=dict(color=r.theme.ink, width=3), thickness=0.75,
                                      value=row[target]) if target else None)))
    fig.update_layout(height=r.height or max(120 * n, 220))
    return fig


# ===================================================================== dot plot
@chart("dot_plot", "comparison", alt_names=("dot chart", "cleveland dot plot", "dot plot"),
       use="Compare one or several values per category without a zero baseline.",
       avoid="When a baseline/bar mass is needed for context.")
def dot_plot(r, data, x, y, group=None, *, sort=True, agg="mean"):
    """Category ``y`` per row, value ``x`` as dot(s); ``group`` adds one coloured dot per level."""
    df = as_frame(data)
    need(df, x, y, group)
    keys = [y] + ([group] if group else [])
    d = df.groupby(keys, observed=True, sort=False)[x].agg(agg).reset_index()
    cats = ordered(df[y])
    if sort:
        cats = list(d.groupby(y, observed=True)[x].mean().sort_values(ascending=False).index)
    groups = ordered(df[group]) if group else [None]
    cmap = r.color_map(groups) if group else {None: r.colors(1)[0]}
    if r.mpl:
        fig, ax = r.subplots(figsize=r.figsize or (7.5, max(3, 0.32 * len(cats) + 1)))
        pos = {c: i for i, c in enumerate(cats)}
        lo = d.groupby(y, observed=True)[x].min()
        hi = d.groupby(y, observed=True)[x].max()
        for c in cats:
            ax.hlines(pos[c], lo[c], hi[c], color=r.theme.grid, lw=2, zorder=1)
        for g in groups:
            sub = d if g is None else d[d[group] == g]
            ax.scatter(sub[x], sub[y].map(pos), s=64, color=cmap[g], label=None if g is None else str(g),
                       edgecolor=r.theme.surface, linewidth=2, zorder=3)
        ax.set_yticks(range(len(cats)), [str(c) for c in cats])
        ax.invert_yaxis()
        ax.set_xlabel(x)
        ax.grid(True, axis="both")
        ax.tick_params(axis="y", length=0)
        mpl_legend(ax, r, title=group)
        return fig
    fig = go().Figure()
    for g in groups:
        sub = d if g is None else d[d[group] == g]
        fig.add_trace(go().Scatter(x=sub[x], y=sub[y].astype(str), mode="markers", name=str(g),
                                   marker=dict(size=11, color=cmap[g], line=dict(color=r.theme.surface, width=2)),
                                   showlegend=g is not None,
                                   hovertemplate=f"<b>%{{y}}</b><br>{x}: %{{x:,.4~g}}<extra>{g or ''}</extra>"))
    fig.update_yaxes(categoryorder="array", categoryarray=[str(c) for c in cats], autorange="reversed",
                     showgrid=True)
    fig.update_xaxes(title_text=x)
    fig.update_layout(legend_title_text=group or "", height=r.height or max(360, 26 * len(cats) + 120))
    return fig


# ===================================================================== dumbbell
@chart("dumbbell", "comparison",
       alt_names=("dumbbell plot", "dumbbell chart", "connected dot plot", "dna chart",
                  "barbell chart", "dumbbell dot plot"),
       use="Emphasise the change/gap between exactly two values per category.",
       avoid="Data that is not naturally paired.")
def dumbbell(r, data, category, value, group, *, sort=True, agg="mean"):
    """Two dots per ``category`` (the two levels of ``group``) joined by a connector."""
    df = as_frame(data)
    need(df, category, value, group)
    levels = ordered(df[group])
    if len(levels) != 2:
        raise ValueError(f"dumbbell needs exactly 2 levels in {group!r}, found {levels}")
    w = df.pivot_table(index=category, columns=group, values=value, aggfunc=agg, observed=True)[levels]
    cats = list(w.index)
    if sort:
        cats = list((w[levels[1]] - w[levels[0]]).sort_values(ascending=False).index)
    w = w.loc[cats]
    cmap = r.color_map(levels)
    if r.mpl:
        fig, ax = r.subplots(figsize=r.figsize or (7.5, max(3, 0.35 * len(cats) + 1)))
        pos = np.arange(len(cats))
        ax.hlines(pos, w[levels[0]], w[levels[1]], color=r.theme.baseline, lw=3, zorder=1)
        for lv in levels:
            ax.scatter(w[lv], pos, s=80, color=cmap[lv], label=str(lv), zorder=3,
                       edgecolor=r.theme.surface, linewidth=2)
        ax.set_yticks(pos, [str(c) for c in cats])
        ax.invert_yaxis()
        ax.set_xlabel(value)
        mpl_value_grid(ax, "x")
        mpl_legend(ax, r, title=group)
        return fig
    fig = go().Figure()
    xs, ys = [], []
    for c in cats:
        xs += [w.loc[c, levels[0]], w.loc[c, levels[1]], None]
        ys += [str(c), str(c), None]
    fig.add_trace(go().Scatter(x=xs, y=ys, mode="lines", line=dict(color=r.theme.baseline, width=3),
                               hoverinfo="skip", showlegend=False))
    for lv in levels:
        fig.add_trace(go().Scatter(x=w[lv], y=[str(c) for c in cats], mode="markers", name=str(lv),
                                   marker=dict(size=13, color=cmap[lv], line=dict(color=r.theme.surface, width=2)),
                                   hovertemplate=f"<b>%{{y}}</b><br>{lv}: %{{x:,.4~g}}<extra></extra>"))
    fig.update_yaxes(categoryorder="array", categoryarray=[str(c) for c in cats], autorange="reversed")
    fig.update_xaxes(title_text=value)
    fig.update_layout(legend_title_text=group, height=r.height or max(360, 28 * len(cats) + 120))
    return fig


# ====================================================================== pictogram
def _clip_poly_x(poly, xmax):
    from ..layout import clip_halfplane
    return clip_halfplane(poly, (1.0, 0.0), xmax)


@chart("pictogram", "comparison",
       alt_names=("pictogram chart", "pictorial chart", "proportional unit chart", "picture graph", "pictograph"),
       use="Simple counts for a general / multilingual audience; each icon = one unit.",
       avoid="Expert audiences, or values needing precision.")
def pictogram(r, data, category, value, *, unit=None, icon="person", per_row=10, show_values=True):
    """Rows of icons per category; each icon represents ``unit`` (partial icons are cropped)."""
    df = aggregate(as_frame(data), category, value)
    cats, vals = [str(c) for c in df[category]], df[value].to_numpy(float)
    unit = unit or max(1.0, 10 ** math.floor(math.log10(max(vals.max(), 1) / per_row + 1e-12)))
    cmap = r.color_map(cats)
    shapes = []  # (polys, colour, category, full?)
    rows_used = 0
    row_y = {}
    for c, v in zip(cats, vals):
        n_icons = v / unit
        nrow = max(1, math.ceil(n_icons / per_row))
        row_y[c] = rows_used
        for k in range(math.ceil(n_icons)):
            frac = min(1.0, n_icons - k)
            rr, cc = divmod(k, per_row)
            x0, y0 = cc * 1.15, -(rows_used + rr) * 1.25
            for p in icon_polygons(icon, x0, y0, 1.0):
                pp = _clip_poly_x(p, x0 + frac) if frac < 1 else p
                if len(pp) >= 3:
                    shapes.append((pp, cmap[c], c))
            # empty remainder ghost
            if frac < 1:
                for p in icon_polygons(icon, x0, y0, 1.0):
                    shapes.append((p, blend(r.theme.ink, r.theme.surface, 0.08), None))
        rows_used += nrow
    width = per_row * 1.15
    if r.mpl:
        fig, ax = r.subplots(figsize=r.figsize or (8, 0.55 * rows_used + 1.2))
        for p, col, _ in shapes:
            mpl_poly(ax, p, col)
        for c, v in zip(cats, vals):
            ax.text(-0.4, -row_y[c] * 1.25 + 0.5, c, ha="right", va="center", color=r.theme.ink_secondary)
            if show_values:
                ax.text(width + 0.2, -row_y[c] * 1.25 + 0.5, compact(v), ha="left", va="center",
                        color=r.theme.ink_secondary, fontsize=9)
        ax.set_xlim(-0.2 - 0.35 * max(len(c) for c in cats), width + 2)
        ax.set_ylim(-rows_used * 1.25 + 0.1, 1.2)
        mpl_blank(ax)
        ax.text(0, -rows_used * 1.25 - 0.2, f"each icon = {compact(unit)}", color=r.theme.muted, fontsize=8, va="top")
        return fig
    fig = go().Figure()
    for p, col, c in shapes:
        fig.add_trace(pl_poly(p, col, hover=None if c is None else f"{c}: {compact(vals[cats.index(c)])}"))
    for c, v in zip(cats, vals):
        fig.add_annotation(x=-0.3, y=-row_y[c] * 1.25 + 0.5, text=c, showarrow=False, xanchor="right",
                           font=dict(color=r.theme.ink_secondary))
        if show_values:
            fig.add_annotation(x=width + 0.2, y=-row_y[c] * 1.25 + 0.5, text=compact(v), showarrow=False,
                               xanchor="left", font=dict(color=r.theme.ink_secondary))
    fig.add_annotation(x=0, y=-rows_used * 1.25 - 0.1, text=f"each icon = {compact(unit)}", showarrow=False,
                       xanchor="left", yanchor="top", font=dict(color=r.theme.muted, size=11))
    pl_hide_axes(fig)
    fig.update_layout(showlegend=False, height=r.height or int(60 * rows_used + 140),
                      margin=dict(l=140))
    return fig


@chart("icon_chart", "comparison", alt_names=("proportional area chart", "icon chart"),
       use="Values spanning orders of magnitude; area (not length) encodes value.",
       avoid="Small differences - area is hard to compare precisely.")
def icon_chart(r, data, category, value, *, icon="circle", sort=True):
    """One icon per category with *area* proportional to value, placed side by side."""
    df = aggregate(as_frame(data), category, value)
    if sort:
        df = df.sort_values(value, ascending=False)
    cats, vals = [str(c) for c in df[category]], df[value].to_numpy(float)
    sizes = np.sqrt(vals / vals.max()) * 2.0
    xs = np.concatenate([[0], np.cumsum(sizes[:-1] + 0.35)])
    cmap = r.color_map(cats)
    if r.mpl:
        fig, ax = r.subplots(figsize=r.figsize or (max(6, 1.3 * len(cats)), 3.6))
        for c, v, s, x0 in zip(cats, vals, sizes, xs):
            for p in icon_polygons(icon, x0, 0, s):
                mpl_poly(ax, p, cmap[c])
            ax.text(x0 + s / 2, -0.15, f"{c}\n{compact(v)}", ha="center", va="top", fontsize=9,
                    color=r.theme.ink_secondary)
        ax.set_xlim(-0.2, xs[-1] + sizes[-1] + 0.2)
        ax.set_ylim(-0.75, 2.1)
        mpl_blank(ax)
        return fig
    fig = go().Figure()
    for c, v, s, x0 in zip(cats, vals, sizes, xs):
        for p in icon_polygons(icon, x0, 0, s):
            fig.add_trace(pl_poly(p, cmap[c], hover=f"{c}: {compact(v)}"))
        fig.add_annotation(x=x0 + s / 2, y=-0.1, text=f"{c}<br>{compact(v)}", showarrow=False, yanchor="top",
                           font=dict(color=r.theme.ink_secondary, size=11))
    pl_hide_axes(fig)
    fig.update_yaxes(range=[-0.7, 2.1])
    fig.update_layout(showlegend=False)
    return fig


# =================================================================== range plot
@chart("range_plot", "comparison", alt_names=("range chart", "range bar chart", "floating bar chart"),
       use="Focus on the spread/gap between two values per category (e.g. pay gap, min-max).",
       avoid="When individual values matter more than the range.")
def range_plot(r, data, category, low, high, *, sort=True):
    """Floating bar from ``low`` to ``high`` per category."""
    df = as_frame(data).copy()
    need(df, category, low, high)
    if sort:
        df = df.assign(_w=df[high] - df[low]).sort_values("_w", ascending=False)
    cats = [str(c) for c in df[category]]
    lo, hi = df[low].to_numpy(float), df[high].to_numpy(float)
    col = r.colors(1)[0]
    if r.mpl:
        fig, ax = r.subplots(figsize=r.figsize or (7.5, max(3, 0.35 * len(cats) + 1)))
        pos = np.arange(len(cats))
        ax.barh(pos, hi - lo, left=lo, height=0.5, color=col)
        for p, a, b in zip(pos, lo, hi):
            ax.annotate(compact(a), (a, p), xytext=(-4, 0), textcoords="offset points", ha="right",
                        va="center", fontsize=8, color=r.theme.ink_secondary)
            ax.annotate(compact(b), (b, p), xytext=(4, 0), textcoords="offset points", ha="left",
                        va="center", fontsize=8, color=r.theme.ink_secondary)
        ax.set_yticks(pos, cats)
        ax.invert_yaxis()
        mpl_value_grid(ax, "x")
        span = hi.max() - lo.min()
        ax.set_xlim(lo.min() - 0.15 * span, hi.max() + 0.12 * span)
        return fig
    fig = go().Figure(go().Bar(
        y=cats, x=hi - lo, base=lo, orientation="h", marker_color=col, width=0.5,
        customdata=np.column_stack([lo, hi]),
        hovertemplate=f"<b>%{{y}}</b><br>{low}: %{{customdata[0]:,.4~g}}<br>{high}: %{{customdata[1]:,.4~g}}<extra></extra>"))
    fig.update_yaxes(categoryorder="array", categoryarray=cats, autorange="reversed")
    pl_value_grid(fig, "x")
    return fig


# =================================================================== radial bar
@chart("radial_bar", "comparison", alt_names=("circular bar chart", "radial bar chart"),
       use="Eye-catching ranking where accuracy is secondary.",
       avoid="Precise comparison - outer rings look longer than inner ones for equal values.")
def radial_bar(r, data, category, value, *, max_angle=270, sort=True):
    """Concentric arcs: arc length (angle) proportional to value, largest on the outer ring."""
    df = aggregate(as_frame(data), category, value)
    if sort:
        df = df.sort_values(value, ascending=True)
    cats, vals = [str(c) for c in df[category]], df[value].to_numpy(float)
    vmax = vals.max()
    ang = vals / vmax * max_angle
    n = len(cats)
    cmap = r.color_map(cats[::-1])
    if r.mpl:
        fig, ax = r.subplots(polar=True, figsize=r.figsize or (6.5, 6.5))
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        for i, (c, a) in enumerate(zip(cats, ang)):
            rad = 1 + i
            ax.barh(rad, np.deg2rad(a), left=0, height=0.72, color=cmap[c])
            ax.text(np.deg2rad(-2), rad, f"{c}  {compact(vals[i])} ", ha="right", va="center", fontsize=8.5,
                    color=r.theme.ink_secondary)
        ax.set_ylim(0, n + 0.6)
        ax.set_yticks([])
        ax.set_xticks(np.deg2rad(np.linspace(0, max_angle, 4)),
                      [compact(v) for v in np.linspace(0, vmax, 4)])
        ax.spines["polar"].set_visible(False)
        return fig
    fig = go().Figure()
    for i, (c, a, v) in enumerate(zip(cats, ang, vals)):
        fig.add_trace(go().Barpolar(r=[0.72], base=[1 + i - 0.36], theta=[a / 2], width=[a], name=c,
                                    marker_color=cmap[c], marker_line_width=0,
                                    hovertemplate=f"<b>{c}</b>: {compact(v)}<extra></extra>"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, n + 0.6], tickvals=[1 + i for i in range(n)],
                                                 ticktext=cats, angle=90, tickangle=90, showgrid=False, showline=False),
                                 angularaxis=dict(rotation=90, direction="clockwise", tickvals=list(np.linspace(0, max_angle, 4)),
                                                  ticktext=[compact(v) for v in np.linspace(0, vmax, 4)])),
                      showlegend=False)
    return fig


# =========================================================== parallel coordinates
@chart("parallel_coordinates", "comparison",
       alt_names=("parallel plot", "parallel coordinates plot", "parallel coordinates"),
       use="Many numeric dimensions with their own scales; spot clusters across dimensions.",
       avoid="When the audience is not data-literate; order of axes changes the story.")
def parallel_coordinates(r, data, dimensions=None, color=None, *, alpha=0.45, max_rows=2000):
    """One polyline per row across ``dimensions`` axes (each axis scaled min-max on its own)."""
    df = as_frame(data)
    dims = list(dimensions) if dimensions else [c for c in df.columns
                                               if pd.api.types.is_numeric_dtype(df[c]) and c != color]
    d = df.dropna(subset=dims).head(max_rows)
    groups = ordered(d[color]) if color else [None]
    cmap = r.color_map(groups) if color else {None: r.colors(1)[0]}
    mins, maxs = d[dims].min(), d[dims].max()
    norm = (d[dims] - mins) / (maxs - mins).replace(0, 1)
    if r.mpl:
        fig, ax = r.subplots(figsize=r.figsize or (max(7.5, 1.4 * len(dims)), 4.8))
        xs = np.arange(len(dims))
        for g in groups:
            sub = norm if g is None else norm[d[color] == g]
            for k, row in enumerate(sub.to_numpy()):
                ax.plot(xs, row, color=cmap[g], alpha=alpha, lw=1, label=str(g) if (k == 0 and g is not None) else None)
        for i, dname in enumerate(dims):
            ax.axvline(i, color=r.theme.baseline, lw=1)
            ax.text(i, 1.03, compact(maxs[dname]), ha="center", fontsize=8, color=r.theme.muted)
            ax.text(i, -0.06, compact(mins[dname]), ha="center", va="top", fontsize=8, color=r.theme.muted)
        ax.set_xticks(xs, dims)
        ax.xaxis.set_ticks_position("top")
        ax.tick_params(axis="x", pad=18, length=0)
        ax.set_yticks([])
        ax.grid(False)
        ax.spines[["left", "bottom"]].set_visible(False)
        ax.set_xlim(-0.15, len(dims) - 0.85)
        mpl_legend(ax, r, title=color, outside=True)
        for line in (ax.get_legend().get_lines() if ax.get_legend() else []):
            line.set_alpha(1)
            line.set_linewidth(2.5)
        return fig
    if color:
        codes = d[color].map({g: i for i, g in enumerate(groups)}).to_numpy()
        k = len(groups)
        cs = []
        for i, g in enumerate(groups):
            cs += [[i / k, cmap[g]], [(i + 1) / k, cmap[g]]]
        line = dict(color=codes, colorscale=cs, cmin=-0.5 + 0.5, cmax=k, showscale=False)
    else:
        line = dict(color=cmap[None])
    fig = go().Figure(go().Parcoords(
        line=line, dimensions=[dict(label=c, values=d[c]) for c in dims],
        labelfont=dict(color=r.theme.ink), tickfont=dict(color=r.theme.ink_secondary),
        rangefont=dict(color=r.theme.muted)))
    if color:
        legend = "   ".join(f"<span style='color:{cmap[g]}'>■</span> {g}" for g in groups)
        fig.add_annotation(text=legend, x=0, y=-0.12, xref="paper", yref="paper", showarrow=False,
                           xanchor="left", font=dict(color=r.theme.ink_secondary))
    fig.update_layout(margin=dict(l=60, r=60, t=90, b=70))
    return fig


# ========================================================================= radar
@chart("radar", "comparison",
       alt_names=("spider chart", "spider graph", "web chart", "spider web chart", "star chart",
                  "star plot", "cobweb chart", "irregular polygon", "kiviat diagram", "radar chart"),
       use="A rough profile of a few items across >= 3 attributes.",
       avoid="Precise reading; use bars (same scale) or parallel coordinates (different scales).")
def radar(r, data, series, dimensions=None, *, normalize=True, agg="mean", fill=True):
    """Polygon per ``series`` level over ``dimensions`` spokes (wide data, aggregated by ``agg``)."""
    df = as_frame(data)
    dims = list(dimensions) if dimensions else [c for c in df.columns
                                               if pd.api.types.is_numeric_dtype(df[c]) and c != series]
    if len(dims) < 3:
        raise ValueError("radar needs at least 3 dimensions")
    prof = df.groupby(series, observed=True, sort=False)[dims].agg(agg)
    raw = prof.copy()
    if normalize:
        lo, hi = prof.min(), prof.max()
        prof = (prof - lo) / (hi - lo).replace(0, 1) * 0.8 + 0.2
    names = list(prof.index)
    cmap = r.color_map(names)
    ang = np.linspace(0, 2 * np.pi, len(dims), endpoint=False)
    if r.mpl:
        fig, ax = r.subplots(polar=True, figsize=r.figsize or (6.5, 6))
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        for nme in names:
            v = prof.loc[nme].to_numpy(float)
            a, vv = np.append(ang, ang[0]), np.append(v, v[0])
            ax.plot(a, vv, color=cmap[nme], lw=2, label=str(nme))
            if fill:
                ax.fill(a, vv, color=cmap[nme], alpha=0.1)
            ax.scatter(ang, v, s=28, color=cmap[nme], zorder=3)
        ax.set_xticks(ang, dims)
        ax.tick_params(axis="x", pad=14)
        ax.set_yticklabels([])
        ax.set_ylim(0, 1.05 if normalize else None)
        ax.spines["polar"].set_color(r.theme.grid)
        mpl_legend(ax, r, title=series)
        return fig
    fig = go().Figure()
    for nme in names:
        v = prof.loc[nme].to_numpy(float)
        fig.add_trace(go().Scatterpolar(
            r=np.append(v, v[0]), theta=dims + [dims[0]], name=str(nme), mode="lines+markers",
            line=dict(color=cmap[nme], width=2), marker=dict(size=6),
            fill="toself" if fill else None, fillcolor=rgba(cmap[nme], 0.1),
            customdata=np.append(raw.loc[nme].to_numpy(float), raw.loc[nme].iloc[0]),
            hovertemplate="<b>%{theta}</b>: %{customdata:,.4~g}<extra>" + str(nme) + "</extra>"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=not normalize, range=[0, 1.05] if normalize else None),
                                 angularaxis=dict(rotation=90, direction="clockwise")),
                      legend_title_text=series)
    return fig


# =================================================================== nightingale
@chart("nightingale", "comparison",
       alt_names=("nightingale's graph", "nightingale rose chart", "rose diagram", "coxcomb chart",
                  "polar area chart", "nightingale chart"),
       use="Cyclic categories (months, directions) where a striking form is wanted.",
       avoid="Precise comparison; similar-valued sectors.")
def nightingale(r, data, category, value, group=None, *, agg="sum"):
    """Equal-angle wedges whose *area* (radius = sqrt) is proportional to value; ``group`` stacks."""
    df = as_frame(data)
    need(df, category, value, group)
    cats = ordered(df[category])
    groups = ordered(df[group]) if group else [value]
    if group:
        w = df.pivot_table(index=category, columns=group, values=value, aggfunc=agg, observed=True)
    else:
        w = df.groupby(category, observed=True)[value].agg(agg).to_frame(value)
    w = w.reindex(index=cats, columns=groups).fillna(0)
    cum = w.cumsum(axis=1)
    rad_out = np.sqrt(cum)
    rad_in = np.sqrt(cum - w)
    cmap = r.color_map(groups) if group else {value: r.colors(1)[0]}
    n = len(cats)
    if r.mpl:
        fig, ax = r.subplots(polar=True, figsize=r.figsize or (6.5, 6.5))
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        for g in groups:
            ax.bar(th, rad_out[g] - rad_in[g], bottom=rad_in[g], width=2 * np.pi / n * 0.97, color=cmap[g],
                   edgecolor=r.theme.gap, linewidth=1, label=str(g) if group else None, align="edge")
        ax.set_xticks(th + np.pi / n, [str(c) for c in cats])
        ax.set_yticklabels([])
        ax.spines["polar"].set_visible(False)
        mpl_legend(ax, r, title=group)
        return fig
    fig = go().Figure()
    for g in groups:
        fig.add_trace(go().Barpolar(r=rad_out[g] - rad_in[g], base=rad_in[g], theta=[str(c) for c in cats],
                                    name=str(g), marker_color=cmap[g], marker_line=dict(color=r.theme.gap, width=1),
                                    customdata=w[g].to_numpy(),
                                    hovertemplate="<b>%{theta}</b>: %{customdata:,.4~g}<extra>" + (str(g) if group else "") + "</extra>"))
    fig.update_layout(polar=dict(radialaxis=dict(showticklabels=False), bargap=0.03,
                                 angularaxis=dict(rotation=90, direction="clockwise")),
                      showlegend=bool(group), legend_title_text=group or "")
    return fig


# ===================================================================== waterfall
@chart("waterfall", "comparison",
       alt_names=("flying bricks chart", "mario chart", "bridge chart", "cascade chart",
                  "waterfall chart"),
       use="Show how positive/negative changes build from a start value to an end value.",
       avoid="When absolute values of each item matter more than their contribution.")
def waterfall(r, data, category, value, measure=None, *, total_label="Total", connectors=True):
    """Floating bars of changes. ``measure`` column: 'absolute'/'relative'/'total' per row
    (default: first row absolute, the rest relative, plus a final total bar)."""
    df = as_frame(data).reset_index(drop=True)
    need(df, category, value, measure)
    cats = [str(c) for c in df[category]]
    vals = df[value].to_numpy(float)
    if measure:
        meas = list(df[measure].astype(str).str.lower())
    else:
        meas = ["absolute"] + ["relative"] * (len(df) - 1)
        if total_label:
            cats.append(total_label)
            vals = np.append(vals, 0.0)
            meas.append("total")
    up, down, tot = r.theme.color(0), r.theme.color(1), r.theme.ink_secondary
    if r.plotly:
        fig = go().Figure(go().Waterfall(
            x=cats, y=vals, measure=meas, connector=dict(line=dict(color=r.theme.baseline, width=1)) if connectors
            else dict(visible=False),
            increasing=dict(marker=dict(color=up)), decreasing=dict(marker=dict(color=down)),
            totals=dict(marker=dict(color=tot)), textposition="outside",
            text=[compact(v) if m != "total" else "" for v, m in zip(vals, meas)],
            textfont=dict(color=r.theme.ink_secondary)))
        fig.update_layout(showlegend=False)
        pl_value_grid(fig, "y")
        return fig
    fig, ax = r.subplots()
    run = 0.0
    bottoms, heights, colors, ends = [], [], [], []
    for v, m in zip(vals, meas):
        if m == "absolute":
            b, h, c = 0, v, tot
            run = v
        elif m == "total":
            b, h, c = 0, run, tot
        else:
            b, h, c = run, v, up if v >= 0 else down
            run += v
        bottoms.append(b)
        heights.append(h)
        colors.append(c)
        ends.append(run)
    pos = np.arange(len(cats))
    ax.bar(pos, heights, bottom=bottoms, color=colors, width=0.62)
    for p, b, h in zip(pos, bottoms, heights):
        ax.annotate(compact(h), (p, max(b, b + h)), xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=8.5, color=r.theme.ink_secondary)
    if connectors:
        for i in range(len(pos) - 1):
            ax.plot([pos[i] + 0.31, pos[i + 1] - 0.31], [ends[i], ends[i]], color=r.theme.baseline, lw=1)
    rot = 0 if sum(len(c) for c in cats) < 70 else 45
    ax.set_xticks(pos, cats, rotation=rot, ha="right" if rot else "center")
    mpl_value_grid(ax, "y")
    mpl_compact_axis(ax.yaxis)
    ax.axhline(0, color=r.theme.baseline, lw=1)
    return fig


# ======================================================================== matrix
@chart("matrix_chart", "comparison", alt_names=("matrix diagram", "matrix chart"),
       use="Presence/strength of relationships between two categorical variables.",
       avoid="More than two categorical dimensions.")
def matrix_chart(r, data, row, col, value=None, color=None, *, agg="sum", max_size=1.0):
    """Grid of circles at each (row, col) pair, area proportional to ``value`` (or count)."""
    df = as_frame(data)
    need(df, row, col, value, color)
    keys = [row, col] + ([color] if color else [])
    d = aggregate(df, keys, value, agg)
    vcol = value or "count"
    rows_, cols_ = ordered(df[row]), ordered(df[col])
    ri = {c: i for i, c in enumerate(rows_)}
    ci = {c: i for i, c in enumerate(cols_)}
    vmax = d[vcol].abs().max() or 1
    groups = ordered(d[color]) if color else [None]
    cmap = r.color_map(groups) if color else {None: r.colors(1)[0]}
    if r.mpl:
        fig, ax = r.subplots(figsize=r.figsize or (max(5, 0.8 * len(cols_) + 2), max(4, 0.6 * len(rows_) + 1.5)))
        ax_px = 72 * min(fig.get_size_inches()[0] / max(len(cols_), 1), fig.get_size_inches()[1] / max(len(rows_), 1))
        for g in groups:
            sub = d if g is None else d[d[color] == g]
            s = (np.sqrt(sub[vcol].abs() / vmax) * ax_px * 0.42 * max_size) ** 2
            ax.scatter(sub[col].map(ci), sub[row].map(ri), s=s, color=cmap[g], label=None if g is None else str(g),
                       edgecolor=r.theme.surface, linewidth=1.5)
        ax.set_xticks(range(len(cols_)), [str(c) for c in cols_], rotation=30 if len(cols_) > 5 else 0,
                      ha="right" if len(cols_) > 5 else "center")
        ax.set_yticks(range(len(rows_)), [str(c) for c in rows_])
        ax.set_xlim(-0.6, len(cols_) - 0.4)
        ax.set_ylim(len(rows_) - 0.4, -0.6)
        ax.set_xlabel(col)
        ax.set_ylabel(row)
        ax.grid(True, axis="both")
        ax.tick_params(length=0)
        mpl_legend(ax, r, title=color)
        return fig
    fig = go().Figure()
    for g in groups:
        sub = d if g is None else d[d[color] == g]
        fig.add_trace(go().Scatter(
            x=sub[col].astype(str), y=sub[row].astype(str), mode="markers", name=str(g),
            marker=dict(size=np.sqrt(sub[vcol].abs() / vmax) * 48 * max_size, color=cmap[g],
                        line=dict(color=r.theme.surface, width=1.5)),
            customdata=sub[vcol], showlegend=g is not None,
            hovertemplate=f"{row}: %{{y}}<br>{col}: %{{x}}<br>{vcol}: %{{customdata:,.4~g}}<extra></extra>"))
    fig.update_xaxes(categoryorder="array", categoryarray=[str(c) for c in cols_], title_text=col, showgrid=True)
    fig.update_yaxes(categoryorder="array", categoryarray=[str(c) for c in rows_], autorange="reversed",
                     title_text=row, showgrid=True)
    return fig


# =============================================================== small multiples
@chart("small_multiples", "comparison",
       alt_names=("small multiples chart", "trellis chart", "lattice chart", "panel chart", "facet grid", "small multiple"),
       use="Split a cluttered chart (spaghetti lines, many groups) into a grid of simple ones.",
       avoid="Too little space for each panel, or when one chart tells the story.")
def small_multiples(r, data, facet, kind="line", *, cols=3, sharex=True, sharey=True, color_by=None,
                    facet_order=None, **chart_kwargs):
    """Repeat chart ``kind`` (any registered chart name) for each level of ``facet``.

    ``color_by`` names the column used for colour so every panel shares one colour mapping.
    """
    from ..registry import get_spec
    df = as_frame(data)
    need(df, facet)
    spec = get_spec(kind)
    levels = list(facet_order) if facet_order else ordered(df[facet])
    rows = math.ceil(len(levels) / cols)
    palette = r.color_map(df[color_by]) if color_by else r.palette
    common = dict(theme=r.theme, palette=palette)
    if r.mpl:
        polar = kind in {"radar", "radial_bar", "nightingale", "radial_histogram"}
        fig, axes = r.subplots(rows, cols, figsize=r.figsize or (3.4 * cols, 2.8 * rows + 0.4),
                               sharex=sharex and not polar, sharey=sharey and not polar, polar=polar)
        axes = np.atleast_1d(axes).ravel()
        for ax, lv in zip(axes, levels):
            spec.func(df[df[facet] == lv], ax=ax, **common, **chart_kwargs)
            ax.set_title(str(lv), fontsize=10, loc="left")
            leg = ax.get_legend()
            if leg and ax is not axes[0]:
                leg.remove()
        for ax in axes[len(levels):]:
            ax.set_visible(False)
        return fig
    from plotly.subplots import make_subplots
    sub_figs = [spec.func(df[df[facet] == lv], backend="plotly", **common, **chart_kwargs).figure for lv in levels]
    polar = any(t.type in ("barpolar", "scatterpolar") for t in sub_figs[0].data)
    specs = [[{"type": "polar" if polar else "xy"}] * cols for _ in range(rows)]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[str(lv) for lv in levels], shared_xaxes=sharex,
                        shared_yaxes=sharey, specs=specs, horizontal_spacing=0.06, vertical_spacing=0.12)
    seen = set()
    for i, sf in enumerate(sub_figs):
        rr, cc = divmod(i, cols)
        for t in sf.data:
            key = t.name
            t.legendgroup = key
            t.showlegend = bool(key) and key not in seen and t.showlegend is not False
            if key:
                seen.add(key)
            fig.add_trace(t, row=rr + 1, col=cc + 1)
        if sf.layout.barmode:
            fig.update_layout(barmode=sf.layout.barmode)
    for a in fig.layout.annotations:
        a.update(x=a.x - 0.5 / cols + 0.01, xanchor="left", font=dict(size=12, color=r.theme.ink_secondary))
    fig.update_layout(height=r.height or 260 * rows + 80)
    return fig


# ===================================================================== word cloud
@chart("word_cloud", "comparison", alt_names=("tag cloud", "word collage", "wordle", "word cloud"),
       use="A quick, approximate overview of the most frequent words in a text.",
       avoid="Anything needing exact frequencies - word area is a poor value encoding.")
def word_cloud(r, data, word=None, count=None, *, max_words=120, stopwords=None, rotate_frac=0.0,
               canvas=(820, 460), max_font=72, min_font=9, seed=7):
    """Words sized by frequency, spiral-packed. ``data`` = raw text, a {word: count} dict, a
    Series, or a DataFrame with ``word``/``count`` columns."""
    if isinstance(data, str):
        freqs = word_frequencies(data, stopwords=stopwords, top=max_words)
    elif isinstance(data, dict):
        freqs = dict(sorted(data.items(), key=lambda kv: -kv[1])[:max_words])
    elif isinstance(data, pd.Series):
        freqs = data.sort_values(ascending=False).head(max_words).to_dict()
    else:
        df = as_frame(data)
        need(df, word, count)
        freqs = df.groupby(word)[count].sum().nlargest(max_words).to_dict()
    W, H = canvas
    if r.mpl and r.ax is not None:
        bb = r.ax.get_window_extent()
        W, H = bb.width * 72 / r.ax.figure.dpi, bb.height * 72 / r.ax.figure.dpi
    words = wordcloud_layout(freqs, W, H, max_font=max_font, min_font=min_font, rotate_frac=rotate_frac, seed=seed)
    steps = r.theme.seq_steps(4)
    sizes = np.array([w["size"] for w in words])
    tiers = np.digitize(sizes, np.quantile(sizes, [0.5, 0.8, 0.95])) if len(sizes) else []
    r.meta["words"] = pd.DataFrame(words)
    if r.mpl:
        fig, ax = r.subplots(figsize=r.figsize or (W / 72, H / 72))
        if r.ax is None:
            ax.set_position([0, 0, 1, 1])
            fig.set_layout_engine("none")
        for w, t in zip(words, tiers):
            ax.text(w["x"], w["y"], w["word"], fontsize=w["size"], rotation=w["rotation"], ha="center",
                    va="center_baseline", color=steps[t], fontweight="bold")
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        mpl_blank(ax, equal=False)
        return fig
    fig = go().Figure()
    for t in range(4):
        sel = [w for w, tt in zip(words, tiers) if tt == t]
        if not sel:
            continue
        fig.add_trace(go().Scatter(
            x=[w["x"] for w in sel], y=[w["y"] for w in sel], mode="text", text=[w["word"] for w in sel],
            textfont=dict(size=[w["size"] * 96 / 72 for w in sel], color=steps[t], family="DejaVu Sans, sans-serif",
                          weight="bold"),
            customdata=[w["count"] for w in sel], hovertemplate="%{text}: %{customdata}<extra></extra>",
            showlegend=False))
    pl_hide_axes(fig, equal=False)
    top = 50 if r.title else 10
    fig.update_xaxes(range=[0, W])
    fig.update_yaxes(range=[H, 0])
    fig.update_layout(width=r.width or int(W * 96 / 72), height=r.height or int(H * 96 / 72) + top,
                      margin=dict(l=0, r=0, b=0, t=top))
    return fig


# ======================================================================== slope
@chart("slope", "comparison", alt_names=("slopegraph", "slope chart", "slope graph"),
       use="Change between exactly two time points / conditions for several categories.",
       avoid="More than two measurement points.")
def slope(r, data, category, x, y, *, highlight=None, color_by="direction", label_fmt=compact):
    """Two vertical axes (the two levels of ``x``) joined by one line per ``category``.

    Colour by ``"direction"`` (up vs down) or ``"category"``; ``highlight`` = list of categories
    drawn in colour while the rest are greyed.
    """
    df = as_frame(data)
    need(df, category, x, y)
    levels = ordered(df[x])
    if len(levels) != 2:
        raise ValueError(f"slope needs exactly two levels of {x!r}, found {levels}")
    w = df.pivot_table(index=category, columns=x, values=y, aggfunc="mean", observed=True)[levels].dropna()
    cats = list(w.index)
    if color_by == "category":
        cmap = r.color_map(cats)
    else:
        cmap = {c: r.theme.color(0) if w.loc[c, levels[1]] >= w.loc[c, levels[0]] else r.theme.color(1) for c in cats}
    if highlight is not None:
        cmap = {c: (cmap[c] if c in highlight else blend(r.theme.ink, r.theme.surface, 0.25)) for c in cats}
    if r.mpl:
        fig, ax = r.subplots(figsize=r.figsize or (6, 6.5))
        for c in cats:
            a, b = w.loc[c, levels[0]], w.loc[c, levels[1]]
            ax.plot([0, 1], [a, b], color=cmap[c], lw=2, marker="o", markersize=7,
                    markeredgecolor=r.theme.surface, markeredgewidth=1.5)
            ax.text(-0.04, a, f"{c}  {label_fmt(a)}", ha="right", va="center", fontsize=8.5, color=r.theme.ink_secondary)
            ax.text(1.04, b, f"{label_fmt(b)}  {c}", ha="left", va="center", fontsize=8.5, color=r.theme.ink_secondary)
        ax.set_xticks([0, 1], [str(levels[0]), str(levels[1])])
        ax.xaxis.set_ticks_position("top")
        ax.set_yticks([])
        ax.grid(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_visible(False)
        for xv in (0, 1):
            ax.axvline(xv, color=r.theme.baseline, lw=1, zorder=0)
        ax.set_xlim(-0.6, 1.6)
        return fig
    fig = go().Figure()
    for c in cats:
        a, b = w.loc[c, levels[0]], w.loc[c, levels[1]]
        fig.add_trace(go().Scatter(x=[str(levels[0]), str(levels[1])], y=[a, b], mode="lines+markers", name=str(c),
                                   line=dict(color=cmap[c], width=2),
                                   marker=dict(size=9, line=dict(color=r.theme.surface, width=1.5)),
                                   hovertemplate=f"<b>{c}</b><br>%{{x}}: %{{y:,.4~g}}<extra></extra>", showlegend=False))
        fig.add_annotation(x=str(levels[0]), y=a, text=f"{c} {label_fmt(a)}", xanchor="right", showarrow=False,
                           xshift=-8, font=dict(size=11, color=r.theme.ink_secondary))
        fig.add_annotation(x=str(levels[1]), y=b, text=f"{label_fmt(b)} {c}", xanchor="left", showarrow=False,
                           xshift=8, font=dict(size=11, color=r.theme.ink_secondary))
    fig.update_xaxes(side="top", showgrid=True, gridcolor=r.theme.baseline, showline=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=170, r=170), height=r.height or 620)
    return fig


# =================================================================== table chart
@chart("table_chart", "comparison", alt_names=("table chart", "heat table", "conditional formatting table"),
       use="Exact values with per-column conditional colour; a first overview of data.",
       avoid="When different colour rules per column would confuse readers without notes.")
def table_chart(r, data, *, heat=None, diverging=None, fmt=None, max_rows=40):
    """Table with conditional colours: each column in ``heat`` gets its own sequential scale,
    each in ``diverging`` a two-pole scale centred at 0."""
    df = as_frame(data).head(max_rows)
    heat, diverging = list(heat or []), list(diverging or [])
    fmt = fmt or {}
    seq, div = r.theme.seq_cmap(), r.theme.div_cmap()
    import matplotlib.colors as mcolors

    def cell_colors(c):
        if c in heat:
            v = df[c].astype(float)
            n = (v - v.min()) / ((v.max() - v.min()) or 1)
            return [mcolors.to_hex(seq(0.08 + 0.8 * x)) for x in n]
        if c in diverging:
            v = df[c].astype(float)
            m = v.abs().max() or 1
            return [mcolors.to_hex(div(0.5 + 0.5 * x / m)) for x in v]
        return [r.theme.surface] * len(df)

    def fmt_cell(c, v):
        if c in fmt:
            return fmt[c](v) if callable(fmt[c]) else format(v, fmt[c])
        if isinstance(v, float | np.floating):
            return f"{v:,.0f}" if integral[c] else f"{v:,.2f}"
        if isinstance(v, int | np.integer):
            return f"{v:,}"
        return str(v)

    cols = list(df.columns)
    integral = {c: pd.api.types.is_numeric_dtype(df[c]) and bool(np.all(np.mod(df[c].dropna(), 1) == 0))
                for c in cols}
    fills = {c: cell_colors(c) for c in cols}
    texts = {c: [fmt_cell(c, v) for v in df[c]] for c in cols}
    if r.mpl:
        fig, ax = r.subplots(figsize=r.figsize or (max(6, 1.25 * len(cols)), 0.3 * len(df) + 0.8))
        tb = ax.table(cellText=[[texts[c][i] for c in cols] for i in range(len(df))], colLabels=cols,
                      cellColours=[[fills[c][i] for c in cols] for i in range(len(df))], loc="upper center",
                      cellLoc="right", colLoc="right")
        tb.auto_set_font_size(False)
        tb.set_fontsize(9)
        tb.scale(1, 1.25)
        for (i, j), cell in tb.get_celld().items():
            cell.set_edgecolor(r.theme.grid)
            cell.set_linewidth(0.6)
            if i == 0:
                cell.set_text_props(color=r.theme.ink, fontweight="bold")
                cell.set_facecolor(r.theme.surface)
            else:
                cell.get_text().set_color(text_on(fills[cols[j]][i - 1], r.theme) if cols[j] in heat + diverging
                                          else r.theme.ink_secondary)
        mpl_blank(ax, equal=False)
        return fig
    fig = go().Figure(go().Table(
        header=dict(values=[f"<b>{c}</b>" for c in cols], fill_color=r.theme.surface, align="right",
                    line_color=r.theme.grid, font=dict(color=r.theme.ink)),
        cells=dict(values=[texts[c] for c in cols], fill_color=[fills[c] for c in cols], align="right",
                   line_color=r.theme.grid,
                   font=dict(color=[[text_on(f, r.theme) for f in fills[c]] if c in heat + diverging
                                    else r.theme.ink_secondary for c in cols]))))
    fig.update_layout(height=r.height or min(900, 28 * len(df) + 140), margin=dict(l=10, r=10))
    return fig


# ======================================================== categorical scatter
@chart("categorical_scatter", "comparison", alt_names=("categorical scatter plot", "bubble grid"),
       use="Two categorical variables and one numeric (e.g. poll results by group x question).",
       avoid="More than ~20-30 categories per axis.")
def categorical_scatter(r, data, x, y, size=None, color=None, *, agg="mean"):
    """Bubble at each (``x``, ``y``) category pair, area ~ ``size`` (or count), coloured by ``color``."""
    return matrix_chart.__wrapped__(r, data, y, x, size, color, agg=agg)


# ===================================================================== quadrant
@chart("quadrant", "comparison",
       alt_names=("quadrant chart", "4-quadrant matrix chart", "magic quadrant", "swot matrix"),
       use="Benchmark items on two dimensions split into meaningful regions (e.g. SWOT).",
       avoid="When the split thresholds are arbitrary or forced.")
def quadrant(r, data, x, y, label=None, color=None, *, x_split="median", y_split="median",
             quadrant_labels=None, max_labels=25):
    """Scatter divided at ``x_split``/``y_split`` (number, 'median' or 'mean') into four regions."""
    df = as_frame(data).dropna(subset=[x, y])
    need(df, x, y, label, color)
    xs = df[x].agg(x_split) if isinstance(x_split, str) else x_split
    ys = df[y].agg(y_split) if isinstance(y_split, str) else y_split
    groups = ordered(df[color]) if color else [None]
    cmap = r.color_map(groups) if color else {None: r.colors(1)[0]}
    ql = quadrant_labels or ("high " + y + " / low " + x, "high / high", "low / low", "low " + y + " / high " + x)
    if r.mpl:
        fig, ax = r.subplots()
        for g in groups:
            sub = df if g is None else df[df[color] == g]
            ax.scatter(sub[x], sub[y], s=60, color=cmap[g], label=None if g is None else str(g),
                       edgecolor=r.theme.surface, linewidth=1.5, zorder=3)
        ax.axvline(xs, color=r.theme.ink_secondary, lw=1)
        ax.axhline(ys, color=r.theme.ink_secondary, lw=1)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        for (tx, ty, ha, va), t in zip([(x0, y1, "left", "top"), (x1, y1, "right", "top"),
                                        (x0, y0, "left", "bottom"), (x1, y0, "right", "bottom")], ql):
            ax.text(tx, ty, t, ha=ha, va=va, color=r.theme.muted, fontsize=9, fontstyle="italic")
        if label and len(df) <= max_labels:
            for _, row in df.iterrows():
                ax.annotate(str(row[label]), (row[x], row[y]), xytext=(5, 4), textcoords="offset points",
                            fontsize=8, color=r.theme.ink_secondary)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        mpl_legend(ax, r, title=color)
        return fig
    fig = go().Figure()
    for g in groups:
        sub = df if g is None else df[df[color] == g]
        fig.add_trace(go().Scatter(
            x=sub[x], y=sub[y], mode="markers+text" if label and len(df) <= max_labels else "markers",
            text=sub[label] if label else None, textposition="top right", name=str(g),
            textfont=dict(size=10, color=r.theme.ink_secondary), showlegend=g is not None,
            marker=dict(size=11, color=cmap[g], line=dict(color=r.theme.surface, width=1.5)),
            hovertemplate=("<b>%{text}</b><br>" if label else "") + f"{x}: %{{x:,.4~g}}<br>{y}: %{{y:,.4~g}}<extra></extra>"))
    fig.add_vline(x=xs, line=dict(color=r.theme.ink_secondary, width=1))
    fig.add_hline(y=ys, line=dict(color=r.theme.ink_secondary, width=1))
    for (px, py, xa, ya), t in zip([(0, 1, "left", "top"), (1, 1, "right", "top"),
                                    (0, 0, "left", "bottom"), (1, 0, "right", "bottom")], ql):
        fig.add_annotation(x=px, y=py, xref="paper", yref="paper", text=f"<i>{t}</i>", showarrow=False,
                           xanchor=xa, yanchor=ya, font=dict(color=r.theme.muted, size=11))
    fig.update_xaxes(title_text=x)
    fig.update_yaxes(title_text=y)
    return fig
