"""3. Part-to-whole & hierarchical charts."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from matplotlib.colors import to_hex

from .._util import as_frame, blend, compact, need, ordered, rgba, text_on
from ..layout import (build_tree, convex_layout, euler_circles, icon_polygons, pack_layout,
                      polygon_centroid, region_label_points, set_regions, treemap_layout,
                      venn_circles)
from ..layout.hierarchy import polygon_area, regular_polygon
from ..registry import chart
from ._draw import (go, mpl_blank, mpl_poly, mpl_value_grid,
                    pl_hide_axes, pl_poly, pl_value_grid)
from .comparison import _bar_engine


def _largest_remainder(values, total: int) -> np.ndarray:
    """Integer units per category summing exactly to ``total`` (Hamilton apportionment)."""
    v = np.asarray(values, float)
    raw = v / v.sum() * total
    base = np.floor(raw).astype(int)
    rem = total - base.sum()
    base[np.argsort(-(raw - base))[:rem]] += 1
    return base


# ================================================================ stacked bar
@chart("stacked_bar", "part_to_whole",
       alt_names=("stacked bar chart", "stacked column chart", "100% stacked bar chart",
                  "percent stacked bar", "stacked bar"),
       use="Totals per category and their composition; normalize=True compares shares.",
       avoid="Comparing inner segments precisely (only the first segment shares a baseline).")
def stacked_bar(r, data, x, y, group, *, orient="v", normalize=False, sort=None, agg="sum", labels=False):
    """Bars per ``x`` divided into ``group`` segments; ``normalize=True`` gives 100% bars."""
    return _bar_engine(r, data, x, y, group, orient, "stack", sort, labels, normalize=normalize, agg=agg)


# ============================================================== diverging bar
@chart("diverging_bar", "part_to_whole",
       alt_names=("diverging bar chart", "likert chart", "diverging stacked bar chart", "deviation chart"),
       use="Values around a meaningful centre: positive vs negative, or Likert-scale survey answers.",
       avoid="Data without a natural midpoint.")
def diverging_bar(r, data, category, value, group=None, *, levels=None, neutral=None, sort=True):
    """Two modes.

    * ``group=None``: one bar per category coloured by sign (e.g. deviation from average).
    * ``group`` given (Likert): shares of ordered ``levels`` (negative -> positive) stacked left and
      right of zero; the ``neutral`` level is split across the centre.
    """
    df = as_frame(data)
    need(df, category, value, group)
    neg_c, mid_c, pos_c = r.theme.diverging
    if group is None:
        d = df.groupby(category, observed=True, sort=False)[value].sum().reset_index()
        if sort:
            d = d.sort_values(value)
        cats, vals = [str(c) for c in d[category]], d[value].to_numpy(float)
        cols = [pos_c if v >= 0 else neg_c for v in vals]
        if r.mpl:
            fig, ax = r.subplots(figsize=r.figsize or (7.5, max(3, 0.32 * len(cats) + 1)))
            pos = np.arange(len(cats))
            ax.barh(pos, vals, color=cols, height=0.6)
            for p, v in zip(pos, vals):
                ax.annotate(compact(v), (v, p), xytext=(4 if v >= 0 else -4, 0), textcoords="offset points",
                            ha="left" if v >= 0 else "right", va="center", fontsize=8, color=r.theme.ink_secondary)
            ax.set_yticks(pos, cats)
            ax.axvline(0, color=r.theme.ink_secondary, lw=1)
            mpl_value_grid(ax, "x")
            ax.margins(x=0.12)
            return fig
        fig = go().Figure(go().Bar(y=cats, x=vals, orientation="h", marker_color=cols,
                                   hovertemplate="<b>%{y}</b>: %{x:,.4~g}<extra></extra>"))
        fig.update_yaxes(categoryorder="array", categoryarray=cats)
        fig.add_vline(x=0, line=dict(color=r.theme.ink_secondary, width=1))
        pl_value_grid(fig, "x")
        return fig
    lv = list(levels) if levels else ordered(df[group])
    w = df.pivot_table(index=category, columns=group, values=value, aggfunc="sum", observed=True)
    w = w.reindex(columns=lv).fillna(0)
    w = w.div(w.sum(axis=1), axis=0) * 100
    n = len(lv)
    mid = lv.index(neutral) if neutral in lv else None
    neg_levels = lv[: (mid if mid is not None else n // 2)]
    pos_levels = lv[(mid + 1 if mid is not None else n // 2):]
    # colour steps: strongest at the outer (extreme) answers, lighter towards the centre;
    # the neutral level takes a mid grey that still reads against the surface
    def ramp(pole, k):
        return [blend(pole, r.theme.surface, a) for a in np.linspace(1.0, 0.45, max(k, 1))][:k]
    colors = dict(zip(neg_levels, ramp(neg_c, len(neg_levels))))
    colors.update(zip(pos_levels, ramp(pos_c, len(pos_levels))[::-1]))
    if mid is not None:
        colors[neutral] = blend(r.theme.ink, r.theme.surface, 0.18)
    left_mass = w[neg_levels].sum(axis=1) + (w[neutral] / 2 if mid is not None else 0)
    if sort:
        w = w.loc[(w[pos_levels].sum(axis=1) - w[neg_levels].sum(axis=1)).sort_values().index]
        left_mass = left_mass.loc[w.index]
    cats = [str(c) for c in w.index]
    order = neg_levels + ([neutral] if mid is not None else []) + pos_levels
    starts = {}
    acc = -left_mass.to_numpy()
    for l in order:
        starts[l] = acc.copy()
        acc = acc + w[l].to_numpy()
    if r.mpl:
        fig, ax = r.subplots(figsize=r.figsize or (8, max(3, 0.4 * len(cats) + 1.2)))
        pos = np.arange(len(cats))
        for l in order:
            ax.barh(pos, w[l], left=starts[l], color=colors[l], height=0.6, label=str(l),
                    edgecolor=r.theme.gap, linewidth=1)
        ax.set_yticks(pos, cats)
        ax.axvline(0, color=r.theme.ink_secondary, lw=1)
        ax.xaxis.set_major_formatter(__import__("matplotlib").ticker.FuncFormatter(lambda v, _: f"{abs(v):.0f}%"))
        mpl_value_grid(ax, "x")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(order), frameon=False,
                  fontsize=8.5)
        return fig
    fig = go().Figure()
    for l in order:
        fig.add_trace(go().Bar(y=cats, x=w[l], base=starts[l], orientation="h", name=str(l), marker_color=colors[l],
                               marker_line=dict(color=r.theme.gap, width=1), customdata=w[l],
                               hovertemplate="<b>%{y}</b><br>" + str(l) + ": %{customdata:.1f}%<extra></extra>"))
    fig.update_layout(barmode="overlay", legend=dict(orientation="h", y=1.08, x=0, traceorder="normal"))
    fig.update_yaxes(categoryorder="array", categoryarray=cats)
    fig.update_xaxes(ticksuffix="%")
    fig.add_vline(x=0, line=dict(color=r.theme.ink_secondary, width=1))
    return fig


# =========================================================== population pyramid
@chart("population_pyramid", "part_to_whole", alt_names=("age pyramid", "age-sex pyramid", "population pyramid"),
       use="Distribution of a population by age band and sex (back-to-back bars).",
       avoid="More than two comparison groups.")
def population_pyramid(r, data, age, value, sex, *, percent=False, age_order=None):
    """Back-to-back horizontal bars: first ``sex`` level to the left, second to the right."""
    df = as_frame(data)
    need(df, age, value, sex)
    sides = ordered(df[sex])
    if len(sides) != 2:
        raise ValueError(f"population_pyramid needs 2 levels of {sex!r}, found {sides}")
    w = df.pivot_table(index=age, columns=sex, values=value, aggfunc="sum", observed=True)[sides]
    if age_order is not None:
        w = w.reindex(age_order)
    if percent:
        w = w / w.to_numpy().sum() * 100
    ages = [str(a) for a in w.index]
    cmap = r.color_map(sides)
    left, right = w[sides[0]].to_numpy(float), w[sides[1]].to_numpy(float)
    m = max(left.max(), right.max())
    fmt = (lambda v: f"{abs(v):.1f}%") if percent else (lambda v: compact(abs(v)))
    if r.mpl:
        fig, ax = r.subplots(figsize=r.figsize or (7.5, max(4, 0.28 * len(ages) + 1.2)))
        pos = np.arange(len(ages))
        ax.barh(pos, -left, color=cmap[sides[0]], height=0.85, label=str(sides[0]), edgecolor=r.theme.gap, lw=0.8)
        ax.barh(pos, right, color=cmap[sides[1]], height=0.85, label=str(sides[1]), edgecolor=r.theme.gap, lw=0.8)
        ax.set_yticks(pos, ages)
        ax.set_xlim(-m * 1.08, m * 1.08)
        ax.xaxis.set_major_formatter(__import__("matplotlib").ticker.FuncFormatter(lambda v, _: fmt(v)))
        ax.axvline(0, color=r.theme.surface, lw=2)
        ax.set_ylabel(age)
        mpl_value_grid(ax, "x")
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)
        return fig
    fig = go().Figure()
    fig.add_trace(go().Bar(y=ages, x=-left, orientation="h", name=str(sides[0]), marker_color=cmap[sides[0]],
                           customdata=left, hovertemplate="%{y}: %{customdata:,.4~g}<extra>" + str(sides[0]) + "</extra>"))
    fig.add_trace(go().Bar(y=ages, x=right, orientation="h", name=str(sides[1]), marker_color=cmap[sides[1]],
                           customdata=right, hovertemplate="%{y}: %{customdata:,.4~g}<extra>" + str(sides[1]) + "</extra>"))
    ticks = np.linspace(-m, m, 7)
    fig.update_layout(barmode="relative", bargap=0.08, legend=dict(orientation="h", y=1.06, x=0))
    fig.update_xaxes(tickvals=ticks, ticktext=[fmt(t) for t in ticks], range=[-m * 1.08, m * 1.08])
    fig.update_yaxes(categoryorder="array", categoryarray=ages, title_text=age)
    pl_value_grid(fig, "x")
    return fig


# ================================================== icon array / waffle (unit grids)
def _unit_grid(r, data, category, value, total, rows, per_row, icon, kind):
    df = as_frame(data)
    if value is None:
        d = df[category].value_counts(sort=False)
        cats, vals = [str(c) for c in d.index], d.to_numpy(float)
    else:
        d = df.groupby(category, observed=True, sort=False)[value].sum()
        cats, vals = [str(c) for c in d.index], d.to_numpy(float)
    units = _largest_remainder(vals, total)
    cmap = r.color_map(cats)
    owner = np.repeat(np.arange(len(cats)), units)
    gap = 0.12 if kind == "waffle" else 0.25
    cells = []
    for k, ci in enumerate(owner):
        if kind == "waffle":     # fill column by column, bottom-up (reads left -> right)
            col, row = divmod(k, rows)
        else:                    # fill row by row, top-down
            row_from_top, col = divmod(k, per_row)
            row = rows - 1 - row_from_top
        cells.append((col * (1 + gap), row * (1 + gap), cats[ci]))
    shares = {c: u for c, u in zip(cats, units)}
    r.meta["units"] = shares
    labels = {c: f"{c}  {u}{'%' if total == 100 else ''}" for c, u in shares.items()}
    if r.mpl:
        ncols = math.ceil(total / rows) if kind == "waffle" else per_row
        fig, ax = r.subplots(figsize=r.figsize or (max(5, 0.45 * ncols + 2.6), max(3.5, 0.45 * rows + 0.6)))
        for x0, y0, c in cells:
            for p in icon_polygons(icon, x0, y0, 1.0):
                mpl_poly(ax, p, cmap[c])
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=cmap[c], label=labels[c]) for c in cats], loc="upper left",
                  bbox_to_anchor=(1.01, 1), frameon=False)
        mpl_blank(ax)
        ax.autoscale_view()
        return fig
    fig = go().Figure()
    seen = set()
    for x0, y0, c in cells:
        for p in icon_polygons(icon, x0, y0, 1.0):
            fig.add_trace(pl_poly(p, cmap[c], name=labels[c], hover=labels[c], legendgroup=c,
                                  showlegend=c not in seen))
            seen.add(c)
    pl_hide_axes(fig)
    fig.update_layout(legend=dict(x=1.02, y=1))
    return fig


@chart("icon_array", "part_to_whole", alt_names=("isotype chart", "icon array"),
       use="Part-to-whole with countable units (e.g. 'x in 100 people') for any audience.",
       avoid="Very large or uneven counts, small differences, more than ~5 categories.")
def icon_array(r, data, category, value=None, *, total=100, per_row=10, icon="person"):
    """``total`` icons (default 100) split between categories by largest-remainder rounding."""
    rows = math.ceil(total / per_row)
    return _unit_grid(r, data, category, value, total, rows, per_row, icon, "icon")


@chart("waffle", "part_to_whole", alt_names=("square pie chart", "square area chart", "gridplot", "waffle chart"),
       use="Part-to-whole as a grid of squares (often 10x10 = percentages); easier than pies.",
       avoid="Many categories or tiny shares (< 1 cell).")
def waffle(r, data, category, value=None, *, rows=10, columns=10):
    """A ``rows`` x ``columns`` grid of squares; each square = 1/(rows*columns) of the total."""
    return _unit_grid(r, data, category, value, rows * columns, rows, columns, "square", "waffle")


# =============================================================== pie family
def _pie_core(r, data, category, value, hole, semicircle, sort, max_slices):
    df = as_frame(data)
    if value is None:
        d = df[category].value_counts()
    else:
        d = df.groupby(category, observed=True, sort=False)[value].sum()
        if sort:
            d = d.sort_values(ascending=False)
    if max_slices and len(d) > max_slices:
        top = d.iloc[: max_slices - 1]
        d = pd.concat([top, pd.Series({"Other": d.iloc[max_slices - 1:].sum()})])
    cats, vals = [str(c) for c in d.index], d.to_numpy(float)
    cmap = r.color_map(cats)
    total = vals.sum()
    if r.mpl:
        fig, ax = r.subplots(figsize=r.figsize or (6.8, 4.8 if not semicircle else 4.0))
        v = list(vals) + ([total] if semicircle else [])
        cols = [cmap[c] for c in cats] + (["none"] if semicircle else [])
        wedges, *_ = ax.pie(v, colors=cols, startangle=180 if semicircle else 90, counterclock=False,
                            wedgeprops=dict(width=1 - hole if hole else 1, edgecolor=r.theme.gap, linewidth=2))
        for wdg, c, val in zip(wedges, cats, vals):
            ang = np.deg2rad((wdg.theta1 + wdg.theta2) / 2)
            share = val / total
            if share < 0.03:
                continue
            rad = 1 - (1 - hole) / 2 if hole else 0.65
            ax.text(rad * np.cos(ang), rad * np.sin(ang), f"{share:.0%}", ha="center", va="center", fontsize=9,
                    color=text_on(cmap[c], r.theme))
        if semicircle:
            wedges[-1].set_visible(False)
            ax.set_ylim(-0.1, 1.05)
        if hole:
            ax.text(0, 0.12 if semicircle else 0, compact(total), ha="center", va="center", fontsize=16,
                    fontweight="bold", color=r.theme.ink)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=cmap[c], label=f"{c}  {compact(v)}") for c, v in zip(cats, vals)],
                  loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
        ax.set_aspect("equal")
        ax.set_axis_off()
        return fig
    labels = cats + ([""] if semicircle else [])
    values = list(vals) + ([total] if semicircle else [])
    colors = [cmap[c] for c in cats] + (["rgba(0,0,0,0)"] if semicircle else [])
    fig = go().Figure(go().Pie(
        labels=labels, values=values, hole=hole, sort=False, direction="clockwise",
        rotation=90 if semicircle else 0,
        marker=dict(colors=colors, line=dict(color=r.theme.gap, width=2)),
        textinfo="percent", textposition="inside", insidetextorientation="horizontal",
        texttemplate=["%{percent:.0%}" for _ in cats] + ([""] if semicircle else []),
        hovertemplate="<b>%{label}</b><br>%{value:,.4~g} (%{percent})<extra></extra>"))
    if semicircle:
        fig.data[0].update(showlegend=True)
        fig.update_layout(legend=dict(traceorder="normal"))
    if hole:
        fig.add_annotation(text=f"<b>{compact(total)}</b>", x=0.5, y=0.58 if semicircle else 0.5, xref="paper",
                           yref="paper", showarrow=False, font=dict(size=22, color=r.theme.ink))
    return fig


@chart("pie", "part_to_whole", alt_names=("pie chart", "circle chart"),
       use="Composition of a whole with few (2-5) categories.",
       avoid="Many slices, similar values, or comparing several pies.")
def pie(r, data, category, value=None, *, sort=True, max_slices=7):
    """Slices of a whole; categories beyond ``max_slices`` fold into 'Other'."""
    return _pie_core(r, data, category, value, 0.0, False, sort, max_slices)


@chart("donut", "part_to_whole", alt_names=("donut chart", "doughnut chart", "ring chart"),
       use="A pie with the total (or a KPI) in the centre.",
       avoid="Same as pie: many or similar slices.")
def donut(r, data, category, value=None, *, hole=0.55, sort=True, max_slices=7):
    """Pie with a hole; the total is printed in the centre."""
    return _pie_core(r, data, category, value, hole, False, sort, max_slices)


@chart("semicircle_donut", "part_to_whole",
       alt_names=("semicircle donut chart", "half donut", "semi-circle pie", "parliament chart"),
       use="Compact part-to-whole, e.g. seat shares or progress towards a total.",
       avoid="Many categories; precise comparison.")
def semicircle_donut(r, data, category, value=None, *, hole=0.55, sort=False, max_slices=7):
    """Half donut spanning 180 degrees."""
    return _pie_core(r, data, category, value, hole, True, sort, max_slices)


# ============================================================= marimekko / mosaic
def _marimekko_geom(df, x, y, value):
    w = df.pivot_table(index=x, columns=y, values=value, aggfunc="sum", observed=True).fillna(0)
    w = w.reindex(index=ordered(df[x]), columns=ordered(df[y]))
    col_tot = w.sum(axis=1)
    widths = col_tot / col_tot.sum()
    lefts = widths.cumsum() - widths
    shares = w.div(col_tot, axis=0)
    bottoms = shares.cumsum(axis=1) - shares
    return w, widths, lefts, shares, bottoms


def _mekko_draw(r, df, x, y, value, color_fn, legend_title, hover_extra=None, colorbar=None):
    w, widths, lefts, shares, bottoms = _marimekko_geom(df, x, y, value)
    cats, groups = list(w.index), list(w.columns)
    gap = 0.004
    if r.mpl:
        fig, ax = r.subplots()
        from matplotlib.patches import Rectangle
        for g in groups:
            for c in cats:
                h = shares.loc[c, g]
                if h <= 0:
                    continue
                col = color_fn(c, g)
                ax.add_patch(Rectangle((lefts[c] + gap, bottoms.loc[c, g]), widths[c] - 2 * gap, h, facecolor=col,
                                       edgecolor=r.theme.gap, linewidth=1.5))
                if h > 0.06 and widths[c] > 0.07:
                    ax.text(lefts[c] + widths[c] / 2, bottoms.loc[c, g] + h / 2, f"{h:.0%}", ha="center",
                            va="center", fontsize=8, color=text_on(col, r.theme))
        ax.set_xticks(lefts + widths / 2, [f"{c}\n{widths[c]:.0%}" for c in cats])
        ax.set_yticks(bottoms.iloc[-1] + shares.iloc[-1] / 2, [str(g) for g in groups])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(False)
        ax.tick_params(length=0)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        if colorbar is not None:
            cb = fig.colorbar(colorbar, ax=ax, shrink=0.8, pad=0.02, label="Pearson residual")
            cb.outline.set_visible(False)
        return fig
    fig = go().Figure()
    for g in groups:
        fig.add_trace(go().Bar(
            x=(lefts + widths / 2).to_numpy(), y=shares[g].to_numpy(), width=(widths - 2 * gap).to_numpy(),
            base=bottoms[g].to_numpy(), name=str(g), marker_color=[color_fn(c, g) for c in cats],
            marker_line=dict(color=r.theme.gap, width=1.5),
            customdata=np.column_stack([cats, w[g].to_numpy(), (hover_extra[g].to_numpy() if hover_extra is not None
                                                                else np.zeros(len(cats)))]),
            hovertemplate=f"{x}: %{{customdata[0]}}<br>{y}: {g}<br>share: %{{y:.1%}}<br>value: %{{customdata[1]:,.4~g}}"
                          + ("<br>residual: %{customdata[2]:.2f}" if hover_extra is not None else "") + "<extra></extra>",
            showlegend=hover_extra is None))
    fig.update_layout(barmode="overlay", bargap=0, legend_title_text=legend_title)
    fig.update_xaxes(tickvals=(lefts + widths / 2).to_numpy(), ticktext=[f"{c}<br>{widths[c]:.0%}" for c in cats],
                     range=[0, 1], showgrid=False, title_text=x)
    fig.update_yaxes(range=[0, 1], tickformat=".0%", title_text=y)
    return fig


@chart("marimekko", "part_to_whole", alt_names=("mekko chart", "variable width stacked bar", "marimekko chart"),
       use="Two-level composition where both column totals and inner shares matter (market maps).",
       avoid="Many segments - small rectangles become unreadable.")
def marimekko(r, data, x, y, value=None):
    """Column width = share of ``x`` in the total; segment height = share of ``y`` inside each column."""
    df = as_frame(data)
    if value is None:
        df = df.assign(_n=1)
        value = "_n"
    cmap = r.color_map(ordered(df[y]))
    return _mekko_draw(r, df, x, y, value, lambda c, g: cmap[g], y)


@chart("mosaic", "part_to_whole", alt_names=("mosaic plot", "mekko plot", "spineplot"),
       use="Association between two categorical variables; cells coloured by deviation from independence.",
       avoid="Many levels per variable.")
def mosaic(r, data, x, y, *, residuals=True):
    """Marimekko of counts. With ``residuals`` cells are coloured by Pearson residual
    (observed - expected)/sqrt(expected): blue = fewer than independence predicts, red = more."""
    from scipy.stats import chi2_contingency
    df = as_frame(data).dropna(subset=[x, y]).assign(_n=1)
    if not residuals:
        cmap = r.color_map(ordered(df[y]))
        return _mekko_draw(r, df, x, y, "_n", lambda c, g: cmap[g], y)
    tab = pd.crosstab(df[x], df[y]).reindex(index=ordered(df[x]), columns=ordered(df[y]))
    res = chi2_contingency(tab)
    resid = (tab - res.expected_freq) / np.sqrt(res.expected_freq)
    r.meta["chi2"] = {"statistic": float(res.statistic), "pvalue": float(res.pvalue), "dof": int(res.dof)}
    r.meta["residuals"] = resid
    m = max(2.0, float(np.abs(resid.to_numpy()).max()))
    import matplotlib.colors as mcolors
    cm = r.theme.div_cmap()
    norm = mcolors.Normalize(-m, m)
    sm = None
    if r.mpl:
        import matplotlib.cm as mcm
        sm = mcm.ScalarMappable(norm=norm, cmap=cm)
    fig = _mekko_draw(r, df, x, y, "_n", lambda c, g: to_hex(cm(norm(resid.loc[c, g]))), y,
                      hover_extra=resid if r.plotly else None, colorbar=sm)
    if r.plotly:
        fig.add_annotation(text=f"χ² = {res.statistic:.1f}, p = {res.pvalue:.2g}", x=1, y=1.06, xref="paper",
                           yref="paper", showarrow=False, xanchor="right", font=dict(color=r.theme.ink_secondary))
    else:
        fig.axes[0].text(1, 1.02, f"χ² = {res.statistic:.1f}, p = {res.pvalue:.2g}", transform=fig.axes[0].transAxes,
                         ha="right", fontsize=9, color=r.theme.ink_secondary)
    return fig


# ===================================================================== treemaps
def _tree(data, path, value):
    df = as_frame(data)
    path = [path] if isinstance(path, str) else list(path)
    need(df, *path, value)
    return build_tree(df, path, value), path


def _node_colors(r, root):
    tops = [c.label for c in root.children]
    cmap = r.color_map(tops)
    out = {}
    for n in root.walk():
        if n is root:
            continue
        base = cmap[n.top().label]
        out[n.id] = base if n.depth == 1 or n.is_leaf else blend(base, r.theme.surface, 0.75)
    return out, cmap


def _plotly_hier(r, root, kind, colors, **kw):
    ids, labels, parents, values, cols = [], [], [], [], []
    for n in root.walk():
        if n is root:
            continue
        ids.append(n.id)
        labels.append(n.label)
        parents.append(n.parent.id if n.parent is not root else "")
        values.append(n.value)
        cols.append(colors[n.id])
    trace = (go().Treemap if kind == "treemap" else go().Sunburst)(
        ids=ids, labels=labels, parents=parents, values=values, branchvalues="total",
        marker=dict(colors=cols, line=dict(color=r.theme.gap, width=2)),
        hovertemplate="<b>%{label}</b><br>%{value:,.4~g} (%{percentRoot:.1%} of total)<extra></extra>", **kw)
    return go().Figure(trace)


@chart("treemap", "part_to_whole", alt_names=("tree map", "rectangular treemap", "treemap chart"),
       use="Hierarchical part-to-whole with many items; area = value.",
       avoid="Precise comparison or few categories (a bar chart is clearer).")
def treemap(r, data, path, value, *, max_depth=None):
    """Squarified treemap of the hierarchy ``path`` (outer -> inner columns), area = ``value``."""
    root, path = _tree(data, path, value)
    colors, _ = _node_colors(r, root)
    if r.plotly:
        fig = _plotly_hier(r, root, "treemap", colors, textinfo="label+value+percent root",
                           tiling=dict(packing="squarify", pad=3), maxdepth=max_depth or -1,
                           pathbar=dict(visible=True))
        fig.update_layout(margin=dict(l=10, r=10, b=10))
        return fig
    W, H = (r.figsize or (8, 5))
    treemap_layout(root, 0, 0, W, H, pad=0.03, pad_top=0.22 if len(path) > 1 else 0)
    fig, ax = r.subplots(figsize=(W, H))
    from matplotlib.patches import Rectangle
    for n in root.walk():
        if n is root or (max_depth and n.depth > max_depth):
            continue
        w, h = n.x1 - n.x0, n.y1 - n.y0
        ax.add_patch(Rectangle((n.x0, H - n.y1), w, h, facecolor=colors[n.id], edgecolor=r.theme.gap, lw=1.5))
        if n.is_leaf or n.depth == (max_depth or -1):
            fs = min(10, max(6, 9 * min(w, h) ** 0.5))
            if w > 0.45 and h > 0.3:
                ax.text(n.x0 + 0.05, H - n.y0 - 0.05, f"{n.label}\n{compact(n.value)}", ha="left", va="top",
                        fontsize=fs, color=text_on(colors[n.id], r.theme), clip_on=True)
        elif w > 0.4:
            ax.text(n.x0 + 0.05, H - n.y0 - 0.03, n.label, ha="left", va="top", fontsize=8.5, fontweight="bold",
                    color=r.theme.ink_secondary, clip_on=True)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    mpl_blank(ax)
    return fig


@chart("circular_treemap", "part_to_whole",
       alt_names=("circle packing", "circular packing", "packed circles", "bubble treemap", "circular treemap"),
       use="Hierarchy shown by nesting; good for organic, attention-grabbing overviews.",
       avoid="Precise area comparison; wastes space compared to a rectangular treemap.")
def circular_treemap(r, data, path, value, *, labels=True, pad=0.0):
    """Nested circle packing (port of d3.pack): leaf area = ``value``, parents enclose children."""
    root, path = _tree(data, path, value)
    pack_layout(root, 1.0, pad=pad or max(root.value, 1) ** 0.5 * 0.008)
    colors, cmap = _node_colors(r, root)
    nodes = [n for n in root.walk()]
    depth_max = max(n.depth for n in nodes)

    def fill(n):
        if n is root:
            return blend(r.theme.ink, r.theme.surface, 0.04)
        return colors[n.id] if n.is_leaf else blend(cmap[n.top().label], r.theme.surface, 0.18 + 0.1 * n.depth)

    if r.mpl:
        fig, ax = r.subplots(figsize=r.figsize or (6.8, 6.8))
        from matplotlib.patches import Circle
        for n in nodes:
            ax.add_patch(Circle((n.cx, n.cy), n.r, facecolor=fill(n), edgecolor=r.theme.surface, lw=1))
            if labels and n.is_leaf and n.r > 0.06:
                ax.text(n.cx, n.cy, n.label, ha="center", va="center", fontsize=min(9, 60 * n.r),
                        color=text_on(fill(n), r.theme), clip_on=True)
        if labels and depth_max > 1:
            for n in root.children:
                ax.text(n.cx, n.cy + n.r, n.label, ha="center", va="bottom", fontsize=9, fontweight="bold",
                        color=r.theme.ink_secondary)
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.02, 1.08)
        mpl_blank(ax)
        return fig
    fig = go().Figure()
    for n in nodes:
        fig.add_shape(type="circle", x0=n.cx - n.r, x1=n.cx + n.r, y0=n.cy - n.r, y1=n.cy + n.r,
                      fillcolor=fill(n), line=dict(color=r.theme.surface, width=1), layer="below")
    leaves = [n for n in nodes if n is not root]
    fig.add_trace(go().Scatter(
        x=[n.cx for n in leaves], y=[n.cy for n in leaves], mode="text",
        text=[n.label if (labels and n.is_leaf and n.r > 0.06) else "" for n in leaves],
        textfont=dict(color=[text_on(fill(n), r.theme) for n in leaves], size=11),
        customdata=[[n.id.replace("/", " / "), n.value] for n in leaves], showlegend=False,
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:,.4~g}<extra></extra>"))
    pl_hide_axes(fig)
    fig.update_xaxes(range=[-1.05, 1.05])
    fig.update_yaxes(range=[-1.05, 1.05])
    fig.update_layout(height=r.height or 640, width=r.width or 640)
    return fig


@chart("convex_treemap", "part_to_whole", alt_names=("voronoi treemap", "polygon treemap", "convex treemap"),
       use="A treemap that must fill a non-rectangular outline (circle, polygon) with an organic look.",
       avoid="Exact area comparison.")
def convex_treemap(r, data, path, value, *, outline="circle", sides=6, angles=12, labels=True):
    """Recursive straight-line bisection of a convex outline into convex polygons with area
    proportional to value (balanced splits, cut angle chosen for the fattest pieces)."""
    root, path = _tree(data, path, value)
    shape = regular_polygon(96) if outline == "circle" else regular_polygon(sides) if outline == "polygon" \
        else np.asarray(outline, float)
    convex_layout(root, shape, angles=angles)
    colors, _ = _node_colors(r, root)
    leaves = root.leaves()
    tops = root.children
    if r.mpl:
        fig, ax = r.subplots(figsize=r.figsize or (6.8, 6.8))
        for n in leaves:
            mpl_poly(ax, n.polygon, colors[n.id], edge=r.theme.surface, lw=1.2)
            if labels and polygon_area(n.polygon) > 0.04:
                cx, cy = polygon_centroid(n.polygon)
                ax.text(cx, cy, n.label, ha="center", va="center", fontsize=8, color=text_on(colors[n.id], r.theme))
        if len(path) > 1:
            for n in tops:
                mpl_poly(ax, n.polygon, "none", edge=r.theme.surface, lw=3.5, zorder=4)
        mpl_blank(ax)
        ax.autoscale_view()
        return fig
    fig = go().Figure()
    for n in leaves:
        fig.add_trace(pl_poly(n.polygon, colors[n.id], line_color=r.theme.surface, line_width=1,
                              hover=f"{n.id.replace('/', ' / ')}: {compact(n.value)}"))
    if len(path) > 1:
        for n in tops:
            p = np.vstack([n.polygon, n.polygon[:1]])
            fig.add_trace(go().Scatter(x=p[:, 0], y=p[:, 1], mode="lines", line=dict(color=r.theme.surface, width=3.5),
                                       hoverinfo="skip", showlegend=False))
    cents = [polygon_centroid(n.polygon) for n in leaves]
    fig.add_trace(go().Scatter(x=[c[0] for c in cents], y=[c[1] for c in cents], mode="text",
                               text=[n.label if labels and polygon_area(n.polygon) > 0.04 else "" for n in leaves],
                               textfont=dict(size=10, color=[text_on(colors[n.id], r.theme) for n in leaves]),
                               hoverinfo="skip", showlegend=False))
    pl_hide_axes(fig)
    fig.update_layout(showlegend=False, height=r.height or 640, width=r.width or 640)
    return fig


# =================================================================== dendrogram
@chart("dendrogram", "part_to_whole", alt_names=("phylogenetic tree", "cluster tree", "hierarchical clustering"),
       use="Hierarchical clustering result: which items merge, and at what distance.",
       avoid="Hundreds of leaves (labels become unreadable).")
def dendrogram(r, data, labels=None, *, method="ward", metric="euclidean", orientation="left",
               color_threshold=None, scale=True):
    """Hierarchical clustering (scipy) of the rows of the numeric columns of ``data``.

    ``labels`` = column used for leaf labels. Clusters below ``color_threshold`` (default 0.7 x the
    largest merge distance) are coloured.
    """
    from scipy.cluster import hierarchy as sch
    df = as_frame(data)
    leaf = df[labels].astype(str).tolist() if labels else [str(i) for i in df.index]
    X = df.select_dtypes("number").drop(columns=[labels] if labels in df else [], errors="ignore")
    X = X.dropna()
    leaf = [leaf[i] for i in range(len(df)) if df.index[i] in X.index]
    Xv = X.to_numpy(float)
    if scale:
        Xv = (Xv - Xv.mean(0)) / np.where(Xv.std(0) == 0, 1, Xv.std(0))
    Z = sch.linkage(Xv, method=method, metric=metric)
    thr = color_threshold if color_threshold is not None else 0.7 * Z[:, 2].max()
    sch.set_link_color_palette(list(r.theme.categorical))
    dn = sch.dendrogram(Z, labels=leaf, no_plot=True, color_threshold=thr, above_threshold_color=r.theme.muted)
    sch.set_link_color_palette(None)
    r.meta["linkage"] = Z
    r.meta["clusters"] = pd.Series(sch.fcluster(Z, thr, "distance"), index=leaf, name="cluster")
    horiz = orientation in ("left", "right")
    n = len(dn["ivl"])
    if r.mpl:
        fig, ax = r.subplots(figsize=r.figsize or ((7.5, max(4, 0.22 * n + 1)) if horiz else (max(7.5, 0.25 * n), 5)))
        for xs, ys, c in zip(dn["icoord"], dn["dcoord"], dn["color_list"]):
            if horiz:
                ax.plot(ys, xs, color=c, lw=1.5)
            else:
                ax.plot(xs, ys, color=c, lw=1.5)
        ticks = np.arange(5, 10 * n, 10)
        if horiz:
            ax.set_yticks(ticks, dn["ivl"], fontsize=8)
            ax.set_xlabel(f"{method} distance")
            if orientation == "left":
                ax.invert_xaxis()
                ax.yaxis.tick_right()
            ax.grid(False)
            ax.grid(True, axis="x")
        else:
            ax.set_xticks(ticks, dn["ivl"], rotation=90, fontsize=8)
            ax.set_ylabel(f"{method} distance")
            ax.grid(False)
            ax.grid(True, axis="y")
        ax.axvline(thr, color=r.theme.baseline, lw=1, ls="--") if horiz else ax.axhline(thr, color=r.theme.baseline, lw=1, ls="--")
        ax.tick_params(length=0)
        return fig
    fig = go().Figure()
    for xs, ys, c in zip(dn["icoord"], dn["dcoord"], dn["color_list"]):
        fig.add_trace(go().Scatter(x=ys if horiz else xs, y=xs if horiz else ys, mode="lines",
                                   line=dict(color=c, width=1.5), hovertemplate="distance %{x:.3g}<extra></extra>"
                                   if horiz else "distance %{y:.3g}<extra></extra>", showlegend=False))
    ticks = list(np.arange(5, 10 * n, 10))
    if horiz:
        fig.update_yaxes(tickvals=ticks, ticktext=dn["ivl"], side="right" if orientation == "left" else "left",
                         showgrid=False)
        fig.update_xaxes(autorange="reversed" if orientation == "left" else True, title_text=f"{method} distance")
        fig.update_layout(height=r.height or max(420, 16 * n + 120))
    else:
        fig.update_xaxes(tickvals=ticks, ticktext=dn["ivl"], tickangle=-90, showgrid=False)
        fig.update_yaxes(title_text=f"{method} distance")
    return fig


# ================================================================== venn / euler
def _sets_from(data, sets):
    """{name: set} from a dict of iterables, or from boolean membership columns of a DataFrame."""
    if isinstance(data, dict):
        return {k: set(v) for k, v in data.items()}
    df = as_frame(data)
    cols = list(sets) if sets else [c for c in df.columns if df[c].dtype == bool]
    return {c: set(df.index[df[c].astype(bool)]) for c in cols}


def _circles_chart(r, circles, regions, title_sets):
    names = list(circles)
    cmap = r.color_map(names)
    anchors = region_label_points(circles)
    if r.mpl:
        fig, ax = r.subplots(figsize=r.figsize or (6.5, 5.5))
        from matplotlib.patches import Circle
        for n, (x, y, rad) in circles.items():
            ax.add_patch(Circle((x, y), rad, facecolor=cmap[n], alpha=0.28, edgecolor=cmap[n], lw=2))
        for key, (x, y) in anchors.items():
            v = regions.get(key, 0)
            ax.text(x, y, f"{v:,}", ha="center", va="center", fontsize=10, color=r.theme.ink)
        for n, (x, y, rad) in circles.items():
            ang = math.atan2(y, x) if (x or y) else math.pi / 2
            ax.text(x + 1.08 * rad * math.cos(ang), y + 1.08 * rad * math.sin(ang), f"{n}\n({len(title_sets[n]):,})",
                    ha="center", va="center", fontsize=10, fontweight="bold", color=r.theme.ink_secondary)
        mpl_blank(ax)
        ax.autoscale_view()
        ax.margins(0.18)
        return fig
    fig = go().Figure()
    for n, (x, y, rad) in circles.items():
        fig.add_shape(type="circle", x0=x - rad, x1=x + rad, y0=y - rad, y1=y + rad, fillcolor=rgba(cmap[n], 0.28),
                      line=dict(color=cmap[n], width=2))
        ang = math.atan2(y, x) if (x or y) else math.pi / 2
        fig.add_annotation(x=x + 1.12 * rad * math.cos(ang), y=y + 1.12 * rad * math.sin(ang),
                           text=f"<b>{n}</b><br>({len(title_sets[n]):,})", showarrow=False,
                           font=dict(color=r.theme.ink_secondary))
    keys = list(anchors)
    fig.add_trace(go().Scatter(
        x=[anchors[k][0] for k in keys], y=[anchors[k][1] for k in keys], mode="text",
        text=[f"{regions.get(k, 0):,}" for k in keys], textfont=dict(size=13, color=r.theme.ink), showlegend=False,
        customdata=[" ∩ ".join(k) + (" only" if len(k) < len(names) else "") for k in keys],
        hovertemplate="%{customdata}: %{text}<extra></extra>"))
    xs = [c[0] - c[2] for c in circles.values()] + [c[0] + c[2] for c in circles.values()]
    ys = [c[1] - c[2] for c in circles.values()] + [c[1] + c[2] for c in circles.values()]
    pad = 0.35 * max(c[2] for c in circles.values())
    pl_hide_axes(fig)
    fig.update_xaxes(range=[min(xs) - pad, max(xs) + pad])
    fig.update_yaxes(range=[min(ys) - pad, max(ys) + pad])
    return fig


@chart("venn", "part_to_whole", alt_names=("venn diagram", "set diagram", "logic diagram"),
       use="Overlaps between 2-3 sets with every possible intersection shown.",
       avoid="More than 3 sets (use an UpSet-style matrix) or when areas should be proportional (euler).")
def venn(r, data, sets=None):
    """Symmetric 2- or 3-set Venn diagram with the size of each exclusive region.

    ``data`` = {name: iterable} or a DataFrame of boolean membership columns (``sets`` selects them).
    """
    s = _sets_from(data, sets)
    regions = set_regions(s)
    r.meta["regions"] = regions
    return _circles_chart(r, venn_circles(list(s)), regions, s)


@chart("euler", "part_to_whole", alt_names=("euler diagram", "proportional venn diagram", "area-proportional venn"),
       use="Set overlaps where circle and overlap areas reflect sizes; empty intersections omitted.",
       avoid="More than 3 sets; 3-set layouts are approximate.")
def euler(r, data, sets=None):
    """Area-proportional circles: exact for 2 sets, best-effort pairwise for 3."""
    s = _sets_from(data, sets)
    regions = set_regions(s)
    names = list(s)
    sizes = {n: len(s[n]) for n in names}
    overlaps = {(a, b): len(s[a] & s[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    circ = euler_circles(sizes, overlaps)
    r.meta["regions"] = regions
    return _circles_chart(r, circ, regions, s)


# ======================================================================== gauge
@chart("gauge", "part_to_whole", alt_names=("circular gauge", "speedometer chart", "dial chart", "gauge chart"),
       use="One KPI checked instantly against good / warning / critical zones.",
       avoid="Several metrics side by side or exact reading - use bullet charts.")
def gauge(r, value, *, min_value=0, max_value=100, bands=None, label="", target=None, fmt=compact):
    """Half-circle gauge with a needle. ``bands`` = [(upper_bound, status), ...] where status is
    'good' | 'warning' | 'serious' | 'critical' or a colour."""
    bands = bands or [(max_value * 0.6, "critical"), (max_value * 0.8, "warning"), (max_value, "good")]
    st = r.theme.status
    band_cols = [st.get(c, c) for _, c in bands]
    if r.plotly:
        steps, prev = [], min_value
        for (ub, _), c in zip(bands, band_cols):
            steps.append(dict(range=[prev, ub], color=rgba(c, 0.35)))
            prev = ub
        fig = go().Figure(go().Indicator(
            mode="gauge+number", value=value, title=dict(text=label, font=dict(color=r.theme.ink_secondary)),
            number=dict(font=dict(color=r.theme.ink, size=44)),
            gauge=dict(axis=dict(range=[min_value, max_value], tickcolor=r.theme.muted,
                                 tickfont=dict(color=r.theme.ink_secondary)),
                       bar=dict(color=r.theme.ink, thickness=0.18), steps=steps, borderwidth=0,
                       bgcolor=r.theme.plotly_bg,
                       threshold=dict(line=dict(color=r.theme.ink_secondary, width=3), value=target)
                       if target is not None else None)))
        fig.update_layout(height=r.height or 340)
        return fig
    from matplotlib.patches import Wedge, Polygon as MPoly
    fig, ax = r.subplots(figsize=r.figsize or (5.5, 3.4))

    def ang(v):
        return 180 - 180 * (v - min_value) / (max_value - min_value)

    prev = min_value
    for (ub, _), c in zip(bands, band_cols):
        ax.add_patch(Wedge((0, 0), 1, ang(ub), ang(prev), width=0.22, facecolor=blend(c, r.theme.surface, 0.55),
                           edgecolor=r.theme.surface, lw=2))
        prev = ub
    filled_c = band_cols[next((i for i, (ub, _) in enumerate(bands) if value <= ub), len(bands) - 1)]
    ax.add_patch(Wedge((0, 0), 0.76, ang(value), 180, width=0.08, facecolor=filled_c, edgecolor="none"))
    a = math.radians(ang(value))
    tip = (0.9 * math.cos(a), 0.9 * math.sin(a))
    perp = (0.03 * math.cos(a + math.pi / 2), 0.03 * math.sin(a + math.pi / 2))
    ax.add_patch(MPoly([tip, (perp[0], perp[1]), (-perp[0], -perp[1])], closed=True, facecolor=r.theme.ink))
    ax.add_patch(__import__("matplotlib").patches.Circle((0, 0), 0.05, facecolor=r.theme.ink))
    if target is not None:
        t = math.radians(ang(target))
        ax.plot([0.74 * math.cos(t), 1.02 * math.cos(t)], [0.74 * math.sin(t), 1.02 * math.sin(t)],
                color=r.theme.ink_secondary, lw=2.5)
    ax.text(0, -0.2, fmt(value), ha="center", va="center", fontsize=26, fontweight="bold", color=r.theme.ink)
    ax.text(0, -0.42, label, ha="center", va="center", fontsize=10, color=r.theme.ink_secondary)
    ax.text(-1, -0.08, fmt(min_value), ha="center", fontsize=8, color=r.theme.muted)
    ax.text(1, -0.08, fmt(max_value), ha="center", fontsize=8, color=r.theme.muted)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.55, 1.08)
    mpl_blank(ax)
    return fig


# ===================================================================== sunburst
def _partition(root):
    """Angular partition layout (d3.partition in polar form): x0/x1 in [0, 2π), y = depth."""
    root.x0, root.x1 = 0.0, 2 * math.pi

    def rec(n):
        a = n.x0
        for c in n.children:
            span = (n.x1 - n.x0) * (c.value / n.value if n.value else 0)
            c.x0, c.x1 = a, a + span
            a += span
            rec(c)
    rec(root)


@chart("sunburst", "part_to_whole",
       alt_names=("multi-level pie chart", "multilayer pie chart", "sunburst graph", "radial treemap", "sunburst chart"),
       use="Multi-level hierarchy as concentric rings (inner = top level).",
       avoid="Precise comparison; outer rings exaggerate size.")
def sunburst(r, data, path, value, *, max_depth=None):
    """Concentric rings: angle = share of the parent, ring = hierarchy level."""
    root, path = _tree(data, path, value)
    colors, _ = _node_colors(r, root)
    if r.plotly:
        fig = _plotly_hier(r, root, "sunburst", colors, maxdepth=max_depth or -1, insidetextorientation="radial")
        fig.update_layout(margin=dict(l=10, r=10, b=10), height=r.height or 620)
        return fig
    _partition(root)
    depth = max_depth or max(n.depth for n in root.walk())
    top_cmap = r.color_map([c.label for c in root.children])
    fig, ax = r.subplots(polar=True, figsize=r.figsize or (6.8, 6.8))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    for n in root.walk():
        if n is root or n.depth > depth:
            continue
        # level 1 in the full group colour; deeper rings lighter, the outermost ring full again
        top_col = top_cmap[n.top().label]
        col = top_col if n.depth == 1 or n.depth == depth else blend(top_col, r.theme.surface, 0.6)
        ax.bar((n.x0 + n.x1) / 2, 1, width=n.x1 - n.x0, bottom=n.depth, color=col, edgecolor=r.theme.gap, lw=1.5)
        span = n.x1 - n.x0
        if span * (n.depth + 0.5) > 0.35:
            rot = np.degrees(-(n.x0 + n.x1) / 2) + 90
            rot = rot + 180 if (rot % 360) > 90 and (rot % 360) < 270 else rot
            ax.text((n.x0 + n.x1) / 2, n.depth + 0.5, n.label, ha="center", va="center", fontsize=8,
                    rotation=rot % 360, rotation_mode="anchor", color=text_on(col, r.theme))
    ax.set_ylim(0, depth + 1)
    ax.set_axis_off()
    return fig


# ======================================================================= funnel
@chart("funnel", "part_to_whole", alt_names=("funnel chart", "conversion funnel", "sales funnel"),
       use="Values shrinking through ordered stages (conversion pipelines).",
       avoid="Stages that are not sequential subsets of each other.")
def funnel(r, data, stage, value):
    """Centred bars per ordered stage, labelled with value and % of the first stage."""
    df = as_frame(data)
    d = df.groupby(stage, observed=True, sort=False)[value].sum()
    stages, vals = [str(s) for s in d.index], d.to_numpy(float)
    cols = list(reversed(r.theme.seq_steps(len(stages))))   # first (largest) stage = strongest step
    if r.plotly:
        fig = go().Figure(go().Funnel(
            y=stages, x=vals, marker=dict(color=cols), textinfo="value+percent initial",
            textfont=dict(color=[text_on(c, r.theme) for c in cols]),
            connector=dict(fillcolor=blend(r.theme.ink, r.theme.surface, 0.06)),
            hovertemplate="<b>%{y}</b><br>%{x:,.4~g}<br>%{percentInitial:.1%} of first stage<extra></extra>"))
        return fig
    fig, ax = r.subplots(figsize=r.figsize or (7.5, 0.6 * len(stages) + 1.2))
    pos = np.arange(len(stages))
    ax.barh(pos, vals, left=-vals / 2, color=cols, height=0.78)
    for p, v, c in zip(pos, vals, cols):
        ax.text(0, p, f"{compact(v)}  ({v / vals[0]:.0%})", ha="center", va="center", fontsize=9,
                color=text_on(c, r.theme))
    for i in range(len(pos) - 1):
        ax.fill_betweenx([pos[i] + 0.39, pos[i + 1] - 0.39], [-vals[i] / 2, -vals[i + 1] / 2],
                         [vals[i] / 2, vals[i + 1] / 2], color=blend(r.theme.ink, r.theme.surface, 0.06), lw=0)
    ax.set_yticks(pos, stages)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    return fig
