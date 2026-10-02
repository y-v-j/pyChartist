"""2. Correlation (relational) charts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from .._util import as_frame, blend, compact, need, ordered, rgba, text_on
from ..registry import chart
from ._draw import go, mpl_compact_axis, mpl_legend


# ===================================================================== heatmap
@chart("heatmap", "correlation", seaborn=True,
       alt_names=("heat map", "heat table", "density table", "correlation matrix", "correlogram"),
       use="Patterns across a matrix of two categorical/binned variables, or a correlation matrix.",
       avoid="Precise value reading (add annotations) or colour-only audiences.")
def heatmap(r, data, x=None, y=None, value=None, *, agg="mean", corr=False, method="pearson",
            annotate=None, diverging=None, fmt=None, zmin=None, zmax=None):
    """Colour-coded matrix.

    * long data: ``x``, ``y`` categories and ``value`` (aggregated by ``agg``);
    * wide data: the DataFrame itself is the matrix (index = rows);
    * ``corr=True``: correlation matrix of the numeric columns (``method`` pearson/spearman/kendall).
    """
    df = as_frame(data)
    if corr:
        num = df.select_dtypes("number")
        mat = num.corr(method=method)
        diverging = True if diverging is None else diverging
        zmin, zmax = -1, 1
        r.meta["matrix"] = mat
        fmt = fmt or "{:.2f}"
    elif x is not None and y is not None:
        need(df, x, y, value)
        mat = df.pivot_table(index=y, columns=x, values=value, aggfunc=agg if value else "size", observed=True)
        mat = mat.reindex(index=ordered(df[y]), columns=ordered(df[x]))
    else:
        mat = df.select_dtypes("number")
    z = mat.to_numpy(float)
    if diverging:
        m = np.nanmax(np.abs(z)) if zmax is None else max(abs(zmin or 0), abs(zmax))
        zmin, zmax = (-m, m) if zmin is None else (zmin, zmax)
    zmin = np.nanmin(z) if zmin is None else zmin
    zmax = np.nanmax(z) if zmax is None else zmax
    if annotate is None:
        annotate = z.shape[0] <= 20 and z.shape[1] <= 20
    fmt = fmt or (lambda v: compact(v, 1))
    f = fmt.format if isinstance(fmt, str) else fmt
    rows = [str(i) for i in mat.index]
    cols = [str(c) for c in mat.columns]
    cmap = r.theme.div_cmap() if diverging else r.theme.seq_cmap()
    if r.sns:
        import seaborn as sns
        fig, ax = r.subplots(figsize=r.figsize or (max(5, 0.6 * len(cols) + 2.5), max(4, 0.45 * len(rows) + 1.5)))
        annot = np.vectorize(lambda v: f(v) if np.isfinite(v) else "")(z) if annotate else False
        sns.heatmap(pd.DataFrame(z, index=rows, columns=cols), cmap=cmap, vmin=zmin, vmax=zmax, annot=annot,
                    fmt="", linewidths=1.5, linecolor=r.theme.surface, ax=ax, annot_kws={"fontsize": 8},
                    cbar_kws={"shrink": 0.8, "pad": 0.02})
        ax.tick_params(length=0)
        ax.collections[0].colorbar.outline.set_visible(False)
        ax.set_xlabel(x or "")
        ax.set_ylabel(y or "")
        return fig
    if r.mpl:
        import matplotlib.colors as mcolors
        fig, ax = r.subplots(figsize=r.figsize or (max(5, 0.6 * len(cols) + 2.5), max(4, 0.45 * len(rows) + 1.5)))
        im = ax.imshow(z, cmap=cmap, vmin=zmin, vmax=zmax, aspect="auto")
        ax.set_xticks(range(len(cols)), cols, rotation=45 if max(map(len, cols)) * len(cols) > 50 else 0,
                      ha="right" if max(map(len, cols)) * len(cols) > 50 else "center")
        ax.set_yticks(range(len(rows)), rows)
        ax.grid(False)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        # 1px surface gap between cells
        ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
        ax.grid(which="minor", color=r.theme.surface, linewidth=1.5)
        ax.tick_params(which="minor", length=0)
        if annotate:
            norm = mcolors.Normalize(zmin, zmax)
            for i in range(z.shape[0]):
                for j in range(z.shape[1]):
                    if np.isfinite(z[i, j]):
                        ax.text(j, i, f(z[i, j]), ha="center", va="center", fontsize=8,
                                color=text_on(mcolors.to_hex(cmap(norm(z[i, j]))), r.theme))
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=r.theme.muted, labelcolor=r.theme.ink_secondary)
        if x:
            ax.set_xlabel(x)
        if y:
            ax.set_ylabel(y)
        return fig
    text = [[f(v) if np.isfinite(v) else "" for v in row] for row in z] if annotate else None
    fig = go().Figure(go().Heatmap(
        z=z, x=cols, y=rows, colorscale=r.theme.div_colorscale() if diverging else r.theme.seq_colorscale(),
        zmin=zmin, zmax=zmax, text=text, texttemplate="%{text}" if annotate else None, xgap=2, ygap=2,
        hovertemplate=f"{y or ''} %{{y}}<br>{x or ''} %{{x}}<br>value: %{{z:,.4~g}}<extra></extra>",
        colorbar=dict(outlinewidth=0)))
    fig.update_yaxes(autorange="reversed", showgrid=False, title_text=y)
    fig.update_xaxes(showgrid=False, title_text=x)
    return fig


# ===================================================================== scatter
def _fit_line(x, y, kind):
    xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    if kind == "lowess":
        from statsmodels.nonparametric.smoothers_lowess import lowess
        fit = lowess(y, x, frac=0.6, return_sorted=True)
        return fit[:, 0], fit[:, 1], None, None
    res = sps.linregress(x, y)
    yhat = res.intercept + res.slope * xs
    n = len(x)
    resid = y - (res.intercept + res.slope * x)
    s = np.sqrt(np.sum(resid ** 2) / (n - 2))
    t = sps.t.ppf(0.975, n - 2)
    se = s * np.sqrt(1 / n + (xs - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    return xs, yhat, yhat - t * se, yhat + t * se


def _scatter_impl(r, df, x, y, color=None, size=None, trendline=None, label=None, alpha=0.85,
                  show_stats=False, size_max=40, log_x=False, log_y=False):
    need(df, x, y, color, size, label)
    d = df.dropna(subset=[x, y])
    groups = ordered(d[color]) if color else [None]
    cmap = r.color_map(groups) if color else {None: r.colors(1)[0]}
    if size:
        smax = d[size].abs().max() or 1
    if show_stats or trendline:
        from ..stats import correlation
        r.meta["stats"] = correlation(d[x], d[y])
    if r.sns:
        import seaborn as sns
        fig, ax = r.subplots()
        kw = dict(data=d, x=x, y=y, ax=ax, alpha=alpha, edgecolor=r.theme.surface, linewidth=1)
        if color:
            kw.update(hue=color, hue_order=groups, palette=cmap)
        else:
            kw.update(color=cmap[None])
        if size:
            kw.update(size=size, sizes=(12, size_max ** 2), size_norm=(0, smax))
        sns.scatterplot(**kw)
        if trendline:
            for g in groups:
                sub = d if g is None else d[d[color] == g]
                if len(sub) > 2:
                    sns.regplot(data=sub, x=x, y=y, scatter=False, ax=ax, ci=95, lowess=trendline == "lowess",
                                color=cmap[g] if color else r.theme.ink_secondary, line_kws={"lw": 2})
        if ax.get_legend():
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    if r.mpl:
        if not r.sns:
            fig, ax = r.subplots()
        for g in ([] if r.sns else groups):
            sub = d if g is None else d[d[color] == g]
            s = (np.sqrt(sub[size].abs() / smax) * size_max) ** 2 if size else 40
            ax.scatter(sub[x], sub[y], s=s, color=cmap[g], alpha=alpha, label=None if g is None else str(g),
                       edgecolor=r.theme.surface, linewidth=1)
            if trendline and len(sub) > 2:
                xs, yh, lo, hi = _fit_line(sub[x].to_numpy(float), sub[y].to_numpy(float), trendline)
                ax.plot(xs, yh, color=cmap[g] if color else r.theme.ink_secondary, lw=2)
                if lo is not None:
                    ax.fill_between(xs, lo, hi, color=cmap[g] if color else r.theme.ink_secondary, alpha=0.1, lw=0)
        if label:
            for _, row in d.iterrows():
                ax.annotate(str(row[label]), (row[x], row[y]), xytext=(4, 3), textcoords="offset points",
                            fontsize=7.5, color=r.theme.ink_secondary)
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        if not log_x:
            mpl_compact_axis(ax.xaxis)
        if not log_y:
            mpl_compact_axis(ax.yaxis)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        if show_stats:
            st = r.meta["stats"]
            ax.text(0.01, 0.99, f"Pearson r = {st['pearson_r']:.2f} (p = {st['pearson_p']:.2g})\n"
                                f"Spearman ρ = {st['spearman_rho']:.2f} (p = {st['spearman_p']:.2g})",
                    transform=ax.transAxes, va="top", fontsize=8.5, color=r.theme.ink_secondary)
        if not r.sns:
            leg = mpl_legend(ax, r, title=color)
            if leg:
                for h in leg.legend_handles:
                    h.set_sizes([40])
        return fig
    fig = go().Figure()
    for g in groups:
        sub = d if g is None else d[d[color] == g]
        msize = (np.sqrt(sub[size].abs() / smax) * size_max) if size else 9
        hover = ("<b>%{text}</b><br>" if label else "") + f"{x}: %{{x:,.4~g}}<br>{y}: %{{y:,.4~g}}"
        if size:
            hover += f"<br>{size}: %{{customdata:,.4~g}}"
        fig.add_trace(go().Scatter(
            x=sub[x], y=sub[y], mode="markers", name=str(g), showlegend=g is not None,
            text=sub[label] if label else None, customdata=sub[size] if size else None,
            marker=dict(size=msize, color=cmap[g], opacity=alpha, line=dict(color=r.theme.surface, width=1)),
            hovertemplate=hover + f"<extra>{g if g is not None else ''}</extra>"))
        if trendline and len(sub) > 2:
            xs, yh, lo, hi = _fit_line(sub[x].to_numpy(float), sub[y].to_numpy(float), trendline)
            lc = cmap[g] if color else r.theme.ink_secondary
            if lo is not None:
                fig.add_trace(go().Scatter(x=np.r_[xs, xs[::-1]], y=np.r_[hi, lo[::-1]], fill="toself",
                                           fillcolor=rgba(lc, 0.1), line=dict(width=0), hoverinfo="skip",
                                           showlegend=False, legendgroup=str(g)))
            fig.add_trace(go().Scatter(x=xs, y=yh, mode="lines", line=dict(color=lc, width=2), hoverinfo="skip",
                                       showlegend=False, legendgroup=str(g)))
    if show_stats:
        st = r.meta["stats"]
        fig.add_annotation(x=0, y=1, xref="paper", yref="paper", xanchor="left", yanchor="top", showarrow=False,
                           align="left", font=dict(size=11, color=r.theme.ink_secondary),
                           text=f"Pearson r = {st['pearson_r']:.2f} (p = {st['pearson_p']:.2g})<br>"
                                f"Spearman ρ = {st['spearman_rho']:.2f} (p = {st['spearman_p']:.2g})")
    fig.update_xaxes(title_text=x, type="log" if log_x else None)
    fig.update_yaxes(title_text=y, type="log" if log_y else None)
    fig.update_layout(legend_title_text=color or "")
    return fig


@chart("scatter", "correlation", alt_names=("scatter plot", "scatter graph", "scatter chart", "xy plot"),
       seaborn=True,
       use="Direction, strength and shape of the relationship between two numeric variables.",
       avoid="Very many overlapping points (use hexbin / contour) or no real pairing.")
def scatter(r, data, x, y, color=None, *, size=None, trendline=None, label=None, alpha=0.85,
            show_stats=False, log_x=False, log_y=False):
    """Points at (``x``, ``y``). ``trendline`` = 'ols' (with 95% CI band) or 'lowess';
    ``show_stats`` annotates Pearson r and Spearman ρ (also in ``chart.meta['stats']``)."""
    return _scatter_impl(r, as_frame(data), x, y, color, size, trendline, label, alpha, show_stats,
                         log_x=log_x, log_y=log_y)


@chart("bubble", "correlation", alt_names=("bubble chart", "bubble plot", "bubble graph"), seaborn=True,
       use="Three numeric variables: x, y and bubble area.",
       avoid="Many overlapping bubbles, or when precise size comparison matters.")
def bubble(r, data, x, y, size, color=None, *, label=None, alpha=0.7, size_max=48, log_x=False,
           log_y=False):
    """Scatter with marker *area* proportional to ``size``."""
    d = as_frame(data).sort_values(size, ascending=False)
    return _scatter_impl(r, d, x, y, color, size, None, label, alpha, False, size_max, log_x, log_y)


# ======================================================== connected scatter
@chart("connected_scatter", "correlation", alt_names=("connected scatter plot", "connected scatterplot"),
       use="Two numeric variables that evolve in a clear sequence (e.g. year by year).",
       avoid="No clear order, or so many points that the path becomes spaghetti.")
def connected_scatter(r, data, x, y, order, group=None, *, label_every=None, arrows=True):
    """Points joined in ``order`` (e.g. year); ``group`` draws one path per level."""
    df = as_frame(data)
    need(df, x, y, order, group)
    df = df.dropna(subset=[x, y]).sort_values(order)
    groups = ordered(df[group]) if group else [None]
    cmap = r.color_map(groups) if group else {None: r.colors(1)[0]}
    if r.mpl:
        fig, ax = r.subplots()
        for g in groups:
            sub = df if g is None else df[df[group] == g]
            xv, yv, ov = sub[x].to_numpy(float), sub[y].to_numpy(float), sub[order].to_numpy()
            ax.plot(xv, yv, color=cmap[g], lw=1.6, label=None if g is None else str(g))
            ax.scatter(xv, yv, s=24, color=cmap[g], edgecolor=r.theme.surface, linewidth=1, zorder=3)
            if arrows and len(xv) > 1:
                ax.annotate("", xy=(xv[-1], yv[-1]), xytext=(xv[-2], yv[-2]),
                            arrowprops=dict(arrowstyle="-|>", color=cmap[g], lw=1.6))
            step = label_every or max(1, len(xv) // 4)
            for i in list(range(0, len(xv), step)) + [len(xv) - 1]:
                ax.annotate(str(ov[i]), (xv[i], yv[i]), xytext=(4, 4), textcoords="offset points",
                            fontsize=7.5, color=r.theme.ink_secondary)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        mpl_compact_axis(ax.xaxis)
        mpl_legend(ax, r, title=group)
        return fig
    fig = go().Figure()
    for g in groups:
        sub = df if g is None else df[df[group] == g]
        step = label_every or max(1, len(sub) // 4)
        txt = [str(o) if (i % step == 0 or i == len(sub) - 1) else "" for i, o in enumerate(sub[order])]
        fig.add_trace(go().Scatter(
            x=sub[x], y=sub[y], mode="lines+markers+text", name=str(g), showlegend=g is not None,
            text=txt, textposition="top right", textfont=dict(size=10, color=r.theme.ink_secondary),
            line=dict(color=cmap[g], width=2), customdata=sub[order],
            marker=dict(size=8, color=cmap[g], line=dict(color=r.theme.surface, width=1),
                        symbol=["circle"] * (len(sub) - 1) + ["arrow"], angleref="previous"),
            hovertemplate=f"{order}: %{{customdata}}<br>{x}: %{{x:,.4~g}}<br>{y}: %{{y:,.4~g}}<extra></extra>"))
    fig.update_xaxes(title_text=x)
    fig.update_yaxes(title_text=y)
    return fig


# ===================================================================== contour
def _kde_grid(x, y, n=100, pad=0.1):
    x, y = np.asarray(x, float), np.asarray(y, float)
    dx, dy = x.max() - x.min(), y.max() - y.min()
    gx = np.linspace(x.min() - pad * dx, x.max() + pad * dx, n)
    gy = np.linspace(y.min() - pad * dy, y.max() + pad * dy, n)
    xx, yy = np.meshgrid(gx, gy)
    kde = sps.gaussian_kde(np.vstack([x, y]))
    return gx, gy, kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)


@chart("contour", "correlation", seaborn=True, alt_names=("contour plot", "density contour", "topographic plot", "isoline plot"),
       use="A continuous surface z(x, y), or the 2-D density of many points.",
       avoid="Few observations or audiences unused to reading isolines.")
def contour(r, data, x, y, z=None, *, levels=10, filled=True, points=False, max_points=3000):
    """Contours of ``z`` over a regular (``x``, ``y``) grid, or of the 2-D KDE density of points
    when ``z`` is None."""
    df = as_frame(data).dropna(subset=[c for c in (x, y, z) if c])
    if z is None:
        gx, gy, zz = _kde_grid(df[x], df[y])
        zlabel = "density"
        if isinstance(levels, int):   # start above ~0 so the empty background stays the surface colour
            levels = np.linspace(zz.max() * 0.04, zz.max(), levels + 1)
    else:
        piv = df.pivot_table(index=y, columns=x, values=z, aggfunc="mean")
        gx, gy, zz = piv.columns.to_numpy(float), piv.index.to_numpy(float), piv.to_numpy(float)
        zlabel = z
    if r.sns and z is None:
        import seaborn as sns
        fig, ax = r.subplots()
        sns.kdeplot(data=df, x=x, y=y, fill=filled, levels=levels if np.isscalar(levels) else len(levels),
                    thresh=0.04, cmap=r.theme.seq_cmap(), ax=ax, cbar=True, cbar_kws={"label": "density"})
        if points:
            sub = df.sample(min(len(df), max_points), random_state=0)
            ax.scatter(sub[x], sub[y], s=4, color=r.theme.ink, alpha=0.3, lw=0)
        ax.grid(False)
        return fig
    if r.mpl:
        fig, ax = r.subplots()
        cmap = r.theme.seq_cmap()
        if filled:
            cs = ax.contourf(gx, gy, zz, levels=levels, cmap=cmap, extend="neither")
            ax.contour(gx, gy, zz, levels=levels, colors=r.theme.surface, linewidths=0.6)
        else:
            cs = ax.contour(gx, gy, zz, levels=levels, cmap=cmap, linewidths=1.4)
        if points and z is None:
            sub = df.sample(min(len(df), max_points), random_state=0)
            ax.scatter(sub[x], sub[y], s=4, color=r.theme.ink, alpha=0.3, lw=0)
        cb = fig.colorbar(cs, ax=ax, shrink=0.85, pad=0.02, label=zlabel)
        cb.outline.set_visible(False)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(False)
        return fig
    fig = go().Figure(go().Contour(
        x=gx, y=gy, z=zz, ncontours=levels if np.isscalar(levels) else None,
        contours_start=None if np.isscalar(levels) else levels[0],
        contours_end=None if np.isscalar(levels) else levels[-1],
        contours_size=None if np.isscalar(levels) else levels[1] - levels[0], colorscale=r.theme.seq_colorscale(),
        contours_coloring="fill" if filled else "lines", line=dict(color=r.theme.surface, width=0.6),
        colorbar=dict(title=dict(text=zlabel), outlinewidth=0),
        hovertemplate=f"{x}: %{{x:,.4~g}}<br>{y}: %{{y:,.4~g}}<br>{zlabel}: %{{z:.3g}}<extra></extra>"))
    if points and z is None:
        sub = df.sample(min(len(df), max_points), random_state=0)
        fig.add_trace(go().Scattergl(x=sub[x], y=sub[y], mode="markers", showlegend=False, hoverinfo="skip",
                                     marker=dict(size=3, color=blend(r.theme.ink, r.theme.surface, 0.3))))
    fig.update_xaxes(title_text=x, showgrid=False)
    fig.update_yaxes(title_text=y, showgrid=False)
    return fig


# ====================================================================== hexbin
@chart("hexbin", "correlation", alt_names=("hexagonal binning", "hexagonal plot", "hexagonal bin plot", "hex bin"),
       use="Replace an over-plotted scatter of thousands of points with binned density.",
       avoid="Small samples or analyses depending on exact values/outliers.")
def hexbin(r, data, x, y, *, gridsize=30, C=None, reduce="mean", mincnt=1, log=False):
    """Hexagonal bins coloured by point count (or by ``reduce`` of column ``C``)."""
    df = as_frame(data).dropna(subset=[c for c in (x, y, C) if c])
    red = {"mean": np.mean, "sum": np.sum, "median": np.median, "max": np.max}[reduce]
    kw = dict(gridsize=gridsize, mincnt=mincnt, C=df[C] if C else None, reduce_C_function=red,
              bins="log" if log else None, cmap=r.theme.seq_cmap(), edgecolors=r.theme.surface, linewidths=0.4)
    if r.mpl:
        fig, ax = r.subplots()
        hb = ax.hexbin(df[x], df[y], **kw)
        cb = fig.colorbar(hb, ax=ax, shrink=0.85, pad=0.02, label=f"{reduce} of {C}" if C else "count")
        cb.outline.set_visible(False)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(False)
        return fig
    from matplotlib.figure import Figure
    tmp_ax = Figure().subplots()
    hb = tmp_ax.hexbin(df[x], df[y], **kw)
    centres, vals = hb.get_offsets(), np.asarray(hb.get_array(), float)
    hexv = hb.get_paths()[0].vertices[:-1]
    nb = 9
    edges = np.linspace(vals.min(), vals.max(), nb + 1)
    idx = np.clip(np.digitize(vals, edges[1:-1]), 0, nb - 1)
    cmap = r.theme.seq_cmap()
    import matplotlib.colors as mcolors
    fig = go().Figure()
    for b in range(nb):
        sel = centres[idx == b]
        if not len(sel):
            continue
        xs, ys = [], []
        for cx, cy in sel:
            xs += list(cx + hexv[:, 0]) + [cx + hexv[0, 0], None]
            ys += list(cy + hexv[:, 1]) + [cy + hexv[0, 1], None]
        col = mcolors.to_hex(cmap((b + 0.5) / nb))
        fig.add_trace(go().Scatter(x=xs, y=ys, mode="lines", fill="toself", fillcolor=col,
                                   line=dict(color=r.theme.surface, width=0.5), hoverinfo="skip", showlegend=False))
    fig.add_trace(go().Scatter(
        x=centres[:, 0], y=centres[:, 1], mode="markers", showlegend=False,
        marker=dict(size=1, color=vals, colorscale=r.theme.seq_colorscale(), opacity=0,
                    colorbar=dict(title=dict(text=f"{reduce} of {C}" if C else "count"), outlinewidth=0),
                    showscale=True, cmin=vals.min(), cmax=vals.max()),
        customdata=vals, hovertemplate=f"{x}: %{{x:,.4~g}}<br>{y}: %{{y:,.4~g}}<br>value: %{{customdata:,.4~g}}<extra></extra>"))
    fig.update_xaxes(title_text=x, showgrid=False)
    fig.update_yaxes(title_text=y, showgrid=False)
    return fig
