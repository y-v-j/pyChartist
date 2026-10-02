"""4. Data over time (temporal) charts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .._util import as_frame, blend, compact, need, ordered, rgba
from ..registry import chart
from ._draw import go, mpl_compact_axis, mpl_legend


def _series(df, x, y, group, agg="mean"):
    """{group: (x array, y array)} aggregated per x and sorted by x."""
    need(df, x, y, group)
    groups = ordered(df[group]) if group else [None]
    out = {}
    for g in groups:
        sub = df if g is None else df[df[group] == g]
        s = sub.groupby(x, observed=True)[y].agg(agg).sort_index()
        out[g] = (s.index.to_numpy(), s.to_numpy(float))
    return out


def _smooth(xv, yv, n=8):
    """Monotone cubic (PCHIP, like d3.curveMonotoneX): smooth, no overshoot."""
    from scipy.interpolate import PchipInterpolator
    if len(xv) < 3:
        return xv, yv
    is_dt = np.issubdtype(np.asarray(xv).dtype, np.datetime64)
    xn = np.asarray(xv).astype("datetime64[ns]").astype("int64").astype(float) if is_dt else np.asarray(xv, float)
    fine = np.linspace(xn.min(), xn.max(), (len(xn) - 1) * n + 1)
    yf = PchipInterpolator(xn, yv)(fine)
    if is_dt:
        fine = fine.astype("int64").astype("datetime64[ns]")
    return fine, yf


def _line_core(r, data, x, y, group, shape, markers, end_labels, agg, fill, ci):
    df = as_frame(data)
    series = _series(df, x, y, group, agg)
    keys = list(series)
    cmap = r.color_map(keys) if group else {None: r.colors(1)[0]}
    label_ends = end_labels if end_labels is not None else (group is not None and len(keys) <= 4)
    if r.sns and shape in ("linear", "step"):
        import seaborn as sns
        fig, ax = r.subplots()
        kw = dict(data=df, x=x, y=y, ax=ax, estimator=agg, errorbar=("ci", 95) if ci else None, lw=2,
                  marker="o" if markers else None, markeredgecolor=r.theme.surface)
        if shape == "step":
            kw["drawstyle"] = "steps-post"
        if group:
            kw.update(hue=group, hue_order=keys, palette=cmap)
        else:
            kw["color"] = cmap[None]
        sns.lineplot(**kw)
        if ci:
            r.meta["band"] = f"95% bootstrap CI of the {agg} per {x}"
        _mpl_line_finish(r, ax, series, cmap, label_ends, group, x, y, legend=not label_ends)
        if label_ends and ax.get_legend():
            ax.get_legend().remove()
        elif ax.get_legend():
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1.01, 1), frameon=False)
        return fig
    if r.mpl:
        fig, ax = r.subplots()
        for g in keys:
            xv, yv = series[g]
            if shape == "spline":
                xv, yv = _smooth(xv, yv)
            draw = "steps-post" if shape == "step" else "default"
            ax.plot(xv, yv, color=cmap[g], lw=2, drawstyle=draw, label=None if g is None else str(g))
            if fill:
                ax.fill_between(xv, yv, 0, color=cmap[g], alpha=0.1, lw=0, step="post" if shape == "step" else None)
            if markers:
                ox, oy = series[g]
                ax.plot(ox, oy, "o", color=cmap[g], ms=6, mec=r.theme.surface, mew=1.5)
        _mpl_line_finish(r, ax, series, cmap, label_ends, group, x, y, legend=not label_ends)
        return fig
    fig = go().Figure()
    pshape = {"linear": "linear", "spline": "spline", "step": "hv"}[shape]
    for g in keys:
        xv, yv = series[g]
        fig.add_trace(go().Scatter(
            x=xv, y=yv, mode="lines+markers" if markers else "lines", name=str(g), showlegend=g is not None
            and not label_ends, line=dict(color=cmap[g], width=2, shape=pshape, smoothing=0.8 if shape == "spline" else None),
            fill="tozeroy" if fill else None, fillcolor=rgba(cmap[g], 0.1),
            marker=dict(size=7, line=dict(color=r.theme.surface, width=1.5)),
            hovertemplate=f"%{{x}}<br>{y}: %{{y:,.4~g}}<extra>{g if g is not None else ''}</extra>"))
        if label_ends:
            fig.add_annotation(x=xv[-1], y=yv[-1], text=f"{g}", showarrow=False, xanchor="left", xshift=6,
                               font=dict(color=r.theme.ink_secondary, size=11))
    fig.update_layout(hovermode="x unified", legend_title_text=group or "",
                      margin=dict(r=110 if label_ends else 30))
    fig.update_xaxes(title_text=x, showgrid=False, showspikes=True, spikecolor=r.theme.baseline, spikethickness=1)
    fig.update_yaxes(title_text=y)
    return fig


def _mpl_line_finish(r, ax, series, cmap, label_ends, group, x, y, legend=True):
    if label_ends:
        for g, (xv, yv) in series.items():
            ax.annotate(f"{g}", (xv[-1], yv[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=r.theme.ink_secondary)
            ax.plot(xv[-1], yv[-1], "o", color=cmap[g], ms=6, mec=r.theme.surface, mew=1.5)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(False)
    ax.grid(True, axis="y")
    mpl_compact_axis(ax.yaxis)
    ax.margins(x=0.08 if label_ends else 0.02)
    if legend and not r.sns:
        mpl_legend(ax, r, title=group)


@chart("line", "temporal", alt_names=("line graph", "line plot", "time series", "line chart"), seaborn=True,
       use="Trends over a continuous axis (usually time); several series on one scale.",
       avoid="Unordered categories, or more than ~5-8 series (use small multiples).")
def line(r, data, x, y, group=None, *, markers=False, end_labels=None, agg="mean", ci=True):
    """Lines of ``y`` over ``x`` (aggregated by ``agg`` when x repeats), one per ``group``.

    Up to 4 series are labelled at their ends instead of a legend. With ``backend="seaborn"``
    repeated observations get a 95% CI band (``ci``)."""
    return _line_core(r, data, x, y, group, "linear", markers, end_labels, agg, False, ci)


@chart("spline", "temporal", alt_names=("spline chart", "smooth line chart", "curved line chart"),
       use="A smoothed line where the general shape matters more than exact intermediate values.",
       avoid="When interpolated values could be mistaken for data.")
def spline(r, data, x, y, group=None, *, markers=True, end_labels=None, agg="mean"):
    """Line smoothed by monotone cubic interpolation (no overshoot), observed points marked."""
    return _line_core(r, data, x, y, group, "spline", markers, end_labels, agg, False, False)


@chart("step_line", "temporal", alt_names=("step chart", "step line chart", "staircase chart", "step plot"),
       seaborn=True,
       use="Values that change at discrete moments and stay constant between (prices, rates, stock).",
       avoid="Continuously changing quantities.")
def step_line(r, data, x, y, group=None, *, markers=False, end_labels=None, agg="mean"):
    """Horizontal-then-vertical steps (value holds until the next observation)."""
    return _line_core(r, data, x, y, group, "step", markers, end_labels, agg, False, False)


@chart("area", "temporal", alt_names=("area chart", "area graph", "filled line chart"),
       use="Volume/magnitude of a trend over time for one or two series.",
       avoid="Many overlapping series (use stacked area or small multiples).")
def area(r, data, x, y, group=None, *, agg="sum", end_labels=None):
    """Line with a light fill down to zero (overlapping, not stacked)."""
    return _line_core(r, data, x, y, group, "linear", False, end_labels, agg, True, False)


def _wide(df, x, y, group, agg="sum"):
    need(df, x, y, group)
    w = df.pivot_table(index=x, columns=group, values=y, aggfunc=agg, observed=True).sort_index().fillna(0)
    return w[ordered(df[group])]


@chart("stacked_area", "temporal", alt_names=("stacked area chart", "100% stacked area chart"),
       use="How a total and its composition evolve over time.",
       avoid="Reading the inner layers precisely, or many layers.")
def stacked_area(r, data, x, y, group, *, normalize=False, agg="sum"):
    """Layers stacked on each other; ``normalize=True`` shows shares (100%)."""
    w = _wide(as_frame(data), x, y, group, agg)
    if normalize:
        w = w.div(w.sum(axis=1), axis=0) * 100
    cmap = r.color_map(list(w.columns))
    if r.mpl:
        fig, ax = r.subplots()
        ax.stackplot(w.index, w.T.to_numpy(), labels=[str(c) for c in w.columns], colors=[cmap[c] for c in w.columns],
                     edgecolor=r.theme.gap, linewidth=0.8)
        ax.set_xlabel(x)
        ax.set_ylabel(f"{y} (%)" if normalize else y)
        ax.grid(False)
        ax.grid(True, axis="y")
        mpl_compact_axis(ax.yaxis)
        ax.margins(x=0)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, title=group)
        return fig
    fig = go().Figure()
    for c in w.columns:
        fig.add_trace(go().Scatter(x=w.index, y=w[c], name=str(c), stackgroup="one", mode="lines",
                                   line=dict(color=cmap[c], width=0.8), fillcolor=cmap[c],
                                   hovertemplate=f"%{{x}}<br>{c}: %{{y:,.4~g}}" + ("%" if normalize else "") + "<extra></extra>"))
    fig.update_layout(hovermode="x unified", legend=dict(traceorder="reversed"), legend_title_text=group)
    fig.update_xaxes(title_text=x, showgrid=False)
    fig.update_yaxes(title_text=f"{y} (%)" if normalize else y, range=[0, 100] if normalize else None)
    return fig


def _stream_baseline(Y, offset):
    """Baseline under the first layer (matplotlib's stackplot formulas; Byron & Wattenberg 2008)."""
    m = Y.shape[0]
    if offset == "zero":
        return np.zeros(Y.shape[1])
    if offset == "silhouette":
        return -Y.sum(0) / 2
    return -(Y * (m - 0.5 - np.arange(m))[:, None]).sum(0) / m   # wiggle


@chart("streamgraph", "temporal", alt_names=("stream graph", "theme river", "streamgraph chart"),
       use="Organic overview of how many categories rise and fall over time.",
       avoid="Precise values - there is no common baseline.")
def streamgraph(r, data, x, y, group, *, offset="wiggle", smooth=True, agg="sum", order="inside_out"):
    """Stacked, smoothed layers around a moving centre baseline (``offset`` = wiggle | silhouette | zero).

    ``order="inside_out"`` puts the largest layers in the middle, as in Byron & Wattenberg."""
    w = _wide(as_frame(data), x, y, group, agg)
    cols = list(w.columns)
    if order == "inside_out":
        tot = w.sum().sort_values(ascending=False)
        top, bot, tsum, bsum = [], [], 0, 0
        for c in tot.index:
            if tsum <= bsum:
                top.append(c)
                tsum += tot[c]
            else:
                bot.append(c)
                bsum += tot[c]
        cols = bot[::-1] + top
    cmap = r.color_map(list(w.columns))
    xv = w.index.to_numpy()
    Y = w[cols].T.to_numpy(float)
    if smooth and len(xv) > 2:
        xs, _ = _smooth(xv, Y[0])
        Y = np.vstack([np.maximum(_smooth(xv, row)[1], 0) for row in Y])
        xv = xs
    base = _stream_baseline(Y, offset)
    stack = np.vstack([base, base + np.cumsum(Y, axis=0)])
    if r.mpl:
        fig, ax = r.subplots()
        for i, c in enumerate(cols):
            ax.fill_between(xv, stack[i], stack[i + 1], color=cmap[c], label=str(c), lw=0.6, edgecolor=r.theme.gap)
        ax.set_yticks([])
        ax.grid(False)
        ax.grid(True, axis="x")
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(x)
        ax.margins(x=0)
        handles, labels = ax.get_legend_handles_labels()
        lab_order = sorted(range(len(labels)), key=lambda i: list(w.columns).index(cols[i]))
        ax.legend([handles[i] for i in lab_order], [labels[i] for i in lab_order], loc="upper left",
                  bbox_to_anchor=(1.01, 1), frameon=False, title=group)
        return fig
    fig = go().Figure()
    fig.add_trace(go().Scatter(x=xv, y=stack[0], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    for i, c in enumerate(cols):
        fig.add_trace(go().Scatter(x=xv, y=stack[i + 1], mode="lines", fill="tonexty", fillcolor=cmap[c],
                                   line=dict(width=0.6, color=r.theme.gap), name=str(c), customdata=Y[i],
                                   hovertemplate=f"{c}: %{{customdata:,.4~g}}<extra></extra>"))
    fig.update_layout(hovermode="x unified", legend_title_text=group)
    fig.update_yaxes(visible=False)
    fig.update_xaxes(title_text=x, showgrid=True)
    return fig


def _ranks(w, ascending=False):
    return w.rank(axis=1, ascending=ascending, method="first")


@chart("bump", "temporal", alt_names=("bump chart", "rank chart", "ranking chart"),
       use="How rankings of several items change over time.",
       avoid="When absolute values matter (ranks hide magnitude).")
def bump(r, data, x, y, group, *, ascending=False, top_n=None, highlight=None):
    """Rank (1 = largest ``y`` unless ``ascending``) of each ``group`` at each ``x``, joined by lines."""
    w = _wide(as_frame(data), x, y, group)
    rk = _ranks(w, ascending)
    if top_n:
        keep = rk.iloc[-1].nsmallest(top_n).index
        rk = rk[keep]
    names = list(rk.columns)
    cmap = r.color_map(names)
    if highlight is not None:
        cmap = {n: (cmap[n] if n in highlight else blend(r.theme.ink, r.theme.surface, 0.25)) for n in names}
    xv = rk.index.to_numpy()
    if r.mpl:
        fig, ax = r.subplots(figsize=r.figsize or (8, max(4, 0.4 * len(names) + 1.5)))
        for n in names:
            xs, ys = _smooth(xv, rk[n].to_numpy(float), 10) if len(xv) > 2 else (xv, rk[n])
            ax.plot(xs, ys, color=cmap[n], lw=2.5, solid_capstyle="round")
            ax.plot(xv, rk[n], "o", color=cmap[n], ms=8, mec=r.theme.surface, mew=2)
            ax.annotate(str(n), (xv[-1], rk[n].iloc[-1]), xytext=(10, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=r.theme.ink_secondary)
            ax.annotate(str(n), (xv[0], rk[n].iloc[0]), xytext=(-10, 0), textcoords="offset points", va="center",
                        ha="right", fontsize=8.5, color=r.theme.ink_secondary)
        ax.set_ylim(len(names) + 0.5, 0.5)
        ax.set_yticks(range(1, len(names) + 1))
        ax.set_ylabel("rank")
        ax.set_xlabel(x)
        ax.grid(False)
        ax.margins(x=0.18)
        return fig
    fig = go().Figure()
    for n in names:
        fig.add_trace(go().Scatter(
            x=xv, y=rk[n], mode="lines+markers", name=str(n), line=dict(color=cmap[n], width=3, shape="spline", smoothing=0.6),
            marker=dict(size=11, line=dict(color=r.theme.surface, width=2)), customdata=w.loc[rk.index, n],
            hovertemplate=f"<b>{n}</b><br>%{{x}}: rank %{{y}} ({y} %{{customdata:,.4~g}})<extra></extra>", showlegend=False))
        fig.add_annotation(x=xv[-1], y=rk[n].iloc[-1], text=str(n), xanchor="left", xshift=10, showarrow=False,
                           font=dict(size=11, color=r.theme.ink_secondary))
        fig.add_annotation(x=xv[0], y=rk[n].iloc[0], text=str(n), xanchor="right", xshift=-10, showarrow=False,
                           font=dict(size=11, color=r.theme.ink_secondary))
    fig.update_yaxes(autorange="reversed", dtick=1, title_text="rank", showgrid=False)
    fig.update_xaxes(title_text=x, showgrid=False)
    fig.update_layout(margin=dict(l=120, r=120), height=r.height or max(420, 34 * len(names) + 120))
    return fig


@chart("bump_area", "temporal", alt_names=("bump area chart", "ribbon chart", "rank flow"),
       use="Both the size of each category and its changing rank over time.",
       avoid="Audiences who only need rank (bump chart) or many categories.")
def bump_area(r, data, x, y, group, *, gap=0.04, ascending=False, samples=24):
    """Ribbons stacked in rank order at every ``x`` (largest on top); thickness = value; ribbons
    cross smoothly when ranks change."""
    w = _wide(as_frame(data), x, y, group)
    names = list(w.columns)
    cmap = r.color_map(names)
    xv = w.index.to_numpy()
    tot_max = w.sum(axis=1).max()
    g = gap * tot_max
    bottoms = pd.DataFrame(index=w.index, columns=names, dtype=float)
    for xi in w.index:
        row = w.loc[xi].sort_values(ascending=ascending)    # largest first -> top
        acc = 0.0
        for n in reversed(list(row.index)):                 # build from the bottom up
            bottoms.loc[xi, n] = acc
            acc += row[n] + g
    is_dt = np.issubdtype(xv.dtype, np.datetime64)
    xn = xv.astype("datetime64[ns]").astype("int64").astype(float) if is_dt else xv.astype(float)
    polys = {}
    t = np.linspace(0, 1, samples)
    s = t * t * (3 - 2 * t)  # smoothstep
    for n in names:
        b, h = bottoms[n].to_numpy(float), w[n].to_numpy(float)
        top_x, top_y, bot_y = [], [], []
        for i in range(len(xn) - 1):
            seg_x = xn[i] + (xn[i + 1] - xn[i]) * t
            top_x.append(seg_x)
            top_y.append(b[i] + h[i] + (b[i + 1] + h[i + 1] - b[i] - h[i]) * s)
            bot_y.append(b[i] + (b[i + 1] - b[i]) * s)
        X = np.concatenate(top_x)
        T, B = np.concatenate(top_y), np.concatenate(bot_y)
        polys[n] = (X, T, B)
    if r.mpl:
        fig, ax = r.subplots()
        for n in names:
            X, T, B = polys[n]
            Xp = X.astype("int64").astype("datetime64[ns]") if is_dt else X
            ax.fill_between(Xp, B, T, color=cmap[n], lw=0.6, edgecolor=r.theme.gap, label=str(n))
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(False)
        ax.grid(True, axis="x")
        ax.set_xlabel(x)
        ax.margins(x=0)
        mpl_legend(ax, r, title=group)
        return fig
    fig = go().Figure()
    for n in names:
        X, T, B = polys[n]
        Xp = X.astype("int64").astype("datetime64[ns]") if is_dt else X
        fig.add_trace(go().Scatter(x=np.concatenate([Xp, Xp[::-1]]), y=np.concatenate([T, B[::-1]]), fill="toself",
                                   fillcolor=cmap[n], mode="lines", line=dict(width=0.6, color=r.theme.gap), name=str(n),
                                   hoveron="fills", text=f"{n}: total {compact(w[n].sum())}", hoverinfo="text"))
    fig.update_yaxes(visible=False)
    fig.update_xaxes(title_text=x)
    fig.update_layout(legend_title_text=group)
    return fig


# =================================================================== finance
def _ohlc_cols(df, date, o, h, l, c):
    need(df, date, o, h, l, c)
    d = df[[date, o, h, l, c]].dropna().sort_values(date)
    return d[date].to_numpy(), d[o].to_numpy(float), d[h].to_numpy(float), d[l].to_numpy(float), d[c].to_numpy(float)


def _finance(r, data, date, open, high, low, close, kind, up_color, down_color):
    t, o, h, l, c = _ohlc_cols(as_frame(data), date, open, high, low, close)
    up = up_color or r.theme.color(2)
    down = down_color or r.theme.color(7)
    if r.plotly:
        tr = go().Candlestick if kind == "candle" else go().Ohlc
        fig = go().Figure(tr(x=t, open=o, high=h, low=l, close=c, increasing_line_color=up,
                             decreasing_line_color=down, name=""))
        if kind == "candle":
            fig.data[0].update(increasing_fillcolor=up, decreasing_fillcolor=down)
        fig.update_layout(xaxis_rangeslider_visible=False, showlegend=False)
        fig.update_yaxes(title_text="price")
        return fig
    import matplotlib.dates as mdates
    fig, ax = r.subplots()
    tn = mdates.date2num(pd.to_datetime(t))
    step = np.median(np.diff(tn)) if len(tn) > 1 else 1
    width = 0.65 * step
    rising = c >= o
    cols = np.where(rising, up, down)
    if kind == "candle":
        ax.vlines(tn, l, h, colors=cols, lw=1)
        ax.bar(tn, np.maximum(np.abs(c - o), 1e-9 * np.abs(c).max()), bottom=np.minimum(o, c), width=width, color=cols,
               edgecolor=cols, lw=0.5)
    else:
        ax.vlines(tn, l, h, colors=cols, lw=1.2)
        ax.hlines(o, tn - width / 2, tn, colors=cols, lw=1.2)
        ax.hlines(c, tn, tn + width / 2, colors=cols, lw=1.2)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.set_ylabel("price")
    ax.grid(False)
    ax.grid(True, axis="y")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=up, lw=6, label="close ≥ open"),
                       Line2D([], [], color=down, lw=6, label="close < open")], loc="upper left", frameon=False)
    return fig


@chart("candlestick", "temporal", alt_names=("candlestick chart", "japanese candlestick", "k-line chart"),
       use="Daily (or periodic) open-high-low-close price movement.",
       avoid="Non-financial data or audiences unfamiliar with the encoding.")
def candlestick(r, data, date, open, high, low, close, *, up_color=None, down_color=None):
    """Body = open..close (coloured by direction), wick = low..high."""
    return _finance(r, data, date, open, high, low, close, "candle", up_color, down_color)


@chart("ohlc", "temporal", alt_names=("ohlc chart", "open-high-low-close chart", "bar chart (finance)"),
       use="Like a candlestick but lighter: left tick = open, right tick = close.",
       avoid="Dense series where ticks merge.")
def ohlc(r, data, date, open, high, low, close, *, up_color=None, down_color=None):
    """Vertical low..high line with an open tick (left) and close tick (right)."""
    return _finance(r, data, date, open, high, low, close, "ohlc", up_color, down_color)


# ===================================================================== gantt
@chart("gantt", "temporal", alt_names=("gantt chart", "timeline chart", "project schedule"),
       use="Schedules: tasks with start/end dates, grouped by phase or owner.",
       avoid="Hundreds of tasks or tasks without durations (use a timeline/barcode).")
def gantt(r, data, task, start, end, group=None, *, sort=True, today=None):
    """Horizontal bars from ``start`` to ``end`` per ``task``; ``today`` draws a reference line."""
    df = as_frame(data).copy()
    need(df, task, start, end, group)
    df[start], df[end] = pd.to_datetime(df[start]), pd.to_datetime(df[end])
    if sort:
        df = df.sort_values(start)
    tasks = [str(t) for t in df[task]]
    groups = ordered(df[group]) if group else [None]
    cmap = r.color_map(groups) if group else {None: r.colors(1)[0]}
    if r.mpl:
        import matplotlib.dates as mdates
        fig, ax = r.subplots(figsize=r.figsize or (8.5, max(3, 0.34 * len(tasks) + 1.2)))
        s = mdates.date2num(df[start])
        e = mdates.date2num(df[end])
        cols = [cmap[g] for g in (df[group] if group else [None] * len(df))]
        ax.barh(np.arange(len(tasks)), e - s, left=s, color=cols, height=0.6)
        ax.set_yticks(np.arange(len(tasks)), tasks)
        ax.invert_yaxis()
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.grid(False)
        ax.grid(True, axis="x")
        ax.tick_params(axis="y", length=0)
        if today is not None:
            ax.axvline(mdates.date2num(pd.Timestamp(today)), color=r.theme.status["critical"], lw=1.5)
        if group:
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(color=cmap[g], label=str(g)) for g in groups], loc="upper left",
                      bbox_to_anchor=(1.01, 1), frameon=False, title=group)
        return fig
    fig = go().Figure()
    for g in groups:
        sub = df if g is None else df[df[group] == g]
        dur = (sub[end] - sub[start]).dt.total_seconds() * 1000
        fig.add_trace(go().Bar(
            y=sub[task].astype(str), x=dur, base=sub[start], orientation="h", name=str(g), marker_color=cmap[g],
            showlegend=g is not None, width=0.6, customdata=np.column_stack([sub[start].dt.strftime("%Y-%m-%d"),
                                                                             sub[end].dt.strftime("%Y-%m-%d")]),
            hovertemplate="<b>%{y}</b><br>%{customdata[0]} → %{customdata[1]}<extra></extra>"))
    fig.update_xaxes(type="date")
    fig.update_yaxes(categoryorder="array", categoryarray=tasks, autorange="reversed")
    fig.update_layout(barmode="overlay", legend_title_text=group or "", height=r.height or max(360, 26 * len(tasks) + 120))
    if today is not None:
        fig.add_vline(x=pd.Timestamp(today).timestamp() * 1000, line=dict(color=r.theme.status["critical"], width=1.5))
    return fig


# =================================================================== barcode
@chart("barcode", "temporal", alt_names=("barcode chart", "barcode plot", "rug plot", "event plot"),
       use="Many observations on one long axis (events in time) in very little space.",
       avoid="When exact values or many colour classes are needed.")
def barcode(r, data, x, group=None, *, highlight=None, alpha=0.6):
    """One thin vertical mark per observation along ``x``; one row per ``group`` level.
    ``highlight`` = boolean column drawn darker/thicker."""
    df = as_frame(data).dropna(subset=[x])
    need(df, x, group, highlight)
    groups = ordered(df[group]) if group else [None]
    col = r.colors(1)[0]
    if r.mpl:
        fig, ax = r.subplots(figsize=r.figsize or (8.5, 0.55 * len(groups) + 1.2))
        for i, g in enumerate(groups):
            sub = df if g is None else df[df[group] == g]
            norm = sub if not highlight else sub[~sub[highlight].astype(bool)]
            ax.vlines(norm[x], i - 0.35, i + 0.35, color=col, alpha=alpha, lw=0.8)
            if highlight:
                hi = sub[sub[highlight].astype(bool)]
                ax.vlines(hi[x], i - 0.42, i + 0.42, color=r.theme.color(1), lw=2)
        ax.set_yticks(range(len(groups)), [str(g) if g is not None else "" for g in groups])
        ax.set_ylim(len(groups) - 0.5, -0.5)
        ax.grid(False)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(x)
        return fig
    fig = go().Figure()
    for g in groups:
        sub = df if g is None else df[df[group] == g]
        yv = [str(g) if g is not None else ""] * len(sub)
        sizes = 22 if not highlight else np.where(sub[highlight].astype(bool), 28, 22)
        cols = col if not highlight else np.where(sub[highlight].astype(bool), r.theme.color(1), col)
        fig.add_trace(go().Scatter(x=sub[x], y=yv, mode="markers", showlegend=False,
                                   marker=dict(symbol="line-ns", size=sizes, line=dict(color=cols, width=1), opacity=alpha),
                                   hovertemplate="%{x}<extra></extra>"))
    fig.update_yaxes(showgrid=False, autorange="reversed")
    fig.update_xaxes(title_text=x)
    fig.update_layout(height=r.height or 60 * len(groups) + 160)
    return fig
