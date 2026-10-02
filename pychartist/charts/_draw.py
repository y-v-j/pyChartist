"""Backend drawing helpers shared by the chart modules (thin wrappers, theme-aware)."""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter

from .._util import compact, rgba


# ------------------------------------------------------------------ matplotlib
def mpl_value_grid(ax, axis: str = "y") -> None:
    """Gridlines only along the value axis; hide the category-axis grid and its spine."""
    ax.grid(False)
    ax.grid(True, axis=axis)
    if axis == "x":
        ax.spines["left"].set_visible(True)
        ax.tick_params(axis="y", length=0)
    else:
        ax.tick_params(axis="x", length=0)


def mpl_compact_axis(axis) -> None:
    axis.set_major_formatter(FuncFormatter(lambda v, _: compact(v)))


def mpl_legend(ax, r, title=None, outside: bool = True, **kw):
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) < 2 and not kw.get("force"):
        return None
    kw.pop("force", None)
    if outside:
        kw.setdefault("loc", "upper left")
        kw.setdefault("bbox_to_anchor", (1.01, 1.0))
    leg = ax.legend(title=title, **kw)
    if leg.get_title():
        leg.get_title().set_color(r.theme.ink_secondary)
    return leg


def mpl_poly(ax, pts, color, alpha=1.0, edge=None, lw=0.0, zorder=2, label=None, **kw):
    patch = Polygon(np.asarray(pts), closed=True, facecolor=color, alpha=alpha,
                    edgecolor=edge if edge is not None else "none", linewidth=lw, zorder=zorder,
                    label=label, **kw)
    ax.add_patch(patch)
    return patch


def mpl_blank(ax, equal: bool = True) -> None:
    ax.set_axis_off()
    if equal:
        ax.set_aspect("equal")


def mpl_bar_labels(ax, r, container, fmt=compact, inside=False, **kw):
    kw.setdefault("padding", 3)
    kw.setdefault("fontsize", 8.5)
    kw.setdefault("color", r.theme.ink_secondary)
    return ax.bar_label(container, labels=[fmt(v) for v in container.datavalues],
                        label_type="center" if inside else "edge", **kw)


# ---------------------------------------------------------------------- plotly
def go():
    import plotly.graph_objects as _go
    return _go


def pl_poly(pts, color, name=None, hover=None, opacity=1.0, line_color=None, line_width=0,
            legendgroup=None, showlegend=False, **kw):
    pts = np.asarray(pts)
    xs = np.append(pts[:, 0], pts[0, 0])
    ys = np.append(pts[:, 1], pts[0, 1])
    return go().Scatter(
        x=xs, y=ys, mode="lines", fill="toself", fillcolor=rgba(color, opacity) if opacity < 1 else color,
        line=dict(color=line_color or color, width=line_width), name=name,
        hoveron="fills", text=hover, hoverinfo="text" if hover else "skip",
        legendgroup=legendgroup, showlegend=showlegend, **kw)


def pl_hide_axes(fig, equal: bool = True, row=None, col=None) -> None:
    kw = dict(visible=False, showgrid=False, zeroline=False)
    fig.update_xaxes(**kw, row=row, col=col)
    fig.update_yaxes(**kw, row=row, col=col)
    if equal:
        fig.update_yaxes(scaleanchor="x", scaleratio=1, row=row, col=col)


def pl_value_grid(fig, axis: str = "y") -> None:
    if axis == "y":
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True)
    else:
        fig.update_yaxes(showgrid=False)
        fig.update_xaxes(showgrid=True)


def pl_legend_right(fig) -> None:
    fig.update_layout(legend=dict(orientation="v", x=1.02, xanchor="left", y=1, yanchor="top"))
