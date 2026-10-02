"""6. Geospatial & other (flow / network) charts."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .._util import as_frame, blend, compact, need, rgba, text_on
from ..layout import (chord_layout, layered_layout, ribbon_polygon, sankey_band,
                      sankey_layout, semicircle)
from ..layout.flows import annulus_polygon
from ..layout.shapes import US_STATE_NAMES, US_TILE_GRID
from ..registry import chart
from ._draw import go, mpl_blank, mpl_poly, pl_hide_axes, pl_poly


# ---------------------------------------------------------------- geometry I/O
def _geo_rings(geom):
    """Exterior rings of a GeoJSON Polygon / MultiPolygon as (n, 2) arrays."""
    if geom is None:
        return []
    if geom["type"] == "Polygon":
        return [np.asarray(geom["coordinates"][0], float)]
    if geom["type"] == "MultiPolygon":
        return [np.asarray(p[0], float) for p in geom["coordinates"]]
    return []


def _feature_key(scope, locationmode, featureidkey):
    if featureidkey:
        return featureidkey.replace("properties.", "")
    if scope == "usa":
        return "name"
    return "NAME" if locationmode == "country names" else "ADM0_A3"


def _load_geo(scope, geojson):
    if geojson is not None:
        return geojson
    from .. import datasets
    return datasets.load("us_states_geo" if scope == "usa" else "world_countries")


def _mpl_map_base(ax, r, geo, key, values=None, cmap=None, norm=None, missing=None, lw=0.4):
    from matplotlib.collections import PolyCollection
    polys, cols = [], []
    for f in geo["features"]:
        k = f["properties"].get(key)
        for ring in _geo_rings(f["geometry"]):
            polys.append(ring)
            if values is None:
                cols.append(missing or r.theme.grid)
            else:
                v = values.get(k)
                cols.append(missing or blend(r.theme.ink, r.theme.surface, 0.08) if v is None or not np.isfinite(v)
                            else cmap(norm(v)))
    pc = PolyCollection(polys, facecolors=cols, edgecolors=r.theme.surface, linewidths=lw)
    ax.add_collection(pc)
    return pc


def _lonlat_aspect(ax, lat0):
    ax.set_aspect(1 / max(math.cos(math.radians(lat0)), 0.2))


# ================================================================= choropleth
@chart("choropleth", "geo_other", alt_names=("choropleth map", "filled map", "thematic map", "shaded map"),
       use="Rates or ratios (per capita!) across administrative areas.",
       avoid="Raw counts that just mirror population, or when large areas would dominate (use a tile map).")
def choropleth(r, data, location, value, *, scope="world", locationmode=None, geojson=None, featureidkey=None,
               diverging=False, log=False):
    """Areas shaded by ``value``. ``location`` = ISO-3 codes (world, default), country names
    (``locationmode="country names"``), US state codes/names (``scope="usa"``), or ids matching
    ``featureidkey`` of your own ``geojson``."""
    df = as_frame(data)
    need(df, location, value)
    vals = df[value].astype(float)
    z = np.log10(vals.where(vals > 0)) if log else vals
    zlabel = f"log10 {value}" if log else value
    usa = scope == "usa"
    if locationmode is None:
        locationmode = "USA-states" if usa else "ISO-3"
    if r.plotly:
        cs = r.theme.div_colorscale() if diverging else r.theme.seq_colorscale()
        kw = dict(z=z, colorscale=cs, marker_line_color=r.theme.surface, marker_line_width=0.5,
                  colorbar=dict(title=dict(text=zlabel), outlinewidth=0), customdata=vals,
                  hovertemplate="<b>%{location}</b><br>" + value + ": %{customdata:,.4~g}<extra></extra>")
        if geojson is not None:
            kw.update(geojson=geojson, locations=df[location], featureidkey=featureidkey or "id")
        else:
            kw.update(locations=df[location], locationmode=locationmode)
        if diverging:
            m = np.nanmax(np.abs(z))
            kw.update(zmin=-m, zmax=m)
        fig = go().Figure(go().Choropleth(**kw))
        fig.update_geos(scope="usa" if usa else "world", projection_type="albers usa" if usa else "natural earth",
                        showframe=False, showcoastlines=False, bgcolor=r.theme.plotly_bg,
                        showland=True, landcolor=blend(r.theme.ink, r.theme.surface, 0.06),
                        fitbounds="locations" if geojson is not None else False)
        fig.update_layout(margin=dict(l=0, r=0, b=0), height=r.height or 520)
        return fig
    geo = _load_geo(scope, geojson)
    key = _feature_key(scope, locationmode, featureidkey)
    locs = df[location].astype(str)
    if usa and locationmode == "USA-states":
        locs = locs.map(lambda s: US_STATE_NAMES.get(s, s))
    values = dict(zip(locs, z))
    import matplotlib.colors as mcolors
    m = np.nanmax(np.abs(z)) if diverging else None
    norm = mcolors.Normalize(-m, m) if diverging else mcolors.Normalize(np.nanmin(z), np.nanmax(z))
    cmap = r.theme.div_cmap() if diverging else r.theme.seq_cmap()
    fig, ax = r.subplots(figsize=r.figsize or (9, 5))
    _mpl_map_base(ax, r, geo, key, values, cmap, norm)
    import matplotlib.cm as mcm
    cb = fig.colorbar(mcm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.7, pad=0.01, label=zlabel)
    cb.outline.set_visible(False)
    if usa:
        ax.set_xlim(-125, -66)
        ax.set_ylim(24, 50)
        _lonlat_aspect(ax, 38)
        ax.text(0.0, 0.0, "Alaska & Hawaii not shown", transform=ax.transAxes, fontsize=7, color=r.theme.muted)
    else:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-58, 85)
        ax.set_aspect(1.15)
    mpl_blank(ax, equal=False)
    missing = sorted(set(locs) - {f["properties"].get(key) for f in geo["features"]})
    if missing:
        r.meta["unmatched_locations"] = missing
    return fig


# ================================================================ geo heatmap
@chart("geo_heatmap", "geo_other", alt_names=("geographic heatmap", "geo heatmap", "density map", "heat map (geo)"),
       use="Where events concentrate in space, independent of administrative borders.",
       avoid="When boundaries matter (choropleth) or extremes would dominate.")
def geo_heatmap(r, data, lat, lon, z=None, *, radius=8, basemap="tiles", bins=180, center=None, zoom=None):
    """Point density (or ``z``-weighted density) on a map.

    Plotly: ``basemap="tiles"`` uses an interactive MapLibre tile map (needs internet for tiles);
    ``"outline"`` draws a KDE over country outlines and works offline/static. matplotlib always
    draws the outline variant."""
    df = as_frame(data).dropna(subset=[lat, lon])
    need(df, lat, lon, z)
    if r.plotly and basemap == "tiles":
        fig = go().Figure(go().Densitymap(lat=df[lat], lon=df[lon], z=df[z] if z else None, radius=radius,
                                          colorscale=r.theme.seq_colorscale(), colorbar=dict(outlinewidth=0)))
        c = center or dict(lat=float(df[lat].median()), lon=float(df[lon].median()))
        fig.update_layout(map=dict(style="carto-darkmatter" if r.theme.dark else "carto-positron", center=c,
                                   zoom=zoom if zoom is not None else 0.6),
                          margin=dict(l=0, r=0, b=0), height=r.height or 520)
        return fig
    w = df[z].to_numpy(float) if z else None
    H, xe, ye = np.histogram2d(df[lon], df[lat], bins=[bins, bins // 2], range=[[-180, 180], [-90, 90]], weights=w)
    from scipy.ndimage import gaussian_filter
    Hs = gaussian_filter(H, sigma=max(radius / 4, 0.5)).T
    Hs = np.where(Hs > Hs.max() * 0.02, Hs, np.nan)
    geo = _load_geo("world", None)
    if r.mpl:
        fig, ax = r.subplots(figsize=r.figsize or (9, 4.8))
        _mpl_map_base(ax, r, geo, "ADM0_A3", missing=blend(r.theme.ink, r.theme.surface, 0.08))
        im = ax.imshow(Hs, extent=(-180, 180, -90, 90), origin="lower", cmap=r.theme.seq_cmap(), alpha=0.9,
                       interpolation="bilinear", zorder=3)
        cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.01, label=f"{z} density" if z else "density")
        cb.outline.set_visible(False)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        ax.set_aspect(1.15)
        mpl_blank(ax, equal=False)
        return fig
    fig = go().Figure()
    for f in geo["features"]:
        for ring in _geo_rings(f["geometry"]):
            fig.add_trace(go().Scatter(x=ring[:, 0], y=ring[:, 1], mode="lines", fill="toself",
                                       fillcolor=blend(r.theme.ink, r.theme.surface, 0.08),
                                       line=dict(color=r.theme.surface, width=0.5), hoverinfo="skip", showlegend=False))
    fig.add_trace(go().Heatmap(x=(xe[:-1] + xe[1:]) / 2, y=(ye[:-1] + ye[1:]) / 2, z=Hs, colorscale=r.theme.seq_colorscale(),
                               zsmooth="best", opacity=0.9, colorbar=dict(outlinewidth=0),
                               hovertemplate="lon %{x:.0f}, lat %{y:.0f}<br>density %{z:.3g}<extra></extra>"))
    pl_hide_axes(fig, equal=False)
    fig.update_xaxes(range=[-180, 180])
    fig.update_yaxes(range=[-60, 85], scaleanchor="x", scaleratio=1.15)
    return fig


# =================================================================== tile map
@chart("tile_map", "geo_other", alt_names=("tile grid map", "tilegram", "statebins", "cartogram grid", "tile map"),
       use="Equal-size tile per region so small regions are as visible as large ones.",
       avoid="When real shape/size of regions matters.")
def tile_map(r, data, location, value, *, grid=None, diverging=False, fmt=compact):
    """One square per region on a fixed grid (default: US states by 2-letter code)."""
    df = as_frame(data)
    need(df, location, value)
    grid = grid or US_TILE_GRID
    vals = dict(zip(df[location].astype(str), df[value].astype(float)))
    import matplotlib.colors as mcolors
    v = np.array([x for x in vals.values() if np.isfinite(x)])
    if diverging:
        m = np.abs(v).max()
        norm = mcolors.Normalize(-m, m)
        cmap = r.theme.div_cmap()
    else:
        norm = mcolors.Normalize(v.min(), v.max())
        cmap = r.theme.seq_cmap()
    empty = blend(r.theme.ink, r.theme.surface, 0.06)
    tiles = []
    for code, (c, rw) in grid.items():
        val = vals.get(code)
        fill = mcolors.to_hex(cmap(norm(val))) if val is not None and np.isfinite(val) else empty
        tiles.append((code, c, -rw, val, fill))
    if r.mpl:
        fig, ax = r.subplots(figsize=r.figsize or (9, 5.6))
        from matplotlib.patches import FancyBboxPatch
        for code, c, rw, val, fill in tiles:
            ax.add_patch(FancyBboxPatch((c + 0.05, rw + 0.05), 0.9, 0.9, boxstyle="round,pad=0,rounding_size=0.08",
                                        facecolor=fill, edgecolor="none"))
            ink = text_on(fill, r.theme)
            ax.text(c + 0.5, rw + 0.62, code, ha="center", va="center", fontsize=9, fontweight="bold", color=ink)
            if val is not None:
                ax.text(c + 0.5, rw + 0.3, fmt(val), ha="center", va="center", fontsize=7, color=ink)
        import matplotlib.cm as mcm
        cb = fig.colorbar(mcm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6, pad=0.01, label=value)
        cb.outline.set_visible(False)
        mpl_blank(ax)
        ax.autoscale_view()
        return fig
    fig = go().Figure()
    for code, c, rw, val, fill in tiles:
        fig.add_shape(type="rect", x0=c + 0.05, x1=c + 0.95, y0=rw + 0.05, y1=rw + 0.95, fillcolor=fill,
                      line=dict(width=0), layer="below")
    fig.add_trace(go().Scatter(
        x=[t[1] + 0.5 for t in tiles], y=[t[2] + 0.5 for t in tiles], mode="text",
        text=[f"<b>{t[0]}</b><br>{fmt(t[3]) if t[3] is not None else ''}" for t in tiles],
        textfont=dict(size=10, color=[text_on(t[4], r.theme) for t in tiles]), showlegend=False,
        customdata=[[t[0], t[3] if t[3] is not None else float("nan")] for t in tiles],
        hovertemplate="<b>%{customdata[0]}</b>: %{customdata[1]:,.4~g}<extra></extra>"))
    fig.add_trace(go().Scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
                               marker=dict(colorscale=r.theme.div_colorscale() if diverging else r.theme.seq_colorscale(),
                                           cmin=norm.vmin, cmax=norm.vmax, color=[norm.vmin], showscale=True,
                                           colorbar=dict(title=dict(text=value), outlinewidth=0))))
    pl_hide_axes(fig)
    return fig


# ===================================================================== sankey
def _links_frame(data, source, target, value):
    if isinstance(data, dict) and "nodes" in data and "links" in data:
        nodes = [n.get("name", n.get("id", str(i))) for i, n in enumerate(data["nodes"])]
        links = [(nodes[l["source"]] if isinstance(l["source"], int) else l["source"],
                  nodes[l["target"]] if isinstance(l["target"], int) else l["target"], l.get("value", 1))
                 for l in data["links"]]
        return pd.DataFrame(links, columns=["source", "target", "value"]), "source", "target", "value"
    df = as_frame(data)
    need(df, source, target, value)
    return df, source, target, value


@chart("sankey", "geo_other", alt_names=("sankey diagram", "flow diagram", "alluvial diagram", "sankey chart"),
       use="Quantities flowing between stages / categories; widths are proportional to flow.",
       avoid="Cyclic flows or very many tiny links.")
def sankey(r, data, source="source", target="target", value="value", *, node_colors=None):
    """Flow diagram from a links table (``source``, ``target``, ``value``) or a {nodes, links} dict."""
    df, s, t, v = _links_frame(data, source, target, value)
    df = df.groupby([s, t], sort=False)[v].sum().reset_index()
    df = df[df[v] > 0]
    nodes = list(dict.fromkeys(list(df[s].astype(str)) + list(df[t].astype(str))))
    idx = {n: i for i, n in enumerate(nodes)}
    links = [(idx[str(a)], idx[str(b)], float(w)) for a, b, w in zip(df[s], df[t], df[v])]
    lay = sankey_layout(nodes, links)
    if node_colors is not None:
        ncol = [node_colors.get(n, r.theme.muted) for n in nodes]
    else:
        # colour by first-column ancestor so flows keep the identity of where they came from
        src_of = {}
        for a, b, _ in sorted(links, key=lambda l: lay["nodes"][l[0]]["depth"]):
            src_of.setdefault(b, src_of.get(a, a))
        roots = sorted({src_of.get(i, i) for i in range(len(nodes))})
        rc = dict(zip(roots, r.colors(len(roots))))
        ncol = [rc[src_of.get(i, i)] for i in range(len(nodes))]
    if r.plotly:
        fig = go().Figure(go().Sankey(
            arrangement="snap", valueformat=",.4~g",
            node=dict(label=nodes, color=ncol, pad=14, thickness=14, line=dict(width=0),
                      hovertemplate="%{label}: %{value}<extra></extra>"),
            link=dict(source=[l[0] for l in links], target=[l[1] for l in links], value=[l[2] for l in links],
                      color=[rgba(ncol[l[0]], 0.35) for l in links])))
        fig.update_traces(textfont=dict(color=r.theme.ink, size=11), selector=dict(type="sankey"))
        fig.update_layout(height=r.height or 560)
        return fig
    fig, ax = r.subplots(figsize=r.figsize or (10, 6))
    for g in lay["links"]:
        poly = sankey_band(g)
        mpl_poly(ax, np.column_stack([poly[:, 0], 1 - poly[:, 1]]), ncol[g["source"]], alpha=0.35)
    for i, n in enumerate(lay["nodes"]):
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((n["x0"], 1 - n["y1"]), n["x1"] - n["x0"], n["y1"] - n["y0"], facecolor=ncol[i]))
        right = n["x0"] < 0.5
        if n["y1"] - n["y0"] > 0.012:
            ax.text(n["x1"] + 0.006 if right else n["x0"] - 0.006, 1 - (n["y0"] + n["y1"]) / 2,
                    f"{n['name']}", ha="left" if right else "right", va="center", fontsize=7.5,
                    color=r.theme.ink_secondary)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    mpl_blank(ax, equal=False)
    return fig


# ====================================================================== chord
def _matrix(data, source, target, value):
    if isinstance(data, pd.DataFrame) and data.shape[0] == data.shape[1] and source is None:
        return [str(c) for c in data.columns], data.to_numpy(float)
    df, s, t, v = _links_frame(data, source or "source", target or "target", value or "value")
    names = list(dict.fromkeys(list(df[s].astype(str)) + list(df[t].astype(str))))
    m = pd.DataFrame(0.0, index=names, columns=names)
    for a, b, w in zip(df[s].astype(str), df[t].astype(str), df[v]):
        m.loc[a, b] += w
    return names, m.to_numpy(float)


@chart("chord", "geo_other", alt_names=("chord diagram", "radial network", "circular flow diagram", "chord chart"),
       use="Flows / relationships between a handful of groups (e.g. migration between regions).",
       avoid="More than ~10-12 groups; precise values.")
def chord(r, data, source=None, target=None, value=None, *, gap=0.04, labels=True):
    """Circle of group arcs (size = total outflow) with ribbons between them. ``data`` = square
    matrix DataFrame, or a links table with ``source``/``target``/``value``."""
    names, m = _matrix(data, source, target, value)
    lay = chord_layout(m, gap=gap)
    cmap = r.color_map(names)
    R = 1.0
    arcs = [(names[g["index"]], annulus_polygon(g["start"], g["end"], R, R * 1.07)) for g in lay["groups"]]
    ribbons = []
    for c in lay["chords"]:
        src = c["source"] if m[c["source"], c["target"]] >= m[c["target"], c["source"]] else c["target"]
        poly = ribbon_polygon(c["s"], c["t"], R * 0.99) if c["source"] != c["target"] else \
            ribbon_polygon(c["s"], c["s"], R * 0.99)
        ribbons.append((names[src], poly, c))
    if r.mpl:
        fig, ax = r.subplots(figsize=r.figsize or (7, 7))
        for n, poly, c in ribbons:
            mpl_poly(ax, poly, cmap[n], alpha=0.55, edge=r.theme.surface, lw=0.4)
        for n, poly in arcs:
            mpl_poly(ax, poly, cmap[n])
        if labels:
            for g in lay["groups"]:
                a = (g["start"] + g["end"]) / 2
                x, y = 1.12 * math.sin(a), 1.12 * math.cos(a)
                rot = 90 - math.degrees(a)
                ax.text(x, y, names[g["index"]], ha="left" if x >= 0 else "right", va="center", fontsize=8.5,
                        rotation=rot if x >= 0 else rot + 180, rotation_mode="anchor", color=r.theme.ink_secondary)
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.45, 1.45)
        mpl_blank(ax)
        return fig
    fig = go().Figure()
    for n, poly, c in ribbons:
        fig.add_trace(pl_poly(poly, cmap[n], opacity=0.55, line_color=r.theme.surface, line_width=0.4,
                              hover=f"{names[c['source']]} → {names[c['target']]}: {compact(m[c['source'], c['target']])}"
                                    + (f"<br>{names[c['target']]} → {names[c['source']]}: {compact(m[c['target'], c['source']])}"
                                       if c["source"] != c["target"] else "")))
    for (n, poly), g in zip(arcs, lay["groups"]):
        fig.add_trace(pl_poly(poly, cmap[n], hover=f"<b>{n}</b>: {compact(g['value'])} total"))
        if labels:
            a = (g["start"] + g["end"]) / 2
            fig.add_annotation(x=1.14 * math.sin(a), y=1.14 * math.cos(a), text=n, showarrow=False,
                               xanchor="left" if math.sin(a) >= 0 else "right", font=dict(size=11, color=r.theme.ink_secondary))
    pl_hide_axes(fig)
    fig.update_xaxes(range=[-1.5, 1.5])
    fig.update_yaxes(range=[-1.3, 1.3])
    fig.update_layout(showlegend=False, height=r.height or 640)
    return fig


# ======================================================================== arc
def _graph(links, source, target, value, nodes, node_id, group):
    import networkx as nx
    if isinstance(links, dict) and "nodes" in links:
        raw = links
        G = nx.Graph()
        for i, n in enumerate(raw["nodes"]):
            G.add_node(n.get("name", i), group=n.get("group"))
        names = [n.get("name", i) for i, n in enumerate(raw["nodes"])]
        for l in raw["links"]:
            a = names[l["source"]] if isinstance(l["source"], int) else l["source"]
            b = names[l["target"]] if isinstance(l["target"], int) else l["target"]
            G.add_edge(a, b, weight=l.get("value", 1))
        return G
    df = as_frame(links)
    need(df, source, target)
    G = nx.Graph()
    if nodes is not None:
        nd = as_frame(nodes)
        for _, row in nd.iterrows():
            G.add_node(row[node_id], group=row[group] if group else None)
    for a, b, w in zip(df[source], df[target], df[value] if value else np.ones(len(df))):
        if G.has_edge(a, b):
            G[a][b]["weight"] += w
        else:
            G.add_edge(a, b, weight=w)
    return G


@chart("arc", "geo_other", alt_names=("arc diagram", "arc chart", "linear network"),
       use="Connections between items ordered along one axis (clusters show as nested arcs).",
       avoid="Dense graphs where arcs overlap into noise.")
def arc(r, links, source="source", target="target", value=None, *, nodes=None, node_id="id", group="group",
        order="group", max_labels=80):
    """Nodes on a line, links as semicircular arcs (width ~ weight). ``links`` may be a
    {nodes, links} dict (d3 format) or a DataFrame."""
    G = _graph(links, source, target, value, nodes, node_id, group if nodes is not None or isinstance(links, dict) else None)
    deg = dict(G.degree(weight="weight"))
    grp = {n: G.nodes[n].get("group") for n in G.nodes}
    names = list(G.nodes)
    if order == "group":
        names.sort(key=lambda n: (str(grp[n]), -deg[n]))
    elif order == "degree":
        names.sort(key=lambda n: -deg[n])
    pos = {n: i for i, n in enumerate(names)}
    groups = list(dict.fromkeys(grp[n] for n in names))
    cmap = r.color_map(groups) if any(g is not None for g in groups) else {None: r.colors(1)[0]}
    wmax = max((d["weight"] for *_, d in G.edges(data=True)), default=1)
    if r.mpl:
        fig, ax = r.subplots(figsize=r.figsize or (10, 5))
        for a, b, d in G.edges(data=True):
            pts = semicircle(pos[a], pos[b])
            ax.plot(pts[:, 0], pts[:, 1], color=cmap[grp[a]] if grp[a] == grp[b] else r.theme.muted,
                    lw=0.4 + 2.5 * d["weight"] / wmax, alpha=0.55)
        sizes = np.array([deg[n] for n in names], float)
        ax.scatter(range(len(names)), np.zeros(len(names)), s=12 + 80 * sizes / sizes.max(),
                   color=[cmap[grp[n]] for n in names], edgecolor=r.theme.surface, lw=1, zorder=3)
        if len(names) <= max_labels:
            for i, n in enumerate(names):
                ax.text(i, -0.6, str(n), rotation=90, ha="center", va="top", fontsize=6.5, color=r.theme.ink_secondary)
        ax.set_ylim(-max(4.0, len(names) * 0.12), len(names) / 2 + 1)
        mpl_blank(ax, equal=False)
        return fig
    fig = go().Figure()
    for a, b, d in G.edges(data=True):
        pts = semicircle(pos[a], pos[b])
        fig.add_trace(go().Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines", hoverinfo="text", showlegend=False,
                                   text=f"{a} — {b}: {d['weight']:,.4g}",
                                   line=dict(color=rgba(cmap[grp[a]] if grp[a] == grp[b] else r.theme.muted, 0.55),
                                             width=0.4 + 2.5 * d["weight"] / wmax)))
    sizes = np.array([deg[n] for n in names], float)
    fig.add_trace(go().Scatter(x=list(range(len(names))), y=[0] * len(names), mode="markers+text" if len(names) <= max_labels
                               else "markers", text=[str(n) for n in names], textposition="bottom center",
                               textfont=dict(size=9, color=r.theme.ink_secondary), showlegend=False,
                               marker=dict(size=5 + 12 * sizes / sizes.max(), color=[cmap[grp[n]] for n in names],
                                           line=dict(color=r.theme.surface, width=1)),
                               customdata=sizes, hovertemplate="<b>%{text}</b><br>weighted degree %{customdata:.4g}<extra></extra>"))
    pl_hide_axes(fig, equal=False)
    return fig


# ==================================================================== network
@chart("network", "geo_other", alt_names=("network diagram", "network graph", "node-link diagram", "graph", "force-directed graph"),
       use="Structure of relationships: clusters, hubs and bridges.",
       avoid="Hairballs of thousands of nodes; precise quantities.")
def network(r, links, source="source", target="target", value=None, *, nodes=None, node_id="id", group="group",
            layout="spring", seed=7, labels="auto"):
    """Node-link diagram (networkx layout: spring | kamada_kawai | circular | shell);
    node size ~ weighted degree, colour = ``group``."""
    import networkx as nx
    G = _graph(links, source, target, value, nodes, node_id, group if nodes is not None or isinstance(links, dict) else None)
    if layout == "spring":
        P = nx.spring_layout(G, seed=seed, weight="weight", k=1.5 / math.sqrt(max(G.number_of_nodes(), 1)))
    else:
        P = getattr(nx, f"{layout}_layout")(G)
    deg = dict(G.degree(weight="weight"))
    grp = {n: G.nodes[n].get("group") for n in G.nodes}
    groups = list(dict.fromkeys(grp.values()))
    cmap = r.color_map(groups) if any(g is not None for g in groups) else {None: r.colors(1)[0]}
    names = list(G.nodes)
    dmax = max(deg.values()) or 1
    wmax = max((d["weight"] for *_, d in G.edges(data=True)), default=1)
    show = labels is True or (labels == "auto" and len(names) <= 40)
    top = set(sorted(names, key=lambda n: -deg[n])[:12]) if labels == "auto" else set(names)
    if r.mpl:
        fig, ax = r.subplots(figsize=r.figsize or (8, 7))
        from matplotlib.collections import LineCollection
        segs = [[P[a], P[b]] for a, b in G.edges()]
        lws = [0.3 + 2.2 * d["weight"] / wmax for *_, d in G.edges(data=True)]
        ax.add_collection(LineCollection(segs, colors=r.theme.baseline, linewidths=lws, zorder=1))
        xy = np.array([P[n] for n in names])
        ax.scatter(xy[:, 0], xy[:, 1], s=[20 + 260 * deg[n] / dmax for n in names],
                   color=[cmap[grp[n]] for n in names], edgecolor=r.theme.surface, lw=1.5, zorder=2)
        for n in names:
            if show or n in top:
                ax.annotate(str(n), P[n], xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7.5,
                            color=r.theme.ink)
        mpl_blank(ax)
        ax.autoscale_view()
        return fig
    fig = go().Figure()
    ex, ey = [], []
    for a, b in G.edges():
        ex += [P[a][0], P[b][0], None]
        ey += [P[a][1], P[b][1], None]
    fig.add_trace(go().Scatter(x=ex, y=ey, mode="lines", line=dict(color=r.theme.baseline, width=0.8), hoverinfo="skip",
                               showlegend=False))
    for g in groups:
        ns = [n for n in names if grp[n] == g]
        fig.add_trace(go().Scatter(
            x=[P[n][0] for n in ns], y=[P[n][1] for n in ns], mode="markers+text", name=str(g), showlegend=g is not None,
            text=[str(n) if (show or n in top) else "" for n in ns], textposition="top center",
            textfont=dict(size=9, color=r.theme.ink), customdata=[[str(n), deg[n]] for n in ns],
            marker=dict(size=[6 + 22 * deg[n] / dmax for n in ns], color=cmap[g], line=dict(color=r.theme.surface, width=1.5)),
            hovertemplate="<b>%{customdata[0]}</b><br>weighted degree %{customdata[1]:.4g}<extra></extra>"))
    pl_hide_axes(fig)
    fig.update_layout(legend_title_text="group", height=r.height or 640)
    return fig


# ================================================================== flowchart
_SHAPES = {"process", "decision", "terminal", "io"}


def _shape_poly(kind, cx, cy, w, h):
    if kind == "decision":
        return np.array([[cx, cy + h / 2 * 1.25], [cx + w / 2, cy], [cx, cy - h / 2 * 1.25], [cx - w / 2, cy]])
    if kind == "io":
        s = 0.12
        return np.array([[cx - w / 2 + s, cy + h / 2], [cx + w / 2 + s, cy + h / 2], [cx + w / 2 - s, cy - h / 2],
                         [cx - w / 2 - s, cy - h / 2]])
    if kind == "terminal":
        t = np.linspace(-math.pi / 2, math.pi / 2, 12)
        rad = h / 2
        right = np.column_stack([cx + w / 2 - rad + rad * np.cos(t), cy + rad * np.sin(t)])
        left = np.column_stack([cx - w / 2 + rad - rad * np.cos(t), cy - rad * np.sin(t)])
        return np.vstack([right, left])
    return np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])


@chart("flowchart", "geo_other", alt_names=("flow chart", "process diagram", "workflow diagram", "flowchart"),
       use="Steps and decisions of a process (customer journey, procedure, pipeline).",
       avoid="Quantitative data.")
def flowchart(r, steps, edges, *, direction="TB", box=(1.5, 0.55)):
    """Layered process diagram.

    ``steps`` = DataFrame/list of dicts with ``id``, ``label`` and optional ``shape``
    (process | decision | terminal | io); ``edges`` = list of (from, to) or (from, to, label)."""
    st = as_frame(steps) if not isinstance(steps, pd.DataFrame) else steps
    ids = [str(i) for i in st["id"]]
    label = dict(zip(ids, st["label"].astype(str) if "label" in st else ids))
    shape = dict(zip(ids, st["shape"] if "shape" in st else ["process"] * len(ids)))
    E = [(str(e[0]), str(e[1]), e[2] if len(e) > 2 else "") for e in edges]
    pos = layered_layout(ids, [(a, b) for a, b, _ in E], direction=direction)
    sx, sy = (2.1, 1.25) if direction == "TB" else (2.4, 1.1)
    P = {k: (v[0] * sx, v[1] * sy) for k, v in pos.items()}
    w, h = box
    fills = {"process": r.theme.color(0), "decision": r.theme.color(1), "terminal": r.theme.ink_secondary,
             "io": r.theme.color(2)}

    def border(a, b):
        """Point on node a's boundary towards b (rectangle approximation)."""
        (x0, y0), (x1, y1) = P[a], P[b]
        dx, dy = x1 - x0, y1 - y0
        k = min((w / 2) / abs(dx) if dx else np.inf, (h / 2 * (1.25 if shape[a] == "decision" else 1)) / abs(dy) if dy else np.inf)
        return x0 + dx * k, y0 + dy * k

    if r.mpl:
        xs = [p[0] for p in P.values()]
        ys = [p[1] for p in P.values()]
        fig, ax = r.subplots(figsize=r.figsize or (max(5, (max(xs) - min(xs)) * 1.3 + 3), max(4, (max(ys) - min(ys)) * 1.2 + 2)))
        for a, b, lab in E:
            p0, p1 = border(a, b), border(b, a)
            back = P[b][1] > P[a][1] if direction == "TB" else P[b][0] < P[a][0]
            ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=r.theme.ink_secondary, lw=1.3,
                                                              connectionstyle="arc3,rad=0.35" if back else "arc3"))
            if lab:
                ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, lab, fontsize=8, color=r.theme.ink_secondary,
                        ha="center", va="center", bbox=dict(boxstyle="round,pad=0.15", fc=r.theme.surface, ec="none"))
        for k in ids:
            kind = shape[k] if shape[k] in _SHAPES else "process"
            col = blend(fills[kind], r.theme.surface, 0.22)
            mpl_poly(ax, _shape_poly(kind, *P[k], w, h), col, edge=fills[kind], lw=1.5, zorder=3)
            ax.text(*P[k], label[k], ha="center", va="center", fontsize=8.5, color=r.theme.ink, zorder=4, wrap=True)
        mpl_blank(ax)
        ax.autoscale_view()
        ax.margins(0.08)
        return fig
    fig = go().Figure()
    for a, b, lab in E:
        p0, p1 = border(a, b), border(b, a)
        fig.add_annotation(x=p1[0], y=p1[1], ax=p0[0], ay=p0[1], xref="x", yref="y", axref="x", ayref="y", text="",
                           showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.3, arrowcolor=r.theme.ink_secondary)
        if lab:
            fig.add_annotation(x=(p0[0] + p1[0]) / 2, y=(p0[1] + p1[1]) / 2, text=lab, showarrow=False,
                               bgcolor=r.theme.plotly_bg, font=dict(size=10, color=r.theme.ink_secondary))
    for k in ids:
        kind = shape[k] if shape[k] in _SHAPES else "process"
        fig.add_trace(pl_poly(_shape_poly(kind, *P[k], w, h), blend(fills[kind], r.theme.surface, 0.22),
                              line_color=fills[kind], line_width=1.5, hover=f"{label[k]} ({kind})"))
    fig.add_trace(go().Scatter(x=[P[k][0] for k in ids], y=[P[k][1] for k in ids], mode="text",
                               text=[label[k] for k in ids], textfont=dict(size=11, color=r.theme.ink),
                               hoverinfo="skip", showlegend=False))
    pl_hide_axes(fig)
    fig.update_layout(showlegend=False)
    return fig
