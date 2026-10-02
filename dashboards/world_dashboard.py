"""Gapminder world dashboard: linked pyChartist charts with cross-filtering.

* year slider + continent filter drive every chart;
* click a country on the map or in the bubble chart to follow it in the line charts;
* light/dark switch re-renders all figures in the matching theme.

Run:  python dashboards/world_dashboard.py   ->  http://127.0.0.1:8052
"""
from __future__ import annotations

import os
import warnings

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, clientside_callback, ctx, dcc, html

import pychartist as pc

warnings.filterwarnings("ignore", category=UserWarning)

GAP = pc.datasets.load("gapminder")
ISO = pc.datasets.load("world_gdp")[["COUNTRY", "CODE"]].rename(columns={"COUNTRY": "country"})
GAP = GAP.merge(ISO, on="country", how="left")
YEARS = sorted(GAP.year.unique())
CONTINENTS = sorted(GAP.continent.unique())
LIGHT, DARK = pc.get_theme("lux").with_transparent(), pc.get_theme("cyborg").with_transparent()
CMAP = dict(zip(CONTINENTS, pc.get_theme("light").categorical))     # one colour per continent everywhere

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX, dbc.icons.BOOTSTRAP], title="World in numbers")
server = app.server


def card(title, cid, height="420px"):
    return dbc.Card([dbc.CardHeader(title), dbc.CardBody(dcc.Graph(id=cid, style={"height": height},
                                                                   config={"displaylogo": False}), className="p-1")],
                    className="shadow-sm h-100")


def kpi(label, cid):
    return dbc.Card(dbc.CardBody([html.Div(label, className="small text-body-secondary"),
                                  html.Div(id=cid, className="fs-3 fw-bold")]), className="shadow-sm text-center")


app.layout = dbc.Container([
    dcc.Store(id="dark-store"), dcc.Store(id="country", data="China"),
    dbc.Row([dbc.Col(html.H3("World in numbers, 1952-2007", className="my-3"), width="auto"),
             dbc.Col(html.Span([html.I(className="bi bi-sun me-2"), dbc.Switch(id="dark", value=False,
                                                                              className="d-inline-block"),
                                html.I(className="bi bi-moon-stars")], className="d-flex align-items-center"),
                     width="auto")], justify="between", align="center"),
    dbc.Row([
        dbc.Col([dbc.Label("Year"), dcc.Slider(min(YEARS), max(YEARS), 5, value=2007, id="year",
                                               marks={int(y): str(y) for y in YEARS[::2]})], md=7),
        dbc.Col([dbc.Label("Continents"), dbc.Checklist(id="continents", options=CONTINENTS, value=CONTINENTS, inline=True, switch=True)], md=5),
    ], className="mb-3"),
    dbc.Row([dbc.Col(kpi("World population", "k-pop"), md=3), dbc.Col(kpi("Median life expectancy", "k-life"), md=3),
             dbc.Col(kpi("Median GDP per capita", "k-gdp"), md=3), dbc.Col(kpi("Selected country", "k-country"), md=3)],
            className="mb-3"),
    dbc.Row([dbc.Col(card("Life expectancy (click a country)", "map"), lg=7, className="mb-3"),
             dbc.Col(card("Wealth vs health (bubble = population)", "bubble"), lg=5, className="mb-3")]),
    dbc.Row([dbc.Col(card("Selected country vs its continent: life expectancy", "trend", "360px"), lg=6, className="mb-3"),
             dbc.Col(card("Population share", "tree", "360px"), lg=6, className="mb-3")]),
    html.Small("Data: Gapminder (CC BY 4.0) via plotly/datasets. Charts: pyChartist.", className="text-body-secondary"),
], fluid=True)

clientside_callback(
    """(dark) => { document.documentElement.setAttribute('data-bs-theme', dark ? 'dark' : 'light'); return dark; }""",
    Output("dark-store", "data"), Input("dark", "value"))


@app.callback(Output("country", "data"), Input("map", "clickData"), Input("bubble", "clickData"),
              State("country", "data"), prevent_initial_call=True)
def _pick(map_click, bubble_click, current):
    click = map_click if ctx.triggered_id == "map" else bubble_click
    if not click:
        return current
    pt = click["points"][0]
    if ctx.triggered_id == "map":
        hit = GAP.loc[GAP.CODE == pt.get("location"), "country"]
        return hit.iloc[0] if len(hit) else current
    return pt.get("text") or current


@app.callback(Output("map", "figure"), Output("bubble", "figure"), Output("trend", "figure"), Output("tree", "figure"),
              Output("k-pop", "children"), Output("k-life", "children"), Output("k-gdp", "children"),
              Output("k-country", "children"),
              Input("year", "value"), Input("continents", "value"), Input("country", "data"), Input("dark-store", "data"))
def _update(year, continents, country, dark):
    th = DARK if dark else LIGHT
    d = GAP[(GAP.year == year) & GAP.continent.isin(continents or CONTINENTS)]
    kw = dict(backend="plotly", theme=th)
    m = pc.choropleth(d.dropna(subset=["CODE"]), "CODE", "lifeExp", **kw).figure
    m.update_layout(height=None, margin=dict(l=0, r=0, t=10, b=0))
    b = pc.bubble(d, "gdpPercap", "lifeExp", "pop", "continent", label="country", log_x=True,
                  palette=CMAP, **kw).figure
    b.update_traces(mode="markers")
    b.update_layout(height=None, margin=dict(t=10), legend=dict(orientation="h", y=-0.2))
    cont = GAP.loc[GAP.country == country, "continent"].iloc[0]
    hist = GAP[GAP.continent == cont].groupby("year", as_index=False).lifeExp.median().assign(series=f"{cont} median")
    mine = GAP[GAP.country == country][["year", "lifeExp"]].assign(series=country)
    t = pc.line(__import__("pandas").concat([mine, hist]), "year", "lifeExp", "series", markers=True, **kw).figure
    t.add_vline(x=year, line=dict(color=th.muted, width=1, dash="dot"))
    t.update_layout(height=None, margin=dict(t=10))
    tr = pc.treemap(d, ["continent", "country"], "pop", palette=CMAP, **kw).figure
    tr.update_layout(height=None, margin=dict(t=10, l=5, r=5, b=5))
    from pychartist._util import compact
    sel = GAP[(GAP.country == country) & (GAP.year == year)]
    sel_txt = f"{country}: {sel.lifeExp.iloc[0]:.1f} y" if len(sel) else country
    return (m, b, t, tr, compact(d["pop"].sum()), f"{d.lifeExp.median():.1f} y", f"${compact(d.gdpPercap.median())}",
            sel_txt)


if __name__ == "__main__":
    app.run(debug=bool(os.environ.get("DASH_DEBUG")), port=int(os.environ.get("PORT", 8052)))
