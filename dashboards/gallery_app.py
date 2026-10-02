"""pyChartist gallery: browse every chart, switch backend (Plotly / matplotlib / seaborn) and theme.

Run:  python dashboards/gallery_app.py   ->  http://127.0.0.1:8054
"""
from __future__ import annotations

import inspect
import os
import warnings

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, clientside_callback, dcc, html

import pychartist as pc

warnings.filterwarnings("ignore", category=UserWarning)

CAT = pc.catalog()
CATEGORIES = list(dict.fromkeys(CAT.category))
LIGHT, DARK = pc.get_theme("flatly").with_transparent(), pc.get_theme("darkly").with_transparent()
EXTRA_THEMES = ["(match page)", "light", "dark", "d3", "chartist", "nvd3", "solar", "vapor", "jupyter-dark",
                "vscode-dark", "github-light"]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP], title="pyChartist gallery")
server = app.server

controls = dbc.Card(dbc.CardBody([
    dbc.Label("Category", className="fw-semibold"),
    dbc.Select(id="category", options=CATEGORIES, value=CATEGORIES[0], className="mb-3"),
    dbc.Label("Chart", className="fw-semibold"),
    dbc.Select(id="chart", className="mb-3"),
    dbc.Label("Backend", className="fw-semibold"),
    dbc.RadioItems(id="backend", value="plotly", className="mb-3",
                   options=[{"label": " Plotly (interactive)", "value": "plotly"},
                            {"label": " matplotlib (static)", "value": "matplotlib"},
                            {"label": " seaborn (static)", "value": "seaborn"}]),
    dbc.Label("Chart theme", className="fw-semibold"),
    dbc.Select(id="theme", options=EXTRA_THEMES, value="(match page)", className="mb-3"),
    html.Small("'(match page)' re-renders charts with the page's light/dark colours.", className="text-body-secondary"),
]), className="shadow-sm")

app.layout = dbc.Container([
    dcc.Store(id="dark-store"),
    dbc.Row([
        dbc.Col(html.H3([html.I(className="bi bi-bar-chart-line me-2"), "pyChartist gallery"], className="my-3"),
                width="auto"),
        dbc.Col(html.Span([html.I(className="bi bi-sun me-2"), dbc.Switch(id="dark", value=False, persistence=True,
                                                                         className="d-inline-block"),
                           html.I(className="bi bi-moon-stars")], className="d-flex align-items-center"), width="auto"),
    ], justify="between", align="center"),
    dbc.Row([
        dbc.Col(controls, lg=3, className="mb-3"),
        dbc.Col([
            dbc.Card([dbc.CardHeader(html.Span(id="chart-title", className="fw-semibold")),
                      dbc.CardBody(dcc.Loading(html.Div(id="chart-slot"), type="dot"), className="p-2")],
                     className="shadow-sm mb-3"),
            dbc.Row([dbc.Col(dbc.Alert(id="use", color="success", className="small mb-2"), md=6),
                     dbc.Col(dbc.Alert(id="avoid", color="warning", className="small mb-2"), md=6)]),
            html.Details([html.Summary("Function signature and docs"), html.Pre(id="doc", className="small mt-2")]),
        ], lg=9),
    ]),
], fluid=True)

clientside_callback(
    """(dark) => { document.documentElement.setAttribute('data-bs-theme', dark ? 'dark' : 'light'); return dark; }""",
    Output("dark-store", "data"), Input("dark", "value"))


@app.callback(Output("chart", "options"), Output("chart", "value"), Input("category", "value"), State("chart", "value"))
def _charts(category, current):
    names = list(CAT[CAT.category == category].chart)
    return [{"label": n.replace("_", " "), "value": n} for n in names], current if current in names else names[0]


@app.callback(Output("chart-slot", "children"), Output("chart-title", "children"), Output("use", "children"),
              Output("avoid", "children"), Output("doc", "children"),
              Input("chart", "value"), Input("backend", "value"), Input("theme", "value"), Input("dark-store", "data"))
def _render(name, backend, theme, dark):
    spec = pc.get_spec(name)
    if backend not in spec.backends and not (backend == "seaborn" and "matplotlib" in spec.backends):
        backend = spec.backends[0]
    th = (DARK if dark else LIGHT) if theme == "(match page)" else theme
    info = pc.examples.info(name)
    try:
        chart = pc.examples.demo(name, backend=backend, theme=th, title=False)
        comp = chart.to_dash(id="gallery-graph")
        if chart.is_plotly:
            comp.style = {"height": "min(70vh, 640px)"} if not chart.figure.layout.width else None
            chart.figure.update_layout(height=None if not chart.figure.layout.width else chart.figure.layout.height)
            comp.figure = chart.figure
        else:
            chart.close()
    except Exception as exc:  # show the error in the page rather than a dead callback
        comp = dbc.Alert(f"{type(exc).__name__}: {exc}", color="danger")
    sig = f"pc.{spec.func.__name__}{inspect.signature(spec.func)}\n\n{inspect.getdoc(spec.func)}"
    title = f"{info['title']}  ·  {name.replace('_', ' ')} · {backend} · data: {info['source']}"
    return comp, title, [html.B("Use when: "), spec.use], [html.B("Avoid when: "), spec.avoid], sig


if __name__ == "__main__":
    app.run(debug=bool(os.environ.get("DASH_DEBUG")), port=int(os.environ.get("PORT", 8054)))
