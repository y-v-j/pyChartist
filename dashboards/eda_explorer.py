"""EDA explorer: pick a dataset and one or two variables; get the right charts *and* tests.

The analysis follows the variable-type table (numerical / categorical) of pychartist.eda.
Run:  python dashboards/eda_explorer.py   ->  http://127.0.0.1:8051
"""
from __future__ import annotations

import os
import warnings

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, clientside_callback, dcc, html

import pychartist as pc
from pychartist import stats

warnings.filterwarnings("ignore", category=UserWarning)

DATASETS = ["penguins", "tips", "titanic", "mpg", "iris", "healthexp", "gapminder"]
LIGHT, DARK = pc.get_theme("cosmo").with_transparent(), pc.get_theme("darkly").with_transparent()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO, dbc.icons.BOOTSTRAP], title="pyChartist EDA")
server = app.server

app.layout = dbc.Container([
    dcc.Store(id="dark-store"),
    dbc.Row([dbc.Col(html.H3("Exploratory data analysis", className="my-3"), width="auto"),
             dbc.Col(html.Span([html.I(className="bi bi-sun me-2"),
                                dbc.Switch(id="dark", value=False, className="d-inline-block"),
                                html.I(className="bi bi-moon-stars")], className="d-flex align-items-center"),
                     width="auto")], justify="between", align="center"),
    dbc.Row([
        dbc.Col([dbc.Label("Dataset"), dbc.Select(id="ds", options=DATASETS, value="penguins")], md=3),
        dbc.Col([dbc.Label("Variable X"), dbc.Select(id="x")], md=3),
        dbc.Col([dbc.Label("Variable Y (optional)"), dbc.Select(id="y")], md=3),
        dbc.Col([dbc.Label("Backend"), dbc.RadioItems(id="backend", value="plotly", inline=True,
                                                      options=["plotly", "matplotlib", "seaborn"])], md=3),
    ], className="mb-3"),
    html.Div(id="kinds", className="mb-2 text-body-secondary small"),
    dcc.Loading(html.Div(id="result"), type="dot"),
    html.Hr(),
    html.Small(["Decision table: 1 numerical → histogram, box, Q-Q + Shapiro-Wilk / KS · 1 categorical → bar, pie + "
                "χ² goodness of fit · 2 numerical → scatter, correlation heatmap + Pearson / Spearman · categorical × "
                "numerical → box, violin + t-test / Mann-Whitney or ANOVA / Kruskal-Wallis · 2 categorical → stacked "
                "bar, mosaic + χ² independence / Fisher."], className="text-body-secondary"),
], fluid=True)

clientside_callback(
    """(dark) => { document.documentElement.setAttribute('data-bs-theme', dark ? 'dark' : 'light'); return dark; }""",
    Output("dark-store", "data"), Input("dark", "value"))


def _load(name):
    df = pc.datasets.load(name)
    return df.drop(columns=[c for c in df.columns if df[c].nunique() > 50 and df[c].dtype == object])


@app.callback(Output("x", "options"), Output("x", "value"), Output("y", "options"), Output("y", "value"),
              Input("ds", "value"))
def _cols(name):
    df = _load(name)
    cols = list(df.columns)
    num = [c for c in cols if stats.variable_kind(df[c]) == "numerical"]
    cat = [c for c in cols if c not in num]
    none = [{"label": "(none)", "value": ""}]
    return cols, (cat[0] if cat else cols[0]), none + [{"label": c, "value": c} for c in cols], (num[0] if num else "")


@app.callback(Output("result", "children"), Output("kinds", "children"),
              Input("ds", "value"), Input("x", "value"), Input("y", "value"), Input("backend", "value"),
              Input("dark-store", "data"))
def _analyze(name, x, y, backend, dark):
    if not x:
        return dash.no_update, ""
    df = _load(name)
    y = None if (not y or y == x) else y
    kinds = f"{x}: {stats.variable_kind(df[x])}" + (f" · {y}: {stats.variable_kind(df[y])}" if y else "")
    try:
        res = pc.eda.analyze(df, x, y, backend=backend, theme=DARK if dark else LIGHT)
    except Exception as exc:
        return dbc.Alert(f"{type(exc).__name__}: {exc}", color="warning"), kinds
    notes = [dbc.Alert(n, color="info", className="small") for n in res.notes]
    return html.Div(notes + [res.to_dash()]), f"Analysis: {res.kind} — {kinds}"


if __name__ == "__main__":
    app.run(debug=bool(os.environ.get("DASH_DEBUG")), port=int(os.environ.get("PORT", 8051)))
