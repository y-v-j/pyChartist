"""Dash helpers: chart cards, theme-matched figures and a light/dark-switching app builder.

Charts become Dash components with ``chart.to_dash()``. For dashboards that toggle between light
and dark mode, give the builder *functions* ``theme -> Chart`` so every chart is re-rendered in
the theme that matches the page::

    import pychartist as pc
    from pychartist.dashboard import make_app

    pen = pc.datasets.load("penguins")
    app = make_app("Penguins", {
        "Body mass": lambda theme: pc.violin(pen, "species", "body_mass_g", backend="plotly", theme=theme),
        "Bill size": lambda theme: pc.scatter(pen, "bill_length_mm", "bill_depth_mm", "species",
                                             backend="plotly", theme=theme),
    }, bootstrap="FLATLY", dark_bootstrap="DARKLY")
    app.run(debug=True)

The chart themes are derived from the Bootstrap theme names (``pc.from_bootstrap``), with
transparent backgrounds so the cards' own background shows through.
"""
from __future__ import annotations

from typing import Callable

from .chart import Chart
from .themes import Theme, from_bootstrap, get_theme

ChartBuilder = Callable[[Theme], Chart]


def bootstrap_url(name: str) -> str:
    import dash_bootstrap_components as dbc
    return getattr(dbc.themes, name.upper())


def chart_theme_for(bootstrap: str, transparent: bool = True) -> Theme:
    """pyChartist theme matching a dbc theme name (e.g. "DARKLY"), transparent by default."""
    th = from_bootstrap(bootstrap)
    return th.with_transparent() if transparent else th


def chart_card(chart: Chart, title: str | None = None, footer: str | None = None, id: str | None = None,
               height: str | None = None):
    """A ``dbc.Card`` holding the chart (Graph for Plotly, PNG for matplotlib/seaborn)."""
    import dash_bootstrap_components as dbc
    from dash import html
    style = {"height": height} if height else None
    body = [chart.to_dash(id=id, style=style) if chart.is_plotly and style else chart.to_dash(id=id)]
    parts = []
    if title:
        parts.append(dbc.CardHeader(html.Span(title, className="fw-semibold")))
    parts.append(dbc.CardBody(body, className="p-2"))
    if footer:
        parts.append(dbc.CardFooter(html.Small(footer, className="text-body-secondary")))
    return dbc.Card(parts, className="h-100 shadow-sm")


def _render(builder: ChartBuilder, theme: Theme, idx: int):
    chart = builder(theme)
    return chart.to_dash(id={"type": "pc-chart", "index": idx})


def make_app(title: str, charts: dict[str, ChartBuilder], *, bootstrap: str = "BOOTSTRAP",
             dark_bootstrap: str | None = "DARKLY", columns: int = 2, start_dark: bool = False,
             header=None, app=None):
    """Build a Dash app: one card per chart, a light/dark switch, charts re-rendered per mode.

    ``charts`` maps card titles to functions ``theme -> Chart``. Light mode uses the ``bootstrap``
    stylesheet; dark mode switches Bootstrap 5.3's ``data-bs-theme`` and re-renders charts with
    the ``dark_bootstrap`` palette (ink/surface), so figures always match the page.
    """
    import dash
    import dash_bootstrap_components as dbc
    from dash import ALL, Input, Output, State, clientside_callback, dcc, html

    light_theme = chart_theme_for(bootstrap)
    dark_theme = chart_theme_for(dark_bootstrap) if dark_bootstrap else get_theme("dark").with_transparent()
    app = app or dash.Dash(__name__, external_stylesheets=[bootstrap_url(bootstrap), dbc.icons.BOOTSTRAP],
                           title=title, suppress_callback_exceptions=True)
    names = list(charts)
    builders = [charts[n] for n in names]
    width = max(1, 12 // columns)
    initial = dark_theme if start_dark else light_theme
    cards = [dbc.Col(dbc.Card([dbc.CardHeader(html.Span(n, className="fw-semibold")),
                               dbc.CardBody(html.Div(_render(b, initial, i), id={"type": "pc-slot", "index": i}),
                                            className="p-2")], className="h-100 shadow-sm"),
                     lg=width, md=12, className="mb-3") for i, (n, b) in enumerate(zip(names, builders))]
    switch = html.Span([dbc.Label(className="bi bi-sun-fill me-2", html_for="pc-color-mode"),
                        dbc.Switch(id="pc-color-mode", value=start_dark, className="d-inline-block",
                                   persistence=True),
                        dbc.Label(className="bi bi-moon-stars-fill", html_for="pc-color-mode")],
                       className="d-flex align-items-center")
    app.layout = dbc.Container([
        dcc.Store(id="pc-mode-store"),
        dbc.Row([dbc.Col(html.H3(title, className="my-3"), width="auto"), dbc.Col(switch, width="auto")],
                justify="between", align="center"),
        header or html.Div(),
        dbc.Row(cards),
        html.Footer(html.Small("Built with pyChartist", className="text-body-secondary"), className="my-3"),
    ], fluid=True)

    clientside_callback(
        """(dark) => { document.documentElement.setAttribute('data-bs-theme', dark ? 'dark' : 'light');
                       return dark; }""",
        Output("pc-mode-store", "data"), Input("pc-color-mode", "value"))

    @app.callback(Output({"type": "pc-slot", "index": ALL}, "children"), Input("pc-mode-store", "data"),
                  State({"type": "pc-slot", "index": ALL}, "id"), prevent_initial_call=True)
    def _rerender(dark, slots):
        th = dark_theme if dark else light_theme
        return [_render(builders[s["index"]], th, s["index"]) for s in slots]

    app.pychartist = {"light": light_theme, "dark": dark_theme, "names": names}
    return app
