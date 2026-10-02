# Architecture

```
pychartist/
  __init__.py        public API (all chart functions, themes, options, catalog)
  _config.py         global options: backend, theme, figure sizes; pc.options() context manager
  registry.py        @chart decorator, Render context, catalog / plot / get_spec
  chart.py           Chart result object: show, save, to_html, to_image, to_dash, notebook repr
  themes.py          Theme tokens -> matplotlib rcParams and Plotly templates; inherit / host presets
  _util.py           data coercion, colour helpers, number formatting, fold_other
  layout/            backend-free geometry (tested on its own)
    hierarchy.py     tree building, squarified treemap, d3 circle packing, convex (polygon) treemap
    placement.py     beeswarm, word cloud spiral, Venn/Euler circles and region label anchors
    flows.py         sankey, chord, arc, layered DAG (flowchart) layouts, Bézier helpers
    shapes.py        polygon icons (person, star, heart, ...) and the US tile grid
  charts/            one module per chart family; each function draws on every backend
    _draw.py         small backend helpers (legend placement, polygons, value grids)
    comparison.py correlation.py part_to_whole.py temporal.py distribution.py geo_other.py
    statistical.py   seaborn figure-level charts (pair, joint, cluster map) + Plotly versions
  stats.py           EDA tests returning TestResult objects
  eda.py             one-call analyses combining charts and tests (EDAResult)
  dashboard.py       Dash helpers: chart_card, make_app with light/dark re-rendering
  datasets.py        registry + downloader for the example datasets
  examples.py        a realistic demo per chart (used by the gallery, notebooks and dashboards)
```

## A chart function

```python
@chart("lollipop", "comparison", alt_names=("lollipop plot",), use="...", avoid="...", seaborn=False)
def lollipop(r, data, x, y, *, orient="h", sort="desc"):
    df = ...                                  # tidy data -> what to draw
    if r.mpl:                                 # matplotlib *or* seaborn backend
        fig, ax = r.subplots()                # reuses the caller's ax= if given
        ...                                   # draw with r.theme tokens / r.color_map(...)
        return fig
    fig = go.Figure(...)                      # plotly
    return fig
```

The decorator:

* adds the common keyword arguments (`backend, theme, title, ax, figsize, width, height, palette`) to
  the public signature and docstring;
* resolves the backend and theme (`"inherit"` reads the host's styling);
* wraps the matplotlib drawing in `rc_context(theme.mpl_rc())`, except for inherited themes;
* applies the title, size and Plotly template, and wraps the result in a `Chart`;
* registers the chart and its aliases for `pc.catalog()` and `pc.plot(name)`.

The `Render` object `r` carries the backend (`r.mpl`, `r.sns`, `r.plotly`), the theme, a stable
category-to-colour mapping (`r.color_map`) and `r.meta` for extra results. Charts with a
seaborn-native path pass `seaborn=True` and branch on `r.sns`. Every other matplotlib chart also
accepts `backend="seaborn"` and draws with matplotlib under the same theme.

matplotlib figures are created with `matplotlib.figure.Figure`, not pyplot, so notebooks never
show a chart twice. `Chart.show()` borrows a pyplot manager only when needed. seaborn's
figure-level functions do register with pyplot, so their figures are detached with `plt.close()`.

## Adding a chart

1. Write the geometry in `layout/` if it is non-trivial, so both backends share it, and add an
   invariant test in `tests/test_layout.py`.
2. Add the function to the relevant `charts/*.py` module with the `@chart` decorator, export it in
   `__init__.py`, add a demo in `examples.py` and a case in `tests/test_charts.py`.
3. Add an oracle test if the chart computes values (aggregation, binning, statistics).
4. Run `python -m pytest` and `python scripts/build_gallery.py`.
