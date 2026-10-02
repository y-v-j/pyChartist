"""Every chart builds on every backend, and the drawn numbers equal independently computed ones."""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pychartist as pc

BACKENDS = ["matplotlib", "seaborn", "plotly"]


# ------------------------------------------------------------- catalogue coverage
# Optional: point PYCHARTIST_REFERENCE_NOTE at the chart-guide note the catalogue was built from.
SOURCE = Path(os.environ.get("PYCHARTIST_REFERENCE_NOTE", "reference/chart_types.md"))


def _source_chart_names():
    """Chart names as headed in the reference note ('What is a <name>?')."""
    text = SOURCE.read_text(encoding="utf-8")
    names = re.findall(r"What is (?:a |an |the )?([^?*_\n]+?)\?", text)
    return sorted({n.strip().lower() for n in names if len(n) < 40})


@pytest.mark.skipif(not SOURCE.exists(), reason="reference note not available")
def test_every_reference_chart_is_in_the_catalog():
    names = _source_chart_names()
    assert len(names) == 77          # the note says "eighty" but documents 77 charts
    missing = []
    for n in names:
        try:
            pc.get_spec(n)
        except KeyError:
            missing.append(n)
    assert not missing, f"charts from the reference without an implementation: {missing}"


def test_catalog_lists_backends():
    cat = pc.catalog()
    assert len(cat) == 82            # 77 reference charts + mosaic, qq, pair, joint, cluster map
    assert set(cat.category) == set(pc.CATEGORIES.values())
    assert "seaborn" in cat.set_index("chart").loc["violin", "backends"]
    assert pc.plot("spider chart", pd.DataFrame({"s": ["a", "b"], "x": [1, 2], "y": [3, 1], "z": [2, 2]}),
                   "s").kind == "radar"


# ------------------------------------------------------------- smoke: all charts
def _cases(pen, tips, titanic, gap07):
    stocks = pc.datasets.load("stocks").query("date.dt.year >= 2008")
    sea = pc.datasets.load("seattle_weather").head(400)
    kpi = pd.DataFrame({"m": ["A", "B"], "v": [5, 8], "t": [6, 7], "lo": [3, 4], "hi": [9, 10]})
    gantt = pd.DataFrame({"task": ["a", "b"], "s": ["2026-01-01", "2026-01-05"], "e": ["2026-01-10", "2026-01-20"]})
    ohlc = pc.datasets.load("apple_stock").tail(20)
    links = pd.DataFrame({"source": ["a", "a", "b"], "target": ["b", "c", "c"], "value": [3, 1, 2]})
    eu = gap07[gap07.continent == "Europe"].head(10)
    return {
        "bar": lambda b: pc.bar(pen, "species", "body_mass_g", agg="mean", backend=b),
        "column": lambda b: pc.column(tips, "day", "tip", backend=b),
        "grouped_bar": lambda b: pc.grouped_bar(tips, "day", "tip", "sex", backend=b),
        "lollipop": lambda b: pc.lollipop(eu, "country", "pop", backend=b),
        "bullet": lambda b: pc.bullet(kpi, "m", "v", "t", ["lo", "hi"], backend=b),
        "dot_plot": lambda b: pc.dot_plot(tips, "tip", "day", "sex", backend=b),
        "dumbbell": lambda b: pc.dumbbell(tips, "day", "tip", "sex", backend=b),
        "pictogram": lambda b: pc.pictogram(eu.head(3), "country", "pop", unit=1e7, backend=b),
        "icon_chart": lambda b: pc.icon_chart(eu.head(4), "country", "pop", backend=b),
        "range_plot": lambda b: pc.range_plot(kpi, "m", "lo", "hi", backend=b),
        "radial_bar": lambda b: pc.radial_bar(eu.head(5), "country", "pop", backend=b),
        "parallel_coordinates": lambda b: pc.parallel_coordinates(pen.head(60), None, "species", backend=b),
        "radar": lambda b: pc.radar(pen, "species", ["bill_length_mm", "bill_depth_mm", "flipper_length_mm"], backend=b),
        "nightingale": lambda b: pc.nightingale(tips, "day", "tip", "sex", backend=b),
        "waterfall": lambda b: pc.waterfall(kpi, "m", "v", backend=b),
        "matrix_chart": lambda b: pc.matrix_chart(tips, "day", "time", backend=b),
        "small_multiples": lambda b: pc.small_multiples(stocks, "symbol", "line", x="date", y="price", backend=b),
        "word_cloud": lambda b: pc.word_cloud("alpha beta beta gamma gamma gamma delta", backend=b),
        "slope": lambda b: pc.slope(pd.DataFrame({"c": list("aabb"), "t": [1, 2, 1, 2], "v": [1, 3, 4, 2]}), "c", "t", "v", backend=b),
        "table_chart": lambda b: pc.table_chart(eu[["country", "pop", "lifeExp"]], heat=["lifeExp"], backend=b),
        "categorical_scatter": lambda b: pc.categorical_scatter(tips, "day", "time", "tip", backend=b),
        "quadrant": lambda b: pc.quadrant(eu, "gdpPercap", "lifeExp", "country", backend=b),
        "heatmap": lambda b: pc.heatmap(pen, corr=True, backend=b),
        "scatter": lambda b: pc.scatter(pen, "bill_length_mm", "bill_depth_mm", "species", trendline="ols", backend=b),
        "bubble": lambda b: pc.bubble(gap07, "gdpPercap", "lifeExp", "pop", "continent", backend=b),
        "connected_scatter": lambda b: pc.connected_scatter(pc.datasets.load("healthexp"), "Spending_USD",
                                                            "Life_Expectancy", "Year", "Country", backend=b),
        "contour": lambda b: pc.contour(pen, "bill_length_mm", "bill_depth_mm", backend=b),
        "hexbin": lambda b: pc.hexbin(pen, "bill_length_mm", "bill_depth_mm", gridsize=10, backend=b),
        "stacked_bar": lambda b: pc.stacked_bar(tips, "day", "tip", "sex", normalize=True, backend=b),
        "diverging_bar": lambda b: pc.diverging_bar(tips, "day", "tip", "smoker", backend=b),
        "population_pyramid": lambda b: pc.population_pyramid(tips, "day", "tip", "sex", backend=b),
        "icon_array": lambda b: pc.icon_array(titanic, "class", backend=b),
        "waffle": lambda b: pc.waffle(titanic, "class", backend=b),
        "pie": lambda b: pc.pie(titanic, "class", backend=b),
        "donut": lambda b: pc.donut(titanic, "class", backend=b),
        "semicircle_donut": lambda b: pc.semicircle_donut(titanic, "class", backend=b),
        "marimekko": lambda b: pc.marimekko(titanic, "class", "sex", backend=b),
        "mosaic": lambda b: pc.mosaic(titanic, "class", "alive", backend=b),
        "treemap": lambda b: pc.treemap(gap07, ["continent", "country"], "pop", backend=b),
        "circular_treemap": lambda b: pc.circular_treemap(eu, ["continent", "country"], "pop", backend=b),
        "convex_treemap": lambda b: pc.convex_treemap(eu, ["country"], "pop", backend=b),
        "dendrogram": lambda b: pc.dendrogram(eu, "country", backend=b),
        "venn": lambda b: pc.venn({"a": {1, 2, 3}, "b": {3, 4}}, backend=b),
        "euler": lambda b: pc.euler({"a": {1, 2, 3}, "b": {3, 4}, "c": {4, 5, 6, 7}}, backend=b),
        "gauge": lambda b: pc.gauge(42, backend=b),
        "sunburst": lambda b: pc.sunburst(titanic.assign(n=1), ["class", "sex"], "n", backend=b),
        "funnel": lambda b: pc.funnel(pd.DataFrame({"s": list("abc"), "n": [10, 6, 2]}), "s", "n", backend=b),
        "line": lambda b: pc.line(stocks, "date", "price", "symbol", backend=b),
        "spline": lambda b: pc.spline(stocks, "date", "price", "symbol", backend=b),
        "step_line": lambda b: pc.step_line(stocks, "date", "price", "symbol", backend=b),
        "area": lambda b: pc.area(stocks, "date", "price", "symbol", backend=b),
        "stacked_area": lambda b: pc.stacked_area(stocks, "date", "price", "symbol", backend=b),
        "streamgraph": lambda b: pc.streamgraph(stocks, "date", "price", "symbol", backend=b),
        "bump": lambda b: pc.bump(stocks, "date", "price", "symbol", backend=b),
        "bump_area": lambda b: pc.bump_area(stocks[stocks.date.dt.month == 1], "date", "price", "symbol", backend=b),
        "candlestick": lambda b: pc.candlestick(ohlc, "Date", "AAPL.Open", "AAPL.High", "AAPL.Low", "AAPL.Close", backend=b),
        "ohlc": lambda b: pc.ohlc(ohlc, "Date", "AAPL.Open", "AAPL.High", "AAPL.Low", "AAPL.Close", backend=b),
        "gantt": lambda b: pc.gantt(gantt, "task", "s", "e", backend=b),
        "barcode": lambda b: pc.barcode(sea, "date", "weather", backend=b),
        "histogram": lambda b: pc.histogram(pen, "body_mass_g", "species", backend=b),
        "density": lambda b: pc.density(pen, "body_mass_g", "species", backend=b),
        "ridgeline": lambda b: pc.ridgeline(pen, "body_mass_g", "species", backend=b),
        "horizon": lambda b: pc.horizon(stocks, "date", "price", "symbol", center="mean", backend=b),
        "radial_histogram": lambda b: pc.radial_histogram(sea, "wind", bins=12, backend=b),
        "strip": lambda b: pc.strip(tips, "day", "tip", backend=b),
        "jitter": lambda b: pc.jitter(tips, "day", "tip", "sex", backend=b),
        "beeswarm": lambda b: pc.beeswarm(tips, "day", "tip", backend=b),
        "heatmap_1d": lambda b: pc.heatmap_1d(sea, "date", "temp_max", backend=b),
        "box": lambda b: pc.box(pen, "species", "body_mass_g", show_stats=True, backend=b),
        "violin": lambda b: pc.violin(pen, "species", "body_mass_g", "sex", backend=b),
        "qq_plot": lambda b: pc.qq_plot(pen, "body_mass_g", backend=b),
        "geo_heatmap": lambda b: pc.geo_heatmap(pc.datasets.load("earthquakes").head(500), "Latitude", "Longitude",
                                                basemap="outline", backend=b),
        "choropleth": lambda b: pc.choropleth(pc.datasets.load("world_gdp"), "CODE", "GDP (BILLIONS)", log=True, backend=b),
        "tile_map": lambda b: pc.tile_map(pc.datasets.load("us_ag_exports"), "code", "beef", backend=b),
        "chord": lambda b: pc.chord(links, "source", "target", "value", backend=b),
        "arc": lambda b: pc.arc(links, value="value", backend=b),
        "sankey": lambda b: pc.sankey(links, backend=b),
        "network": lambda b: pc.network(links, value="value", backend=b),
        "flowchart": lambda b: pc.flowchart([{"id": "a", "label": "A", "shape": "terminal"},
                                            {"id": "b", "label": "B?", "shape": "decision"},
                                            {"id": "c", "label": "C"}], [("a", "b"), ("b", "c", "yes"), ("b", "a", "no")],
                                           backend=b),
        "pair_plot": lambda b: pc.pair_plot(pen, ["bill_length_mm", "bill_depth_mm"], "species", backend=b),
        "joint_plot": lambda b: pc.joint_plot(pen, "bill_length_mm", "bill_depth_mm", "species", backend=b),
        "cluster_map": lambda b: pc.cluster_map(eu[["country", "pop", "lifeExp", "gdpPercap"]], "country", backend=b),
    }


@pytest.fixture(scope="module")
def cases(penguins, tips, titanic, gap07):
    return _cases(penguins, tips, titanic, gap07)


CHART_NAMES = [
    "bar", "column", "grouped_bar", "lollipop", "bullet", "dot_plot", "dumbbell", "pictogram", "icon_chart",
    "range_plot", "radial_bar", "parallel_coordinates", "radar", "nightingale", "waterfall", "matrix_chart",
    "small_multiples", "word_cloud", "slope", "table_chart", "categorical_scatter", "quadrant", "heatmap", "scatter",
    "bubble", "connected_scatter", "contour", "hexbin", "stacked_bar", "diverging_bar", "population_pyramid",
    "icon_array", "waffle", "pie", "donut", "semicircle_donut", "marimekko", "mosaic", "treemap", "circular_treemap",
    "convex_treemap", "dendrogram", "venn", "euler", "gauge", "sunburst", "funnel", "line", "spline", "step_line",
    "area", "stacked_area", "streamgraph", "bump", "bump_area", "candlestick", "ohlc", "gantt", "barcode",
    "histogram", "density", "ridgeline", "horizon", "radial_histogram", "strip", "jitter", "beeswarm",
    "heatmap_1d", "box", "violin", "qq_plot", "geo_heatmap", "choropleth", "tile_map", "chord", "arc", "sankey",
    "network", "flowchart", "pair_plot", "joint_plot", "cluster_map",
]


def test_case_list_covers_registry():
    assert set(CHART_NAMES) == set(pc.catalog().chart)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", CHART_NAMES)
def test_chart_builds(cases, name, backend):
    c = cases[name](backend)
    assert isinstance(c, pc.Chart)
    assert c.kind == name
    if backend == "plotly":
        assert c.is_plotly and len(c.figure.data) > 0
        c.figure.to_json()                       # serialisable for Dash / HTML
    else:
        assert c.backend in ("matplotlib", "seaborn") and len(c.figure.axes) > 0
        png = c.to_image("png", dpi=40)
        assert png[:4] == b"\x89PNG"
        c.close()


# ---------------------------------------------------------- values = data (oracles)
def _mpl_bar_lengths(chart, horizontal):
    ax = chart.figure.axes[0]
    return [p.get_width() if horizontal else p.get_height() for p in ax.patches]


def test_bar_values_match_groupby(penguins):
    expect = penguins.groupby("species")["body_mass_g"].mean()
    c = pc.bar(penguins, "species", "body_mass_g", agg="mean")
    assert _mpl_bar_lengths(c, True) == pytest.approx(list(expect[["Adelie", "Chinstrap", "Gentoo"]]))
    p = pc.bar(penguins, "species", "body_mass_g", agg="mean", backend="plotly")
    assert dict(zip(p.figure.data[0].y, p.figure.data[0].x)) == pytest.approx(expect.to_dict())
    s = pc.bar(penguins, "species", "body_mass_g", agg="mean", backend="seaborn")
    assert sorted(_mpl_bar_lengths(s, True)) == pytest.approx(sorted(expect))
    assert "confidence" in s.meta["error_bars"]


def test_stacked_percent_sums_to_100(titanic):
    c = pc.stacked_bar(titanic.assign(n=1), "class", "n", "alive", normalize=True, backend="plotly")
    tot = np.sum([np.asarray(t.y, float) for t in c.figure.data], axis=0)
    assert tot == pytest.approx(100)
    expect = pd.crosstab(titanic["class"], titanic.alive, normalize="index") * 100
    yes = dict(zip(c.figure.data[1].x, c.figure.data[1].y))
    assert yes["First"] == pytest.approx(expect.loc["First", "yes"])


def test_histogram_counts_match_numpy(penguins):
    v = penguins.body_mass_g.dropna()
    c = pc.histogram(penguins, "body_mass_g", bins=12)
    heights = [p.get_height() for p in c.figure.axes[0].patches]
    assert heights == pytest.approx(np.histogram(v, bins=np.histogram_bin_edges(v, 12))[0])


def test_heatmap_corr_matches_pandas(penguins):
    c = pc.heatmap(penguins, corr=True, method="spearman", backend="plotly")
    expect = penguins.select_dtypes("number").corr(method="spearman").to_numpy()
    assert np.asarray(c.figure.data[0].z) == pytest.approx(expect)


def test_waffle_units_and_pie_values(titanic):
    c = pc.waffle(titanic, "class")
    share = titanic["class"].value_counts(normalize=True) * 100
    assert sum(c.meta["units"].values()) == 100
    for k, u in c.meta["units"].items():
        assert abs(u - share[k]) < 1
    p = pc.pie(titanic, "class", backend="plotly")
    assert dict(zip(p.figure.data[0].labels, p.figure.data[0].values)) == titanic["class"].value_counts().to_dict()


def test_treemap_and_sunburst_values(gap07):
    t = pc.treemap(gap07, ["continent", "country"], "pop", backend="plotly").figure.data[0]
    vals = dict(zip(t.ids, t.values))
    assert vals["Asia"] == pytest.approx(gap07.query("continent=='Asia'")["pop"].sum())
    assert vals["Europe/France"] == pytest.approx(gap07.query("country=='France'")["pop"].iloc[0])


def test_sankey_link_values_preserved():
    links = pd.DataFrame({"source": ["a", "a", "b", "a"], "target": ["b", "c", "c", "b"], "value": [3, 1, 2, 4]})
    s = pc.sankey(links, backend="plotly").figure.data[0]
    labels = list(s.node.label)
    got = {(labels[a], labels[b]): v for a, b, v in zip(s.link.source, s.link.target, s.link.value)}
    assert got == {("a", "b"): 7, ("a", "c"): 1, ("b", "c"): 2}


def test_population_pyramid_sides(tips):
    c = pc.population_pyramid(tips, "day", "tip", "sex", backend="plotly")
    left, right = c.figure.data
    expect = tips.pivot_table(index="day", columns="sex", values="tip", aggfunc="sum")
    assert dict(zip(left.y, -np.asarray(left.x))) == pytest.approx(expect[left.name].to_dict())
    assert dict(zip(right.y, right.x)) == pytest.approx(expect[right.name].to_dict())


def test_scatter_stats_meta(penguins):
    c = pc.scatter(penguins, "flipper_length_mm", "body_mass_g", show_stats=True)
    d = penguins.dropna(subset=["flipper_length_mm", "body_mass_g"])
    assert c.meta["stats"]["pearson_r"] == pytest.approx(d.flipper_length_mm.corr(d.body_mass_g))


def test_mosaic_residuals(titanic):
    c = pc.mosaic(titanic, "class", "alive")
    tab = pd.crosstab(titanic["class"], titanic.alive)
    exp = np.outer(tab.sum(1), tab.sum(0)) / tab.to_numpy().sum()
    resid = (tab - exp) / np.sqrt(exp)
    assert c.meta["residuals"].loc["First", "yes"] == pytest.approx(resid.loc["First", "yes"])


def test_errors_are_helpful(penguins):
    with pytest.raises(KeyError, match="not in data"):
        pc.bar(penguins, "nope", "body_mass_g")
    with pytest.raises(ValueError, match="backend"):
        pc.bar(penguins, "species", "body_mass_g", backend="bokeh")
    with pytest.raises(ValueError, match="exactly 2"):
        pc.dumbbell(penguins, "island", "body_mass_g", "species")


def test_ax_reuse_and_saving(tmp_path, penguins):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2)
    c1 = pc.box(penguins, "species", "body_mass_g", ax=axes[0])
    pc.violin(penguins, "species", "body_mass_g", ax=axes[1], backend="seaborn")
    assert c1.figure is fig
    for ext in ("png", "svg", "pdf", "html"):
        assert c1.save(tmp_path / f"x.{ext}").stat().st_size > 0
    p = pc.bar(penguins, "species", "body_mass_g", backend="plotly")
    assert "plotly" in p.save(tmp_path / "x.html").read_text()
    assert p.save(tmp_path / "x.json").stat().st_size > 0
