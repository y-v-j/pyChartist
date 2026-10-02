"""A realistic demo for every chart, built on the downloaded example datasets.

    >>> import pychartist as pc
    >>> pc.examples.demo("sankey", backend="plotly", theme="dark").show()
    >>> pc.examples.names()

The gallery dashboard (``dashboards/gallery_app.py``) and ``scripts/build_gallery.py`` use these.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from . import datasets as ds


@lru_cache(maxsize=None)
def _d(name):
    return ds.load(name)


def _g07():
    g = _d("gapminder")
    return g[g.year == 2007]


def _kpi():
    return pd.DataFrame({"metric": ["Revenue (k$)", "Profit (k$)", "New customers"], "actual": [270, 22, 1650],
                         "target": [250, 26, 1500], "poor": [150, 10, 1000], "satisfactory": [225, 20, 1400],
                         "good": [300, 30, 2000]})


def _likert():
    rows = {"Easy to use": [5, 10, 20, 40, 25], "Fast": [10, 15, 30, 30, 15], "Reliable": [20, 25, 25, 20, 10],
            "Good docs": [3, 7, 15, 45, 30]}
    levels = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
    return pd.DataFrame([(q, l, n) for q, ns in rows.items() for l, n in zip(levels, ns)],
                        columns=["question", "answer", "respondents"])


def _gantt():
    return pd.DataFrame({
        "task": ["Scope", "Data collection", "Cleaning", "EDA", "Modelling", "Dashboard", "Report"],
        "start": pd.to_datetime(["2026-01-05", "2026-01-12", "2026-02-02", "2026-02-16", "2026-03-02",
                                 "2026-03-23", "2026-04-06"]),
        "end": pd.to_datetime(["2026-01-16", "2026-02-06", "2026-02-20", "2026-03-06", "2026-03-27",
                               "2026-04-10", "2026-04-24"]),
        "phase": ["Plan", "Data", "Data", "Analysis", "Analysis", "Delivery", "Delivery"]})


def _flow():
    steps = pd.DataFrame({"id": ["s", "in", "v", "ok", "fix", "out", "e"],
                          "label": ["Start", "Load data", "Valid?", "Analyse", "Clean data", "Publish", "End"],
                          "shape": ["terminal", "io", "decision", "process", "process", "io", "terminal"]})
    edges = [("s", "in"), ("in", "v"), ("v", "ok", "yes"), ("v", "fix", "no"), ("fix", "v"), ("ok", "out"),
             ("out", "e")]
    return steps, edges


def _energy():
    raw = _d("energy_sankey")["data"][0]
    labels = raw["node"]["label"]
    return pd.DataFrame({"source": [labels[i] for i in raw["link"]["source"]],
                         "target": [labels[i] for i in raw["link"]["target"]], "value": raw["link"]["value"]})


def _titanic_sets():
    t = _d("titanic")
    return {"Survived": set(t.index[t.survived == 1]), "Female": set(t.index[t.sex == "female"]),
            "First class": set(t.index[t.pclass == 1])}


def _region_flows():
    names = ["North", "South", "East", "West", "Centre"]
    m = [[0, 30, 10, 5, 12], [20, 0, 15, 8, 6], [12, 6, 0, 22, 9], [4, 9, 18, 0, 14], [10, 7, 11, 13, 0]]
    return pd.DataFrame(m, index=names, columns=names)


def _sea_month():
    s = _d("seattle_weather")
    return s.assign(month=s.date.dt.strftime("%b"), year=s.date.dt.year)


def _unemp(n=6):
    from ._util import fold_other
    u = _d("unemployment_industries")
    return fold_other(u, "series", "count", n=n)


def _demos():
    import pychartist as pc
    D = {}

    def add(name, title, fn, source):
        D[name] = (title, fn, source)

    pen, tips, titanic, gap = _d("penguins"), _d("tips"), _d("titanic"), _d("gapminder")
    stocks, aapl, health = _d("stocks"), _d("apple_stock"), _d("healthexp")
    # 1. comparison
    add("bar", "Mean body mass by penguin species", lambda **k: pc.bar(pen, "species", "body_mass_g", agg="mean", **k), "penguins")
    add("column", "Total restaurant bills by day", lambda **k: pc.column(tips, "day", "total_bill", **k), "tips")
    add("grouped_bar", "Mean tip by day and sex", lambda **k: pc.grouped_bar(tips, "day", "tip", "sex", agg="mean", **k), "tips")
    add("lollipop", "Most populous countries, 2007", lambda **k: pc.lollipop(_g07().nlargest(12, "pop"), "country", "pop", **k), "gapminder")
    add("bullet", "KPIs against targets", lambda **k: pc.bullet(_kpi(), "metric", "actual", "target", ["poor", "satisfactory", "good"], **k), "synthetic")
    add("dot_plot", "Barley yield by site, 1931 vs 1932", lambda **k: pc.dot_plot(_d("barley"), "yield", "site", "year", **k), "barley")
    add("dumbbell", "Change in barley yield by site", lambda **k: pc.dumbbell(_d("barley"), "site", "yield", "year", **k), "barley")
    add("pictogram", "Penguins measured per species (1 icon = 10)", lambda **k: pc.pictogram(
        pen.species.value_counts().rename_axis("species").reset_index(name="n"), "species", "n", unit=10, **k), "penguins")
    add("icon_chart", "Population by continent, 2007", lambda **k: pc.icon_chart(
        _g07().groupby("continent", as_index=False)["pop"].sum(), "continent", "pop", **k), "gapminder")
    add("range_plot", "Body-mass range per species", lambda **k: pc.range_plot(
        pen.groupby("species", as_index=False).agg(lightest=("body_mass_g", "min"), heaviest=("body_mass_g", "max")),
        "species", "lightest", "heaviest", **k), "penguins")
    add("radial_bar", "Population by continent, 2007", lambda **k: pc.radial_bar(
        _g07().groupby("continent", as_index=False)["pop"].sum(), "continent", "pop", **k), "gapminder")
    add("parallel_coordinates", "Penguin morphology across four measures", lambda **k: pc.parallel_coordinates(
        pen, ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"], "species", **k), "penguins")
    add("radar", "Species profiles (min-max scaled)", lambda **k: pc.radar(
        pen, "species", ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"], **k), "penguins")
    add("nightingale", "Seattle precipitation by month, 2012-2015", lambda **k: pc.nightingale(
        _sea_month(), "month", "precipitation", **k), "seattle_weather")
    add("waterfall", "Profit bridge", lambda **k: pc.waterfall(pd.DataFrame(
        {"item": ["Revenue", "Services", "Cost of sales", "Opex", "Tax"], "k$": [420, 85, -210, -150, -40]}), "item", "k$", **k), "synthetic")
    add("matrix_chart", "Number of tables by day and service", lambda **k: pc.matrix_chart(tips, "day", "time", **k), "tips")
    add("small_multiples", "Stock prices, one panel per company", lambda **k: pc.small_multiples(
        stocks, "symbol", "line", x="date", y="price", cols=3, sharey=False, **k), "stocks")
    add("word_cloud", "Pride and Prejudice: most frequent words", lambda **k: pc.word_cloud(_d("pride_prejudice"), **k), "Project Gutenberg")
    add("slope", "Barley yield by site: 1931 → 1932", lambda **k: pc.slope(
        _d("barley").groupby(["site", "year"], as_index=False)["yield"].sum(), "site", "year", "yield", **k), "barley")
    add("table_chart", "Ten most populous countries, 2007", lambda **k: pc.table_chart(
        _g07().nlargest(10, "pop")[["country", "continent", "pop", "lifeExp", "gdpPercap"]], heat=["lifeExp", "gdpPercap"], **k), "gapminder")
    add("categorical_scatter", "Mean tip by day x service", lambda **k: pc.categorical_scatter(tips, "day", "time", "tip", **k), "tips")
    add("quadrant", "European countries: wealth vs life expectancy, 2007", lambda **k: pc.quadrant(
        _g07()[_g07().continent == "Europe"], "gdpPercap", "lifeExp", "country", **k), "gapminder")
    # 2. correlation
    add("heatmap", "Monthly airline passengers 1949-1960", lambda **k: pc.heatmap(_d("flights"), "year", "month", "passengers", **k), "flights")
    add("scatter", "Flipper length vs body mass", lambda **k: pc.scatter(
        pen, "flipper_length_mm", "body_mass_g", "species", trendline="ols", show_stats=True, **k), "penguins")
    add("bubble", "Wealth, health and population, 2007", lambda **k: pc.bubble(
        _g07(), "gdpPercap", "lifeExp", "pop", "continent", log_x=True, **k), "gapminder")
    add("connected_scatter", "Health spending vs life expectancy, 1970-2020", lambda **k: pc.connected_scatter(
        health, "Spending_USD", "Life_Expectancy", "Year", "Country", **k), "healthexp")
    add("contour", "Bill length vs depth: point density", lambda **k: pc.contour(pen, "bill_length_mm", "bill_depth_mm", points=True, **k), "penguins")
    add("hexbin", "Where 23k significant earthquakes happened", lambda **k: pc.hexbin(_d("earthquakes"), "Longitude", "Latitude", gridsize=45, **k), "USGS")
    # 3. part-to-whole
    add("stacked_bar", "Titanic passengers by class and sex", lambda **k: pc.stacked_bar(
        titanic.assign(n=1), "class", "n", "sex", **k), "titanic")
    add("diverging_bar", "Survey: product statements", lambda **k: pc.diverging_bar(
        _likert(), "question", "respondents", "answer", neutral="Neutral", **k), "synthetic")
    add("population_pyramid", "US population by age and sex, 2000", lambda **k: pc.population_pyramid(
        _d("population").query("year == 2000").assign(sex=lambda d: d.sex.map({1: "Male", 2: "Female"})),
        "age", "people", "sex", percent=True, **k), "IPUMS")
    add("icon_array", "Titanic passengers per 100, by class", lambda **k: pc.icon_array(titanic, "class", **k), "titanic")
    add("waffle", "World population share by continent, 2007", lambda **k: pc.waffle(_g07(), "continent", "pop", **k), "gapminder")
    add("pie", "Port of embarkation", lambda **k: pc.pie(titanic, "embark_town", **k), "titanic")
    add("donut", "World population by continent, 2007", lambda **k: pc.donut(_g07(), "continent", "pop", **k), "gapminder")
    add("semicircle_donut", "Seats by party", lambda **k: pc.semicircle_donut(pd.DataFrame(
        {"party": ["Left", "Centre", "Right", "Other"], "seats": [120, 85, 140, 25]}), "party", "seats", **k), "synthetic")
    add("marimekko", "Titanic passengers: class x person type", lambda **k: pc.marimekko(titanic, "class", "who", **k), "titanic")
    add("mosaic", "Survival by class (residuals from independence)", lambda **k: pc.mosaic(titanic, "class", "alive", **k), "titanic")
    add("treemap", "World population, 2007", lambda **k: pc.treemap(_g07(), ["continent", "country"], "pop", **k), "gapminder")
    add("circular_treemap", "World population, 2007", lambda **k: pc.circular_treemap(_g07(), ["continent", "country"], "pop", **k), "gapminder")
    add("convex_treemap", "Population of Europe, Africa and the Americas, 2007", lambda **k: pc.convex_treemap(
        _g07()[_g07().continent.isin(["Europe", "Africa", "Americas"])], ["continent", "country"], "pop", **k), "gapminder")
    add("dendrogram", "Clustering 40 cars by specification", lambda **k: pc.dendrogram(
        _d("mpg").dropna().sample(40, random_state=1)[["name", "mpg", "horsepower", "weight", "acceleration"]], "name", **k), "mpg")
    add("venn", "Titanic: survivors, women and first class", lambda **k: pc.venn(_titanic_sets(), **k), "titanic")
    add("euler", "Titanic: survivors, women and first class (area-proportional)", lambda **k: pc.euler(_titanic_sets(), **k), "titanic")
    add("gauge", "Service uptime score", lambda **k: pc.gauge(72, label="Uptime score", target=80, **k), "synthetic")
    add("sunburst", "Titanic: class › sex › survival", lambda **k: pc.sunburst(titanic.assign(n=1), ["class", "sex", "alive"], "n", **k), "titanic")
    add("funnel", "Conversion funnel", lambda **k: pc.funnel(pd.DataFrame(
        {"stage": ["Visits", "Sign-ups", "Trials", "Purchases", "Renewals"], "users": [12000, 4300, 2100, 830, 560]}),
        "stage", "users", **k), "synthetic")
    # 4. temporal
    add("line", "Monthly stock prices 2000-2010", lambda **k: pc.line(stocks, "date", "price", "symbol", **k), "stocks")
    add("spline", "Yearly airline passengers", lambda **k: pc.spline(
        _d("flights").groupby("year", as_index=False)["passengers"].sum(), "year", "passengers", **k), "flights")
    add("step_line", "Life expectancy, USA vs Japan", lambda **k: pc.step_line(
        health[health.Country.isin(["USA", "Japan"])], "Year", "Life_Expectancy", "Country", **k), "healthexp")
    add("area", "Yearly airline passengers", lambda **k: pc.area(
        _d("flights").groupby("year", as_index=False)["passengers"].sum(), "year", "passengers", **k), "flights")
    add("stacked_area", "US unemployment by industry (top 6 + other)", lambda **k: pc.stacked_area(
        _unemp(6), "date", "count", "series", **k), "BLS")
    add("streamgraph", "US unemployment by industry (top 7 + other)", lambda **k: pc.streamgraph(
        _unemp(7), "date", "count", "series", **k), "BLS")
    add("bump", "GDP per capita rank among seven European countries", lambda **k: pc.bump(
        gap[gap.country.isin(["Germany", "France", "Italy", "Spain", "Sweden", "Norway", "Poland"])],
        "year", "gdpPercap", "country", **k), "gapminder")
    add("bump_area", "Stock price ranking each January", lambda **k: pc.bump_area(
        stocks[stocks.date.dt.month == 1], "date", "price", "symbol", **k), "stocks")
    add("candlestick", "AAPL, last 60 trading days", lambda **k: pc.candlestick(
        aapl.tail(60), "Date", "AAPL.Open", "AAPL.High", "AAPL.Low", "AAPL.Close", **k), "plotly/datasets")
    add("ohlc", "AAPL, last 60 trading days", lambda **k: pc.ohlc(
        aapl.tail(60), "Date", "AAPL.Open", "AAPL.High", "AAPL.Low", "AAPL.Close", **k), "plotly/datasets")
    add("gantt", "Analytics project schedule", lambda **k: pc.gantt(_gantt(), "task", "start", "end", "phase", today="2026-03-10", **k), "synthetic")
    add("barcode", "Seattle: snow and fog days, 2012-2015", lambda **k: pc.barcode(
        _d("seattle_weather").query("weather in ['snow', 'fog']"), "date", "weather", **k), "seattle_weather")
    # 5. distribution
    add("histogram", "Flipper length by species", lambda **k: pc.histogram(pen, "flipper_length_mm", "species", kde=True, **k), "penguins")
    add("density", "Flipper length by species", lambda **k: pc.density(pen, "flipper_length_mm", "species", **k), "penguins")
    add("ridgeline", "Seattle daily maximum temperature by month", lambda **k: pc.ridgeline(
        _sea_month(), "temp_max", "month", **k), "seattle_weather")
    add("horizon", "Stock prices relative to each company's mean", lambda **k: pc.horizon(
        stocks, "date", "price", "symbol", center="mean", **k), "stocks")
    add("radial_histogram", "Seattle precipitation by day of year", lambda **k: pc.radial_histogram(
        _d("seattle_weather").assign(doy=lambda d: d.date.dt.dayofyear), "doy", bins=24, period=366, **k), "seattle_weather")
    add("strip", "Restaurant bills by day", lambda **k: pc.strip(tips, "day", "total_bill", **k), "tips")
    add("jitter", "Restaurant bills by day and sex", lambda **k: pc.jitter(tips, "day", "total_bill", "sex", **k), "tips")
    add("beeswarm", "Body mass of every penguin", lambda **k: pc.beeswarm(pen, "species", "body_mass_g", **k), "penguins")
    add("heatmap_1d", "Seattle monthly mean maximum temperature (stripes)", lambda **k: pc.heatmap_1d(
        _d("seattle_weather").assign(m=lambda d: d.date.dt.to_period("M").astype(str)), "m", "temp_max", **k), "seattle_weather")
    add("box", "Body mass by species (with test)", lambda **k: pc.box(pen, "species", "body_mass_g", show_stats=True, **k), "penguins")
    add("violin", "Flipper length by species and sex", lambda **k: pc.violin(pen, "species", "flipper_length_mm", "sex", **k), "penguins")
    add("qq_plot", "Is penguin body mass normal?", lambda **k: pc.qq_plot(pen, "body_mass_g", **k), "penguins")
    # 6. geo & other
    add("geo_heatmap", "Earthquake density 1965-2016", lambda **k: pc.geo_heatmap(
        _d("earthquakes"), "Latitude", "Longitude", basemap="outline", **k), "USGS")
    add("choropleth", "Life expectancy, 2007", lambda **k: pc.choropleth(
        _g07().merge(_d("world_gdp"), left_on="country", right_on="COUNTRY"), "CODE", "lifeExp", **k), "gapminder + CIA")
    add("tile_map", "US agricultural exports by state, 2011 (M$)", lambda **k: pc.tile_map(
        _d("us_ag_exports"), "code", "total exports", **k), "USDA")
    add("chord", "Migration between regions", lambda **k: pc.chord(_region_flows(), **k), "synthetic")
    add("arc", "Les Misérables co-occurrence", lambda **k: pc.arc(_d("miserables"), **k), "Knuth")
    add("sankey", "UK energy flows (2050 projection)", lambda **k: pc.sankey(_energy(), **k), "UK DECC")
    add("network", "Les Misérables co-occurrence network", lambda **k: pc.network(_d("miserables"), **k), "Knuth")
    add("flowchart", "Data publishing workflow", lambda **k: pc.flowchart(*_flow(), **k), "synthetic")
    # EDA (seaborn figure-level + plotly)
    add("pair_plot", "Penguin measurements, pairwise", lambda **k: pc.pair_plot(
        pen, ["bill_length_mm", "bill_depth_mm", "flipper_length_mm"], "species", **k), "penguins")
    add("joint_plot", "Bill length vs depth with marginals", lambda **k: pc.joint_plot(
        pen, "bill_length_mm", "bill_depth_mm", "species", **k), "penguins")
    add("cluster_map", "30 cars clustered by specification (z-scores)", lambda **k: pc.cluster_map(
        _d("mpg").dropna().sample(30, random_state=2)[["name", "mpg", "horsepower", "weight", "acceleration", "displacement"]],
        "name", **k), "mpg")
    return D


_CACHE: dict = {}


def demos() -> dict:
    if not _CACHE:
        _CACHE.update(_demos())
    return _CACHE


def names() -> list[str]:
    return list(demos())


def demo(name: str, backend: str | None = None, theme=None, title: bool = True, **kwargs):
    """Build the demo chart ``name`` on ``backend`` / ``theme``."""
    t, fn, _src = demos()[name]
    kw = {k: v for k, v in dict(backend=backend, theme=theme).items() if v is not None}
    if title:
        kw["title"] = t
    kw.update(kwargs)
    return fn(**kw)


def info(name: str) -> dict:
    t, _fn, src = demos()[name]
    return {"title": t, "source": src}
