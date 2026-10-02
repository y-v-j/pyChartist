"""Generate the example notebooks in notebooks/ (then execute them with scripts/run_notebooks.sh)."""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent.parent / "notebooks"
SETUP = """import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
import pychartist as pc
import plotly.io as pio
pio.renderers.default = "plotly_mimetype+notebook_connected"   # renders in JupyterLab/VS Code/Colab; plotly.js from CDN"""


def nb(name, title, intro, cells):
    book = nbf.v4.new_notebook()
    book.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"}
    book.cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n{intro}"), nbf.v4.new_code_cell(SETUP)]
    for kind, src in cells:
        book.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    OUT.mkdir(exist_ok=True)
    nbf.write(book, OUT / f"{name}.ipynb")


M, C = "md", "code"

nb("01_quickstart", "pyChartist quick start",
   "One function per chart type; every call returns a `Chart` that displays inline, saves to "
   "PNG/SVG/PDF/HTML and drops into Dash. Pick the renderer with `backend=`.", [
    (M, "## Data\nAll example data are freely available and downloaded once into `datasets/`."),
    (C, "pc.datasets.info()[['name', 'description', 'licence', 'downloaded']]"),
    (C, "pen = pc.datasets.load('penguins')\npen.head()"),
    (M, "## The same chart, three backends\n`matplotlib` (static, publication), `seaborn` (static, with "
        "statistical estimation - here 95% bootstrap CI error bars) and `plotly` (interactive)."),
    (C, "pc.bar(pen, 'species', 'body_mass_g', agg='mean', title='matplotlib')"),
    (C, "c = pc.bar(pen, 'species', 'body_mass_g', agg='mean', backend='seaborn', title='seaborn: mean ± 95% CI')\n"
        "print(c.meta)\nc"),
    (C, "pc.bar(pen, 'species', 'body_mass_g', agg='mean', backend='plotly', title='plotly: hover me')"),
    (M, "## Set defaults once"),
    (C, "pc.set_backend('plotly')\npc.violin(pen, 'species', 'flipper_length_mm', title='Default backend is now Plotly')"),
    (C, "pc.set_backend('matplotlib')\n"
        "with pc.options(backend='seaborn', theme='dark'):\n"
        "    c = pc.violin(pen, 'species', 'flipper_length_mm', 'sex', title='Temporarily seaborn + dark')\nc"),
    (M, "## Saving and exporting"),
    (C, "from pathlib import Path\nout = Path('_output'); out.mkdir(exist_ok=True)\n"
        "c = pc.scatter(pen, 'bill_length_mm', 'bill_depth_mm', 'species', trendline='ols')\n"
        "for ext in ['png', 'svg', 'pdf', 'html']:\n    print(c.save(out / f'scatter.{ext}'))\n"
        "p = pc.scatter(pen, 'bill_length_mm', 'bill_depth_mm', 'species', backend='plotly')\n"
        "print(p.save(out / 'scatter_interactive.html'), p.save(out / 'scatter_plotly.png', width=900, height=550))"),
    (M, "## Composing with matplotlib\nPass `ax=` to draw into your own figure (matplotlib/seaborn backends)."),
    (C, "import matplotlib.pyplot as plt\nfig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')\n"
        "pc.histogram(pen, 'body_mass_g', 'species', ax=axes[0])\n"
        "pc.box(pen, 'species', 'body_mass_g', ax=axes[1], backend='seaborn')\n"
        "pc.beeswarm(pen, 'species', 'body_mass_g', ax=axes[2])\nfig"),
    (M, "## The catalogue\nEvery chart from the reference list (77) plus five EDA extras, with alternative names and "
        "guidance on when (not) to use it. `pc.plot()` accepts any name or alias."),
    (C, "cat = pc.catalog()\nprint(len(cat), 'charts')\ncat.groupby('category').size()"),
    (C, "cat[['chart', 'function', 'backends', 'alt_names']].head(12)"),
    (C, "pc.plot('spider chart', pen, 'species', ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'])"),
    (M, "## Every chart has a ready-made demo"),
    (C, "pc.examples.demo('sankey', backend='plotly')"),
])

nb("02_comparison_charts", "Comparison charts",
   "Bar family, dot plots, dumbbells, pictograms, radial/polar forms, parallel coordinates, waterfall, slope, "
   "small multiples, word cloud, tables and quadrants.", [
    (C, "pen = pc.datasets.load('penguins'); tips = pc.datasets.load('tips'); barley = pc.datasets.load('barley')\n"
        "gap = pc.datasets.load('gapminder'); g07 = gap[gap.year == 2007]"),
    (M, "## Bars and columns\nSorted, labelled at the tip only when there are few bars; colours follow the entity."),
    (C, "pc.bar(g07.nlargest(10, 'pop'), 'country', 'pop', sort='desc', title='Ten most populous countries, 2007')"),
    (C, "pc.grouped_bar(tips, 'day', 'tip', 'sex', agg='mean', backend='seaborn', title='Mean tip (95% CI)')"),
    (C, "pc.lollipop(g07[g07.continent == 'Europe'].nlargest(15, 'gdpPercap'), 'country', 'gdpPercap', "
        "title='GDP per capita, Europe 2007')"),
    (M, "## Paired values: dot plot, dumbbell, range, slope"),
    (C, "pc.dot_plot(barley, 'yield', 'site', 'year', title='Barley yield by site (Cleveland dot plot)')"),
    (C, "pc.dumbbell(barley, 'site', 'yield', 'year', backend='plotly', title='1931 → 1932: the Morris anomaly')"),
    (C, "pc.slope(barley.groupby(['site', 'year'], as_index=False)['yield'].sum(), 'site', 'year', 'yield',\n"
        "         highlight=['Morris'], title='Only Morris increased')"),
    (M, "## Targets and changes: bullet and waterfall"),
    (C, "pc.examples.demo('bullet')"),
    (C, "pc.examples.demo('waterfall', backend='plotly')"),
    (M, "## Icons (no SVG files needed - icons are polygons)"),
    (C, "pc.pictogram(pen.species.value_counts().rename_axis('species').reset_index(name='n'), 'species', 'n', unit=10,\n"
        "             icon='person', title='Penguins measured (1 icon = 10)')"),
    (C, "pc.icon_chart(g07.groupby('continent', as_index=False)['pop'].sum(), 'continent', 'pop', icon='circle',\n"
        "              title='Population by continent (area-proportional)')"),
    (M, "## Many dimensions: radar and parallel coordinates"),
    (C, "dims = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']\n"
        "pc.radar(pen, 'species', dims, backend='plotly', title='Species profiles')"),
    (C, "pc.parallel_coordinates(pen, dims, 'species', backend='plotly', title='Brush the axes to filter')"),
    (M, "## Circular forms (use sparingly)"),
    (C, "pc.examples.demo('nightingale')"),
    (C, "pc.examples.demo('radial_bar', backend='plotly')"),
    (M, "## Small multiples instead of spaghetti"),
    (C, "stocks = pc.datasets.load('stocks')\n"
        "pc.small_multiples(stocks, 'symbol', 'area', x='date', y='price', cols=3, sharey=False, color_by='symbol')"),
    (M, "## Text, tables, matrices, quadrants"),
    (C, "c = pc.word_cloud(pc.datasets.load('pride_prejudice'), max_words=100, title='Pride and Prejudice')\n"
        "print(c.meta['words'].head(5)[['word', 'count']])\nc"),
    (C, "pc.table_chart(g07.nlargest(10, 'pop')[['country', 'pop', 'lifeExp', 'gdpPercap']], heat=['lifeExp', 'gdpPercap'],\n"
        "               backend='plotly', title='Conditional formatting per column')"),
    (C, "pc.matrix_chart(tips, 'day', 'time', 'tip', title='Total tips by day x service')"),
    (C, "pc.quadrant(g07[g07.continent == 'Europe'], 'gdpPercap', 'lifeExp', 'country',\n"
        "            quadrant_labels=('healthy, poorer', 'healthy & rich', 'less healthy & poorer', 'richer, less healthy'))"),
])

nb("03_correlation_and_distribution", "Relationships and distributions",
   "Scatter/bubble/hexbin/contour for two variables; histograms, KDEs, ridgelines, swarms, box and violin "
   "plots for one. The seaborn backend adds its statistical estimation (CI bands, KDEs, regression fits).", [
    (C, "pen = pc.datasets.load('penguins'); g07 = pc.datasets.load('gapminder').query('year == 2007')\n"
        "quakes = pc.datasets.load('earthquakes')"),
    (C, "c = pc.scatter(pen, 'flipper_length_mm', 'body_mass_g', 'species', trendline='ols', show_stats=True)\n"
        "print(c.meta['stats'])\nc"),
    (C, "pc.scatter(pen, 'flipper_length_mm', 'body_mass_g', 'species', trendline='lowess', backend='seaborn')"),
    (C, "pc.bubble(g07, 'gdpPercap', 'lifeExp', 'pop', 'continent', label=None, log_x=True, backend='plotly',\n"
        "          title='Wealth, health and population (2007)')"),
    (C, "pc.examples.demo('connected_scatter')"),
    (M, "## Too many points? Bin them."),
    (C, "pc.hexbin(quakes, 'Longitude', 'Latitude', gridsize=45, title='23k earthquakes, hexagonal bins')"),
    (C, "pc.contour(pen, 'bill_length_mm', 'bill_depth_mm', points=True, backend='seaborn')"),
    (C, "pc.heatmap(pen, corr=True, method='spearman', title='Spearman correlation')"),
    (C, "pc.joint_plot(pen, 'bill_length_mm', 'bill_depth_mm', 'species')"),
    (C, "pc.pair_plot(pen, ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm'], 'species')"),
    (M, "## Distributions"),
    (C, "pc.histogram(pen, 'flipper_length_mm', 'species', kde=True, backend='seaborn')"),
    (C, "pc.density(pen, 'flipper_length_mm', 'species', backend='plotly')"),
    (C, "pc.examples.demo('ridgeline')"),
    (C, "pc.beeswarm(pen, 'species', 'body_mass_g', 'sex', title='Every penguin, no overlap')"),
    (C, "pc.box(pen, 'species', 'body_mass_g', points='all', show_stats=True, backend='plotly')"),
    (C, "pc.violin(pen, 'species', 'flipper_length_mm', 'sex', split=True, backend='seaborn')"),
    (C, "pc.examples.demo('horizon')"),
    (C, "pc.examples.demo('heatmap_1d', title='Seattle stripes: monthly mean max temperature')"),
    (C, "pc.examples.demo('radial_histogram', backend='plotly')"),
    (C, "pc.qq_plot(pen, 'body_mass_g')"),
])

nb("04_part_to_whole_and_hierarchy", "Part-to-whole and hierarchical charts",
   "Stacks, waffles and icon arrays; pies done carefully; Marimekko and mosaic; treemaps in three geometries "
   "(squarified, circle-packed, convex polygons); sets; gauges; funnels.", [
    (C, "titanic = pc.datasets.load('titanic'); g07 = pc.datasets.load('gapminder').query('year == 2007')"),
    (C, "pc.stacked_bar(titanic.assign(n=1), 'class', 'n', 'alive', normalize=True, orient='h',\n"
        "               title='Survival share by class')"),
    (C, "pc.examples.demo('diverging_bar', backend='plotly')"),
    (C, "pc.examples.demo('population_pyramid')"),
    (C, "pc.icon_array(titanic, 'class', title='Of every 100 passengers…')"),
    (C, "pc.waffle(g07, 'continent', 'pop', title='World population, 2007 (1 square = 1%)')"),
    (C, "pc.donut(g07, 'continent', 'pop', backend='plotly')"),
    (C, "pc.examples.demo('semicircle_donut')"),
    (C, "pc.marimekko(titanic, 'class', 'who', title='Class x person type (width = class size)')"),
    (C, "c = pc.mosaic(titanic, 'class', 'alive', title='Mosaic: red = more than independence predicts')\n"
        "print(c.meta['chi2'])\nc"),
    (M, "## Three treemap geometries"),
    (C, "pc.treemap(g07, ['continent', 'country'], 'pop', backend='plotly', title='Click to zoom')"),
    (C, "pc.circular_treemap(g07, ['continent', 'country'], 'pop')"),
    (C, "pc.convex_treemap(g07[g07.continent.isin(['Europe', 'Africa', 'Americas'])], ['continent', 'country'], 'pop')"),
    (C, "pc.sunburst(titanic.assign(n=1), ['class', 'sex', 'alive'], 'n', backend='plotly')"),
    (C, "c = pc.examples.demo('dendrogram')\nprint(c.meta['clusters'].value_counts())\nc"),
    (C, "pc.cluster_map(pc.datasets.load('mpg').dropna().sample(30, random_state=2)"
        "[['name', 'mpg', 'horsepower', 'weight', 'acceleration']], 'name')"),
    (M, "## Sets"),
    (C, "sets = {'Survived': set(titanic.index[titanic.survived == 1]),\n"
        "        'Female': set(titanic.index[titanic.sex == 'female']),\n"
        "        'First class': set(titanic.index[titanic.pclass == 1])}\npc.venn(sets)"),
    (C, "pc.euler(sets, backend='plotly', title='Area-proportional (Euler)')"),
    (M, "## Single KPIs and pipelines"),
    (C, "pc.gauge(72, label='Uptime score', target=80, bands=[(60, 'critical'), (80, 'warning'), (100, 'good')])"),
    (C, "pc.examples.demo('funnel', backend='plotly')"),
])

nb("05_time_series", "Data over time",
   "Lines (with direct end labels), splines, steps, areas, stacked areas and streamgraphs, rankings over time, "
   "financial OHLC/candlesticks, Gantt schedules and barcode event plots.", [
    (C, "stocks = pc.datasets.load('stocks'); flights = pc.datasets.load('flights')\n"
        "unemp = pc.datasets.load('unemployment_industries'); aapl = pc.datasets.load('apple_stock')"),
    (C, "pc.line(stocks, 'date', 'price', 'symbol', backend='plotly', title='Hover for a unified tooltip')"),
    (C, "pc.line(stocks[stocks.symbol.isin(['AAPL', 'MSFT', 'IBM'])], 'date', 'price', 'symbol',\n"
        "        title='≤ 4 series are labelled at their ends')"),
    (C, "sea = pc.datasets.load('seattle_weather')\n"
        "pc.line(sea.assign(month=sea.date.dt.month), 'month', 'temp_max', backend='seaborn',\n"
        "        title='seaborn: mean with 95% CI band')"),
    (C, "pc.spline(flights.groupby('year', as_index=False)['passengers'].sum(), 'year', 'passengers')"),
    (C, "pc.examples.demo('step_line')"),
    (M, "## Composition over time\nMore than 8 series? Fold the small ones into *Other* to keep colours distinguishable."),
    (C, "top = pc.fold_other(unemp, 'series', 'count', n=6)\n"
        "pc.stacked_area(top, 'date', 'count', 'series', backend='plotly')"),
    (C, "pc.streamgraph(pc.fold_other(unemp, 'series', 'count', n=7), 'date', 'count', 'series')"),
    (M, "## Rankings"),
    (C, "pc.examples.demo('bump', backend='plotly')"),
    (C, "pc.examples.demo('bump_area')"),
    (M, "## Finance"),
    (C, "pc.candlestick(aapl.tail(90), 'Date', 'AAPL.Open', 'AAPL.High', 'AAPL.Low', 'AAPL.Close', backend='plotly')"),
    (C, "pc.ohlc(aapl.tail(40), 'Date', 'AAPL.Open', 'AAPL.High', 'AAPL.Low', 'AAPL.Close')"),
    (M, "## Schedules and events"),
    (C, "pc.examples.demo('gantt', backend='plotly')"),
    (C, "pc.examples.demo('barcode')"),
])

nb("06_geo_flows_networks", "Maps, flows and networks",
   "Choropleths (per-capita rates!), tile maps, density maps; Sankey, chord and arc diagrams; force-directed "
   "networks and flowcharts. matplotlib maps draw the downloaded GeoJSON directly - no geopandas needed.", [
    (C, "g07 = pc.datasets.load('gapminder').query('year == 2007').merge(pc.datasets.load('world_gdp'),\n"
        "                                                            left_on='country', right_on='COUNTRY')\n"
        "us = pc.datasets.load('us_ag_exports')"),
    (C, "pc.choropleth(g07, 'CODE', 'lifeExp', backend='plotly', title='Life expectancy, 2007')"),
    (C, "pc.choropleth(g07, 'CODE', 'gdpPercap', log=True, title='GDP per capita (log10), matplotlib + Natural Earth')"),
    (C, "pc.choropleth(us, 'code', 'total exports', scope='usa', log=True, backend='plotly')"),
    (C, "pc.tile_map(us, 'code', 'total exports', title='Equal-size tiles: small states stay visible')"),
    (C, "quakes = pc.datasets.load('earthquakes')\n"
        "pc.geo_heatmap(quakes, 'Latitude', 'Longitude', basemap='outline', title='Earthquake density')"),
    (M, "Interactive tile basemap (needs internet for map tiles):"),
    (C, "pc.geo_heatmap(quakes.sample(5000, random_state=0), 'Latitude', 'Longitude', 'Magnitude', radius=6,\n"
        "               backend='plotly', zoom=0.4)"),
    (M, "## Flows"),
    (C, "pc.examples.demo('sankey', backend='plotly')"),
    (C, "pc.examples.demo('sankey')"),
    (C, "pc.examples.demo('chord', backend='plotly')"),
    (M, "## Networks"),
    (C, "mis = pc.datasets.load('miserables')\npc.arc(mis, title='Les Misérables: arcs grouped by community')"),
    (C, "pc.network(mis, backend='plotly', title='Drag to zoom; size = weighted degree')"),
    (C, "steps = pd.DataFrame({'id': ['s', 'q', 'a', 'b', 'e'], 'label': ['Order received', 'In stock?', 'Ship',\n"
        "                      'Back-order', 'Done'], 'shape': ['terminal', 'decision', 'process', 'process', 'terminal']})\n"
        "pc.flowchart(steps, [('s', 'q'), ('q', 'a', 'yes'), ('q', 'b', 'no'), ('b', 'a'), ('a', 'e')])"),
])

nb("07_eda_workflow", "Exploratory data analysis: charts + statistical tests",
   "`pc.eda` follows the variable-type decision table: each call chooses the charts *and* the tests.\n\n"
   "| Analysis | Charts | Tests |\n|---|---|---|\n"
   "| 1 numerical | histogram, box, Q-Q | Shapiro-Wilk, Kolmogorov-Smirnov |\n"
   "| 1 categorical | bar, pie | χ² goodness of fit |\n"
   "| 2 numerical | scatter, correlation heatmap | Pearson r, Spearman ρ |\n"
   "| categorical × numerical | grouped box / violin | Welch t / Mann-Whitney (2 groups), ANOVA / Kruskal-Wallis (3+) |\n"
   "| 2 categorical | stacked bar, mosaic | χ² independence, Fisher's exact |", [
    (C, "pen = pc.datasets.load('penguins'); titanic = pc.datasets.load('titanic')"),
    (C, "pc.eda.overview(pen)"),
    (M, "## 1 numerical variable"),
    (C, "pc.eda.analyze(pen, 'body_mass_g')"),
    (M, "## 1 categorical variable (is the split uniform?)"),
    (C, "pc.eda.analyze(pen, 'island')"),
    (M, "## 2 numerical variables"),
    (C, "pc.eda.analyze(pen, 'flipper_length_mm', 'body_mass_g', backend='plotly')"),
    (M, "## Group differences\nThe recommendation switches to the non-parametric test when any group fails "
        "Shapiro-Wilk."),
    (C, "r = pc.eda.analyze(pen, 'species', 'bill_length_mm', backend='seaborn')\nr"),
    (C, "r.tests"),
    (C, "pc.eda.analyze(pen[pen.species == 'Adelie'], 'sex', 'body_mass_g')"),
    (M, "## 2 categorical variables"),
    (C, "pc.eda.analyze(titanic, 'class', 'alive')"),
    (M, "## Using the tests directly"),
    (C, "from pychartist import stats\nprint(stats.shapiro_wilk(pen.body_mass_g))\n"
        "print(stats.chi2_independence(titanic, 'sex', 'alive'))\nstats.eda_tests(pen, 'species', 'flipper_length_mm')"),
])

nb("08_themes_and_host_compatibility", "Themes: fitting into the host",
   "Charts should look native wherever they are shown. pyChartist offers: built-in light/dark themes; "
   "`theme='inherit'` to adopt the host's active matplotlib/seaborn style or Plotly template; presets for "
   "Jupyter, VS Code, Colab, GitHub and every Bootstrap (Dash) theme; `*-transparent` variants; and builders "
   "for your own brand.", [
    (C, "pen = pc.datasets.load('penguins')\nsorted(pc.available_themes())[:20]"),
    (M, "## Light and dark"),
    (C, "pc.scatter(pen, 'bill_length_mm', 'bill_depth_mm', 'species', theme='light', title='light')"),
    (C, "pc.scatter(pen, 'bill_length_mm', 'bill_depth_mm', 'species', theme='dark', title='dark')"),
    (C, "pc.violin(pen, 'species', 'body_mass_g', theme='dark', backend='plotly', title='dark (plotly)')"),
    (M, "## Inherit the host's styling\nWhatever matplotlib style or seaborn theme is active is used as-is: "
        "background, text and the series colour cycle."),
    (C, "import matplotlib.pyplot as plt, seaborn as sns\n"
        "with plt.style.context('ggplot'):\n    c = pc.histogram(pen, 'body_mass_g', 'species', theme='inherit', title='inherit: ggplot')\nc"),
    (C, "with sns.axes_style('darkgrid'), sns.color_palette('Set2'):\n"
        "    c = pc.box(pen, 'species', 'body_mass_g', theme='inherit', backend='seaborn', title='inherit: seaborn darkgrid + Set2')\nc"),
    (C, "pio.templates.default = 'plotly_dark'\n"
        "c = pc.bar(pen, 'species', 'body_mass_g', agg='mean', theme='inherit', backend='plotly', title='inherit: plotly_dark')\n"
        "pio.templates.default = 'plotly'\nc"),
    (M, "## Host presets\nMatch notebook/editor surfaces so the chart background blends in."),
    (C, "pc.line(pc.datasets.load('stocks'), 'date', 'price', 'symbol', theme='vscode-dark', title='vscode-dark')"),
    (C, "pc.treemap(pc.datasets.load('gapminder').query('year == 2007'), ['continent', 'country'], 'pop',\n"
        "           theme='jupyter-dark', title='jupyter-dark')"),
    (M, "## Bootstrap / Dash themes and transparency\nIn dashboards use a `-transparent` theme so the card shows through."),
    (C, "pc.donut(pen, 'species', theme='solar', backend='plotly', title='Bootswatch Solar')"),
    (C, "pc.from_bootstrap('DARKLY').with_transparent()"),
    (M, "## Your own brand theme\n`derive_theme` builds all chrome from a background (and optional text/palette); "
        "`register_theme` makes it available by name."),
    (C, "brand = pc.derive_theme('acme', surface='#0f1b2d', ink='#e8edf5',\n"
        "                        categorical=['#4cc9f0', '#f72585', '#ffd166', '#06d6a0'], register=True)\n"
        "pc.stacked_bar(pc.datasets.load('titanic').assign(n=1), 'class', 'n', 'who', theme='acme', title='theme=\"acme\"')"),
    (C, "pc.from_matplotlib('Solarize_Light2')"),
    (M, "## Library looks: d3, nvd3 and chartist palettes"),
    (C, "for t in ['d3', 'nvd3', 'chartist']:\n"
        "    pc.grouped_bar(pen, 'island', 'body_mass_g', 'species', agg='mean', theme=t, title=t).show()"),
    (M, "## Dashboards\n`pychartist.dashboard.make_app` builds a Dash app whose charts re-render when the page "
        "switches light/dark. See `dashboards/` for complete apps; run one inline with "
        "`app.run(jupyter_mode='inline')`."),
    (C, "from pychartist.dashboard import make_app\n"
        "app = make_app('Penguins', {'Mass': lambda th: pc.violin(pen, 'species', 'body_mass_g', backend='plotly', theme=th)})\n"
        "app.pychartist['light'].name, app.pychartist['dark'].name"),
])

if __name__ == "__main__":
    print("wrote", sorted(p.name for p in OUT.glob("*.ipynb")))
