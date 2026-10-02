"""pyChartist - one API for 80+ chart types, rendered static (matplotlib / seaborn) or interactive (Plotly).

    >>> import pychartist as pc
    >>> pen = pc.datasets.load("penguins")
    >>> pc.bar(pen, "species", "body_mass_g", agg="mean")                        # matplotlib
    >>> pc.bar(pen, "species", "body_mass_g", agg="mean", backend="seaborn")     # + 95% CI bars
    >>> pc.violin(pen, "species", "flipper_length_mm", backend="plotly")        # interactive
    >>> pc.catalog()                                                             # every chart
    >>> pc.eda.analyze(pen, "species", "body_mass_g")                           # charts + tests

Every chart function returns a :class:`Chart` with ``show()``, ``save()``, ``to_html()`` and
``to_dash()``. Themes adapt to the host: ``pc.set_theme("dark")``, ``theme="inherit"`` (use the
active matplotlib/seaborn rcParams or Plotly template), host presets such as ``"jupyter-dark"``
or ``"vscode-dark"``, Bootstrap names such as ``"darkly"``, and ``"*-transparent"`` variants.
"""
from . import datasets, layout, stats
from ._config import get_option, options, set_backend, set_option, set_theme
from ._util import fold_other
from .chart import Chart
from .charts.comparison import (bar, bullet, categorical_scatter, column, dot_plot, dumbbell,
                                grouped_bar, icon_chart, lollipop, matrix_chart, nightingale,
                                parallel_coordinates, pictogram, quadrant, radar, radial_bar,
                                range_plot, slope, small_multiples, table_chart, waterfall,
                                word_cloud)
from .charts.correlation import bubble, connected_scatter, contour, heatmap, hexbin, scatter
from .charts.distribution import (beeswarm, box, density, heatmap_1d, histogram, horizon, jitter,
                                  qq_plot, radial_histogram, ridgeline, strip, violin)
from .charts.geo_other import (arc, choropleth, chord, flowchart, geo_heatmap, network, sankey,
                               tile_map)
from .charts.part_to_whole import (circular_treemap, convex_treemap, dendrogram, diverging_bar,
                                   donut, euler, funnel, gauge, icon_array, marimekko, mosaic, pie,
                                   population_pyramid, semicircle_donut, stacked_bar, sunburst,
                                   treemap, venn, waffle)
from .charts.statistical import cluster_map, joint_plot, pair_plot
from .charts.temporal import (area, barcode, bump, bump_area, candlestick, gantt, line, ohlc,
                              spline, stacked_area, step_line, streamgraph)
from .registry import CATEGORIES, catalog, get_spec, plot
from .themes import (THEMES, Theme, available_themes, derive_theme, from_bootstrap,
                     from_matplotlib, from_plotly, get_theme, register_theme)
from . import eda, examples  # noqa: E402  (need the chart functions above)

__version__ = "0.1.0"

__all__ = [n for n in dir() if not n.startswith("_")]
