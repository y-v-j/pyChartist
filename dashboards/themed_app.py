"""Smallest themed dashboard: ``pychartist.dashboard.make_app`` with a light/dark switch.

Plotly *and* matplotlib/seaborn cards re-render in the theme that matches the page.
Run:  python dashboards/themed_app.py   ->  http://127.0.0.1:8053
"""
import os

import pychartist as pc
from pychartist.dashboard import make_app

pen = pc.datasets.load("penguins")
tips = pc.datasets.load("tips")

app = make_app("Penguins & tips", {
    "Body mass by species (Plotly)": lambda th: pc.violin(pen, "species", "body_mass_g", backend="plotly", theme=th),
    "Bill size (Plotly)": lambda th: pc.scatter(pen, "bill_length_mm", "bill_depth_mm", "species", trendline="ols",
                                               backend="plotly", theme=th),
    "Mean tip with 95% CI (seaborn)": lambda th: pc.grouped_bar(tips, "day", "tip", "sex", agg="mean",
                                                               backend="seaborn", theme=th),
    "Bills per day (matplotlib)": lambda th: pc.beeswarm(tips, "day", "total_bill", theme=th),
}, bootstrap="MINTY", dark_bootstrap="SLATE")
server = app.server

if __name__ == "__main__":
    app.run(debug=bool(os.environ.get("DASH_DEBUG")), port=int(os.environ.get("PORT", 8053)))
