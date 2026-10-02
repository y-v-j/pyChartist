import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import pychartist as pc  # noqa: E402


@pytest.fixture(scope="session")
def penguins():
    return pc.datasets.load("penguins")


@pytest.fixture(scope="session")
def tips():
    return pc.datasets.load("tips")


@pytest.fixture(scope="session")
def titanic():
    return pc.datasets.load("titanic")


@pytest.fixture(scope="session")
def gap07():
    g = pc.datasets.load("gapminder")
    return g[g.year == 2007].reset_index(drop=True)


@pytest.fixture(autouse=True)
def _reset_options():
    import plotly.io as pio
    old = pio.templates.default
    with pc.options(backend="matplotlib", theme="pychartist"):
        yield
    pio.templates.default = old
    import matplotlib.pyplot as plt
    plt.close("all")


@pytest.fixture(scope="session")
def small_ts():
    s = pc.datasets.load("stocks")
    return s[s.date.dt.year >= 2008]


def frame(**cols):
    return pd.DataFrame(cols)
