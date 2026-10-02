"""Freely available example datasets used by the notebooks, dashboards and tests.

Every dataset is downloaded once into ``<project>/datasets/`` (or the directory named by the
``PYCHARTIST_DATASETS`` environment variable) and loaded from disk afterwards::

    >>> from pychartist import datasets
    >>> datasets.fetch_all()            # download everything that is missing
    >>> penguins = datasets.load("penguins")

``datasets.info()`` lists every name with its source URL and licence.
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_SEABORN = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/"
_PLOTLY = "https://raw.githubusercontent.com/plotly/datasets/master/"
_VEGA = "https://cdn.jsdelivr.net/npm/vega-datasets@2/data/"


@dataclass(frozen=True)
class Dataset:
    name: str
    url: str
    filename: str
    description: str
    source: str
    licence: str


_REGISTRY: dict[str, Dataset] = {d.name: d for d in [
    Dataset("penguins", _SEABORN + "penguins.csv", "penguins.csv",
            "Palmer penguins: bill/flipper/mass for 3 species on 3 islands",
            "seaborn-data (Gorman et al. 2014, palmerpenguins)", "CC0"),
    Dataset("tips", _SEABORN + "tips.csv", "tips.csv",
            "Restaurant bills and tips by day/time/sex/smoker",
            "seaborn-data (Bryant & Smith 1995)", "public example data"),
    Dataset("titanic", _SEABORN + "titanic.csv", "titanic.csv",
            "Titanic passengers: class, sex, age, survival",
            "seaborn-data", "public domain"),
    Dataset("flights", _SEABORN + "flights.csv", "flights.csv",
            "Monthly airline passengers 1949-1960 (Box & Jenkins)",
            "seaborn-data", "public domain"),
    Dataset("mpg", _SEABORN + "mpg.csv", "mpg.csv",
            "Auto MPG: fuel economy, horsepower, weight, origin",
            "seaborn-data (UCI Auto MPG)", "CC BY 4.0"),
    Dataset("iris", _SEABORN + "iris.csv", "iris.csv",
            "Fisher's iris measurements", "seaborn-data", "public domain"),
    Dataset("healthexp", _SEABORN + "healthexp.csv", "healthexp.csv",
            "Health spending per capita vs life expectancy, 6 countries 1970-2020",
            "seaborn-data (OECD)", "OECD terms (free reuse)"),
    Dataset("gapminder", _PLOTLY + "gapminderDataFiveYear.csv", "gapminder.csv",
            "Gapminder: population, life expectancy, GDP per capita by country 1952-2007",
            "plotly/datasets (Gapminder)", "CC BY 4.0"),
    Dataset("apple_stock", _PLOTLY + "finance-charts-apple.csv", "apple_stock.csv",
            "AAPL daily OHLC prices 2015-2017", "plotly/datasets", "public example data"),
    Dataset("world_gdp", _PLOTLY + "2014_world_gdp_with_codes.csv", "world_gdp_2014.csv",
            "GDP (billions USD) by country with ISO-3 codes, 2014",
            "plotly/datasets (CIA World Factbook)", "public domain"),
    Dataset("us_ag_exports", _PLOTLY + "2011_us_ag_exports.csv", "us_ag_exports_2011.csv",
            "US agricultural exports by state, 2011", "plotly/datasets (USDA)", "public domain"),
    Dataset("earthquakes", _PLOTLY + "earthquakes-23k.csv", "earthquakes.csv",
            "23k significant earthquakes 1965-2016 (lat/lon/magnitude)",
            "plotly/datasets (USGS via Kaggle)", "public domain (USGS)"),
    Dataset("energy_sankey",
            "https://raw.githubusercontent.com/plotly/plotly.js/master/test/image/mocks/sankey_energy.json",
            "energy_sankey.json", "UK energy flows 2050 projection (nodes + links)",
            "plotly.js test mocks (UK DECC via Mike Bostock)", "MIT / OGL"),
    Dataset("miserables", _VEGA + "miserables.json", "miserables.json",
            "Les Miserables character co-occurrence network",
            "vega-datasets (Knuth, Stanford GraphBase)", "BSD-3"),
    Dataset("population", _VEGA + "population.json", "us_population.json",
            "US census population by age group and sex, 1850-2000",
            "vega-datasets (IPUMS)", "BSD-3"),
    Dataset("barley", _VEGA + "barley.json", "barley.json",
            "Barley yields at 6 Minnesota sites, 1931 vs 1932 (Immer 1934)",
            "vega-datasets", "BSD-3"),
    Dataset("seattle_weather", _VEGA + "seattle-weather.csv", "seattle_weather.csv",
            "Daily Seattle weather 2012-2015", "vega-datasets (NOAA)", "public domain"),
    Dataset("stocks", _VEGA + "stocks.csv", "stocks.csv",
            "Monthly prices of AAPL/AMZN/GOOG/IBM/MSFT 2000-2010", "vega-datasets", "BSD-3"),
    Dataset("unemployment_industries", _VEGA + "unemployment-across-industries.json",
            "unemployment_industries.json",
            "US unemployment counts by industry, monthly 2000-2010",
            "vega-datasets (BLS)", "public domain"),
    Dataset("world_countries",
            "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson",
            "world_countries_110m.geojson", "Natural Earth 1:110m country polygons",
            "Natural Earth", "public domain"),
    Dataset("us_states_geo",
            "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json",
            "us_states.geojson", "US state polygons (GeoJSON)",
            "PublicaMundi MappingAPI (US Census)", "public domain"),
    Dataset("pride_prejudice", "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
            "pride_and_prejudice.txt", "Jane Austen, Pride and Prejudice (full text)",
            "Project Gutenberg #1342", "public domain"),
]}

_MIRRORS = {
    "pride_prejudice": ["https://www.gutenberg.org/files/1342/1342-0.txt",
                        "https://raw.githubusercontent.com/laumann/ds/master/hashing/books/pride-and-prejudice.txt"],
}


def data_dir() -> Path:
    """Directory holding the downloaded files (created on demand)."""
    env = os.environ.get("PYCHARTIST_DATASETS")
    path = Path(env) if env else Path(__file__).resolve().parent.parent / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def names() -> list[str]:
    return sorted(_REGISTRY)


def info() -> pd.DataFrame:
    """Table of all registered datasets, with a column telling whether each is on disk."""
    rows = []
    for d in _REGISTRY.values():
        rows.append({"name": d.name, "file": d.filename, "description": d.description,
                     "source": d.source, "licence": d.licence, "url": d.url,
                     "downloaded": (data_dir() / d.filename).exists()})
    return pd.DataFrame(rows)


def path(name: str, download: bool = True) -> Path:
    """Local path of a dataset, downloading it first if necessary."""
    d = _get(name)
    p = data_dir() / d.filename
    if not p.exists():
        if not download:
            raise FileNotFoundError(f"{p} is missing; call pychartist.datasets.fetch({name!r})")
        fetch(name)
    return p


def fetch(name: str, force: bool = False, timeout: float = 60) -> Path:
    """Download one dataset (tries mirrors when the primary URL fails)."""
    d = _get(name)
    target = data_dir() / d.filename
    if target.exists() and not force:
        return target
    errors = []
    for url in [d.url, *_MIRRORS.get(name, [])]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pychartist-datasets"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = resp.read()
            tmp = target.with_suffix(target.suffix + ".part")
            tmp.write_bytes(payload)
            tmp.replace(target)
            return target
        except Exception as exc:  # network errors differ by platform; try the next mirror
            errors.append(f"{url}: {exc}")
    raise RuntimeError(f"could not download {name!r}:\n  " + "\n  ".join(errors))


def fetch_all(force: bool = False) -> dict[str, str]:
    """Download every registered dataset; returns {name: 'ok' | error message}."""
    status = {}
    for name in _REGISTRY:
        try:
            fetch(name, force=force)
            status[name] = "ok"
        except RuntimeError as exc:
            status[name] = str(exc)
    return status


def load(name: str):
    """Load a dataset: CSV/JSON tables as DataFrames, GeoJSON/graph JSON as dicts, text as str."""
    p = path(name)
    if name == "pride_prejudice":
        return p.read_text(encoding="utf-8-sig", errors="replace")
    if p.suffix == ".csv":
        df = pd.read_csv(p)
        if name in {"apple_stock", "seattle_weather"}:
            df["Date" if name == "apple_stock" else "date"] = pd.to_datetime(
                df["Date" if name == "apple_stock" else "date"])
        if name == "stocks":
            df["date"] = pd.to_datetime(df["date"], format="%b %d %Y")
        return df
    raw = json.loads(p.read_text(encoding="utf-8"))
    if name in {"population", "barley"}:
        return pd.DataFrame(raw)
    if name == "unemployment_industries":
        df = pd.DataFrame(raw)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df
    return raw


def _get(name: str) -> Dataset:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; choose from {names()}") from None


if __name__ == "__main__":
    for k, v in fetch_all().items():
        print(f"{k:26s} {v}")
