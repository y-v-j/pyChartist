"""Theme compatibility with host packages, and the Dash helpers."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.io as pio
import pytest
from matplotlib.colors import to_hex

import pychartist as pc
from pychartist.themes import is_dark_color


def test_light_dark_and_transparent(penguins):
    light = pc.bar(penguins, "species", "body_mass_g", theme="light")
    dark = pc.bar(penguins, "species", "body_mass_g", theme="dark")
    assert to_hex(light.figure.get_facecolor()) == "#fcfcfb"
    assert to_hex(dark.figure.get_facecolor()) == "#1a1a19"
    p = pc.bar(penguins, "species", "body_mass_g", theme="dark-transparent", backend="plotly")
    assert p.figure.layout.template.layout.paper_bgcolor == "rgba(0,0,0,0)"
    t = pc.bar(penguins, "species", "body_mass_g", theme="light-transparent")
    assert t.figure.get_facecolor()[3] == 0


def test_inherit_matplotlib_style(penguins):
    with plt.style.context("dark_background"):
        th = pc.get_theme("inherit")
        assert th.inherit and th.dark
        c = pc.histogram(penguins, "body_mass_g", "species", theme="inherit")
        cycle = [to_hex(d["color"]) for d in mpl.rcParams["axes.prop_cycle"]]
        face = [to_hex(p.get_facecolor()[:3]) for p in c.figure.axes[0].patches[:1]]
    assert face[0] == cycle[0]                         # series colour came from the host's cycle
    assert to_hex(c.figure.get_facecolor()[:3]) == "#000000"


def test_inherit_seaborn_palette(penguins):
    import seaborn as sns
    with sns.axes_style("darkgrid"), sns.color_palette("Set2"):
        c = pc.box(penguins, "species", "body_mass_g", theme="inherit", backend="seaborn")
        assert to_hex(c.figure.axes[0].get_facecolor()) == to_hex(mpl.rcParams["axes.facecolor"])
        assert c.theme.categorical[0] == to_hex(sns.color_palette("Set2")[0])


def test_inherit_plotly_template(penguins):
    pio.templates.default = "plotly_dark"
    c = pc.bar(penguins, "species", "body_mass_g", theme="inherit", backend="plotly")
    assert c.figure.layout.template.layout.paper_bgcolor == pio.templates["plotly_dark"].layout.paper_bgcolor
    assert c.theme.dark


def test_named_and_derived_themes():
    assert is_dark_color(pc.get_theme("darkly").surface)
    assert not is_dark_color(pc.get_theme("flatly").surface)
    th = pc.derive_theme("brand", "#102030", "#e0e0e0")
    assert th.dark and th.surface == "#102030"
    assert pc.from_bootstrap("https://cdn.jsdelivr.net/npm/bootswatch@5.3.3/dist/cyborg/bootstrap.min.css").name == "cyborg"
    assert {"jupyter-dark", "vscode-light", "github-dark", "solar", "d3", "chartist"} <= set(pc.available_themes())
    from_mpl = pc.from_matplotlib("ggplot")
    assert to_hex(from_mpl.surface) == to_hex(plt.style.library["ggplot"]["axes.facecolor"])


def test_set_theme_and_options(penguins):
    with pc.options(theme="vscode-dark", backend="plotly"):
        c = pc.bar(penguins, "species", "body_mass_g")
        assert c.is_plotly and c.theme.name == "vscode-dark"
    assert pc.get_option("backend") == "matplotlib"


def test_validated_palette_order():
    """The default categorical order should keep adjacent pairs apart for CVD readers (OKLab)."""
    def oklab(hexc):
        r, g, b = (np.array(mpl.colors.to_rgb(hexc)))
        lin = np.where(np.array([r, g, b]) <= 0.04045, np.array([r, g, b]) / 12.92,
                       ((np.array([r, g, b]) + 0.055) / 1.055) ** 2.4)
        l, m, s = np.cbrt(np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                                    [0.2119034982, 0.6806995451, 0.1073969566],
                                    [0.0883024619, 0.2817188376, 0.6299787005]]) @ lin)
        return np.array([0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                         1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                         0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s])
    cat = pc.get_theme("light").categorical
    d = [np.linalg.norm(oklab(a) - oklab(b)) * 100 for a, b in zip(cat, cat[1:])]
    assert min(d) >= 15        # normal-vision floor between neighbours


def test_dash_components_and_app(penguins):
    pytest.importorskip("dash_bootstrap_components")
    from dash import dcc, html

    from pychartist.dashboard import chart_card, make_app
    p = pc.bar(penguins, "species", "body_mass_g", backend="plotly")
    assert isinstance(p.to_dash(id="g"), dcc.Graph)
    m = pc.bar(penguins, "species", "body_mass_g")
    assert isinstance(m.to_dash(), html.Img) and m.to_dash().src.startswith("data:image/png;base64,")
    chart_card(p, "Body mass", "source: palmerpenguins")
    app = make_app("Test", {"bar": lambda th: pc.bar(penguins, "species", "body_mass_g", backend="plotly", theme=th),
                            "box": lambda th: pc.box(penguins, "species", "body_mass_g", theme=th)})
    assert app.pychartist["dark"].dark and not app.pychartist["light"].dark
    # the re-render callback returns one component per slot, in the dark theme
    cb = [v for k, v in app.callback_map.items() if "pc-slot" in k][0]
    out = cb["callback"].__wrapped__(True, [{"type": "pc-slot", "index": 0}, {"type": "pc-slot", "index": 1}])
    assert len(out) == 2
    assert out[0].figure.layout.template.layout.paper_bgcolor == "rgba(0,0,0,0)"


def test_eda_results(penguins, titanic):
    r1 = pc.eda.analyze(penguins, "body_mass_g")
    assert r1.kind == "1 numerical" and len(r1.charts) == 3 and "Shapiro-Wilk" in set(r1.tests.test)
    r2 = pc.eda.analyze(titanic, "class", "alive", backend="plotly")
    assert r2.kind == "2 categorical" and all(c.is_plotly for c in r2.charts)
    r3 = pc.eda.analyze(penguins, "species", "flipper_length_mm", backend="seaborn")
    assert "Kruskal-Wallis H" in set(r3.tests.test)
    assert "<h4>" in r3._repr_html_()
    r4 = pc.eda.analyze(penguins, "bill_length_mm", "bill_depth_mm")
    assert set(r4.tests.test) == {"Pearson r", "Spearman rho"}
    pc.eda.overview(penguins)
