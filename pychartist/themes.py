"""Themes: one set of colour/ink/surface tokens rendered to both matplotlib and Plotly.

Built-in themes
---------------
``"light"`` / ``"pychartist"``   default light theme (validated colour-blind-safe series order)
``"dark"`` / ``"pychartist-dark"`` the same hues stepped for a dark surface
``"*-transparent"``              any theme with no background, so charts show the page behind them
``"jupyter-light"``, ``"jupyter-dark"``, ``"vscode-light"``, ``"vscode-dark"``, ``"colab-light"``,
``"colab-dark"``, ``"github-light"``, ``"github-dark"``
                                 match the surfaces of common notebook / editor hosts
``"bootstrap"``, ``"darkly"``, ``"cyborg"``, ``"slate"``, ``"solar"``, ``"superhero"``, ``"flatly"``,
``"lux"``, ``"minty"``, ``"vapor"``, ...
                                 match Dash Bootstrap Components themes (see :func:`from_bootstrap`)
``"d3"``, ``"nvd3"``, ``"chartist"`` those libraries' default series colours (not CVD-validated)

Compatibility with the host package
-----------------------------------
``theme="inherit"`` builds the theme at call time from whatever the host has active and leaves
the host's styling in charge:

* matplotlib backend - the current ``rcParams`` (``plt.style.use(...)``, ``sns.set_theme(...)``,
  a Jupyter/VS Code style) supply background, text and the series colour cycle; pyChartist pushes
  no rc overrides of its own.
* Plotly backend - ``plotly.io.templates.default`` supplies colourway/background/fonts and the
  figure keeps that template.

``from_matplotlib(style)``, ``from_plotly(template)``, ``from_bootstrap(name)`` and
``derive_theme(surface=...)`` build persistent themes from those sources;
``register_theme`` makes them available by name.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_hex, to_rgb, to_rgba


@dataclass(frozen=True)
class Theme:
    name: str
    categorical: tuple[str, ...]
    sequential: tuple[str, ...]          # low -> high magnitude
    diverging: tuple[str, str, str]      # negative pole, neutral midpoint, positive pole
    surface: str
    ink: str                              # primary text
    ink_secondary: str
    muted: str                            # axis labels / ticks
    grid: str
    baseline: str
    status: dict = field(default_factory=lambda: {
        "good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"})
    font: str = "DejaVu Sans"
    dark: bool = False
    transparent: bool = False             # no background fill: the host page shows through
    inherit: bool = False                 # host owns styling: no rc / template pushed
    plotly_base: str | None = None        # template to keep when inherit=True

    # ------------------------------------------------------------------ colours
    def color(self, i: int) -> str:
        return self.categorical[i % len(self.categorical)]

    def colors(self, n: int) -> list[str]:
        return [self.color(i) for i in range(n)]

    def seq_cmap(self) -> LinearSegmentedColormap:
        return LinearSegmentedColormap.from_list(f"{self.name}-seq", list(self.sequential))

    def div_cmap(self) -> LinearSegmentedColormap:
        return LinearSegmentedColormap.from_list(f"{self.name}-div", list(self.diverging))

    def cat_cmap(self) -> ListedColormap:
        return ListedColormap(list(self.categorical), name=f"{self.name}-cat")

    def seq_colorscale(self) -> list[list]:
        n = len(self.sequential)
        return [[i / (n - 1), c] for i, c in enumerate(self.sequential)]

    def div_colorscale(self) -> list[list]:
        return [[0.0, self.diverging[0]], [0.5, self.diverging[1]], [1.0, self.diverging[2]]]

    def seq_steps(self, n: int) -> list[str]:
        """n ordinal steps from the sequential ramp, skipping the near-surface end."""
        cmap = self.seq_cmap()
        lo = 0.25 if n > 1 else 0.6
        return [to_hex(cmap(lo + (1 - lo) * i / max(n - 1, 1))) for i in range(n)]

    @property
    def bg(self) -> str:
        """Background for matplotlib ('none' when transparent)."""
        return "none" if self.transparent else self.surface

    @property
    def plotly_bg(self) -> str:
        return "rgba(0,0,0,0)" if self.transparent else self.surface

    @property
    def gap(self) -> str:
        """Colour of the 2px separating gap between touching marks."""
        return self.surface

    def with_transparent(self, transparent: bool = True) -> "Theme":
        suffix = "-transparent"
        name = self.name if self.name.endswith(suffix) or not transparent else self.name + suffix
        return replace(self, name=name, transparent=transparent)

    # ------------------------------------------------------------- matplotlib
    def mpl_rc(self) -> dict:
        if self.inherit:
            return {}
        return {
            "figure.facecolor": self.bg,
            "axes.facecolor": self.bg,
            "savefig.facecolor": self.bg,
            "savefig.transparent": self.transparent,
            "axes.edgecolor": self.baseline,
            "axes.labelcolor": self.ink_secondary,
            "axes.titlecolor": self.ink,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": mpl.cycler(color=list(self.categorical)),
            "grid.color": self.grid,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": self.muted,
            "ytick.color": self.muted,
            "xtick.labelcolor": self.ink_secondary,
            "ytick.labelcolor": self.ink_secondary,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.color": self.ink,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": self.ink_secondary,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "lines.markersize": 7,
            "patch.edgecolor": self.surface,
            "font.family": "sans-serif",
            "font.sans-serif": [self.font, "DejaVu Sans", "Arial"],
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }

    # ------------------------------------------------------------------ plotly
    def plotly_template(self):
        import plotly.graph_objects as go
        axis = dict(gridcolor=self.grid, linecolor=self.baseline, zerolinecolor=self.baseline,
                    tickcolor=self.muted, tickfont=dict(color=self.ink_secondary),
                    title=dict(font=dict(color=self.ink_secondary)), showline=True,
                    ticks="outside", ticklen=4, automargin=True)
        return go.layout.Template(layout=dict(
            colorway=list(self.categorical),
            paper_bgcolor=self.plotly_bg, plot_bgcolor=self.plotly_bg,
            font=dict(family=f"system-ui, -apple-system, 'Segoe UI', {self.font}, sans-serif",
                      color=self.ink, size=12),
            title=dict(font=dict(size=16, color=self.ink), x=0.0, xanchor="left", xref="paper"),
            xaxis=axis, yaxis=axis,
            polar=dict(bgcolor=self.plotly_bg,
                       angularaxis=dict(gridcolor=self.grid, linecolor=self.baseline,
                                        tickfont=dict(color=self.ink_secondary)),
                       radialaxis=dict(gridcolor=self.grid, linecolor=self.baseline,
                                       tickfont=dict(color=self.ink_secondary))),
            legend=dict(font=dict(color=self.ink_secondary), bgcolor="rgba(0,0,0,0)"),
            colorscale=dict(sequential=self.seq_colorscale(), diverging=self.div_colorscale()),
            coloraxis=dict(colorbar=dict(tickfont=dict(color=self.ink_secondary), outlinewidth=0)),
            hoverlabel=dict(bgcolor=self.surface, font=dict(color=self.ink),
                            bordercolor=self.baseline),
            barcornerradius=3,
            margin=dict(l=60, r=30, t=60, b=50),
            geo=dict(bgcolor=self.plotly_bg, landcolor=self.grid, lakecolor=self.plotly_bg,
                     showframe=False, coastlinecolor=self.baseline, subunitcolor=self.surface,
                     countrycolor=self.surface),
            map=dict(style="carto-darkmatter" if self.dark else "carto-positron"),
        ))


# =============================================================== construction
_SEQ_BLUE = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
_CAT_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
_CAT_DARK = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767")


def _mix(a: str, b: str, t: float) -> str:
    """Blend colour a toward b by fraction t."""
    ra, rb = to_rgb(a), to_rgb(b)
    return to_hex(tuple(x + (y - x) * t for x, y in zip(ra, rb)))


def _lum(c: str) -> float:
    def lin(v):
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(v) for v in to_rgb(c))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def is_dark_color(c: str) -> bool:
    return _lum(c) < 0.18


def derive_theme(name: str, surface: str, ink: str | None = None, *, categorical=None,
                 sequential=None, diverging=None, font: str | None = None,
                 transparent: bool = False, register: bool = False) -> Theme:
    """Build a complete theme from a background colour (and optionally text colour/palette).

    Chrome (secondary ink, muted, grid, baseline) is mixed from ``surface`` and ``ink``; the
    series palette and ramps default to the validated light or dark steps depending on how
    dark ``surface`` is.
    """
    surface = to_hex(to_rgba(surface)[:3])
    dark = is_dark_color(surface)
    ink = to_hex(to_rgb(ink)) if ink else ("#ffffff" if dark else "#0b0b0b")
    seq = tuple(sequential) if sequential else (tuple(reversed(_SEQ_BLUE)) if dark else _SEQ_BLUE)
    if diverging:
        div = tuple(diverging)
    else:
        div = ("#3987e5", _mix(surface, ink, 0.12), "#e66767") if dark else \
              ("#2a78d6", _mix(surface, ink, 0.06), "#e34948")
    th = Theme(
        name=name,
        categorical=tuple(categorical) if categorical else (_CAT_DARK if dark else _CAT_LIGHT),
        sequential=seq, diverging=div, surface=surface, ink=ink,
        ink_secondary=_mix(ink, surface, 0.30), muted=_mix(ink, surface, 0.50),
        grid=_mix(surface, ink, 0.10 if dark else 0.11), baseline=_mix(surface, ink, 0.22),
        font=font or "DejaVu Sans", dark=dark, transparent=transparent)
    if register:
        register_theme(th)
    return th


THEMES: dict[str, Theme] = {}
_ALIASES = {"light": "pychartist", "default": "pychartist", "pychartist-light": "pychartist",
            "dark": "pychartist-dark"}


def register_theme(theme: Theme) -> Theme:
    """Make a theme available by name (also registers its ``-transparent`` variant)."""
    THEMES[theme.name] = theme
    if not theme.transparent:
        t = theme.with_transparent()
        THEMES[t.name] = t
    _REGISTERED_PLOTLY.pop(theme.name, None)
    return theme


_REGISTERED_PLOTLY: dict[str, Theme] = {}

register_theme(Theme(
    name="pychartist", categorical=_CAT_LIGHT, sequential=_SEQ_BLUE,
    diverging=("#2a78d6", "#f0efec", "#e34948"),
    surface="#fcfcfb", ink="#0b0b0b", ink_secondary="#52514e", muted="#898781",
    grid="#e1e0d9", baseline="#c3c2b7"))
register_theme(Theme(
    name="pychartist-dark", categorical=_CAT_DARK, sequential=tuple(reversed(_SEQ_BLUE)),
    diverging=("#3987e5", "#383835", "#e66767"),
    surface="#1a1a19", ink="#ffffff", ink_secondary="#c3c2b7", muted="#898781",
    grid="#2c2c2a", baseline="#383835", dark=True))
for _n, _t in {"light": "pychartist", "dark": "pychartist-dark"}.items():
    _ALIASES[f"{_n}-transparent"] = f"{_t}-transparent"

# Host presets: notebook / editor surfaces (background, foreground, font)
HOST_SURFACES = {
    "jupyter-light": ("#ffffff", "#000000"),
    "jupyter-dark": ("#111111", "#ffffff"),     # JupyterLab Dark --jp-layout-color0
    "vscode-light": ("#ffffff", "#3b3b3b"),
    "vscode-dark": ("#1f1f1f", "#cccccc"),      # VS Code "Dark Modern" editor
    "colab-light": ("#ffffff", "#212121"),
    "colab-dark": ("#383838", "#d5d5d5"),
    "github-light": ("#ffffff", "#1f2328"),
    "github-dark": ("#0d1117", "#e6edf3"),
}
for _n, (_s, _i) in HOST_SURFACES.items():
    register_theme(derive_theme(_n, _s, _i))

# Dash Bootstrap Components / Bootswatch themes: (body background, body text, primary, font)
BOOTSTRAP_THEMES = {
    "bootstrap": ("#ffffff", "#212529", "#0d6efd", None),
    "cerulean": ("#ffffff", "#495057", "#2fa4e7", None),
    "cosmo": ("#ffffff", "#373a3c", "#2780e3", None),
    "flatly": ("#ffffff", "#212529", "#2c3e50", None),
    "journal": ("#ffffff", "#222222", "#eb6864", None),
    "litera": ("#ffffff", "#343a40", "#4582ec", None),
    "lumen": ("#ffffff", "#222222", "#158cba", None),
    "lux": ("#ffffff", "#55595c", "#1a1a1a", None),
    "materia": ("#ffffff", "#444444", "#2196f3", None),
    "minty": ("#ffffff", "#888888", "#78c2ad", None),
    "pulse": ("#ffffff", "#444444", "#593196", None),
    "sandstone": ("#ffffff", "#3e3f3a", "#325d88", None),
    "simplex": ("#fcfcfc", "#212529", "#d9230f", None),
    "spacelab": ("#ffffff", "#777777", "#446e9b", None),
    "united": ("#ffffff", "#333333", "#e95420", None),
    "yeti": ("#ffffff", "#222222", "#008cba", None),
    "zephyr": ("#ffffff", "#495057", "#3459e6", None),
    "darkly": ("#222222", "#ffffff", "#375a7f", None),
    "cyborg": ("#060606", "#adafae", "#2a9fd6", None),
    "slate": ("#272b30", "#aaaaaa", "#3a3f44", None),
    "solar": ("#002b36", "#839496", "#b58900", None),
    "superhero": ("#0f2537", "#ebebeb", "#df6919", None),
    "vapor": ("#1a0933", "#32fbe2", "#6f42c1", None),
    "quartz": ("#686dc3", "#ffffff", "#e83283", None),
    "morph": ("#d9e3f1", "#7b8ab8", "#378dfc", None),
    "sketchy": ("#ffffff", "#212529", "#333333", None),
}


def from_bootstrap(name: str, register: bool = True) -> Theme:
    """Theme matching a Dash Bootstrap Components theme.

    ``name`` may be the theme name (``"DARKLY"``), or a ``dbc.themes.X`` stylesheet URL.
    """
    key = name.rstrip("/").split("/")[-2] if name.startswith("http") else name
    key = key.lower()
    if key not in BOOTSTRAP_THEMES:
        raise KeyError(f"unknown Bootstrap theme {name!r}; known: {sorted(BOOTSTRAP_THEMES)}")
    bg, fg, _primary, font = BOOTSTRAP_THEMES[key]
    return derive_theme(key, bg, fg, font=font, register=register)


for _n in BOOTSTRAP_THEMES:
    from_bootstrap(_n)


def from_matplotlib(style=None, name: str | None = None, inherit: bool = False) -> Theme:
    """Theme from the active matplotlib rcParams, or from a named style / rc dict.

    Works with ``plt.style.use(...)``, ``seaborn.set_theme(...)`` and editor/notebook styles.
    With ``inherit=True`` pyChartist will not push its own rc settings when drawing.
    """
    import matplotlib.pyplot as plt
    if style is None:
        rc = mpl.rcParams
        return _theme_from_rc(dict(rc), name or "matplotlib-inherit", inherit)
    with plt.style.context(style):
        return _theme_from_rc(dict(mpl.rcParams), name or f"mpl-{style}", inherit)


def _theme_from_rc(rc: dict, name: str, inherit: bool) -> Theme:
    face = rc.get("axes.facecolor", "white")
    fig_face = rc.get("figure.facecolor", face)
    transparent = (str(face).lower() in {"none", "transparent"}) or to_rgba(face)[3] == 0
    surface = to_hex(to_rgba(fig_face if transparent else face)[:3]) if not (
        transparent and to_rgba(fig_face)[3] == 0) else "#ffffff"
    ink = rc.get("text.color", "black")
    cycle = [to_hex(c["color"]) for c in rc["axes.prop_cycle"] if "color" in c] if \
        "axes.prop_cycle" in rc else None
    base = derive_theme(name, surface, ink, categorical=cycle or None, transparent=transparent)
    grid = rc.get("grid.color", base.grid)
    try:
        grid = to_hex(grid)
    except ValueError:
        grid = base.grid
    edge = rc.get("axes.edgecolor", base.baseline)
    return replace(base, grid=grid, baseline=to_hex(edge), inherit=inherit,
                   ink_secondary=to_hex(rc.get("axes.labelcolor", base.ink_secondary))
                   if rc.get("axes.labelcolor") not in (None, "auto") else base.ink_secondary)


def from_plotly(template=None, name: str | None = None, inherit: bool = False) -> Theme:
    """Theme from a Plotly template (name, Template, or the current ``pio.templates.default``)."""
    import plotly.io as pio
    tname = template if isinstance(template, str) else None
    if template is None:
        tname = pio.templates.default
        template = pio.templates[tname] if tname else None
    elif isinstance(template, str):
        # composed names like "plotly_dark+presentation": the first one carries the colours
        template = pio.templates[template.split("+")[0]]
    lay = template.layout if template is not None else None
    surface = (lay.plot_bgcolor or lay.paper_bgcolor) if lay is not None else None
    surface = _plotly_color(surface) or "#ffffff"
    ink = _plotly_color(lay.font.color) if lay is not None and lay.font else None
    colorway = [c for c in (lay.colorway or [])] if lay is not None else []
    th = derive_theme(name or f"plotly-{tname or 'custom'}", surface, ink,
                      categorical=colorway or None)
    if lay is not None and lay.xaxis and lay.xaxis.gridcolor:
        th = replace(th, grid=_plotly_color(lay.xaxis.gridcolor) or th.grid)
    return replace(th, inherit=inherit, plotly_base=tname)


def _plotly_color(c):
    if not c:
        return None
    c = str(c).strip()
    if c.startswith("rgb"):
        nums = [float(v) for v in c[c.index("(") + 1:c.index(")")].split(",")[:3]]
        return to_hex(tuple(v / 255 for v in nums))
    try:
        return to_hex(c)
    except ValueError:
        return None


# d3.schemeCategory10 (also the nvd3 default series colours)
register_theme(replace(THEMES["pychartist"], name="d3", categorical=(
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf"), sequential=("#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"),
    diverging=("#2166ac", "#f7f7f7", "#b2182b"), surface="#ffffff"))
register_theme(replace(THEMES["d3"], name="nvd3", grid="#e5e5e5"))
# Chartist.js series colours a..i
register_theme(replace(THEMES["pychartist"], name="chartist", categorical=(
    "#d70206", "#f05b4f", "#f4c63d", "#d17905", "#453d3f", "#59922b", "#0544d3", "#6b0392",
    "#e6805e"), surface="#ffffff", grid="#e0e0e0", baseline="#999999"))


def get_theme(theme: str | Theme | None = None, backend: str = "matplotlib") -> Theme:
    """Resolve a theme name. ``"inherit"`` reads the host's active matplotlib / Plotly styling."""
    from ._config import get_option
    if isinstance(theme, Theme):
        return theme
    name = theme or get_option("theme")
    if name == "inherit":
        return from_plotly(inherit=True) if backend == "plotly" else from_matplotlib(inherit=True)
    name = _ALIASES.get(name, name)
    try:
        return THEMES[name]
    except KeyError:
        raise KeyError(f"unknown theme {name!r}; available: {available_themes()}") from None


def available_themes() -> list[str]:
    return sorted(set(THEMES) | set(_ALIASES) | {"inherit"})


def plotly_template_name(theme: Theme) -> str:
    """Register the theme as a named Plotly template (once) and return its name."""
    import plotly.io as pio
    if theme.inherit:
        return theme.plotly_base or pio.templates.default
    if _REGISTERED_PLOTLY.get(theme.name) != theme or theme.name not in pio.templates:
        pio.templates[theme.name] = theme.plotly_template()
        _REGISTERED_PLOTLY[theme.name] = theme
    return theme.name
