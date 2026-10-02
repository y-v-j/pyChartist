# Themes and host compatibility

A pyChartist theme is a small set of tokens: series palette, sequential and diverging ramps,
surface, text inks, grid and baseline, and status colours. The same tokens are rendered as
matplotlib `rcParams` and as a Plotly template, so a theme looks the same on every backend.

## Choosing a theme

| You are… | Use |
|---|---|
| writing a report on white paper | default (`"light"`) |
| using a dark notebook / editor | `"jupyter-dark"`, `"vscode-dark"`, `"colab-dark"`, `"github-dark"` (or plain `"dark"`) |
| already styling matplotlib (`plt.style.use`, `sns.set_theme`) | `theme="inherit"` |
| already styling Plotly (`pio.templates.default = ...`) | `theme="inherit"` with `backend="plotly"` |
| building a Dash app with a Bootswatch theme | the theme's name (`"flatly"`, `"darkly"`, `"solar"`, …), usually `-transparent` |
| embedding in a page whose background you don't control | any theme + `-transparent` |
| matching d3 / nvd3 / Chartist examples | `"d3"`, `"nvd3"`, `"chartist"` |
| shipping a brand look | `pc.derive_theme(...)` + `pc.register_theme(...)` |

`pc.available_themes()` lists all names; `pc.set_theme(name)` changes the default and `theme=` overrides it per call.

## `theme="inherit"`

The theme is built at call time from the host's active styling, and pyChartist then leaves that
styling in charge.

* **matplotlib/seaborn backends.** The current `rcParams` supply the figure and axes background,
  text colours, grid and the `axes.prop_cycle` series colours. This works with `plt.style.use`,
  `plt.style.context`, `sns.set_theme`, `sns.axes_style` and `sns.color_palette`. pyChartist sets
  no rc values of its own.
* **Plotly backend.** `plotly.io.templates.default` (for example `plotly_dark`,
  `simple_white` or your own template) supplies the colourway and background. The figure keeps
  that template.

Charts that colour by magnitude (heatmaps, choropleths) or by sign (diverging bars) derive
their ramps from the inherited surface, so they stay readable on a dark host.

## Light and dark

`"light"` and `"dark"` use the same eight hues, stepped separately for each surface. The dark
variant is chosen and validated for a dark surface, not inverted from the light one. Series
order is fixed and colours follow the category, so filtering never repaints the remaining
series. Each step was checked for:

* the lightness band;
* colour-blind separation between neighbouring series (OKLab ΔE ≥ 8);
* a normal-vision floor between neighbours (ΔE ≥ 15).

The test suite checks the last of these.

| slot | light | dark |
|---|---|---|
| 1 | `#2a78d6` | `#3987e5` |
| 2 | `#eb6834` | `#d95926` |
| 3 | `#1baf7a` | `#199e70` |
| 4 | `#eda100` | `#c98500` |
| 5 | `#e87ba4` | `#d55181` |
| 6 | `#008300` | `#008300` |
| 7 | `#4a3aa7` | `#9085e9` |
| 8 | `#e34948` | `#e66767` |

For scatter, choropleth and other all-pairs comparisons, the first three slots are the safe set.
Beyond eight categories a warning suggests `pc.fold_other(df, col, value, n=7)` or `small_multiples`.

## Building your own

```python
th = pc.derive_theme("acme", surface="#0f1b2d", ink="#e8edf5",
                     categorical=["#4cc9f0", "#f72585", "#ffd166", "#06d6a0"], register=True)
pc.bar(df, "x", "y", theme="acme")

pc.from_matplotlib("ggplot")          # a theme from a matplotlib style (or the current rcParams)
pc.from_plotly("simple_white")        # from a Plotly template
pc.from_bootstrap("SUPERHERO")        # from a dbc theme name or stylesheet URL
th.with_transparent()                 # no background fill
```

`derive_theme` mixes the secondary ink, muted, grid and baseline colours from the surface and
ink. It chooses the light or dark validated palette by the surface's luminance.

## Dash

```python
from pychartist.dashboard import make_app, chart_theme_for
chart_theme_for("DARKLY")   # transparent theme matching the Darkly Bootstrap theme
```

`make_app(title, {card_title: lambda theme: Chart}, bootstrap="FLATLY", dark_bootstrap="DARKLY")`
adds a light/dark switch. The switch flips Bootstrap 5.3's `data-bs-theme` and re-renders every
card with the matching chart theme. Plotly cards become `dcc.Graph`; matplotlib and seaborn
cards become PNG `html.Img`.
