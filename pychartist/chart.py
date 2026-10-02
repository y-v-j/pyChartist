"""The :class:`Chart` result object returned by every pyChartist chart function.

It wraps either a matplotlib ``Figure`` or a Plotly ``go.Figure`` and gives both the same API:
``show()``, ``save()``, ``to_html()``, ``to_image()``, ``to_dash()`` and rich notebook display.
"""
from __future__ import annotations

import base64
import io
import warnings
from pathlib import Path


class Chart:
    def __init__(self, figure, backend: str, kind: str, theme, meta: dict | None = None,
                 owns_figure: bool = True):
        self.figure = figure
        self.backend = backend
        self.kind = kind
        self.theme = theme
        self.meta = meta or {}
        self._owns_figure = owns_figure   # False when drawn into a user-supplied matplotlib Axes

    # ------------------------------------------------------------------ helpers
    @property
    def fig(self):
        return self.figure

    @property
    def is_plotly(self) -> bool:
        return self.backend == "plotly"

    @property
    def axes(self):
        """matplotlib Axes list (empty for Plotly charts)."""
        return [] if self.is_plotly else list(self.figure.axes)

    def __repr__(self) -> str:
        extra = f", meta={sorted(self.meta)}" if self.meta else ""
        return f"<Chart kind={self.kind!r} backend={self.backend!r} theme={self.theme.name!r}{extra}>"

    # ------------------------------------------------------------------ display
    def show(self, **kwargs):
        """Display the chart (notebook inline, browser for Plotly, a window for matplotlib)."""
        if self.is_plotly:
            self.figure.show(**kwargs)
            return
        if _in_notebook():
            from IPython.display import display
            display(self)
            return
        import matplotlib.pyplot as plt
        if self._owns_figure:
            # A Figure built without pyplot has no GUI manager; borrow one.
            dummy = plt.figure()
            manager = dummy.canvas.manager
            manager.canvas.figure = self.figure
            self.figure.set_canvas(manager.canvas)
        plt.show(**kwargs)

    def _repr_mimebundle_(self, include=None, exclude=None):
        if self.is_plotly:
            return self.figure._repr_mimebundle_(include, exclude)
        return {"image/png": base64.b64encode(self.to_image("png")).decode(),
                "text/plain": repr(self)}

    # ------------------------------------------------------------------ export
    def to_image(self, format: str = "png", width: int | None = None, height: int | None = None,
                 scale: float = 2, dpi: int | None = None) -> bytes:
        format = format.lower().lstrip(".")
        if self.is_plotly:
            from ._config import get_option
            lw, lh = self.figure.layout.width, self.figure.layout.height
            if lw and lh and (width or height):
                # fixed-canvas figures (word cloud, circle packing...) are laid out for their own
                # size: rescale the image instead of re-flowing the layout
                scale *= (width / lw) if width else (height / lh)
                width, height = lw, lh
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                return self.figure.to_image(
                    format=format, scale=scale,
                    width=width or self.figure.layout.width or get_option("width"),
                    height=height or self.figure.layout.height or get_option("height"))
        buf = io.BytesIO()
        self.figure.savefig(buf, format=format, dpi=dpi or 150)
        return buf.getvalue()

    def save(self, path, width: int | None = None, height: int | None = None, scale: float = 2,
             dpi: int | None = None, include_plotlyjs="cdn") -> Path:
        """Save as .png/.svg/.pdf/.jpg/.webp (both backends) or .html/.json (Plotly; matplotlib
        charts saved as .html embed an SVG)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix.lower().lstrip(".")
        if ext == "html":
            path.write_text(self.to_html(full_html=True, include_plotlyjs=include_plotlyjs),
                            encoding="utf-8")
        elif ext == "json":
            if not self.is_plotly:
                raise ValueError("JSON export is only available for Plotly charts")
            path.write_text(self.figure.to_json(), encoding="utf-8")
        else:
            path.write_bytes(self.to_image(ext, width=width, height=height, scale=scale, dpi=dpi))
        return path

    def to_html(self, full_html: bool = False, include_plotlyjs="cdn") -> str:
        if self.is_plotly:
            return self.figure.to_html(full_html=full_html, include_plotlyjs=include_plotlyjs)
        svg = self.to_image("svg").decode("utf-8")
        svg = svg[svg.find("<svg"):]
        return f"<!doctype html><html><body>{svg}</body></html>" if full_html else svg

    def to_dash(self, id: str | None = None, **kwargs):
        """A Dash component: ``dcc.Graph`` for Plotly charts, ``html.Img`` (PNG) for matplotlib."""
        from dash import dcc, html
        if self.is_plotly:
            props = {"figure": self.figure, "config": {"displaylogo": False}}
            if id is not None:
                props["id"] = id
            props.update(kwargs)
            return dcc.Graph(**props)
        src = "data:image/png;base64," + base64.b64encode(self.to_image("png")).decode()
        style = {"maxWidth": "100%", "height": "auto"}
        style.update(kwargs.pop("style", {}))
        return html.Img(src=src, id=id or f"img-{self.kind}", style=style, **kwargs)

    def close(self):
        """Release a matplotlib figure's memory (no-op for Plotly)."""
        if not self.is_plotly:
            self.figure.clear()


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False
