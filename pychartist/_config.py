"""Global options: default backend, theme and figure size."""
from __future__ import annotations

from contextlib import contextmanager

BACKENDS = ("matplotlib", "seaborn", "plotly")
_ALIASES = {"mpl": "matplotlib", "static": "matplotlib", "sns": "seaborn", "px": "plotly",
            "interactive": "plotly", "go": "plotly"}

_OPTIONS = {
    "backend": "matplotlib",
    "theme": "pychartist",
    "figsize": (7.5, 4.8),          # inches, matplotlib
    "width": 820,                    # pixels, plotly
    "height": 520,
}


def normalize_backend(backend: str | None) -> str:
    b = (backend or _OPTIONS["backend"]).lower()
    b = _ALIASES.get(b, b)
    if b not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS} (or aliases {sorted(_ALIASES)}), got {backend!r}")
    return b


def get_option(key: str):
    return _OPTIONS[key]


def set_option(**kwargs) -> None:
    for k, v in kwargs.items():
        if k not in _OPTIONS:
            raise KeyError(f"unknown option {k!r}; options are {sorted(_OPTIONS)}")
        if k == "backend":
            v = normalize_backend(v)
        _OPTIONS[k] = v


def set_backend(backend: str) -> None:
    """Default backend for every chart call.

    ``"matplotlib"`` (static), ``"seaborn"`` (static; seaborn's statistical functions where a chart
    has a seaborn equivalent - confidence intervals, KDEs, regression fits - else matplotlib) or
    ``"plotly"`` (interactive)."""
    set_option(backend=backend)


def set_theme(theme: str) -> None:
    from .themes import get_theme
    get_theme(theme)  # validate
    set_option(theme=theme)


@contextmanager
def options(**kwargs):
    """Temporarily override options: ``with pc.options(backend="plotly"): ...``."""
    old = {k: _OPTIONS[k] for k in kwargs}
    set_option(**kwargs)
    try:
        yield
    finally:
        _OPTIONS.update(old)
