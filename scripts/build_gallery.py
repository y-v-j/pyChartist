"""Render every demo chart into gallery/ and regenerate docs/CATALOG.md, docs/GALLERY.md, datasets/SOURCES.md
and the thumbnail grid in README.md.

    python scripts/build_gallery.py            # static PNGs (matplotlib / seaborn) + docs
    python scripts/build_gallery.py --html     # also interactive Plotly HTML (plotly.js from CDN)

Run it under a memory cap, e.g.
    systemd-run --user --scope -p MemoryMax=5G -p MemorySwapMax=0 python scripts/build_gallery.py --html
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pychartist as pc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GALLERY, DOCS = ROOT / "gallery", ROOT / "docs"
SLUG = {v: k for k, v in pc.CATEGORIES.items()}


def build_images(html: bool) -> list[str]:
    cat = pc.catalog()
    failures = []
    for _, row in cat.iterrows():
        name, folder = row.chart, GALLERY / SLUG[row.category]
        folder.mkdir(parents=True, exist_ok=True)
        spec = pc.get_spec(name)
        static_backend = "seaborn" if "matplotlib" not in spec.backends else "matplotlib"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c = pc.examples.demo(name, backend=static_backend)
                c.save(folder / f"{name}.png", dpi=110)
                c.close()
                if html and "plotly" in spec.backends:
                    pc.examples.demo(name, backend="plotly").save(folder / f"{name}.html")
            print("ok  ", name)
        except Exception as exc:   # keep going; report at the end
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print("FAIL", name, exc)
    return failures


def build_docs(html: bool) -> None:
    DOCS.mkdir(exist_ok=True)
    cat = pc.catalog()
    lines = ["# Chart catalogue", "",
             f"{len(cat)} charts: the 77 charts of the reference list (Datylon's chart guide) plus five EDA "
             "extras (mosaic, Q-Q plot, pair plot, joint plot, cluster map). Every chart accepts "
             "`backend=` (`matplotlib`, `seaborn`, `plotly`) and `theme=`. *seaborn* in the backends column means "
             "a seaborn-native drawing path; every matplotlib chart also accepts `backend=\"seaborn\"`.", ""]
    for category, sub in cat.groupby("category", sort=False):
        lines += [f"## {category}", "", "| Chart | Function | Backends | Also known as | Use when | Avoid when |",
                  "|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            img = f"../gallery/{SLUG[category]}/{r.chart}.png"
            lines.append(f"| [{r.chart.replace('_', ' ')}]({img}) | `{r.function}` | {r.backends} | "
                         f"{r.alt_names} | {r.use_when} | {r.avoid_when} |")
        lines.append("")
    (DOCS / "CATALOG.md").write_text("\n".join(lines), encoding="utf-8")

    g = ["# Gallery", "", "Static renders (matplotlib / seaborn, default light theme) of `pc.examples.demo(name)`. "
         "Click an image to open it full size. Interactive Plotly versions are in `gallery/<family>/<chart>.html`.", ""]
    for category, sub in cat.groupby("category", sort=False):
        g += [f"## {category}", "", "<table>"]
        names = list(sub.chart)
        for i in range(0, len(names), 3):
            g.append("<tr>")
            for n in names[i:i + 3]:
                info = pc.examples.info(n)
                png = f"../gallery/{SLUG[category]}/{n}.png"
                g.append(f"<td width='33%' valign='top'><b>{n}</b><br><a href='{png}'><img src='{png}' "
                         f"width='100%' alt='{n}'></a><br><sub>{info['title']} · data: {info['source']}</sub></td>")
            g.append("</tr>")
        g += ["</table>", ""]
    (DOCS / "GALLERY.md").write_text("\n".join(g), encoding="utf-8")

    info = pc.datasets.info()
    s = ["# Datasets", "", "Downloaded by `python -m pychartist.datasets` (or `pc.datasets.fetch_all()`); "
         "all are freely available.", "", "| name | file | description | source | licence | URL |",
         "|---|---|---|---|---|---|"]
    for _, r in info.iterrows():
        s.append(f"| {r['name']} | {r.file} | {r.description} | {r.source} | {r.licence} | <{r.url}> |")
    (ROOT / "datasets" / "SOURCES.md").write_text("\n".join(s) + "\n", encoding="utf-8")


README_START, README_END = "<!-- gallery:start -->", "<!-- gallery:end -->"


def build_readme_gallery(columns: int = 4) -> None:
    """Write the thumbnail grid between the gallery markers in README.md. Each thumbnail links to its full-size PNG."""
    cat = pc.catalog()
    width = f"{100 // columns}%"
    g = [README_START, ""]
    for category, sub in cat.groupby("category", sort=False):
        g += ["<details open>", f"<summary><b>{category}</b> ({len(sub)} charts)</summary>", "", "<table>"]
        names = list(sub.chart)
        for i in range(0, len(names), columns):
            g.append("<tr>")
            for n in names[i:i + columns]:
                info = pc.examples.info(n)
                png = f"gallery/{SLUG[category]}/{n}.png"
                g.append(f'<td align="center" valign="top" width="{width}"><a href="{png}" title="{info["title"]}">'
                         f'<img src="{png}" alt="{n}" width="100%"></a><br><sub><b>{n.replace("_", " ")}</b></sub></td>')
            g.append("</tr>")
        g += ["</table>", "</details>", ""]
    g.append(README_END)
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    head, rest = text.split(README_START, 1)
    tail = rest.split(README_END, 1)[1]
    readme.write_text(head + "\n".join(g) + tail, encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", action="store_true", help="also export interactive Plotly HTML")
    ap.add_argument("--docs-only", action="store_true")
    a = ap.parse_args()
    fails = [] if a.docs_only else build_images(a.html)
    build_docs(a.html)
    build_readme_gallery()
    print("failures:", fails or "none")
