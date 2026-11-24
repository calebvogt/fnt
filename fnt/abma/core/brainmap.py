"""The brain map: one HTML page that draws an anatomy spec and a run of it.

The page (``fnt/abma/gui/brain_map_page/brain_map_template.html``) shows every area
of a spec (:mod:`fnt.abma.core.anatomy`) as ribbons or as a network of model
neurons, with the evidence behind each projection, receptor maps, firing-rate
traces of single neurons, and a jointed body in an open field.

This module has no Qt in it. It only builds the page:

``build_page(spec)``
    A single self-contained document: the 3D library is written into it, no
    fonts or scripts are fetched, so it opens with no internet connection.
    Opened by itself it runs a *sandbox*: its own JavaScript copy of the
    network in a simplified box, which is not the ABMA engine.

``build_page(spec, bridge_js=..., ws_url=...)``
    The same page, told how to reach an engine. It then stops running its own
    copy and draws what the engine sends. :mod:`fnt.abma.gui.brain_map` is the
    engine side; ``python -m fnt.abma --brain-map`` opens it.

``build_page(spec, hosted=True)``
    The page as a fragment that loads the 3D library from a public CDN, for
    publishing on a web host that adds its own document skeleton.

What a reader changes on the page (projection strengths, receptor levels,
verification notes) is kept beside the user's own specs, in
``~/ABMA_brains/anatomy/<name>.notes.json``. The spec itself is never
rewritten: folding notes back into it is a deliberate act.
"""
from __future__ import annotations

import json
import os

from . import anatomy

_CDN = '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'
_FONTS = '<link rel="stylesheet" href="https://fonts.googleapis.com'


def assets_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gui", "brain_map_page")


def _read(name: str) -> str:
    with open(os.path.join(assets_dir(), name), encoding="utf-8") as f:
        return f.read()


def _ascii_only(html: str) -> str:
    """Every character outside ASCII as an escape, so the page reads the same
    whatever encoding it is served with: an entity in the markup, a ``\\uXXXX``
    escape inside the page's own (last) script."""
    cut = html.rindex("<script>")
    head = "".join(c if ord(c) < 128 else f"&#{ord(c)};" for c in html[:cut])
    tail = "".join(c if ord(c) < 128 else f"\\u{ord(c):04x}" for c in html[cut:])
    return head + tail


def build_page(spec: dict, hosted: bool = False, bridge_js: str = "", ws_url: str = "") -> str:
    anatomy.check_spec(spec)
    html = _read("brain_map_template.html")
    if html.count("__SPEC__") != 1 or html.count(_CDN) != 1:
        raise ValueError("the brain map template has lost its placeholders")
    html = html.replace("__SPEC__", json.dumps(spec, ensure_ascii=True).replace("</", "<\\/"))
    if hosted:
        return _ascii_only(html)
    a = html.index(_FONTS)
    html = html[:a] + html[html.index(">", a) + 1:]           # no web fonts: the fallbacks are named in the CSS
    lib = "<script>" + _read("three.min.js").replace("</script", "<\\/script") + "</script>"
    if bridge_js:
        lib += "<script>" + bridge_js.replace("</script", "<\\/script") + "</script>"
    if ws_url:
        lib += f"<script>window.ABMA_WS={json.dumps(ws_url)};</script>"
    html = _ascii_only(html.replace(_CDN, "__LIB__")).replace("__LIB__", lib)
    return ('<!doctype html><html><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1"></head><body>'
            + html + "</body></html>")


def write_page(path: str, spec: dict, **kw) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(build_page(spec, **kw))
    return path


# --------------------------------------------------------------------------- #
# What the reader changed or verified on the page
# --------------------------------------------------------------------------- #
def notes_path(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0] or "anatomy"
    return os.path.join(anatomy.user_dir(), base + ".notes.json")


def load_notes(name: str) -> dict | None:
    try:
        with open(notes_path(name), encoding="utf-8") as f:
            got = json.load(f)
        return got if isinstance(got, dict) else None
    except (OSError, ValueError):
        return None


def save_notes(name: str, notes: dict) -> str:
    path = notes_path(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(notes, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    return path


def edits_from_notes(notes: dict | None) -> tuple[dict, dict]:
    """The page's edits as ``AnatomicalBrain`` takes them: strengths keyed
    ``"A>B"`` and receptor levels keyed ``"MOD@AREA"``. Anything malformed is
    dropped rather than trusted."""
    ed = (notes or {}).get("ED") or notes or {}
    strengths, receptors = {}, {}
    for k, v in (ed.get("proj") or {}).items():
        try:
            strengths[str(k).replace("~", ">")] = min(3.0, max(0.0, float(v)))
        except (TypeError, ValueError):
            pass
    for k, v in (ed.get("rec") or {}).items():
        try:
            receptors[str(k)] = min(3, max(0, int(v)))
        except (TypeError, ValueError):
            pass
    return strengths, receptors
