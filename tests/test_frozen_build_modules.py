"""The release build (fnt.spec) must bundle every tool window.

Tool windows are opened by module name (fnt.tool_launcher.TOOL_REGISTRY, imported
with importlib in fnt.tool_host), so PyInstaller cannot find them by following
imports. v0.1.15 shipped without CAD, MAD, Compress WAVs and most other tools.
The spec bundles them with collect_submodules('fnt'), which walks the package on
disk: each registered module therefore has to be a real file inside fnt/.

Runs under pytest, or directly.
"""
import ast
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _registry_modules():
    src = open(os.path.join(ROOT, "fnt", "tool_launcher.py"), encoding="utf-8").read()
    mods = re.findall(r'ToolSpec\(\s*"[^"]+",\s*"([\w.]+)"', src)
    assert len(mods) >= 20, mods
    return mods


def _module_file(mod):
    base = os.path.join(ROOT, *mod.split("."))
    return next((p for p in (base + ".py", os.path.join(base, "__init__.py"))
                 if os.path.exists(p)), None)


def test_the_spec_collects_every_fnt_module():
    spec = open(os.path.join(ROOT, "fnt.spec"), encoding="utf-8").read()
    calls = [n for n in ast.walk(ast.parse(spec))
             if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "collect_submodules"]
    assert any(c.args and getattr(c.args[0], "value", None) == "fnt" for c in calls)


def test_every_registered_tool_is_a_module_inside_fnt():
    missing = [m for m in _registry_modules() if not m.startswith("fnt.") or _module_file(m) is None]
    assert not missing, missing


def test_every_tool_folder_is_a_package():
    """collect_submodules only walks folders with an __init__.py (v0.1.15 had
    none in sleapProcessing / gitProcessing, so those tools were left out)."""
    bare = set()
    for m in _registry_modules():
        parts = m.split(".")[:-1]
        for i in range(1, len(parts) + 1):
            d = os.path.join(ROOT, *parts[:i])
            if not os.path.exists(os.path.join(d, "__init__.py")):
                bare.add(".".join(parts[:i]))
    assert not bare, sorted(bare)


if __name__ == "__main__":
    test_the_spec_collects_every_fnt_module()
    test_every_registered_tool_is_a_module_inside_fnt()
    test_every_tool_folder_is_a_package()
    print("ok")
