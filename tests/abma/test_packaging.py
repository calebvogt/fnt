"""ABMA inside FNT: the launcher knows it, and both kinds of install carry
the brain files (pip via package data, the release build via fnt.spec)."""
import ast
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BRAINS = os.path.join(ROOT, "fnt", "abma", "brains")


def test_brains_ship_with_the_package():
    assert sorted(os.path.basename(p) for p in glob.glob(os.path.join(BRAINS, "*.npz"))) \
        >= ["vole_doorways.npz", "vole_forager.npz"]


def test_pip_installs_carry_the_brains():
    text = open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8").read()
    assert '"abma/brains/*.npz"' in text


def test_the_release_build_carries_the_brains():
    """fnt.spec lists data files by hand. Without this entry the frozen app
    opens ABMA with no learned brain to choose."""
    spec = open(os.path.join(ROOT, "fnt.spec"), encoding="utf-8").read()
    pairs = []
    for node in ast.walk(ast.parse(spec)):
        if isinstance(node, ast.Tuple) and len(node.elts) == 2 and all(
                isinstance(e, ast.Constant) for e in node.elts):
            pairs.append((node.elts[0].value, node.elts[1].value))
    src = [s for s, dest in pairs if dest == "fnt/abma/brains"]
    assert src, "fnt.spec does not bundle fnt/abma/brains"
    matched = {os.path.basename(p) for s in src
               for p in glob.glob(os.path.join(ROOT, s))}
    assert {"vole_forager.npz", "vole_doorways.npz"} <= matched


def test_the_launcher_registers_abma():
    from fnt.tool_launcher import TOOL_REGISTRY
    spec = TOOL_REGISTRY["abma_designer"]
    assert spec.module == "fnt.abma.gui.abma_main_pyqt"


def test_abma_imports_no_other_fnt_tool():
    """Each FNT tool stands alone: ABMA may import fnt.abma and nothing else
    from fnt."""
    bad = []
    for path in glob.glob(os.path.join(ROOT, "fnt", "abma", "**", "*.py"),
                          recursive=True):
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            bad += [(os.path.relpath(path, ROOT), n) for n in names
                    if n.startswith("fnt.") and not n.startswith("fnt.abma")]
    assert not bad, bad
