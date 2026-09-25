"""'Label animals by' renames animals in the figures and nowhere else.

The daily path grid shipped with SexID row labels and no way to ask for the
names from Configure Identities. The choice now reaches every figure, while
file names and the behaviour-events table (also an exported CSV) keep the
SexID, the key every analysis joins on.

Runs under pytest, or directly (``python test_plot_labels.py``).
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from PyQt5.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication(sys.argv)

from fnt.uwb.uwb_preprocessing_pyqt import PlotSaverWorker  # noqa: E402

ROSTER = {
    0x06: {'sex': 'F', 'identity': '9905', 'name': 'Hera', 'code': 'HER'},
    0x07: {'sex': 'M', 'identity': '9901', 'name': 'Apollo', 'code': 'APO'},
    0x2A: {'sex': 'F', 'identity': '9806', 'name': 'Athena', 'code': ''},
    0x30: {'sex': 'M', 'identity': '9916', 'name': '', 'code': 'POS'},
}


def _worker(label_type="SexID", roster=ROSTER):
    return PlotSaverWorker(None, None, [], False, "None", plot_types={},
                           tag_identities=roster, use_identities=True,
                           label_type=label_type)


def test_sexid_default_is_unchanged():
    w = _worker()
    assert w._tag_label(0x06) == "F-9905"          # titles keep the hyphen
    assert w._grid_row_label([0x06]) == "F9905"    # grid keeps the join key
    assert w._tag_legend_label(0x06) == "HexID 6 — F9905"
    assert w._animal_label([0x06, 0x07]) == "HexIDs 6, 7 — F9905"


def test_name_labels_every_figure():
    w = _worker("Name")
    assert w._tag_label(0x06) == "Hera"
    assert w._grid_row_label([0x06]) == "Hera"
    assert w._tag_legend_label(0x06) == "HexID 6 — Hera"
    # A tag swap keeps both hardware ids visible alongside the name.
    assert w._animal_label([0x06, 0x07]) == "HexIDs 6, 7 — Hera"


def test_missing_name_falls_back_to_sexid():
    w = _worker("Name")
    assert w._tag_label(0x30) == "M9916"
    w = _worker("Code")
    assert w._tag_label(0x2A) == "F9806"
    assert w._tag_label(0x30) == "POS"


def test_hex_and_short_ids():
    w = _worker("HexID")
    assert w._tag_label(0x2A) == "HexID 2A"
    assert w._tag_legend_label(0x2A) == "HexID 2A"      # not said twice
    assert w._grid_row_label([0x06, 0x07]) == "HexID 6 / HexID 7"
    w = _worker("ShortID")
    assert w._tag_label(0x2A) == "ShortID 42"


def test_unconfigured_tag_still_labelled():
    w = _worker("Name", roster={})
    assert w._tag_label(0x2A) == "HexID 2A"
    assert w._grid_row_label([0x2A]) == "HexID 2A"


def test_behaviour_keys_translate_for_figures_only():
    w = _worker("Name")
    assert w._display_animal("F9905") == "Hera"
    assert w._display_animal("M9916") == "M9916"         # no name set
    assert w._display_animal("HexID2A") == "Athena"
    assert w._display_animal("somethingelse") == "somethingelse"
    assert _worker()._display_animal("F9905") == "F9905"


def test_grid_rows_stay_in_sexid_order():
    """Choosing Name must not re-sort rows alphabetically by name."""
    import pandas as pd
    from fnt.uwb import roi_bouts as RB
    w = _worker("Name")
    groups = [(w._grid_sexid_label(t), w._grid_row_label(t), t)
              for _s, t in w.animal_groups(sorted(ROSTER))]
    groups.sort(key=lambda g: RB.natural_animal_key(g[0]))
    order = [label for _k, label, _t in groups]
    # SexID order: F9806, F9905, M9901, M9916 -> names shown in that order.
    assert order == ["Athena", "Hera", "Apollo", "M9916"], order


def test_file_names_keep_the_sexid():
    w = _worker("Name")
    assert w._tag_file_suffix(0x06) == "F-9905"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all plot label tests passed")
