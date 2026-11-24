"""The brain map: the page is self-contained, and the engine side of the
bridge steps the real engine, applies the page's edits and keeps its notes."""
import base64
import json
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fnt.abma.core import anatomy, brainmap, presets          # noqa: E402
from fnt.abma.run_headless import main                         # noqa: E402


@pytest.fixture(scope="module")
def spec():
    return anatomy.load_spec("c57_male")


def _cfg():
    cfg = presets.open_field_test()
    cfg.brain.enabled, cfg.brain.anatomy = True, "c57_male"
    return cfg


@pytest.fixture
def engine(tmp_path, monkeypatch):
    from fnt.abma.gui.brain_map import Engine, _app

    monkeypatch.setattr(anatomy, "user_dir", lambda: str(tmp_path / "anatomy"))
    _app()
    e = Engine(_cfg())
    e.timer.stop()
    yield e
    e.deleteLater()


def test_the_page_needs_nothing_from_the_internet(spec):
    html = brainmap.build_page(spec)
    assert html.startswith("<!doctype html>") and "__SPEC__" not in html and "__LIB__" not in html
    assert "https://cdnjs" not in html and "fonts.googleapis" not in html
    assert "Three.js Authors" in html                    # the 3D library is written into the page
    assert html.isascii() and "Male C57 Neural Network Map" in html
    assert 'src="http' not in html and "window.ABMA_WS" not in html.split("<script>window.ABMA_WS=")[0][-200:]


def test_the_page_can_be_told_where_the_engine_is(spec):
    html = brainmap.build_page(spec, bridge_js="/*bridge*/var QWebChannel=1;", ws_url="ws://127.0.0.1:5555")
    assert "/*bridge*/" in html and 'window.ABMA_WS="ws://127.0.0.1:5555"' in html


def test_the_hosted_form_is_a_fragment_that_loads_the_library(spec):
    html = brainmap.build_page(spec, hosted=True)
    assert html.lstrip().startswith("<title>") and "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/" in html
    assert html.isascii()


def test_writing_the_page(tmp_path, spec):
    path = brainmap.write_page(str(tmp_path / "sub" / "map.html"), spec)
    assert os.path.getsize(path) > 500_000


def test_notes_are_kept_beside_the_users_specs(tmp_path, monkeypatch):
    monkeypatch.setattr(anatomy, "user_dir", lambda: str(tmp_path / "anatomy"))
    assert brainmap.load_notes("c57_male") is None
    notes = {"ED": {"proj": {"mPFC~DRN": 1.5, "junk": "x"}, "rec": {"OT@LS": 0}}, "VER": {"SC~TURN": [{"kind": "note"}]}}
    path = brainmap.save_notes("c57_male", notes)
    assert path.endswith("c57_male.notes.json") and brainmap.load_notes("c57_male") == notes
    assert brainmap.edits_from_notes(notes) == ({"mPFC>DRN": 1.5}, {"OT@LS": 0})
    assert brainmap.edits_from_notes(None) == ({}, {})


def test_the_engine_sends_a_frame_for_every_step(engine):
    got = []
    engine.frame.connect(got.append)
    for _ in range(5):
        engine.step()
    assert len(got) == 5
    f = json.loads(got[-1])
    assert abs(f["t"] - 1.25) < 1e-9 and 0.0 <= f["x"] <= 0.5 and 0.0 <= f["bladder"] <= 1.0
    assert len(base64.b64decode(f["rates"])) == engine.sim.policy.brain.n
    hello = json.loads(engine.hello())
    assert hello["units"] == engine.sim.policy.brain.n and hello["notes"] is None


def test_the_clock_runs_only_while_playing(engine):
    got = []
    engine.frame.connect(got.append)
    engine._last -= 1.0
    engine.tick()
    assert not got                                          # paused: time passes, nothing steps
    engine.setPlaying(True)
    engine.setRate(4.0)
    engine._last -= 0.5                                     # half a second at 4x = 2 s = 8 steps
    engine.tick()
    assert len(got) == 8


def test_an_edit_from_the_page_changes_the_running_brain(engine):
    w = lambda text: np.frombuffer(base64.b64decode(text), "<f4").reshape(engine.sim.policy.brain.n, -1)  # noqa: E731
    b = engine.sim.policy.brain
    before = w(engine.weights())
    for _ in range(3):
        engine.step()
    state = engine.sim.policy.r.copy()
    after = w(engine.setEdits(json.dumps({"proj": {"SC~TURN": 0}, "rec": {"OT@LS": 0}})))
    assert before[b.sl["TURN"], b.sl["SC"]].any() and not after[b.sl["TURN"], b.sl["SC"]].any()
    np.testing.assert_array_equal(engine.sim.policy.r, state)      # the animal carries on from where it is
    assert engine.cfg.brain.strengths == {"SC>TURN": 0.0}
    # an edit that names nothing real changes nothing
    np.testing.assert_array_equal(w(engine.setEdits(json.dumps({"proj": {"mPFC~MOE": 2}}))), after)


def test_restart_with_another_wiring_draw(engine):
    seeds = []
    engine.cleared.connect(seeds.append)
    first = engine.weights()
    engine.restart(9)
    assert seeds == [9] and engine.k == 0 and engine.weights() != first


def test_the_engine_saves_the_pages_notes(engine):
    engine.saveNotes(json.dumps({"ED": {"proj": {"mPFC~DRN": 2}}, "VER": {}}))
    engine.saveNotes("not json")
    assert json.loads(engine.hello())["notes"]["ED"]["proj"] == {"mPFC~DRN": 2}


def test_it_opens_in_the_browser_form_without_the_web_engine(tmp_path, monkeypatch):
    from fnt.abma.gui import brain_map

    monkeypatch.setattr(anatomy, "user_dir", lambda: str(tmp_path / "anatomy"))
    win, engine, page = brain_map.open_brain_map(_cfg(), where="browser", exec_=False, open_browser=False)
    try:
        with open(page, encoding="utf-8") as f:
            html = f.read()
        assert f"ws://127.0.0.1:{win._server.serverPort()}" in html and "QWebChannel" in html
    finally:
        engine.timer.stop()
        win.close()
    if not brain_map.webengine_available():
        with pytest.raises(RuntimeError, match="PyQtWebEngine"):
            brain_map.open_brain_map(_cfg(), where="window", exec_=False)
    with pytest.raises(ValueError):
        brain_map.Engine(presets.open_field_test())          # no anatomy: nothing to map


def test_command_line_accepts_the_option(capsys):
    assert main(["--brain-map", "browser", "--dry-run"]) == 0
    assert "untrained" in capsys.readouterr().out            # --brain-map alone means the shipped C57 male
    with pytest.raises(SystemExit):
        main(["--brain-map", "sideways", "--dry-run"])


def test_the_page_ships_with_the_package():
    import tomllib

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(here, "pyproject.toml"), "rb") as f:
        proj = tomllib.load(f)
    assert "abma/gui/brain_map_page/*" in proj["tool"]["setuptools"]["package-data"]["fnt"]
    assert proj["project"]["optional-dependencies"]["brainmap"] == ["PyQtWebEngine"]
    for name in ("brain_map_template.html", "three.min.js"):
        assert os.path.isfile(os.path.join(brainmap.assets_dir(), name))


def _short_recording(tmp_path, n=1):
    cfg = _cfg()
    cfg.dt, cfg.scent.enabled = 0.25, True
    cfg.groups[0].count = n
    rec = anatomy.record(cfg, 20.0)
    return rec, anatomy.save_recording(rec, str(tmp_path / "run" / "brain_recording.npz"))


def test_a_recording_becomes_a_page_that_plays_itself(tmp_path):
    rec, path = _short_recording(tmp_path, n=2)
    spec, payload = brainmap.replay_payload(anatomy.load_recording(path))
    assert payload["frames"] == 80 and len(payload["x"]) == 80 and payload["arena"] == [0.5, 0.5]
    assert len(base64.b64decode(payload["rates"])) == 80 * rec["rates"].shape[2]
    assert len(payload["others"]) == 80 and len(payload["others"][0]) == 1      # the second animal, as a position
    assert "Open field test" in payload["title"] and "50" in payload["title"]
    html = brainmap.build_page(spec, replay=payload)
    assert "window.ABMA_REPLAY=" in html and "https://" not in html.split("window.ABMA_REPLAY=")[0][-400:]
    hosted = brainmap.build_page(spec, hosted=True, replay=payload)
    assert "window.ABMA_REPLAY=" in hosted and "cdnjs.cloudflare.com" in hosted


def test_replay_opens_without_an_engine(tmp_path):
    from fnt.abma.gui import brain_map
    _, path = _short_recording(tmp_path)
    page = brain_map.open_replay(path, where="browser", open_browser=False)
    with open(page, encoding="utf-8") as f:
        assert "window.ABMA_REPLAY=" in f.read()


def test_command_line_recording_writes_the_replay_page(tmp_path, capsys):
    out = str(tmp_path)
    assert main(["--anatomy", "c57_male", "--brain-recording", "0.1", "--out", out]) == 0
    assert os.path.isfile(os.path.join(out, "open_field_test_anatomy", "brain_recording.html"))
    with pytest.raises(SystemExit):
        main(["--brain-replay", os.path.join(out, "nope.npz")])


def test_the_engine_reports_the_arena_and_the_other_animals(tmp_path, monkeypatch):
    from fnt.abma.gui.brain_map import Engine, _app
    monkeypatch.setattr(anatomy, "user_dir", lambda: str(tmp_path / "anatomy"))
    _app()
    cfg = presets.get_preset(presets.preset_names()[0]) if False else _cfg()
    cfg.groups[0].count = 3
    e = Engine(cfg)
    e.timer.stop()
    got = []
    e.frame.connect(got.append)
    e.step()
    f = json.loads(got[0])
    assert len(f["others"]) == 2 and "contact" in f and f["voids"] == []
    hello = json.loads(e.hello())
    assert hello["arena"] == [0.5, 0.5] and "cm" in hello["title"]
    assert e.cfg.scent.enabled
    e.deleteLater()


def test_the_abma_window_has_a_brain_map_button(tmp_path, monkeypatch):
    import webbrowser
    from fnt.abma.gui.launch import build_window
    from fnt.abma.gui.brain_map import _app

    monkeypatch.setattr(anatomy, "user_dir", lambda: str(tmp_path / "anatomy"))
    monkeypatch.setattr(webbrowser, "open", lambda *a, **k: True)
    _app()
    win = build_window(presets.open_field_test())
    try:
        assert win.btn_brain_map.toolTip().startswith("Brain map")
        win._open_brain_map()
        got = win._brain_map
        map_win, engine = got[0], got[1]
        assert engine.name == "c57_male"            # a design with no anatomy brain gets the shipped one
        engine.timer.stop()
        map_win.close()
    finally:
        win.close()
