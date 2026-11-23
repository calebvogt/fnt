"""Animated export: filming a run and writing it as a GIF."""
import os

import numpy as np
import pytest
from PIL import Image

from fnt.abma.core import movie
from fnt.abma.core.compose import design
from fnt.abma.core.simulation import Simulation
from fnt.abma.run_headless import main as headless, study_base


def _cfg(males=2, females=1):
    return design(preset="anosmia", males=males, females=females)


def test_film_keeps_what_the_run_did():
    cfg = _cfg()
    clip = movie.film(cfg, seconds=80, sample_s=4.0)
    assert clip.xy.shape == (20, 3, 2) and clip.hunger.shape == (20, 3)
    assert clip.t[0] == 4.0 and clip.t[-1] == 80.0
    assert sorted(clip.sex) == ["F", "M", "M"]
    # the clip is trial 0 of the experiment, not some other run
    sim = Simulation(cfg, 0)
    for k in range(40):
        sim.step(k * 2.0, 2.0)
    assert np.allclose(clip.xy[-1], sim.P)


def test_filming_does_not_touch_the_config():
    cfg = _cfg()
    before = cfg.to_dict()
    movie.film(cfg, seconds=20)
    treated = movie.anosmic(cfg)
    assert cfg.to_dict() == before
    assert all(g.treatment.drug == "methimazole" for g in treated.groups)
    assert Simulation(treated, 0).smell.max() == 0.0


def test_animals_can_be_released_deprived():
    clip = movie.film(_cfg(), seconds=4, prepare=movie.arrive_deprived(70))
    assert clip.hunger[0].min() > 65 and clip.thirst[0].min() > 65


def test_render_writes_a_white_gif(tmp_path):
    a = movie.film(_cfg(), seconds=40, title="left")
    b = movie.film(movie.anosmic(_cfg()), seconds=24, title="right")
    one = movie.render([a], str(tmp_path / "one.gif"), dpi=50)
    two = movie.render([a, b], str(tmp_path / "two.gif"), dpi=50)
    im1, im2 = Image.open(one), Image.open(two)
    assert im1.n_frames == 10, "one frame per sample"
    assert im2.n_frames == 10, "the shorter clip holds its last frame"
    assert im2.size[0] > 1.6 * im1.size[0], "two clips sit side by side"
    im2.seek(5)
    assert im2.convert("RGB").getpixel((2, 2)) == (255, 255, 255), \
        "the page must be pure white, for slides"
    with pytest.raises(ValueError, match="no clips"):
        movie.render([], str(tmp_path / "none.gif"))


def test_headless_movie(tmp_path, capsys):
    rc = headless(["--preset", "anosmia", "--males", "1", "--females", "1",
                   "--name", "clip", "--out", str(tmp_path), "--movie",
                   "--movie-hours", "0.01", "--movie-vs-mmz",
                   "--movie-deprived", "60"])
    assert rc == 0
    path = tmp_path / "clip" / "movie.gif"
    assert path.exists() and "wrote" in capsys.readouterr().out
    assert Image.open(path).n_frames > 5


def test_a_study_can_be_described_in_one_line(capsys):
    """3 males, 3 females, 5 days, 2 intact and 2 methimazole replicates."""
    rc = headless(["--preset", "anosmia", "--males", "3", "--females", "3",
                   "--days", "5", "--anosmia-study", "--doses", "0,1",
                   "--replicates", "2", "--brain", "vole_forager",
                   "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "3 F" in out and "3 M" in out
    assert "2 arms" in out and "2 replicate(s), 5 days" in out
    assert "vole_forager" in out


def test_study_base_carries_the_design_and_the_brain():
    import argparse
    args = argparse.Namespace(preset="anosmia", males=3, females=3,
                              species="Prairie vole", days=5.0, seed=None,
                              name=None, season=None, year=2026, config=None,
                              brain="vole_forager")
    cfg = study_base(args)
    assert sum(g.count for g in cfg.groups) == 6 and cfg.days == 5.0
    assert cfg.brain.enabled and cfg.brain.name == "vole_forager"
    args.preset = args.brain = None
    args.males = args.females = 0
    assert study_base(args) is None, "nothing described: the study's default"
