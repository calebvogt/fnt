"""The held-out audit: drawing clips, and scoring a model on them.

What it has to guarantee, because it is the measurement the anosmia full run
is gated on:

* clips never come from a recording any training project holds, or any
  recording that already carries labels (it could join training any day);
* every trial gets the same mix of microphones, day/night and strata, so a
  between-condition recall comparison is like for like;
* clips are bit-exact copies of their source window;
* only clips marked review complete are scored, and an empty complete clip
  still counts (every detection in it is a false positive);
* recall and precision per trial and condition add up from the per-clip
  matches, with a leak check against the model's own training project.

Runs under pytest, or directly.
"""
import csv
import json
import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("soundfile")
pytest.importorskip("h5py")

import soundfile as sf  # noqa: E402

from fnt.usv.usv_detector import mad_audit as A  # noqa: E402

SR = 250_000


# ------------------------------------------------------------------ helpers
def tone(seconds=1.0, khz=40.0, start=0.2, dur=0.3, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    x = rng.normal(0, noise, t.size)
    on = (t >= start) & (t < start + dur)
    x[on] += 0.3 * np.sin(2 * np.pi * khz * 1000 * t[on])
    return x.astype(np.float32)


def write(path, x):
    sf.write(path, np.clip(np.round(x * 32768), -32768, 32767)
             .astype(np.int16), SR, subtype="PCM_16")


def make_trial(root, name, active=True, seed=0):
    """Two mics x (two day + two night) recordings, 3 s each."""
    d = os.path.join(root, name)
    os.makedirs(d)
    k = 0
    for mic in (1, 2):
        for hh in ("10-00-00", "14-00-00", "22-00-00", "02-00-00"):
            day = "2026-02-08" if hh != "02-00-00" else "2026-02-09"
            x = tone(3.0, start=1.2, seed=seed + k) if active else \
                tone(3.0, dur=0.0, seed=seed + k)
            write(os.path.join(d, f"ch{mic}{day}_{hh}_{k:05d}.wav"), x)
            k += 1
    return d


def draw(root, dirs, **kw):
    out = os.path.join(root, "audit")
    base = dict(per_trial_active=2, per_trial_random=2, window_s=1.0,
                edge_hours=0.0, min_activity=20.0, screen_per_stratum=1)
    base.update(kw)
    return out, A.draw_audit(dirs, out, **base)


# -------------------------------------------------------- activity screen
def test_a_tonal_call_scores_and_silence_does_not():
    act = A.activity_per_window(tone(2.0, start=0.2, dur=0.4), SR, 1.0)
    assert act[0] > 50, act
    assert act[1] < 5, act


def test_a_broadband_click_is_not_mistaken_for_a_call():
    """Knocks and cage noise light the whole band at once; a USV is a ridge."""
    x = tone(1.0, dur=0.0)
    for c in range(20, SR, SR // 40):
        x[c:c + 30] += np.random.default_rng(c).normal(0, 0.5, 30)
    assert A.activity_per_window(x, SR, 1.0)[0] < 5


# ----------------------------------------------------------------- drawing
def test_the_draw_spans_mics_and_day_and_night():
    with tempfile.TemporaryDirectory() as root:
        t5 = make_trial(root, "T005_prairie_anosmic-saline")
        out, rows = draw(root, [t5], per_trial_active=4, per_trial_random=4)
        assert {r["mic"] for r in rows} == {"1", "2"}
        assert {r["phase"] for r in rows} == {"day", "night"}
        assert {r["stratum"] for r in rows} == {"active", "random"}
        assert all(r["condition"] == "saline" and r["trial"] == "T005"
                   for r in rows)


def test_clips_are_bit_exact_copies_of_their_window():
    from fnt.usv.usv_detector.spectrogram import load_audio
    with tempfile.TemporaryDirectory() as root:
        t5 = make_trial(root, "T005_prairie_anosmic-saline")
        out, rows = draw(root, [t5])
        for r in rows:
            clip, _ = load_audio(os.path.join(out, A.CLIP_DIR, r["clip"]))
            src, _ = load_audio(r["source_wav"])
            a = int(r["offset_samples"])
            assert float(r["offset_s"]) == pytest.approx(a / SR, abs=1e-9)
            np.testing.assert_array_equal(clip, src[a:a + clip.size])


def test_training_recordings_are_never_drawn():
    from fnt.usv.usv_detector.mad_project import (
        MADProjectConfig, create_mad_project)
    from fnt.usv.usv_detector.mad_registry import RegisteredFile
    with tempfile.TemporaryDirectory() as root:
        t5 = make_trial(root, "T005_prairie_anosmic-saline")
        wavs = sorted(os.path.join(t5, n) for n in os.listdir(t5))
        proj = os.path.join(root, "train_proj")
        create_mad_project(proj)
        cfg = MADProjectConfig.load(proj)
        cfg.set_audio_entries([RegisteredFile.from_path(w) for w in wavs[:6]])
        cfg.save()
        out, rows = draw(root, [t5], exclude_projects=[proj],
                         per_trial_active=4, per_trial_random=4)
        banned = {os.path.basename(w) for w in wavs[:6]}
        assert rows, "nothing was drawn"
        assert not {os.path.basename(r["source_wav"]) for r in rows} & banned


def test_a_recording_that_already_has_labels_is_never_drawn():
    """It could be added to a training project at any time."""
    from fnt.usv.usv_detector.fnt_mask_store import td_save_example
    with tempfile.TemporaryDirectory() as root:
        t5 = make_trial(root, "T005_prairie_anosmic-saline")
        wavs = sorted(os.path.join(t5, n) for n in os.listdir(t5))
        for w in wavs[:7]:
            td_save_example(w[:-4] + "_FNT.mad",
                            np.zeros((8, 8), np.float32),
                            np.ones((8, 8), np.uint8),
                            {"source_wav": os.path.basename(w)}, None)
        out, rows = draw(root, [t5], per_trial_active=2, per_trial_random=2)
        assert rows
        assert {r["source_wav"] for r in rows} == {wavs[7]}


def test_setup_and_teardown_hours_are_skipped():
    with tempfile.TemporaryDirectory() as root:
        t5 = make_trial(root, "T005_prairie_anosmic-saline")
        out, rows = draw(root, [t5], edge_hours=5.0)
        # The first recording (10:00 day 1) and the last (02:00 day 2) sit
        # within five hours of the trial's ends.
        assert not any("_10-00-00_" in r["clip"] or "_02-00-00_" in r["clip"]
                       for r in rows)


def test_the_audit_is_its_own_project_holding_its_clips():
    from fnt.usv.usv_detector.mad_project import MADProjectConfig
    with tempfile.TemporaryDirectory() as root:
        t5 = make_trial(root, "T005_prairie_anosmic-saline")
        out, rows = draw(root, [t5])
        cfg = MADProjectConfig.load(out)
        names = {os.path.basename(e.path) for e in cfg.audio_entries()}
        assert names == {r["clip"] for r in rows}
        assert os.path.isfile(os.path.join(out, A.README_NAME))


def test_drawing_never_overwrites_an_existing_audit():
    """Someone may already have labelled those clips."""
    with tempfile.TemporaryDirectory() as root:
        t5 = make_trial(root, "T005_prairie_anosmic-saline")
        out, _ = draw(root, [t5])
        with pytest.raises(RuntimeError):
            A.draw_audit([t5], out, window_s=1.0, edge_hours=0.0)


def test_the_draw_is_reproducible():
    with tempfile.TemporaryDirectory() as r1, \
            tempfile.TemporaryDirectory() as r2:
        a = draw(r1, [make_trial(r1, "T005_x-saline")], seed=7)[1]
        b = draw(r2, [make_trial(r2, "T005_x-saline")], seed=7)[1]
        assert [(x["clip"], x["offset_s"]) for x in a] == \
               [(x["clip"], x["offset_s"]) for x in b]


# ----------------------------------------------------------------- scoring
class Plan:
    """Per clip: truth boxes and predicted boxes with scores."""

    def __init__(self):
        self.truth = {}
        self.preds = {}
        self.order = []


def box(t0, f0=40_000.0, score=0.9):
    return {"start_s": t0, "stop_s": t0 + 0.05, "min_freq_hz": f0,
            "max_freq_hz": f0 + 5_000, "score": score}


def stub_pipeline(monkeypatch, plan: Plan):
    """Swap the model for the plan. The clip is identified through the audio
    length, which the stubbed loader sets from the clip's position."""
    import fnt.usv.usv_detector.mad_csv_rebuild as rb
    import fnt.usv.usv_detector.mad_inference as mi
    import fnt.usv.usv_detector.spectrogram as sp

    def load_audio(path):
        i = plan.order.index(os.path.basename(path))
        return np.zeros(1000 + i, np.float32), SR
    monkeypatch.setattr(sp, "load_audio", load_audio)
    monkeypatch.setattr(mi, "load_model",
                        lambda path, device="auto": (None, {}, "cpu"))
    monkeypatch.setattr(mi, "compute_full_spec_image",
                        lambda audio, sr, **kw: np.full((1, 1), audio.size))
    monkeypatch.setattr(mi, "infer_probability_mask",
                        lambda model, spec, **kw: spec)

    def extract(prob, threshold, **kw):
        name = plan.order[int(prob[0, 0]) - 1000]
        return [dict(b) for b in plan.preds.get(name, [])
                if b["score"] >= threshold]
    monkeypatch.setattr(mi, "extract_blobs", extract)
    monkeypatch.setattr(mi, "merge_consecutive_blobs",
                        lambda blobs, **kw: blobs)
    monkeypatch.setattr(mi, "blobs_to_rows", lambda blobs, **kw: blobs)

    def rows_for_wav(path):
        name = os.path.basename(path)
        return [dict(b, blob_id=f"h{k}", status="accepted")
                for k, b in enumerate(plan.truth.get(name, []))]
    monkeypatch.setattr(rb, "rows_for_wav", rows_for_wav)


def fake_audit(root, clips):
    """An audit folder with a manifest and complete/incomplete sidecars.
    ``clips``: (name, trial, condition, stratum, complete)."""
    from fnt.usv.usv_detector.fnt_mask_store import set_review_complete
    out = os.path.join(root, "audit")
    os.makedirs(os.path.join(out, A.CLIP_DIR))
    with open(os.path.join(out, A.AUDIT_MANIFEST), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=A.MANIFEST_FIELDS)
        w.writeheader()
        for name, trial, cond, stratum, done in clips:
            path = os.path.join(out, A.CLIP_DIR, name)
            write(path, tone(0.1, dur=0.0))
            if done:
                set_review_complete(path[:-4] + "_FNT.mad", True)
            w.writerow({"clip": name, "trial": trial, "condition": cond,
                        "mic": "1", "phase": "night", "stratum": stratum,
                        "source_wav": os.path.join(root, "src", name),
                        "offset_s": 0, "offset_samples": 0,
                        "sample_rate": SR, "window_s": 60, "activity": 0,
                        "seed": 1, "drawn_at": ""})
    return out


def scored(monkeypatch, root, clips, plan, **kw):
    out = fake_audit(root, clips)
    plan.order = [c[0] for c in clips]
    stub_pipeline(monkeypatch, plan)
    model = os.path.join(root, "run", "weights.pt")
    os.makedirs(os.path.dirname(model), exist_ok=True)
    return A.score_audit(out, model, thresholds=(0.5, 0.8), n_boot=200,
                         **kw)


def row_at(res, t):
    return next(r for r in res["per_threshold"] if r["threshold"] == t)


def test_recall_and_precision_add_up_per_condition(monkeypatch):
    plan = Plan()
    # saline clip: 4 calls, model finds 3 and adds 1 false positive
    plan.truth["s.wav"] = [box(0.1), box(0.3), box(0.5), box(0.7)]
    plan.preds["s.wav"] = [box(0.1), box(0.3), box(0.5), box(0.9)]
    # mmz clip: 2 calls, model finds 1
    plan.truth["m.wav"] = [box(0.1), box(0.4)]
    plan.preds["m.wav"] = [box(0.1)]
    with tempfile.TemporaryDirectory() as root:
        res = scored(monkeypatch, root,
                     [("s.wav", "T005", "saline", "active", True),
                      ("m.wav", "T006", "mmz", "active", True)], plan)
    r = row_at(res, 0.5)
    assert r["by_condition"]["saline"]["recall"] == pytest.approx(0.75)
    assert r["by_condition"]["saline"]["precision"] == pytest.approx(0.75)
    assert r["by_condition"]["mmz"]["recall"] == pytest.approx(0.5)
    assert r["recall_diff"]["contrast"] == "mmz - saline"
    assert r["recall_diff"]["recall_diff"] == pytest.approx(-0.25)
    assert r["overall"]["tp"] == 4 and r["overall"]["fn"] == 2


def test_a_higher_threshold_is_scored_as_its_own_run(monkeypatch):
    plan = Plan()
    plan.truth["s.wav"] = [box(0.1), box(0.3)]
    plan.preds["s.wav"] = [box(0.1, score=0.9), box(0.3, score=0.6)]
    with tempfile.TemporaryDirectory() as root:
        res = scored(monkeypatch, root,
                     [("s.wav", "T005", "saline", "active", True)], plan)
    assert row_at(res, 0.5)["overall"]["recall"] == pytest.approx(1.0)
    assert row_at(res, 0.8)["overall"]["recall"] == pytest.approx(0.5)


def test_clips_not_marked_complete_are_skipped(monkeypatch):
    plan = Plan()
    plan.truth["a.wav"] = [box(0.1)]
    plan.preds["a.wav"] = [box(0.1)]
    plan.truth["b.wav"] = [box(0.1)]      # half-labelled: must not count
    with tempfile.TemporaryDirectory() as root:
        res = scored(monkeypatch, root,
                     [("a.wav", "T005", "saline", "random", True),
                      ("b.wav", "T005", "saline", "random", False)], plan)
    assert res["n_scored"] == 1 and res["n_incomplete"] == 1
    assert row_at(res, 0.5)["overall"]["recall"] == pytest.approx(1.0)


def test_an_empty_complete_clip_counts_its_false_positives(monkeypatch):
    plan = Plan()
    plan.preds["quiet.wav"] = [box(0.2), box(0.6)]
    with tempfile.TemporaryDirectory() as root:
        res = scored(monkeypatch, root,
                     [("quiet.wav", "T005", "saline", "random", True)], plan)
    o = row_at(res, 0.5)["overall"]
    assert o["fp"] == 2 and o["tp"] == 0 and o["precision"] == 0.0
    assert o["recall"] is None, "no calls: recall is undefined, not zero"


def test_results_are_written_beside_the_audit_and_the_model(monkeypatch):
    plan = Plan()
    plan.truth["s.wav"] = [box(0.1)]
    plan.preds["s.wav"] = [box(0.1)]
    with tempfile.TemporaryDirectory() as root:
        res = scored(monkeypatch, root,
                     [("s.wav", "T005", "saline", "active", True)], plan)
        res_dir = os.path.join(root, "audit", A.RESULTS_DIR)
        assert os.path.isfile(os.path.join(res_dir, "audit_run.json"))
        assert os.path.isfile(os.path.join(res_dir, "audit_run.csv"))
        assert os.path.isfile(os.path.join(root, "run", "audit_audit.json"))
        assert "ALL" in A.format_table(res)


def test_a_model_trained_on_an_audit_source_is_flagged(monkeypatch):
    from fnt.usv.usv_detector.mad_project import (
        MADProjectConfig, create_mad_project)
    from fnt.usv.usv_detector.mad_registry import RegisteredFile
    plan = Plan()
    plan.truth["s.wav"] = [box(0.1)]
    with tempfile.TemporaryDirectory() as root:
        proj = os.path.join(root, "train_proj")
        create_mad_project(proj)
        src = os.path.join(root, "src")
        os.makedirs(src)
        write(os.path.join(src, "s.wav"), tone(0.1))
        cfg = MADProjectConfig.load(proj)
        cfg.set_audio_entries([RegisteredFile.from_path(
            os.path.join(src, "s.wav"))])
        cfg.save()
        os.makedirs(os.path.join(root, "run"))
        with open(os.path.join(root, "run", "training_summary.json"),
                  "w") as f:
            json.dump({"config": {"project_dir": proj}}, f)
        res = scored(monkeypatch, root,
                     [("s.wav", "T005", "saline", "active", True)], plan)
    assert res["training_overlap"] == ["s.wav"]
    assert "LEAK" in A.format_table(res)


def test_the_condition_interval_resamples_trials_not_just_clips():
    """Sixty clips from three trials are not sixty independent observations:
    with trials that disagree, the nested interval must be wider."""
    rng = np.random.default_rng(0)
    by_trial = {"T1": np.array([[9, 0, 1]] * 20, float),
                "T2": np.array([[5, 0, 5]] * 20, float),
                "T3": np.array([[1, 0, 9]] * 20, float)}
    _, nested = A._boot_nested(by_trial, rng, 2000)
    _, flat = A._boot_clips(np.vstack(list(by_trial.values())), rng, 2000)
    width = lambda d: np.subtract(*np.percentile(d, [97.5, 2.5]))
    assert width(nested) > 2 * width(flat)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
