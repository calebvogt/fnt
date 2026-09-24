"""Which calls the 3D view draws — the rules, tested without OpenGL.

``collect_call_paths`` decides what reaches the screen, and two of its rules are
requirements rather than defaults: rejected masks are never drawn (they record
what a call is *not*), and every path is computed from the call's own mask
pixels. Both are checked here on synthetic calls with known shapes, so the test
also catches a path that stops matching the audio it came from.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from fnt.usv.mad_trajectory_view import collect_call_paths

SR, HOP, NPERSEG, NOVERLAP, NFFT = 250_000, 128, 512, 384, 1024
DF = (SR / 2.0) / (NFFT // 2)
DT = HOP / SR
T_ORG = (NPERSEG / 2) / SR

# name, start_s, dur_s, contour u∈[0,1] -> Hz, status, harmonic_n
CALLS = [
    ('upsweep', 0.10, 0.040, lambda u: 30e3 + 30e3 * u, None, 1),
    ('upsweep_H2', 0.10, 0.040, lambda u: 2 * (30e3 + 30e3 * u), None, 2),
    ('downsweep', 0.25, 0.030, lambda u: 70e3 - 30e3 * u, 'prediction', 1),
    ('accepted_flat', 0.40, 0.025, lambda u: 45e3 + 0 * u, 'accepted', 1),
    ('rejected', 0.55, 0.030, lambda u: 35e3 + 10e3 * u, 'rejected', 1),
]


def _annotation(i, name, s0, d, f, status, h):
    k0 = int(np.ceil((s0 - T_ORG) / DT))
    k1 = int(np.floor((s0 + d - T_ORG) / DT))
    ks = np.arange(k0, k1)
    u = np.clip((T_ORG + ks * DT - s0) / d, 0, 1)
    centre = np.round(f(u) / DF).astype(int)
    f0, f1 = centre.min() - 3, centre.max() + 4
    mask = np.zeros((f1 - f0, ks.size), dtype=bool)
    for j, c in enumerate(centre):
        mask[c - 3 - f0:c + 4 - f0, j] = True
    ann = {'id': name, 'status': status, 'f0': int(f0), 'f1': int(f1),
           't0': int(k0), 't1': int(k1), 'mask': mask}
    if h > 1:
        ann['harmonic_n'] = h
    return ann


@pytest.fixture(scope="module")
def sg():
    """A duck-typed spectrogram widget: the attributes the view reads."""
    t = np.arange(int(SR * 0.7)) / SR
    audio = 0.002 * np.random.default_rng(0).standard_normal(t.size)
    for name, s0, d, f, st, h in CALLS:
        m = (t >= s0) & (t < s0 + d)
        u = (t[m] - s0) / d
        audio[m] += 0.05 * np.sin(np.pi * u) ** 0.5 * np.sin(
            2 * np.pi * np.cumsum(f(u)) / SR)
    return SimpleNamespace(
        audio_data=audio.astype(np.float32), sample_rate=SR, hop=HOP,
        nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT,
        annotations=[_annotation(i, *c) for i, c in enumerate(CALLS)])


def _collect(sg, **kw):
    paths, skipped = collect_call_paths(sg, -100.0, -20.0, T_ORG, **kw)
    return {p['id']: p for p in paths}, skipped


def test_rejected_masks_are_never_drawn(sg):
    for pend in (True, False):
        for acc in (True, False):
            for harm in (True, False):
                got, _ = _collect(sg, include_pending=pend,
                                  include_accepted=acc, include_harmonics=harm)
                assert 'rejected' not in got


def test_pending_and_accepted_by_default_harmonics_hidden(sg):
    got, skipped = _collect(sg)
    assert set(got) == {'upsweep', 'downsweep', 'accepted_flat'}
    assert got['downsweep']['cls'] == 'pending'
    # Both a confirmed label (status None) and an accepted prediction count.
    assert got['upsweep']['cls'] == 'accepted'
    assert got['accepted_flat']['cls'] == 'accepted'
    assert skipped['harmonics'] == 1


def test_status_filters(sg):
    only_pending, _ = _collect(sg, include_accepted=False)
    assert set(only_pending) == {'downsweep'}
    only_accepted, _ = _collect(sg, include_pending=False)
    assert set(only_accepted) == {'upsweep', 'accepted_flat'}


def test_harmonics_on_request(sg):
    got, skipped = _collect(sg, include_harmonics=True)
    assert 'upsweep_H2' in got and skipped['harmonics'] == 0
    # A harmonic is the fundamental's contour scaled: twice the pitch.
    ratio = (np.median(got['upsweep_H2']['traj']['pitch'])
             / np.median(got['upsweep']['traj']['pitch']))
    assert ratio == pytest.approx(2.0, rel=0.03)


def test_time_window_keeps_only_overlapping_calls(sg):
    got, _ = _collect(sg, time_window=(0.24, 0.30))
    assert set(got) == {'downsweep'}


def test_paths_match_the_audio_they_came_from(sg):
    """Measured from the real STFT of the synthetic audio, through the mask."""
    got, _ = _collect(sg)
    up = got['upsweep']['traj']
    down = got['downsweep']['traj']
    flat = got['accepted_flat']['traj']
    # 30 kHz over 40 ms = 0.75 kHz/ms; 30 kHz over 30 ms = 1.0 kHz/ms.
    assert np.median(up['pitch_rate']) == pytest.approx(0.75, rel=0.1)
    assert np.median(down['pitch_rate']) == pytest.approx(-1.0, rel=0.1)
    assert abs(np.median(flat['pitch_rate'])) < 0.1
    assert up['pitch'][0] < up['pitch'][-1]
    # Frames are the canvas's global frame indices.
    ann = next(a for a in sg.annotations if a['id'] == 'upsweep')
    assert up['frame'][0] >= ann['t0'] and up['frame'][-1] < ann['t1']


def test_cache_reuses_work_and_misses_on_an_edited_mask(sg):
    """The cache holds each call's frame features — the STFT work. A second
    pass must reuse them; an edited mask must not."""
    cache = {}
    first, _ = _collect(sg, cache=cache, cache_tag='wav')
    n = len(cache)
    cached = dict(cache)
    again, _ = _collect(sg, cache=cache, cache_tag='wav')
    assert len(cache) == n
    assert all(cache[k] is cached[k] for k in cached), "frames recomputed"
    assert np.array_equal(again['upsweep']['traj']['pitch'],
                          first['upsweep']['traj']['pitch'])

    edited = dict(sg.annotations[0])
    edited['mask'] = edited['mask'].copy()
    edited['mask'][:, -3:] = False            # user trimmed the call's tail
    sg2 = SimpleNamespace(**{**vars(sg), 'annotations': [edited]})
    got, _ = _collect(sg2, cache=cache, cache_tag='wav')
    assert len(cache) == n + 1
    assert got['upsweep']['traj']['pitch'].size \
        < first['upsweep']['traj']['pitch'].size


def test_smoothing_is_applied_from_the_cache_without_recomputing(sg):
    """Moving the Smoothing slider re-derives paths from cached frames: the
    cache doesn't grow, the path gets smoother, and the frames it spans stay
    the same (smoothing never trims or shifts a call)."""
    cache = {}
    raw, _ = _collect(sg, cache=cache, cache_tag='wav', smooth_frames=1)
    n = len(cache)
    heavy, _ = _collect(sg, cache=cache, cache_tag='wav', smooth_frames=15)
    assert len(cache) == n
    r = raw['downsweep']['traj']
    h = heavy['downsweep']['traj']
    assert np.array_equal(r['frame'], h['frame'])
    rough = lambda a: float(np.mean(np.abs(np.diff(a, 2))))   # noqa: E731
    assert rough(h['pitch_rate']) < 0.5 * rough(r['pitch_rate'])
    # Heavy smoothing still reads as the same call.
    assert np.median(h['pitch_rate']) == pytest.approx(-1.0, rel=0.15)


def test_no_audio_is_no_paths_not_an_error():
    empty = SimpleNamespace(audio_data=None, sample_rate=None, hop=None,
                            annotations=[])
    assert collect_call_paths(empty, -100.0, -20.0, 0.0) == (
        [], {'harmonics': 0, 'short': 0})
