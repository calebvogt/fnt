"""The preview must not make a slow database slower, or look dead while it waits.

Two things went wrong while scrubbing VT-P002 on 2026-09-24. The source Wiser
export carries no index on (shortid, timestamp), so until the indexed copy is
rebuilt - which a daily re-download forces - every chunk read is a full scan of
a 6.5 GB table, and the preview was firing two extra scans to warm the
neighbours plus a blocking MIN/MAX on the GUI thread whenever a chunk came back
empty. Meanwhile the clock and weather line kept updating (both local
arithmetic), so a load that was merely slow read as a frozen window.

Runs under pytest, or directly (``python test_preview_slow_path.py``).
"""
import os
import sys
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from PyQt5.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication(sys.argv)

from fnt.uwb.uwb_preprocessing_pyqt import UWBQuickVisualizationWindow as W  # noqa: E402


def _win(**kw):
    """A stand-in window carrying only what the methods under test touch."""
    h = types.SimpleNamespace(
        db_path="/data/trial.sqlite",
        preview_db_path="/data/trial_FNT_analysis/trial_indexed.sqlite",
        exporting=False,
        preview_cache={},
        preview_inflight={},
        requested=[],
    )
    h.__dict__.update(kw)
    for name in ("_preview_is_indexed", "_loading_status_text", "_prefetch_neighbors"):
        setattr(h, name, getattr(W, name).__get__(h, types.SimpleNamespace))
    h._request_chunk = lambda idx, make_current=False: h.requested.append(idx)
    return h


def test_indexed_detection():
    assert _win()._preview_is_indexed() is True
    # Falling back to the original (no copy, or one still building) is the slow path.
    assert _win(preview_db_path="/data/trial.sqlite")._preview_is_indexed() is False
    assert _win(preview_db_path=None)._preview_is_indexed() is False


def test_prefetch_only_when_reads_are_cheap():
    h = _win()
    h._prefetch_neighbors(5)
    assert sorted(h.requested) == [4, 6]

    # Unindexed: each prefetch is another full scan competing with the chunk
    # the user is actually waiting on.
    h = _win(preview_db_path="/data/trial.sqlite")
    h._prefetch_neighbors(5)
    assert h.requested == []

    # An export already streams the whole table; don't add to the queue.
    h = _win(exporting=True)
    h._prefetch_neighbors(5)
    assert h.requested == []


def test_prefetch_skips_cached_and_inflight():
    h = _win(preview_cache={4: object()}, preview_inflight={6: object()})
    h._prefetch_neighbors(5)
    assert h.requested == []


def test_loading_status_names_the_cause():
    assert _win()._loading_status_text() == "Loading…"

    slow = _win(preview_db_path="/data/trial.sqlite")._loading_status_text()
    assert "no fast index" in slow

    busy = _win(exporting=True)._loading_status_text()
    assert "export" in busy

    both = _win(preview_db_path="/data/trial.sqlite", exporting=True)._loading_status_text()
    assert "no fast index" in both and "export" in both


def test_stale_chunks_are_dropped_unprocessed():
    """A drag delivers chunks the playhead has left behind; they must be free.

    Measured on VT-P002 before this: one drag delivered 88 chunks, 87 already
    stale, and filtering/smoothing them cost 23 s of main-thread time - which
    is what stopped the scene redrawing while the clock label carried on.
    """
    h = types.SimpleNamespace(
        preview_pending_current=None,
        preview_playhead_ms=0,
        _chunk_index_for=lambda ts: 300,
    )
    wanted = W._chunk_is_wanted.__get__(h, types.SimpleNamespace)

    assert wanted(300) is True                      # the chunk in view
    assert wanted(299) is True and wanted(301) is True   # prefetched neighbours
    assert wanted(298) is False                     # scrubbed past
    assert wanted(742) is False

    # Whatever the user is actually waiting on is always worth paying for.
    h.preview_pending_current = 742
    assert wanted(742) is True


def test_scrub_debounce_has_a_floor():
    """A sustained drag must still get a read, not just one on release."""
    assert W.SCRUB_DEBOUNCE_MS > 0
    assert 0 < W.SCRUB_MAX_WAIT_S <= 1.0


def test_gap_search_runs_on_a_worker():
    """_handle_empty_chunk must hand the MIN/MAX scan to _start_db_query."""
    started = []
    h = types.SimpleNamespace(
        db_path="/data/trial.sqlite",
        preview_db_path="/data/trial_indexed.sqlite",
        table_name="VoleTerra",
        selected_preview_tags=lambda: [1, 2],
        _chunk_bounds=lambda idx: (1000, 2000),
        lbl_preview_status=types.SimpleNamespace(setText=lambda t: None),
        _start_db_query=lambda fn, ok, bad: started.append(fn),
        _on_gap_search_done=lambda res: None,
        _on_gap_search_failed=lambda err: None,
    )
    W._handle_empty_chunk.__get__(h, types.SimpleNamespace)(7)
    assert len(started) == 1, "the scan must not run on the GUI thread"


def test_gap_result_ignored_once_the_user_scrubs_away():
    """A scan that lands after the user moved on must not yank the playhead back."""
    moved = []
    h = types.SimpleNamespace(
        _preview_active=True,
        table_name="VoleTerra",
        preview_playhead_ms=999_000,
        _chunk_index_for=lambda ts: 99 if ts == 999_000 else 7,
        lbl_preview_status=types.SimpleNamespace(setText=lambda t: None),
        log_message=lambda m: None,
        _sync_timeline_to_playhead=lambda: moved.append(True),
        _request_chunk=lambda idx, make_current=False: moved.append(idx),
    )
    done = W._on_gap_search_done.__get__(h, types.SimpleNamespace)
    done({'idx': 7, 'start': 1000, 'table': 'VoleTerra', 'ts': 5000})
    assert moved == [], "stale gap result moved the playhead"
    assert h.preview_playhead_ms == 999_000

    # A torn-down preview must not act on a late result either.
    h._preview_active = False
    h._chunk_index_for = lambda ts: 7
    done({'idx': 7, 'start': 1000, 'table': 'VoleTerra', 'ts': 5000})
    assert moved == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all preview slow-path tests passed")
