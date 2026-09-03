"""Hands-free MuseStudio session: connect, prove data, record, advance.

    conda activate fnt
    python scripts/muse_autorun.py                 # Flight Calibration, subject CV01
    python scripts/muse_autorun.py probe_quick S02 # any session key / subject

Why this exists: macOS grants Bluetooth per-application, and a process spawned by
another tool does not inherit the Terminal's grant — CoreBluetooth aborts it with
SIGABRT before it prints anything. So the scan has to be started from a terminal
that already has Bluetooth permission. This script is that terminal command, and
it removes every click after it: it connects, refuses to continue unless EEG is
genuinely flowing, starts the recording, and advances the set-up phase itself.

It prints a running state line so an operator (or someone reading over their
shoulder) can see exactly where it is without touching the window.
"""

import sys
import time

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from fnt.musestudio.theme import STYLESHEET
from fnt.musestudio.musestudio_pyqt import MuseStudioWindow

SESSION = sys.argv[1] if len(sys.argv) > 1 else "flight_cal"
SUBJECT = sys.argv[2] if len(sys.argv) > 2 else "CV01"

CONNECT_TIMEOUT_S = 90     # OpenMuse find + LSL resolve can take a while
DATA_TIMEOUT_S = 45        # after connecting, how long to wait for real samples
DATA_PROOF_S = 5           # consecutive seconds of samples before we trust it
SETTLE_S = 25              # pause on the set-up phase so the subject can settle


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    app = QApplication(sys.argv[:1])
    app.setStyleSheet(STYLESHEET)
    w = MuseStudioWindow()
    w.resize(1700, 1050)
    w.show()
    w.raise_()
    w.activateWindow()

    w.subject_combo.setCurrentText(SUBJECT)
    for i in range(w.mode_combo.count()):
        if w.mode_combo.itemData(i) == SESSION:
            w.mode_combo.setCurrentIndex(i)
            break
    else:
        log(f"unknown session {SESSION!r}; leaving the current selection")
    if hasattr(w, "speak_check"):
        w.speak_check.setChecked(True)
    log(f"window up · session={w.mode_combo.currentData()} · subject={w.subject_id()}")
    log("connecting — do not click anything")

    st = {"phase": "connect", "t0": time.monotonic(), "good": 0, "last": 0.0,
          "last_phase": None}

    def tick():
        now = time.monotonic()
        el = now - st["t0"]
        connected = w.reader is not None
        rate = w.device_status.rate_label.text() or "—"

        if now - st["last"] >= 3:
            st["last"] = now
            log(f"  t+{el:4.0f}s {st['phase']:<12} connected={connected} "
                f"rate={rate} batt={w.device_status.battery_label.text()}")

        if st["phase"] == "connect":
            if connected and w._eeg_stream:
                st.update(phase="prove", t0=now, good=0)
                log("CONNECTED — verifying that EEG is actually flowing")
            elif el > CONNECT_TIMEOUT_S:
                log("FAILED: no connection.")
                log("  A PULSATING light means the Muse is advertising, not paired.")
                log("  Switch the headband off and on, keep it close, and retry.")
                st["phase"] = "failed"
            return

        if st["phase"] == "prove":
            fresh = any((now - t) < 2.0 for t in (w._last_sample or {}).values())
            st["good"] = st["good"] + 1 if fresh else 0
            if st["good"] >= DATA_PROOF_S:
                log(f"DATA CONFIRMED ({rate}) — starting the recording")
                st.update(phase="start", t0=now)
            elif el > DATA_TIMEOUT_S:
                log("FAILED: connected, but NO EEG DATA arrived.")
                log("  This is the silent-stream state: the socket is open and the")
                log("  headband reports connected while sending nothing. Not starting")
                log("  a session — it would record an empty file.")
                log("  Charge the headband, reseat it, and power-cycle it.")
                st["phase"] = "failed"
            return

        if st["phase"] == "start":
            w.on_start_session(record=True)
            if SESSION == "flight":
                log("FLIGHT: keep your eyes OPEN while the baseline builds")
            name = getattr(w.session, "name", "?")
            log(f"RECORDING STARTED → {name}")
            log(f"  set-up phase holds for {SETTLE_S}s, then advances by itself")
            st.update(phase="setup", t0=now)
            return

        if SESSION == "flight" and st["phase"] in ("setup", "run"):
            # Flight has no phases; narrate the controller's own state instead.
            fl = w.flight
            if fl.is_running():
                cs = fl.craft.state
                if not fl.pipeline.baseline_ready():
                    if int(el) % 5 == 0:
                        log(f"  baseline {100*fl.pipeline.baseline_progress():.0f}%"
                            f" — eyes OPEN")
                elif st.get("told") != "closed":
                    st["told"] = "closed"
                    log("BASELINE SET — now CLOSE YOUR EYES and let yourself settle")
                elif cs.phase.value != st.get("cphase"):
                    st["cphase"] = cs.phase.value
                    log(f"CRAFT: {cs.phase.value}  alt={cs.altitude:.0f}")
            else:
                log("flight ended")
                st["phase"] = "done"
            return

        if st["phase"] == "setup":
            if w.runner.waiting_for_user() and el > SETTLE_S:
                log("advancing to block 1 (eyes open)")
                w.session_banner.continue_clicked.emit()
                st["phase"] = "run"
            return

        if st["phase"] == "run":
            ph = w._current_phase.name if w._current_phase else "?"
            if ph != st["last_phase"]:
                st["last_phase"] = ph
                log(f"PHASE: {ph}")
            if not w._session_active:
                log("SESSION ENDED")
                st["phase"] = "done"
            return

    t = QTimer()
    t.timeout.connect(tick)
    t.start(1000)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
