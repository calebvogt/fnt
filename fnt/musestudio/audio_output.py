"""What is the Mac actually going to play sound through?

Flight Mode and the eyes-closed protocols talk to a subject who cannot see the
screen, so audio is not decoration -- it is the only channel they have. Two
things then matter, and neither is guessable from inside the app:

* **Are they wearing headphones?** Spoken guidance over laptop speakers is fine
  with an experimenter in the room and useless in a shared lab; audio altitude
  feedback over speakers is audible to everyone including the experimenter, which
  is sometimes wanted and sometimes not.
* **Is the level safe?** Anything played into in-ear buds at a level chosen for
  laptop speakers is unpleasant at best. The app must never assume.

macOS answers the first question through ``system_profiler SPAudioDataType``,
which reports every device plus a ``Transport`` field (Built-in / Bluetooth /
USB / Virtual) and marks the current default output. That is enough to tell
built-in speakers from Bluetooth earbuds without adding a dependency.

The second question is NOT answerable -- macOS does not expose output volume to
an unprivileged process in any stable way, and even if it did, the loudness at
the eardrum depends on the transducer. So detection informs the prompt; it never
replaces the human confirmation. ``kind`` is a hint for the UI, not a safety
interlock.
"""

import re
import subprocess
from dataclasses import dataclass

# Substrings that identify a headphone-shaped device by name, for the cases
# transport alone cannot settle (a USB DAC could be either; "AirPods" is not).
_HEADPHONE_HINTS = (
    "airpod", "headphone", "headset", "buds", "beats", "wh-", "wf-",
    "quietcomfort", "bose", "sennheiser", "earphone", "earbud", "hyperx",
)
_SPEAKER_HINTS = ("speaker", "display audio", "studio display", "tv", "hdmi")


# Rough one-way output latency by transport (ms). Bluetooth is the one that
# matters: A2DP buffering puts AirPods-class devices in the 150-250 ms range,
# which is spent straight out of the closed-loop feedback budget. The control
# pipeline already carries ~1 s from its 2 s analysis window, and the agency
# literature puts the break point near 750 ms, so this is not fatal -- but it is
# not free either, and a session flown on Bluetooth is not latency-comparable to
# one flown on wired output.
_LATENCY_MS = {"bluetooth": 200, "usb": 30, "built-in": 10, "virtual": 40}


@dataclass
class AudioCapability:
    """What the available gear lets this session actually do.

    The point of this type is that hardware decides experimental design, not the
    other way round. Stereo separation is the clearest case: over headphones the
    two ears can be driven independently, so audio can carry a direction; over
    speakers both ears hear both channels and lateralization collapses to mono.
    A Flight Mode session flown on speakers therefore has strictly fewer usable
    feedback dimensions than one flown on headphones, and the two are not the
    same experiment. Recording which one happened is what keeps them from being
    silently pooled later.
    """
    tier: str = "unknown"            # "headphones" | "speakers" | "unknown"
    stereo_separation: bool = False  # can audio carry left/right information?
    private: bool = False            # does the room hear it too?
    latency_ms: int = 0
    device: str = ""
    transport: str = ""
    notes: list = None               # operator-facing warnings

    def tokens(self):
        """Capability tokens a Protocol phase can require."""
        t = set()
        if self.stereo_separation:
            t.add("stereo_audio")
        if self.private:
            t.add("private_audio")
        return t

    def as_dict(self):
        return {"tier": self.tier, "stereo_separation": self.stereo_separation,
                "private": self.private, "latency_ms": self.latency_ms,
                "device": self.device, "transport": self.transport,
                "notes": list(self.notes or [])}


@dataclass
class AudioOutput:
    name: str = "unknown"
    transport: str = ""
    source: str = ""
    kind: str = "unknown"        # "headphones" | "speakers" | "unknown"
    detail: str = ""

    @property
    def is_headphones(self):
        return self.kind == "headphones"

    def describe(self):
        bits = [self.name]
        if self.transport:
            bits.append(f"{self.transport}")
        return " — ".join(bits)


def _parse(text):
    """Pull the default output device out of system_profiler's indented text."""
    blocks = re.split(r"\n(?=        \S)", text)
    for block in blocks:
        if "Default Output Device: Yes" not in block:
            continue
        name = block.strip().split(":")[0].strip()
        def field(label):
            m = re.search(rf"{label}:\s*(.+)", block)
            return m.group(1).strip() if m else ""
        return AudioOutput(name=name, transport=field("Transport"),
                           source=field("Output Source"))
    return None


def _classify(out):
    blob = f"{out.name} {out.source}".lower()
    transport = out.transport.lower()
    if any(h in blob for h in _HEADPHONE_HINTS):
        out.kind = "headphones"
        out.detail = "device name identifies a headset"
    elif any(sp in blob for sp in _SPEAKER_HINTS):
        out.kind = "speakers"
        out.detail = "device name identifies a speaker"
    elif transport == "bluetooth":
        # Bluetooth is a strong hint but not proof -- it could be a BT speaker.
        out.kind = "headphones"
        out.detail = "Bluetooth output (assumed headset — confirm)"
    elif transport == "built-in":
        out.kind = "speakers"
        out.detail = "built-in output"
    else:
        out.kind = "unknown"
        out.detail = f"unrecognised transport {out.transport!r}" if out.transport else ""
    return out


def current_output(timeout=6.0):
    """Best guess at the current default output device.

    Never raises: on any failure it returns an ``unknown`` result, because the
    caller's job is to prompt the human either way and a detection failure must
    not block a session.
    """
    try:
        proc = subprocess.run(
            ["system_profiler", "SPAudioDataType"],
            capture_output=True, text=True, timeout=timeout,
        )
        out = _parse(proc.stdout or "")
        return _classify(out) if out else AudioOutput(detail="no default output found")
    except Exception as exc:  # noqa: BLE001
        return AudioOutput(detail=f"detection failed: {exc}")


def capability(out=None):
    """Translate a detected device into what the session may do with it."""
    out = out or current_output()
    transport = (out.transport or "").lower()
    cap = AudioCapability(tier=out.kind, device=out.name, transport=out.transport,
                          latency_ms=_LATENCY_MS.get(transport, 20), notes=[])
    if out.kind == "headphones":
        cap.stereo_separation = True
        cap.private = True
        # Over-ear cups sit exactly where TP9 and TP10 do. Subject M01 lost both
        # ear electrodes to hair alone; adding clamping force over the same
        # sensors is the most avoidable way to repeat that. We cannot detect cup
        # style, so this is raised as a question rather than a conclusion.
        cap.notes.append(
            "If these are OVER-EAR headphones, the cups press on TP9/TP10 — the "
            "two electrodes most likely to fail already. Prefer in-ear buds with "
            "a Muse, and check the ear contact dots after putting them on.")
        if transport == "bluetooth":
            cap.notes.append(
                f"Bluetooth output adds roughly {cap.latency_ms} ms of one-way "
                "latency. Fine for spoken guidance; it comes out of the budget "
                "for any closed-loop audio feedback.")
    elif out.kind == "speakers":
        cap.stereo_separation = False
        cap.private = False
        cap.notes.append(
            "Speakers: both ears hear both channels, so audio cannot carry "
            "left/right information this session — feedback is mono only.")
        cap.notes.append(
            "Room noise reaches the subject during the eyes-closed blocks and "
            "will show up as arousal. Keep the room quiet, and do not compare "
            "these blocks against a headphone session without noting it.")
    else:
        cap.notes.append(
            "Could not identify the output device. Confirm manually; the session "
            "will assume mono, non-private audio.")
    return cap


def advice():
    """One line for the UI: what we think, and what the human must confirm."""
    out = current_output()
    if out.kind == "headphones":
        return out, (f"Output: {out.describe()} — looks like headphones. "
                     "Confirm they are in your ears before starting.")
    if out.kind == "speakers":
        return out, (f"Output: {out.describe()} — this is NOT headphones. "
                     "Switch output to your earbuds, or continue on speakers.")
    return out, (f"Output: {out.describe()} — could not tell whether these are "
                 "headphones. Check before starting.")
