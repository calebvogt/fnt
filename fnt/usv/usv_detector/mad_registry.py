"""MAD training-file registry — recordings referenced by path, not copied.

A MAD project used to copy every training recording into ``<project>/recordings/``.
That made projects self-contained but duplicated hundreds of MB per file (a
10-minute 250 kHz wav is ~300 MB), and it created two sources of truth: edits to
the copy never reached the original and vice versa.

The copies were never needed for *training*. MAD bakes each confirmed call into
``models/training_data/training_data.h5`` as a self-contained spectrogram patch
plus mask, and ``mad_training`` reads only that store — it never opens a wav. So
the audio is needed for exactly one thing: re-opening a file to look at it. That
is a much weaker dependency than SLEAP has on its videos (SLEAP stores only frame
indices and coordinates, so a missing video means it cannot train at all).

Hence: reference by path, and treat a missing file as a **soft** state. Training,
the example store and every existing detection keep working when a path breaks;
only preview and playback for that one file are unavailable until it is relocated.

Each entry carries a cheap content fingerprint so a moved or renamed file can be
re-identified positively rather than by name alone.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


# Bytes read from each end of the file for the content fingerprint. Wav headers
# differ early and audio differs late, so head+tail+size identifies a recording
# without reading hundreds of MB — the whole point is that this stays fast enough
# to run over a whole project on open.
_FP_CHUNK = 1 << 20  # 1 MiB


def file_fingerprint(path: str) -> str:
    """Short content fingerprint: size + first and last 1 MiB.

    Not a cryptographic identity — it is a *re-identification* key for "is this
    the same recording after a move/rename", where the alternative is matching
    on basename alone. Two genuinely different recordings colliding here would
    need identical size and identical head and tail megabytes.
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return ""
    h = hashlib.sha1()
    h.update(str(size).encode())
    try:
        with open(path, 'rb') as f:
            h.update(f.read(_FP_CHUNK))
            if size > 2 * _FP_CHUNK:
                f.seek(-_FP_CHUNK, os.SEEK_END)
                h.update(f.read(_FP_CHUNK))
    except OSError:
        return ""
    return h.hexdigest()[:16]


@dataclass
class RegisteredFile:
    """One recording a project references (never a copy of it)."""
    path: str = ""                  # last known absolute path
    basename: str = ""
    size: int = 0
    fingerprint: str = ""
    # Set when the audio lives inside the project directory — i.e. a legacy
    # recordings/ copy, or one embedded by "Pack project". Those are owned by
    # the project and get deleted when the entry is removed; referenced files
    # never are.
    embedded: bool = False

    @classmethod
    def from_path(cls, path: str, embedded: bool = False) -> 'RegisteredFile':
        ap = os.path.abspath(path)
        try:
            size = os.path.getsize(ap)
        except OSError:
            size = 0
        return cls(path=ap, basename=os.path.basename(ap), size=size,
                   fingerprint=file_fingerprint(ap), embedded=embedded)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'RegisteredFile':
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    def exists(self) -> bool:
        return bool(self.path) and os.path.isfile(self.path)


def _candidate_dirs(entries: Sequence[RegisteredFile],
                    extra_roots: Iterable[str]) -> List[str]:
    """Directories worth searching for a missing file, most likely first.

    Files move in groups — a whole experiment folder gets relocated — so the
    directories of files we *did* resolve are the best hint for the ones we
    didn't.
    """
    seen: List[str] = []

    def add(d: Optional[str]):
        if not d:
            return
        n = os.path.normcase(os.path.abspath(d))
        if n not in {os.path.normcase(x) for x in seen} and os.path.isdir(d):
            seen.append(d)

    for e in entries:
        if e.exists():
            add(os.path.dirname(e.path))
    for r in extra_roots:
        add(r)
    return seen


def resolve_entries(
    entries: Sequence[RegisteredFile],
    extra_roots: Iterable[str] = (),
    verify_fingerprint: bool = True,
) -> Dict[str, Optional[str]]:
    """Map each entry's stored path to a path that exists now (or None).

    Resolution order, cheapest first:
      1. the stored path itself;
      2. same basename in a directory where another entry did resolve, or in one
         of ``extra_roots`` (the project's known source folders);
      3. give up — the caller shows the file as missing.

    A step-2 hit is accepted when the size matches and, if ``verify_fingerprint``
    and both fingerprints are known, the fingerprints match. Size-only matching
    is deliberately allowed for entries registered before fingerprints existed.
    """
    out: Dict[str, Optional[str]] = {}
    unresolved: List[RegisteredFile] = []
    for e in entries:
        if e.exists():
            out[e.path] = e.path
        else:
            out[e.path] = None
            unresolved.append(e)
    if not unresolved:
        return out

    for d in _candidate_dirs(entries, extra_roots):
        still: List[RegisteredFile] = []
        for e in unresolved:
            cand = os.path.join(d, e.basename)
            if not os.path.isfile(cand):
                still.append(e)
                continue
            try:
                if e.size and os.path.getsize(cand) != e.size:
                    still.append(e)
                    continue
            except OSError:
                still.append(e)
                continue
            if (verify_fingerprint and e.fingerprint
                    and file_fingerprint(cand) != e.fingerprint):
                still.append(e)
                continue
            out[e.path] = os.path.abspath(cand)
        unresolved = still
        if not unresolved:
            break
    return out


def _is_under(path: str, prefix: str) -> bool:
    """True when ``path`` is inside directory ``prefix``.

    Compares whole path components, not raw strings: a plain ``startswith``
    would treat ``D:/data/exp_backup`` as living under ``D:/data/exp`` and
    silently repoint an unrelated tree during a bulk relocate.
    """
    p = os.path.normcase(os.path.abspath(path))
    q = os.path.normcase(os.path.abspath(prefix)).rstrip(os.sep + '/')
    if p == q:
        return True
    return p.startswith(q + os.sep) or p.startswith(q + '/')


def remap_prefix(entries: Sequence[RegisteredFile],
                 old_prefix: str, new_prefix: str) -> int:
    """Repoint every entry under ``old_prefix`` to ``new_prefix``, in place.

    This is what makes a broken project cheap to fix: recordings move as whole
    trees, so relocating one file usually tells you where all of them went. Only
    entries that are currently missing and whose rewritten path actually exists
    are changed — so a wrong guess is a no-op rather than a corruption, and an
    already-resolved file is never dragged somewhere else. Returns the number
    repointed.
    """
    n_fixed = 0
    for e in entries:
        if e.exists():
            continue
        if not _is_under(e.path, old_prefix):
            continue
        rel = os.path.relpath(e.path, old_prefix)
        if rel.startswith(os.pardir):     # not genuinely below the prefix
            continue
        cand = os.path.abspath(os.path.join(new_prefix, rel))
        if os.path.isfile(cand):
            e.path = cand
            e.basename = os.path.basename(cand)
            n_fixed += 1
    return n_fixed


def infer_prefix_change(old_path: str, new_path: str) -> Optional[tuple]:
    """Given one relocated file, deduce the (old_prefix, new_prefix) pair.

    Strips the longest common trailing path segments — if
    ``D:/exp1/mic2/a.wav`` became ``E:/data/exp1/mic2/a.wav``, the shared tail is
    ``exp1/mic2/a.wav`` and the change is ``D:/`` -> ``E:/data``. That pair then
    fixes every sibling in one pass via :func:`remap_prefix`.
    """
    a = Path(os.path.abspath(old_path)).parts
    b = Path(os.path.abspath(new_path)).parts
    i = 0
    while i < min(len(a), len(b)) and a[len(a) - 1 - i] == b[len(b) - 1 - i]:
        i += 1
    if i == 0:
        return None
    old_prefix = str(Path(*a[:len(a) - i])) if len(a) - i > 0 else os.sep
    new_prefix = str(Path(*b[:len(b) - i])) if len(b) - i > 0 else os.sep
    return old_prefix, new_prefix


def entries_from_dicts(items: Iterable[Dict]) -> List[RegisteredFile]:
    return [RegisteredFile.from_dict(d) for d in (items or [])]


def entries_to_dicts(entries: Iterable[RegisteredFile]) -> List[Dict]:
    return [e.to_dict() for e in entries]
