import base64
import faulthandler
import io
import os
import re
from collections import OrderedDict
import sys
import tempfile
import time
import sqlite3
import struct
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import pytz
import json
import shutil
import xml.etree.ElementTree as ET

import cv2
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import (Polygon as MplPolygon, Patch,
                                Rectangle as MplRectangle)
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import matplotlib.dates as mdates
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg, NavigationToolbar2QT)
from scipy.signal import savgol_filter
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QMessageBox,
                             QGroupBox, QCheckBox, QScrollArea, QComboBox,
                             QSpinBox, QDoubleSpinBox, QFrame, QLineEdit,
                             QPlainTextEdit,
                             QDialog, QDialogButtonBox, QFormLayout, QTableWidget,
                             QTableWidgetItem, QHeaderView, QTextEdit, QProgressBar,
                             QDateTimeEdit, QTreeWidget, QTreeWidgetItem,
                             QSplitter, QSlider, QProgressDialog, QGridLayout,
                             QListWidget, QListWidgetItem,
                             QColorDialog, QInputDialog, QMenu)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QDateTime
from PyQt5.QtGui import (QFont, QTextCursor, QImage, QPixmap, QColor,
                         QCursor)

from fnt.uwb.uwb_preview_canvas import (
    UWBPreview2D, UWBPreview3D, PreviewArena, fit_arena_to_data,
    BUILTIN_ARENAS, HAVE_GL as PREVIEW_HAVE_GL, GL_ERROR as PREVIEW_GL_ERROR,
    label_halo)
from fnt.uwb import animation as uwb_animation
from fnt.uwb import uwb_roi


# Columns the processing pipeline actually consumes. Wiser tables carry ~20
# columns, most of them unused TEXT fields (zones, alias, groupnames,
# arenaname) that dominate DataFrame memory: reading all of them for one Echo
# tag cost 5.0 s / 387 MB versus 1.6 s / 37 MB for just these five.
#
# This applies only to the *processed* reads (preview and the smoothed /
# downsampled exports). The raw CSV export still does SELECT * so the
# full-fidelity dump of the database is preserved untouched.
PROCESSING_COLUMNS = ("shortid", "timestamp", "location_x", "location_y",
                      "battery_voltage", "anchors_used")
# Without these the pipeline cannot run at all, so a table missing any of them
# falls back to SELECT * rather than failing on a missing column.
REQUIRED_COLUMNS = ("shortid", "timestamp", "location_x", "location_y")


def get_fnt_version():
    """Resolve the installed FNT version for stamping into export artifacts.

    Mirrors the main GUI's resolution order: read pyproject.toml when running
    from a source checkout, then fall back to installed package metadata (used
    by the PyInstaller build). Returns 'unknown' if neither is available so
    stamping never breaks an export.
    """
    try:
        import tomllib
        from pathlib import Path
        toml_path = Path(__file__).resolve().parent.parent.parent / 'pyproject.toml'
        if toml_path.exists():
            with open(toml_path, "rb") as f:
                data = tomllib.load(f)
            version = data.get('project', {}).get('version')
            if version:
                return version
    except Exception:
        pass
    try:
        from importlib.metadata import version
        return version("fnt")
    except Exception:
        return "unknown"


def list_visible_files(directory):
    """List filenames in ``directory``, excluding hidden and macOS sidecar files.

    Network/exFAT shares written from macOS accumulate AppleDouble companions
    ('._<name>') and '.DS_Store' entries alongside the real files. These are not
    configuration or data files and must never be auto-discovered — e.g.
    '._EchoConfiguration_2024.11.6.xml' would otherwise shadow the real
    'EchoConfiguration_2024.11.6.xml' (its leading '.' can even sort first, so a
    naive scan picks it). Filtering any name starting with '.' covers AppleDouble
    files, .DS_Store, and Unix dotfiles in one rule.

    Returns [] if the directory cannot be read.
    """
    try:
        return [f for f in os.listdir(directory) if not f.startswith('.')]
    except OSError:
        return []


# Fallback spatial context layers for exported plots/animation, used only
# before a preview exists (a config restored for a trial that has not been
# opened yet). In normal use the layers are INHERITED from the Preview Options
# 'Show ...' toggles - see plot_layers_from_preview.
DEFAULT_PLOT_LAYERS = {'background': True, 'zones': True, 'anchors': True,
                       'rois': True}


def _polygon_len(region):
    """Corner count of a zone or region, 0 if it has no usable polygon.

    Written as a function because ``region.get('points') or []`` looks right
    and is not: a drawn region's points are a numpy array, and testing one for
    truth raises rather than returning False.
    """
    pts = region.get('points')
    if pts is None:
        return 0
    try:
        return len(np.asarray(pts, dtype=float).reshape(-1, 2))
    except Exception:
        return 0

# ── Displacement (supplant) detection: PARKED, not removed ────────────────
# FUTURE FEATURE. The detector runs and its thresholds are still configurable
# in code, but nothing about it is offered in the GUI and it contributes no
# rows to any exported file. Parked because the detections have not been
# validated against scored video: an unvalidated behaviour that quietly lands
# in behavior_events.csv is worse than one that is absent, because it looks
# like a finding.
#
# Everything needed to bring it back is intact and covered by the tests:
# behavior_detection.classify still computes `displacing`, extract_events still
# recovers the directed bouts, the five threshold spinboxes are still built
# (hidden) and still round-trip through fnt_config.json, and the plots already
# add a displacement panel whenever displacement bouts exist. Flip this to True
# to restore the whole feature - preview overlay, animation layer, its own
# settings group, and the CSV rows. Grep DISPLACEMENT_ENABLED for every site.
#
# When it does come back, the thing to settle first is whether "arrival +
# resident leaves" is separable from two animals happening to swap places,
# which is what the current thresholds cannot distinguish.
DISPLACEMENT_ENABLED = False

# ── clearing an analysis folder ──────────────────────────────────────────
# Everything in the folder is an export EXCEPT these. A keep-list rather than
# a list of things to delete, and that asymmetry is the whole point: a product
# that has been renamed or retired leaves a file FNT no longer produces, so a
# list of "files we write" is blind to exactly the orphans worth removing.
# VT001 carried a _proximity_bouts.csv (renamed) and a _runSummary.csv
# (retired) for months for this reason - stale files that still looked current.
CLEAR_KEEP_SUFFIXES = (
    '_indexed.sqlite',      # the fast-index copy: gigabytes, and a cache of
    '_indexed.json',        # the SOURCE database rather than an export
    '_indexed.sqlite.partial',
)
CLEAR_KEEP_NAMES = (
    'fnt_config.json',      # the settings record; losing it loses provenance
)

# Files FNT currently writes or has written. Only used to tell the user which
# deletions are expected and which are strangers - never to decide what goes.
_KNOWN_EXPORT_PATTERNS = (
    '_smoothed.csv', '_raw.csv', '_SocialOverlapBouts.csv', '_proximity_bouts.csv',
    '_behavior_events.csv', '_ROI_bout_occupancy.csv', '_ROI_DailySummary.csv',
    # Retired products. Kept in the list so a re-export of a trial processed by
    # an older build reports clearing them as expected, not as strangers.
    '_network_GBI.csv', '_network_edgelist',
    '_zone_occupancy.csv', '_roiSummary.csv',
    '_messageLog.txt', '_runSummary.csv', '_sitemap.png', '.partial',
)
_KNOWN_EXPORT_DIRS = ('plots', 'animation_Tracking', 'animation_SocialNetworks',
                      'csvs')

# Every CSV product goes in one subfolder, so the analysis folder's top level
# is the handful of things a person opens (config, message log, site map) plus
# the plots/animation/csv folders - rather than a dozen tables mixed in with
# them.
CSV_SUBDIR = 'csvs'


def ensure_dir(path, attempts=12, delay=0.25):
    """Create a directory, tolerating a delete that has not landed yet.

    On a NETWORK SHARE a removed directory lingers in a pending-delete state
    until every handle closes. CreateDirectory then fails with ERROR_ALREADY
    EXISTS while os.path.isdir() reports False, so makedirs(exist_ok=True)
    re-raises rather than succeeding - and Clear & Re-export, which deletes
    this very folder moments earlier, walks straight into it.

    Retries briefly instead of failing the export over a race that resolves
    itself in well under a second.
    """
    last = None
    for i in range(attempts):
        try:
            os.makedirs(path, exist_ok=True)
            if os.path.isdir(path):
                return path
        except OSError as e:
            last = e
        time.sleep(delay * (i + 1))
        if os.path.isdir(path):
            return path
    if last is not None:
        raise last
    raise OSError(f"could not create {path}")


def csv_output_dir(output_dir, create=False):
    """The folder CSV products are written to."""
    path = os.path.join(output_dir, CSV_SUBDIR)
    if create:
        ensure_dir(path)
    return path


def check_dir_usable(path):
    """Why ``path`` cannot serve as an output folder, or None if it can.

    Creating and writing a file is not enough: a directory can accept writes
    while refusing to be ENUMERATED, which is the state a network share can
    leave one in after a delete that did not complete. Everything still
    exports, but the folder can never be cleared again, so each run's output
    silently accumulates on the last one's.

    Checked before a long export rather than after, because finding out at the
    end costs the whole run.
    """
    if not os.path.isdir(path):
        return None
    try:
        os.listdir(path)
    except OSError as e:
        return (f"{path} cannot be listed ({e.strerror or e}). It still "
                f"accepts files, so an export would appear to work - but it "
                f"can never be cleared, and old products would survive beside "
                f"the new ones looking identical. Delete this folder in "
                f"Explorer; the next export recreates it.")
    return None


def export_csv_path(output_dir, filename, create=False):
    """Where a CSV product goes, or where an existing one already is.

    Writes always land in the subfolder. Reads prefer it but fall back to the
    analysis folder's top level, because trials exported before this change
    have their CSVs there and re-deriving a multi-gigabyte smoothed CSV just
    because it sits one directory up would be a poor trade.
    """
    nested = os.path.join(csv_output_dir(output_dir, create=create), filename)
    if os.path.exists(nested):
        return nested
    flat = os.path.join(output_dir, filename)
    if os.path.exists(flat):
        return flat
    return nested


def classify_analysis_folder(folder):
    """What a Clear & Re-export would remove from ``folder``.

    Returns ``(removable, kept, unknown)``. Each of ``removable`` and ``kept``
    is a list of ``(relative_path, bytes, is_dir)``; ``unknown`` is the subset
    of ``removable`` that does not match anything FNT is known to write, so the
    caller can show those separately - a file you put there yourself should be
    something you get to see before it goes, not a silent casualty.
    """
    removable, kept, unknown = [], [], []
    if not folder or not os.path.isdir(folder):
        return removable, kept, unknown

    def _dir_size(path):
        total = 0
        for root, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return total

    for entry in sorted(os.listdir(folder)):
        full = os.path.join(folder, entry)
        is_dir = os.path.isdir(full)
        try:
            size = _dir_size(full) if is_dir else os.path.getsize(full)
        except OSError:
            size = 0
        if entry in CLEAR_KEEP_NAMES or entry.endswith(CLEAR_KEEP_SUFFIXES):
            kept.append((entry, size, is_dir))
            continue
        removable.append((entry, size, is_dir))
        known = (entry in _KNOWN_EXPORT_DIRS if is_dir
                 else any(p in entry for p in _KNOWN_EXPORT_PATTERNS))
        if not known:
            unknown.append((entry, size, is_dir))
    return removable, kept, unknown


def clear_analysis_folder(folder, log=None):
    """Delete every export in ``folder``, keeping the config and fast index.

    Returns ``(removed_count, freed_bytes, failures)``. A file that cannot be
    deleted - open in Excel, locked by another process - is reported rather
    than raised on, because a partial clear followed by a full re-export is a
    recoverable state while an aborted export is not.
    """
    removable, _kept, _unknown = classify_analysis_folder(folder)
    removed, freed, failures = 0, 0, []
    for name, size, is_dir in removable:
        full = os.path.join(folder, name)
        try:
            if is_dir:
                # EMPTIED, not removed. Deleting a directory and recreating it
                # moments later races on a network share, where the removal is
                # pending until every handle closes - which is exactly what
                # Clear & Re-export does to csvs/. Leaving the (now empty)
                # directory in place removes the race entirely.
                entries = os.listdir(full)
                if not entries:
                    # Already empty, so there is nothing to clear and nothing
                    # to report: a second clear must be a genuine no-op, not a
                    # count of the directories the first one left behind.
                    continue
                for entry in entries:
                    child = os.path.join(full, entry)
                    if os.path.isdir(child):
                        shutil.rmtree(child)
                    else:
                        os.remove(child)
            else:
                os.remove(full)
            removed += 1
            freed += size
        except OSError as e:
            failures.append(f"{name}: {e}")
    if log:
        log(f"Cleared {removed} item(s) from "
            f"{os.path.basename(folder)} ({human_bytes(freed)} freed)")
        if failures:
            # A clear that did not clear is not a detail. Whatever survived is
            # about to sit beside this run's output looking exactly like it,
            # and a product this run does not write would be read as if it did.
            log(f"  ⚠ CLEAR INCOMPLETE - {len(failures)} item(s) could "
                f"NOT be removed:")
            for f in failures:
                log(f"      {f}")
            log("      Files left behind are from an EARLIER run. Anything "
                "this export does not overwrite will survive and look "
                "current. Delete the folder by hand before trusting a "
                "product this run did not write.")
    return removed, freed, failures


def human_bytes(n):
    """A byte count at the precision a person actually reads."""
    n = float(n)
    for unit in ('B', 'KB', 'MB', 'GB'):
        if abs(n) < 1024.0 or unit == 'GB':
            return f"{n:.0f} {unit}" if unit == 'B' else f"{n:.1f} {unit}"
        n /= 1024.0


# Defined in dataview.py so animation.py can format panel cells without
# importing THIS module - which imports animation.py, and would close the
# circle. Re-exported under the name callers already use.
from fnt.uwb.dataview import human_duration          # noqa: E402,F401


# A parked tag reports the same spot over and over, so its MAD is 0 and every
# fix is infinitely many deviations from the median. The floor says "never
# call anything an outlier for being more than 5 cm off a perfectly still
# track", which is below the positional noise of the system anyway.
HAMPEL_MAD_FLOOR_M = 0.05

SOCIAL_BOUTS_TOOLTIP = (
    "FILE: {db}_SocialOverlapBouts.csv\n"
    "\n"
    "THE social product. One row per dyad per contiguous contact bout: "
    "animal1, animal2, Day, Date, bout_start, bout_stop, duration_s, "
    "mean_distance, n_reads.\n"
    "\n"
    "CONTACT = centres within 2x the social radius (the preview's dotted "
    "circles overlapping). Measured on the filtered and smoothed track at "
    "full temporal resolution, so no fixes are dropped before pairwise "
    "distances are computed.\n"
    "\n"
    "STRICTLY DYADIC, no chain rule. A pair appears only if those two animals "
    "were themselves within the radius — and a bout does NOT fragment when a "
    "third animal joins, because the pair's own relationship did not change.\n"
    "\n"
    "GROUPS ARE RECOVERABLE FROM THIS FILE, exactly. Pool every bout_start "
    "and bout_stop, and between consecutive boundaries the contact graph is "
    "constant; connected components on each interval give chain-rule groups "
    "with true durations. The reverse is not possible, which is why the GBI "
    "and the windowed edge list were removed rather than exported alongside: "
    "both were lossy projections of this table. See "
    "SOCIAL_OVERLAP_ANALYSIS.md for worked R.\n"
    "\n"
    "NOT SPLIT AT MIDNIGHT. Day is the day the bout STARTED.\n"
    "\n"
    "THRESHOLD-SPECIFIC. To analyse a different distance, re-export with a "
    "new social radius; raw per-timestamp distances are not written.")

ROI_BOUTS_TOOLTIP = (
    "FILE: {db}_ROI_bout_occupancy.csv\n"
    "\n"
    "One row per VISIT: a run of consecutive reads inside one drawn region. "
    "Columns: SexID, Day, ROI, bout_start, bout_stop, duration_s, n_reads.\n"
    "\n"
    "STRICT. A single fix outside the region ends the bout, and the next fix "
    "back inside starts a new one. Nothing is bridged, no dropout is filled "
    "in, and no duration is rounded — millisecond edges throughout.\n"
    "\n"
    "A LONE READ inside a region reports identical bout_start and bout_stop "
    "and duration 0.0 s, with n_reads = 1. That is what was observed: the "
    "visit is real but its length is not measurable from a single read. So "
    "duration_s IS bout_stop - bout_start on every row without exception, "
    "occupancy totals are a strict LOWER BOUND on the truth, and n_reads is "
    "what tells you how much evidence stands behind a row.\n"
    "\n"
    "Count entrances with the ROW COUNT, not with time — a lone-read visit "
    "adds 1 to the count and 0 to the seconds.\n"
    "\n"
    "NOT SPLIT AT MIDNIGHT. A bout that crosses midnight stays one row and "
    "Day is the day it STARTED. Splitting would destroy the difference "
    "between one long visit and two short ones; you can always split "
    "downstream, but you cannot un-split.\n"
    "\n"
    "Keyed on the ANIMAL, not the tag: two tags configured to one identity "
    "are one animal, and their visits are unioned so a replacement tag "
    "cannot bill the same minute twice.\n"
    "\n"
    "Regions are the ones DRAWN in the preview. Site-XML zones are measured "
    "only once imported with 'Import XML' beside the region list. Where two "
    "regions overlap the fix goes to whichever takes precedence, never to "
    "both, so an animal is in at most one region at a time.")

ROI_DAILY_TOOLTIP = (
    "FILE: {db}_ROI_DailySummary.csv\n"
    "\n"
    "Who held each region, each day. One row per animal per region per day, "
    "for animals that actually entered it. Columns: SexID, Day, ROI, "
    "total_ROI_time_s, focal_ROI_time_s, focal_ROI_perc_time, rank_order, "
    "n_reads.\n"
    "\n"
    "A row may show 0 s with n_reads ≥ 1: the animal was there, in lone reads "
    "whose duration cannot be measured. Ranking is by TIME, so those rows sit "
    "last — n_reads is how you see them at all.\n"
    "\n"
    "total_ROI_time_s is EVERY animal's time in that region that day, so "
    "focal_ROI_perc_time is a share of the region's whole traffic and "
    "rank_order says who owned it — rank 1 is the animal holding the largest "
    "share. Tied shares take the average rank, so two animals level for "
    "third are both 3.5.\n"
    "\n"
    "THE ONLY FILE THAT SPLITS AT MIDNIGHT, because it is the only one that "
    "is definitionally per-day. A bout crossing midnight is divided at the "
    "boundary and its seconds land on both days, so this file's seconds sum "
    "to exactly the bout file's totals.\n"
    "\n"
    "Computed from the same visits as the bout CSV, but exported "
    "independently — tick either, both or neither.")

ROI_IMPORT_XML_TOOLTIP = (
    "Copy the site XML's zones in as regions of interest, so they can be "
    "edited, reordered, recoloured, renamed or deleted like any region you "
    "drew yourself.\n"
    "\n"
    "This is the ONLY route from an XML zone to an occupancy statistic. The "
    "occupancy outputs measure drawn regions and nothing else — one region "
    "pipeline, not two — so a zone that has not been imported is not "
    "measured.\n"
    "\n"
    "The copy is one-way and taken once. Editing an imported region changes "
    "the region, never the XML, and re-importing later will not pull your "
    "edits back out or overwrite them: a zone whose name is already in the "
    "list is skipped, so pressing this twice is safe.\n"
    "\n"
    "Zone boundaries were authored against the raw fix stream, which is noisy "
    "enough that they are often placed by guesswork. Check them against the "
    "smoothed track before trusting the numbers, and drag the corners where "
    "they do not fit.\n"
    "\n"
    "Imported regions can overlap each other or the ones you drew; you will "
    "be asked which takes precedence, since a fix inside two regions is "
    "credited to exactly one.")

DATA_VIEW_TOOLTIP = (
    "Add a panel beside the arena showing time in each region accumulating "
    "in step with the video — one row per animal, one column per region.\n"
    "\n"
    "Counters run from the START OF THE TRIAL, not the start of the clip, so "
    "the last frame carries the trial's real totals: scrub the preview to the "
    "end, render a 10 s clip, and the numbers you read are the ones the "
    "summary CSV holds.\n"
    "\n"
    "The numbers are read from the same intervals the CSV sums, at each "
    "frame's own timestamp, so they cannot drift with the video's speed or "
    "frame rate. Units follow the value (0.0s → m → h → d).\n"
    "\n"
    "SOCIAL OVERLAP IS NOT SHOWN. Recomputing proximity here would give the "
    "trial a second opinion on who was together and for how long, competing "
    "with the exported bouts — and catching that kind of disagreement is what "
    "this panel is for. Social overlap lives in the Social Overlap Bouts CSV, "
    "the GBI and the edge list.\n"
    "\n"
    "The first render after a settings change reads the whole recording to "
    "build the counters, which takes a while on a long trial; it is then "
    "reused. The video is also wider, to fit the panel without shrinking the "
    "arena."
)


ADJACENCY_FILTER_TOOLTIP = (
    "Removes short runs of fixes that only look plausible because their "
    "neighbours were already removed.\n"
    "\n"
    "Runs last, after the three thresholds above — it can only test the joins "
    "their removals create.\n"
    "\n"
    "The other thresholds measure each fix against the one before it and "
    "delete in a single pass. Deleting changes who is next to whom, and "
    "nothing re-checks the new pairs. A RUN of bad fixes therefore loses only "
    "its first and last members — the ones with a visible step — while the "
    "middle, which is internally consistent, survives attached to the real "
    "track by a stride nobody looked at. That is how a tag can appear to "
    "cross the arena in half a second with every threshold satisfied.\n"
    "\n"
    "This re-measures what is left, splits the track wherever a join is still "
    "impossible, and removes a displaced run only when the track joins up "
    "sensibly without it — the test that separates a ghost from an animal "
    "that really went somewhere.\n"
    "\n"
    "WHAT THE NUMBER MEANS. It is a LENGTH LIMIT on what may be deleted, not "
    "a detection threshold. A stranded run SHORTER than this many seconds is "
    "removed; a run LONGER is kept and reported. Raising it deletes more; "
    "lowering it deletes less and leaves more flagged.\n"
    "\n"
    "Nothing is deleted for merely being long. A run is only ever removed "
    "when the join into it is impossible AND the track reads sensibly once "
    "it is gone. The seconds are the cap on how much evidence may be "
    "discarded on that judgement, because deleting minutes of data on an "
    "inference is worse than saying the track is broken there and letting "
    "you look.\n"
    "\n"
    "Judged on SPEED, not distance: 7 m in half a second is a teleport, 7 m "
    "in a minute is a walk.\n"
    "\n"
    "HOW MANY IT TAKES is in the threshold read-out under the preview, on "
    "its own Excursion row. Expect it to remove MORE than the three tests "
    "above on a tag with long ghost runs - those tests strand the runs, this "
    "one clears them. If the read-out still warns that the largest remaining "
    "step exceeds the jump threshold, this filter looked at that join and "
    "declined: the run was longer than the seconds set here."
)

OUTLIER_FILTER_TOOLTIP = (
    "Remove fixes that sit far off the LOCAL track, judged against the median of a window of neighbouring fixes rather than against the previous fix alone (a Hampel filter).\n\nWHY THIS AND NOT JUST THE JUMP THRESHOLD. A jump test asks how far a fix is from the one before it, so a RUN of consecutive ghosts defeats it: the first ghost is caught, the second looks like a small step from the first and survives, and the real fix after the run gets deleted in its place. A median over many samples cannot be moved by a few wild values, so every fix in the run is rejected and the honest ones are kept.\n\nOn this project's pilot data, adding it alongside the jump threshold took physically impossible steps (over a metre inside a second) from 884 down to 69 across six tags, while keeping 99% of fixes.")

OUTLIER_WINDOW_TOOLTIP = (
    "How many neighbouring fixes the local median is taken over. Forced odd so the window has a centre.\n\nIt must be more than TWICE the longest run of consecutive ghosts you want to survive - a run longer than half the window becomes the majority and moves the median it is being judged against. Short windows fail quietly for exactly this reason: on the pilot data a window of 11 left MORE impossible steps than the jump threshold alone, while 31 removed almost all of them.\n\nThis is in SAMPLES, not seconds, so its real-time span follows the reporting rate.")

OUTLIER_SIGMA_TOOLTIP = (
    "How far off the local median a fix may sit before it is rejected, in MADs (median absolute deviations) of the same window.\n\nMAD is the median of each fix's distance from the window's median - a spread measure built from medians, so, unlike a standard deviation, a wild value cannot inflate it and hide inside it. It is scaled by 1.4826 so that on ordinary noise one MAD equals one standard deviation, which is why this reads on the familiar sigma scale.\n\n6 means 'further out than essentially any honest fix'. Lower it to cut harder; raise it if a real fast movement is being clipped at its onset.")

VELOCITY_MIN_DT_TOOLTIP = (
    "The shortest interval the velocity threshold is allowed to divide by.\n\nSpeed is distance over time, and this system's reports can arrive a millisecond apart. Dividing a few centimetres of positional noise by a millisecond gives tens of m/s, so an unfloored velocity threshold mostly measures HOW CLOSELY SPACED THE REPORTS WERE rather than how fast the animal moved - it deletes tightly spaced fixes wherever they occur.\n\nOn this project's pilot data the raw interval put 41% of fixes over 1 m/s with a median step of only 9 cm; flooring at 0.5 s brings that to 2%.\n\nSet 0 for the old unfloored behaviour (what configs written before this option restore, so an old trial reproduces exactly).")


def hampel_deviation(x, y, window=31, group=None,
                     mad_floor=HAMPEL_MAD_FLOOR_M):
    """How many MADs each fix sits from its local median, worst of x and y.

    The score the threshold is compared against, returned separately so the
    live stats can re-price a different sigma without recomputing the medians
    - and so the preview can show what a change would cost before it is made.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.zeros(len(x), dtype=float)
    if len(x) == 0:
        return out
    if group is not None:
        g = np.asarray(group)
        if g.ndim != 1:
            raise ValueError("group must be one scalar key per fix")
        for val in np.unique(g):
            m = g == val
            out[m] = hampel_deviation(x[m], y[m], window, None, mad_floor)
        return out
    k = max(3, int(window) | 1)
    if len(x) < k:
        return out
    pad = k // 2
    for arr in (x, y):
        filled = pd.Series(arr).ffill().bfill().to_numpy()
        win = sliding_window_view(np.pad(filled, pad, mode='edge'), k)
        med = np.median(win, axis=1)
        mad = np.median(np.abs(win - med[:, None]), axis=1) * 1.4826
        out = np.maximum(out, np.abs(filled - med)
                         / np.maximum(mad, float(mad_floor)))
    return out


def hampel_keep(x, y, window=31, n_sigma=6.0, group=None,
                mad_floor=HAMPEL_MAD_FLOOR_M):
    """Boolean mask of fixes that are NOT spatial outliers of the local track.

    A Hampel filter compares each coordinate against the MEDIAN of a window
    centred on it, and measures the distance in units of that window's median
    absolute deviation (MAD). A fix further than ``n_sigma`` MADs from the
    local median in x or y is an outlier.

    Median and MAD are used rather than mean and standard deviation because
    both are ROBUST: a handful of wild values inside the window cannot drag
    them, whereas one 20 m ghost moves a mean and inflates an SD enough to
    hide itself. That is the whole reason this catches what a step-distance
    threshold cannot - a ghost fix is an outlier against the surrounding
    track even when it lands close to the ghost before it, so RUNS of
    consecutive ghosts (which defeat a step test, because the second ghost
    looks like a small step from the first) are still rejected as long as the
    run is shorter than half the window.

    MAD is scaled by 1.4826 so that for normally distributed noise one MAD
    equals one standard deviation; ``n_sigma`` then reads on the familiar
    scale, and 6 means "further out than essentially any honest fix".

    ``window`` is in SAMPLES, not seconds, and is forced odd so it can be
    centred. ``group`` optionally splits the track (per tag, per time-gap
    segment) so no window ever spans two animals or a dropout.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.ones(len(x), dtype=bool)
    if len(x) == 0:
        return keep
    if group is not None:
        g = np.asarray(group)
        if g.ndim != 1:
            raise ValueError("group must be one scalar key per fix")
        for val in np.unique(g):
            m = g == val
            keep[m] = hampel_keep(x[m], y[m], window, n_sigma, None, mad_floor)
        return keep

    k = max(3, int(window) | 1)          # odd, so the window has a centre
    if len(x) < k:
        return keep                      # too short to have a local median
    pad = k // 2
    for arr in (x, y):
        # A NaN inside the window would poison its median, so hold the last
        # good value; a NaN fix is dropped by the coordinate check anyway.
        filled = pd.Series(arr).ffill().bfill().to_numpy()
        win = sliding_window_view(np.pad(filled, pad, mode='edge'), k)
        med = np.median(win, axis=1)
        mad = np.median(np.abs(win - med[:, None]), axis=1) * 1.4826
        keep &= np.abs(filled - med) <= float(n_sigma) * np.maximum(
            mad, float(mad_floor))
    return keep & np.isfinite(x) & np.isfinite(y)


# The smoothing methods that read the window control as something other than a
# window, spelled once so the config writer and the loader cannot drift apart.
EWMA_METHOD = "Forward-Backward Exponentially Weighted Moving Average"
SAVGOL_METHOD = "Savitzky-Golay"


def smoothing_window_fields(method, window, units, *, window_key,
                            units_key, span_key="smoothing_span"):
    """How a method's window settings should be SERIALISED.

    Only the rolling methods actually have a window. Savitzky-Golay derives
    its own from the data (31 samples or the group length, whichever is
    smaller) and never reads the control; EWMA reads the same control as a
    sample SPAN, with no units at all - the units combo is hidden for it, so
    whatever it holds is left over from another method.

    Writing a number and a unit for those two asserted something the run had
    not done: a config reading ``window: 30, window_units: "Seconds"`` under
    EWMA had in fact smoothed with a 30-SAMPLE span, and nothing in the file
    said so. Null means "not applicable to this method", and ``smoothing_span``
    names the one value that did apply.

    Returns a dict of the keys to merge into the config.
    """
    if method == EWMA_METHOD:
        return {window_key: None, units_key: None, span_key: int(window)}
    if method == SAVGOL_METHOD:
        # No span key either: Savitzky-Golay takes no parameter from the user,
        # so there is nothing to record.
        return {window_key: None, units_key: None}
    return {window_key: window, units_key: units}


def smoothing_window_value(cfg, *, window_key, span_key="smoothing_span"):
    """The window/span to restore from a config, or None if it said nothing.

    ``smoothing_span`` wins where present. Falling back to the window key
    keeps every config written before this split loading correctly: those
    stored the EWMA span in the window field, so the number is right even
    though the label was not.
    """
    span = cfg.get(span_key)
    if isinstance(span, (int, float)) and not isinstance(span, bool):
        return int(span)
    val = cfg.get(window_key)
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return int(val)
    return None


def draw_context_layers(ax, layers, *, bg_image=None, bg_extent=None,
                        zones_xml=None, zones_df=None, anchors=None,
                        rois=None):
    """Draw the background image, zone polygons, ROIs and anchors on ``ax``.

    Gated by ``layers`` (a dict with 'background'/'zones'/'anchors'/'rois'
    booleans) so every exported spatial figure honours the same per-export
    choice. Sources are passed explicitly because the worker and the main
    window hold them in different forms (loaded PNG vs. XML map image;
    xml_zones list vs. arena_zones DataFrame).

    ``rois`` are the user's own regions, drawn over the preview rather than
    authored in the site XML. They are outlined in their own colour and given
    only a faint wash, because the point of drawing one is to read tracks
    against it - a solid fill would bury the tracks it exists to frame.
    """
    layers = layers or {}
    if layers.get('background') and bg_image is not None and bg_extent is not None:
        ax.imshow(bg_image, extent=list(bg_extent), origin='upper',
                  aspect='auto', alpha=0.6, zorder=0)
    if layers.get('zones'):
        if zones_xml:
            for z in zones_xml:
                pts = z.get('points')
                if pts is None or len(pts) < 3:
                    continue
                is_bounds = (z.get('name', '') or '').strip().lower() == 'arena'
                ax.add_patch(MplPolygon(
                    pts, closed=True,
                    facecolor='none' if is_bounds else z.get('color', '#888'),
                    edgecolor=z.get('color', '#888'),
                    alpha=1.0 if is_bounds else 0.22,
                    linewidth=1.8 if is_bounds else 1.0, zorder=1))
        elif zones_df is not None and not zones_df.empty:
            for zone_name in zones_df['zone'].unique():
                coords = zones_df[zones_df['zone'] == zone_name][['x', 'y']].values
                if len(coords) >= 3:
                    ax.add_patch(MplPolygon(coords, closed=True, fill=False,
                                            edgecolor='black', linewidth=1.5,
                                            linestyle='--', zorder=1))
    if layers.get('rois', True) and rois:
        for r in rois:
            pts = np.asarray(r.get('points'), dtype=float).reshape(-1, 2)
            if len(pts) < 3:
                continue
            col = r.get('color', '#e6194b')
            ax.add_patch(MplPolygon(pts, closed=True, facecolor=col,
                                    alpha=0.10, edgecolor='none', zorder=1.5))
            ax.add_patch(MplPolygon(pts, closed=True, facecolor='none',
                                    edgecolor=col,
                                    linewidth=float(r.get('linewidth', 2.0)),
                                    zorder=1.6))
            ax.annotate(r.get('name', ''),
                        (pts[:, 0].mean(), pts[:, 1].mean()),
                        color=col, fontsize=7, fontweight='bold',
                        ha='center', va='center', zorder=5,
                        path_effects=label_halo(col, 2.0))
    if layers.get('anchors') and anchors:
        ax.scatter([a['x'] for a in anchors], [a['y'] for a in anchors],
                   marker='^', s=40, c='#f2c24f', edgecolors='none', zorder=2)


def _dyad_type(sex_a, sex_b):
    """MM / FF / MF / NA for a pair of sexes.

    Mixed pairs are ALWAYS 'MF'. Sorting the two letters would yield 'FM',
    which is not the convention used elsewhere and would silently break a
    downstream filter written against 'MF'.
    """
    a = str(sex_a or 'U')[:1].upper()
    b = str(sex_b or 'U')[:1].upper()
    if 'U' in (a, b) or a not in 'MF' or b not in 'MF':
        return 'NA'
    return a + b if a == b else 'MF'


def is_network_path(path):
    """True if ``path`` lives on a network filesystem (SMB/NFS/AFP share).

    Used to warn before the tracking animation writes its temporary frames
    somewhere slow: the renderer churns through frame data continuously, and
    doing that over a share is far slower than local disk.

    Cross-platform: UNC paths anywhere, mapped drive letters on Windows, and
    real mount types on macOS/Linux (an SMB share on macOS lands in /Volumes
    exactly like a USB disk, so the path alone cannot tell them apart).
    """
    try:
        p = os.path.abspath(path)
        if p.startswith('\\\\') or p.startswith('//'):
            return True
        if os.name == 'nt':
            drive = os.path.splitdrive(p)[0]
            if drive:
                import ctypes
                DRIVE_REMOTE = 4
                return ctypes.windll.kernel32.GetDriveTypeW(
                    drive + '\\') == DRIVE_REMOTE
            return False
        # POSIX: ask the OS which mount points are network filesystems.
        import subprocess
        out = subprocess.run(['mount'], capture_output=True, text=True,
                             timeout=5).stdout
        net_types = ('smbfs', 'afpfs', 'nfs', 'cifs', 'webdav', 'ftpfs')
        for line in out.splitlines():
            if ' on ' not in line:
                continue
            rest = line.split(' on ', 1)[1]
            mount_point = rest.split(' (')[0].strip()
            opts = rest[rest.find('('):] if '(' in rest else ''
            if not any(t in opts for t in net_types):
                continue
            if p == mount_point or p.startswith(mount_point.rstrip('/') + '/'):
                return True
        return False
    except Exception:
        pass
    return False

def _onedrive_sync_roots():
    """Folders OneDrive is ACTUALLY syncing, per its own registry state.

    A folder merely named 'OneDrive' proves nothing: Windows' Known Folder Move
    can redirect the Desktop into C:/Users/<you>/OneDrive and leave it there
    after that account is signed out, so the path looks synced while nothing
    syncs. Only the roots OneDrive registers are real.
    """
    roots = []
    if os.name != 'nt':
        return roots
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            "Software" + os.sep + "Microsoft" + os.sep
                            + "OneDrive" + os.sep + "Accounts") as key:
            i = 0
            while True:
                try:
                    account = winreg.EnumKey(key, i)
                except OSError:
                    break
                i += 1
                try:
                    with winreg.OpenKey(
                            key, account + os.sep
                            + "ScopeIdToMountPointPathCache") as scope:
                        j = 0
                        while True:
                            try:
                                _n, val, _t = winreg.EnumValue(scope, j)
                            except OSError:
                                break
                            j += 1
                            if val:
                                roots.append(os.path.abspath(str(val)))
                except OSError:
                    continue
    except Exception:
        pass
    return roots


def cloud_sync_provider(path):
    """Name of the cloud-sync service backing ``path``, or None.

    Writing a multi-GB render into a synced folder makes the sync client
    upload it continuously while the animation is still being written, which
    competes for disk and network and can lock the file mid-render. Worth a
    warning even though the folder is technically local.

    Note Windows' OneDrive Backup redirects Desktop/Documents into OneDrive,
    so a folder the user picked as 'the Desktop' can be synced without any
    obvious sign in the path they chose.
    """
    try:
        p = os.path.abspath(path)
        # Prefer OneDrive's own record of what it syncs. If it lists any roots,
        # trust it completely: a path outside them is NOT synced, however it is
        # named.
        roots = _onedrive_sync_roots()
        if roots:
            for root in roots:
                if p == root or p.startswith(root + os.sep):
                    return 'OneDrive'
            return None
        known = (('onedrive', 'OneDrive'), ('dropbox', 'Dropbox'),
                 ('google drive', 'Google Drive'), ('googledrive', 'Google Drive'),
                 ('icloud drive', 'iCloud Drive'), ('box sync', 'Box'))
        for seg in p.replace('/', os.sep).split(os.sep):
            low = seg.strip().lower()
            for key, name in known:
                if low == key or low.startswith(key + ' -'):
                    return name
    except Exception:
        pass
    return None

def connect_ro(path):
    """Open a SQLite database read-only for querying, safely over a network drive.

    FNT's source databases (and the index/preview DBs written alongside them)
    frequently live on mapped/network drives. SQLite explicitly warns against
    network filesystems: its normal file locking and memory-mapped reads are
    unreliable over SMB and can fault the whole process with a native access
    violation instead of a catchable error (see ~/.fnt/faulthandler_crash.log,
    the crash inside pandas ``_fetchall_as_list``).

    This opens the file with ``mode=ro&immutable=1``: SQLite treats it as stable
    read-only media, skipping ALL locking and change-detection and never touching
    ``-wal``/``-shm``/journal side files on the share. ``PRAGMA mmap_size=0`` then
    forces plain ``read()`` I/O, so a bad remote page surfaces as a normal
    ``sqlite3`` exception rather than a segfault.

    Caveat: ``immutable=1`` means writes to the file while it is open are NOT
    detected. Only use this for databases FNT treats purely as read-only sources
    (all query paths). Writers must keep using ``sqlite3.connect`` directly.
    """
    import pathlib
    try:
        uri = pathlib.Path(path).absolute().as_uri() + "?mode=ro&immutable=1"
        conn = sqlite3.connect(uri, uri=True)
    except Exception:
        # If URI construction/open fails for any reason, fall back to the legacy
        # plain connection so behavior never regresses versus before this fix.
        conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA mmap_size=0")
    except Exception:
        pass
    return conn


def is_corruption_error(err):
    """True if ``err`` is SQLite reporting a damaged database file.

    Corruption usually surfaces mid-scan rather than at open: the header and
    schema pages read fine, so the file opens and lists its tables, and only a
    query that reaches the damaged page fails. Recognising it explicitly lets
    the UI say so once, loudly, instead of leaving an empty tag list behind a
    one-line log message.
    """
    msg = str(err).lower()
    return ("malformed" in msg          # "database disk image is malformed"
            or "not a database" in msg  # wrong/truncated header
            or "corrupt" in msg)


def processing_select_clause(conn, table_name):
    """SELECT list covering PROCESSING_COLUMNS present in ``table_name``.

    Intersects with the real schema so tables written by different Wiser
    versions still work, and degrades to ``*`` if anything essential is absent.
    """
    try:
        have = {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})")}
    except Exception:
        return "*"
    if not have or any(c not in have for c in REQUIRED_COLUMNS):
        return "*"
    return ", ".join(c for c in PROCESSING_COLUMNS if c in have)


def resolve_adjacency(t, x, y, keep, *, group=None, v_thresh=None, min_dt=0.0,
                      j_thresh=None, max_excursion_s=5.0,
                      max_removed_frac=0.10, max_passes=32):
    """Re-apply the step tests to the adjacency that REMOVAL creates.

    The ordinary masks measure every point against its neighbour in the
    original sequence and are applied once. Deleting a point makes its
    neighbours adjacent, and that new pair is never tested - so a RUN of bad
    fixes loses only its first and last members (the ones with visible steps)
    while its interior, which is internally consistent, survives joined to the
    real track by an impossible stride.

    Each pass re-measures the survivors and drops the arrival point of any
    join that is still impossible, until nothing changes. Velocity is the test
    that matters here rather than distance: after a removal the gap in TIME
    has grown too, and 7 m is a teleport across half a second and an ordinary
    walk across a minute. ``j_thresh`` is applied as well when given.

    Bounded by ``max_passes`` because convergence is not guaranteed: where a
    tag has genuinely relocated and stayed, each pass peels one more point off
    the far side, and it only settles once enough time has elapsed to make the
    move plausible. Returns (mask, removed, converged).
    """
    idx = np.flatnonzero(keep)
    removed = 0
    if idx.size < 3:
        return keep, 0, True

    def bad_joins(ii):
        d = np.hypot(np.diff(x[ii]), np.diff(y[ii]))
        dt = np.diff(t[ii])
        b = np.zeros(d.shape, dtype=bool)
        if v_thresh:
            b |= d / np.maximum(dt, min_dt if min_dt else 1e-9) > v_thresh
        if j_thresh:
            b |= d > j_thresh
        if group is not None:
            # A join between two tags, or across a reporting gap, is not a
            # step at all - the existing grouping already says not to measure
            # there, and testing it would delete the first fix after every gap.
            b &= group[ii][1:] == group[ii][:-1]
        return b

    def joins_cleanly(a, b):
        if group is not None and group[a] != group[b]:
            return True
        d = float(np.hypot(x[b] - x[a], y[b] - y[a]))
        gap = float(t[b] - t[a])
        if v_thresh and d / max(gap, min_dt if min_dt else 1e-9) > v_thresh:
            return False
        if j_thresh and d > j_thresh:
            return False
        return True

    for _ in range(max_passes):
        bad = bad_joins(idx)
        if not bad.any():
            break
        # Split at the impossible joins and judge whole SEGMENTS, never single
        # points. Dropping the arrival point of every bad join eats whichever
        # side follows the discontinuity - so a ghost that goes out and comes
        # back would take the real track after it with it.
        cuts = np.flatnonzero(bad) + 1
        starts = np.concatenate([[0], cuts])
        ends = np.concatenate([cuts, [idx.size]])
        segs = list(zip(starts, ends))
        drop = np.zeros(idx.size, dtype=bool)
        # INTERIOR segments first, and by the only criterion that actually
        # justifies deleting data: the track carries on sensibly without this
        # piece. A segment whose removal reconnects the two sides was an
        # excursion; one whose removal leaves the join impossible was not.
        for k, (s, e) in enumerate(segs):
            if not (0 < k < len(segs) - 1):
                continue
            if float(t[idx[e - 1]] - t[idx[s]]) > max_excursion_s:
                continue
            if joins_cleanly(idx[s - 1], idx[e]):
                drop[s:e] = True
        if not drop.any() and len(segs) > 1:
            # An END segment reconnects nothing, so that test cannot vouch for
            # it. Only a clear minority is removed - otherwise a recording
            # that simply begins or ends somewhere else would lose real data
            # on nothing more than being first.
            for k in (0, len(segs) - 1):
                s, e = segs[k]
                other = segs[1] if k == 0 else segs[-2]
                if float(t[idx[e - 1]] - t[idx[s]]) > max_excursion_s:
                    continue
                if (e - s) <= 0.05 * (other[1] - other[0]):
                    drop[s:e] = True
        if not drop.any():
            break                          # discontinuities remain, but real
        # A floor as well as a fraction: on a short stretch any excursion is a
        # large share of it, and a percentage alone would refuse to remove a
        # six-fix ghost from a twenty-fix window.
        budget = max(max_removed_frac * keep.sum(), 50)
        if removed + int(drop.sum()) > budget:
            # A safety valve, not a threshold. Wanting to remove this much
            # means the track is not a good track with a few ghosts in it -
            # it is substantially disputed - and quietly deleting a large
            # share of a recording on a heuristic is not a call this should
            # make. Stop, keep what is left, and let it be reported.
            break
        idx = idx[~drop]
        removed += int(drop.sum())
        if idx.size < 3:
            break

    out = np.zeros_like(keep)
    out[idx] = True
    resolved = bool(idx.size < 2 or not bad_joins(idx).any())
    return out, removed, resolved


def collapse_multi_solutions(df, log=None, label=""):
    """One position per (tag, instant); returns (frame, stats).

    The Wiser export can carry SEVERAL rows for the same shortid and the same
    millisecond, each an independent solution from a different anchor subset
    (different ``reportid`` and ``anchors_list``). They are not duplicate rows:
    in VT001, 3,974,096 instants carry more than one row and exactly 8 of them
    hold identical coordinates.

    Nothing in the schema marks them as alternatives, so read as a time series
    they look like movement between positions in zero elapsed time. That does
    two things: ~10 cm of jitter at every instant roughly DOUBLES measured path
    length, and the ~1% of instants whose solutions diverge past a metre seed
    apparent teleports that survive speed and step filtering.

    The rule is the highest ``anchors_used`` - more ranging constraints, better
    geometry - and, where several tie at the top count (about a fifth of
    multi-solution instants), the coordinate-wise MEDIAN of the tied rows.
    Taking the median of ALL solutions instead gives a smoother track overall
    but a worse >1 m tail, because a minimum-anchor outlier drags the median
    when only two or three solutions exist.

    This must run BEFORE thresholding. Velocity is meaningless while two
    consecutive rows can share a timestamp, and a step-distance test cannot
    tell a solver disagreement from a stride.

    See WISER_MULTI_SOLUTION_REPORT.md.
    """
    stats = {'rows_in': len(df), 'rows_out': len(df), 'dropped': 0, 'tied': 0}
    if df is None or len(df) == 0:
        return df, stats
    if not {'shortid', 'timestamp'} <= set(df.columns):
        return df, stats

    # anchors_used is read ONLY to make this choice, and is dropped on the way
    # out. The export streams every column of the frame it is handed straight
    # into the smoothed CSV, which is meant to stay lightweight, so a column
    # carried past here would silently become a permanent export column.
    def _shed(frame):
        return (frame.drop(columns=['anchors_used'])
                if 'anchors_used' in frame.columns else frame)

    # Fast path, and the common one: eleven of VT001's seventeen tags never do
    # this, so a clean read must not pay for the machinery.
    dup = df.duplicated(subset=['shortid', 'timestamp'], keep=False)
    if not dup.any():
        return _shed(df), stats

    keys = ['shortid', 'timestamp']
    if 'anchors_used' in df.columns:
        best = df.groupby(keys, sort=False)['anchors_used'].transform('max')
        top = df[df['anchors_used'] == best]
    else:
        # Older exports without the column: fall back to the median of every
        # solution, which is still far better than treating them as movement.
        top = df

    grp = top.groupby(keys, sort=False)
    out = grp.first().reset_index()
    coord = [c for c in ('location_x', 'location_y', 'location_z')
             if c in top.columns]
    if coord:
        med = grp[coord].median().reset_index()
        for c in coord:
            out[c] = med[c]
    stats['tied'] = int((grp.size() > 1).sum())
    stats['rows_out'] = len(out)
    stats['dropped'] = stats['rows_in'] - stats['rows_out']

    if log and stats['dropped']:
        pct = 100.0 * stats['dropped'] / max(stats['rows_in'], 1)
        who = f"{label}: " if label else ""
        log(f"  {who}collapsed {stats['dropped']:,} co-timestamped solution(s) "
            f"-> {stats['rows_out']:,} fixes ({pct:.1f}% of rows were "
            f"alternative solutions for an instant already present)")
        if stats['tied']:
            log(f"      {stats['tied']:,} instant(s) tied on anchor count; "
                f"took the median position of the tied solutions")
    return _shed(out), stats


def date_axis(values):
    """Datetimes as the float day-numbers matplotlib plots internally.

    Handing ``ax.plot`` a datetime series makes matplotlib convert every point
    on every call. Converting up front is only worth doing if the conversion
    is itself cheap, and ``mdates.date2num`` is not: it routes a 3 M-row
    series through Python datetime objects and takes ~11 s. Matplotlib's date
    numbers are simply days since its epoch, so the same answer comes out of
    integer nanoseconds by division - 543x faster, and agreeing with
    ``date2num`` to under a microsecond (float64 rounding; a pixel on these
    figures spans minutes).

    The epoch is read from matplotlib rather than hard-coded, so this keeps
    matching if a caller ever moves it.

    Pair every use with ``ax.xaxis_date(tz)``: the axis has no other way to
    know the numbers are dates, and would label them as plain floats.
    """
    ser = pd.Series(values) if not isinstance(values, pd.Series) else values
    if not pd.api.types.is_datetime64_any_dtype(ser):
        return mdates.date2num(values)
    if getattr(ser.dt, 'tz', None) is not None:
        ser = ser.dt.tz_convert('UTC').dt.tz_localize(None)
    epoch = np.datetime64(mdates.get_epoch()).astype('datetime64[ns]').astype('int64')
    ns = ser.dt.as_unit('ns').astype('int64').to_numpy()
    return (ns - epoch) / 86_400_000_000_000.0


def smoothed_csv_timestamps(df, tz):
    """The exported CSV's timestamps as a tz-aware datetime series.

    Reads the INTEGER `timestamp` column (epoch ms) that the export writes
    alongside the human-readable `Timestamp` string, and which the string was
    derived from in the first place. Parsing the string instead cost ~40 s on a
    15 M-row trial, three times over (proximity, plots, animation), and it had
    to use format='mixed' because a recording crossing a DST boundary carries
    two UTC offsets in that column - the same thing that made it fragile, since
    without utc=True pandas hands back an object-dtype column and every later
    .dt access raises. One integer per row has neither problem.

    Falls back to the string column when the integer one is missing (a CSV
    written by an older version) or unusable.
    """
    if 'timestamp' in df.columns:
        epoch = pd.to_numeric(df['timestamp'], errors='coerce')
        if len(epoch) and epoch.notna().all():
            return pd.to_datetime(epoch.astype('int64'), unit='ms',
                                  utc=True).dt.tz_convert(tz)
    return pd.to_datetime(df['Timestamp'], format='mixed',
                          utc=True).dt.tz_convert(tz)


def forward_backward_ewma(series, span):
    """Zero-phase exponential weighted moving average (filtfilt-style cascade).

    The Wiser hardware applies a causal EWMA, which necessarily lags the true
    signal. Post-hoc we know each sample's future, so we run the EWMA forward,
    then run a second EWMA backward over that result. The two passes impose
    equal and opposite group delays, so the lag cancels exactly.

    Reversing with [::-1] changes row order but preserves index labels, so the
    final reversal restores the original ordering and the returned Series still
    aligns correctly when used inside groupby(...).transform(...).
    """
    forward = series.ewm(span=span, adjust=False).mean()
    return forward[::-1].ewm(span=span, adjust=False).mean()[::-1]


def rolling_smooth_xy(data, agg, window_value, time_based):
    """Centred (zero-phase) rolling smoothing of location_x/location_y per tag.

    Shared by the interactive preview/export path and the plot-worker fallback
    so both interpret the smoothing window identically.

    Parameters
    ----------
    data : DataFrame with 'shortid', 'Timestamp', 'location_x', 'location_y'.
    agg : 'mean' (Rolling Average) or 'median' (Rolling Median).
    window_value : the window size. When ``time_based`` it is a duration in
        seconds; otherwise it is a count of consecutive samples.
    time_based : if True the window is a real-time span (e.g. '30s') evaluated
        against each tag's Timestamp, so the degree of smoothing is independent
        of the (irregular, well-under-1 Hz) reporting rate — 30 s of data is
        averaged regardless of how many fixes landed in that interval. If False
        the window is a fixed number of consecutive fixes, so its wall-clock
        span drifts with the reporting rate (the historical behaviour).

    In both modes the window is **centred**, so each output sample uses roughly
    equal amounts of past and future data and smoothing adds no temporal lag.
    Writes 'smoothed_x'/'smoothed_y' back into ``data`` and returns it.
    """
    def _agg(roller):
        return roller.mean() if agg == 'mean' else roller.median()

    if time_based:
        # Time-based rolling needs a monotonic DatetimeIndex, so sort each tag
        # by Timestamp, roll, then scatter the result back to the original rows.
        win = f"{int(window_value)}s"
        data['smoothed_x'] = np.nan
        data['smoothed_y'] = np.nan
        for _, g in data.groupby('shortid'):
            g_sorted = g.sort_values('Timestamp')
            idx = pd.DatetimeIndex(g_sorted['Timestamp'])
            for src, dst in (('location_x', 'smoothed_x'), ('location_y', 'smoothed_y')):
                s = pd.Series(g_sorted[src].to_numpy(), index=idx)
                rolled = _agg(s.rolling(win, center=True, min_periods=1))
                data.loc[g_sorted.index, dst] = rolled.to_numpy()
    else:
        window_size = max(3, int(window_value))  # sample-count floor of 3
        for src, dst in (('location_x', 'smoothed_x'), ('location_y', 'smoothed_y')):
            data[dst] = data.groupby('shortid')[src].transform(
                lambda x: _agg(x.rolling(window=window_size, center=True, min_periods=1)))
    return data


# Shared, math-level description of every smoothing method. Used verbatim as
# the tooltip for BOTH the Export "Smoothing method" combo and the live-preview
# "Smoothing method" combo, so the two are guaranteed identical. Keep it in sync
# with forward_backward_ewma(), rolling_smooth_xy() and apply_smoothing_to_data().
SMOOTHING_METHODS_TOOLTIP = (
    "Smoothing is applied per tag, after threshold filtering. Every method here\n"
    "is symmetric (zero-phase): because this is post-processing, each sample's\n"
    "future is known, so the window is centred and smoothing adds no temporal\n"
    "lag — unlike the causal filter the Wiser hardware must run live.\n"
    "\n"
    "The Smoothing Window / units control sets the window for the two rolling\n"
    "methods; Forward-Backward Exponentially Weighted Moving Average uses it as a sample span; Savitzky-Golay and\n"
    "None ignore it.\n"
    "\n"
    "• None\n"
    "    No smoothing. The raw (already threshold-filtered) fixes are used as-is.\n"
    "\n"
    "• Rolling Average\n"
    "    Centred moving mean — each output is the arithmetic mean of the fixes\n"
    "    inside a window centred on that sample:\n"
    "        out[i] = mean( x[j] : t[j] within the window around t[i] )\n"
    "    With units = Seconds the window is a real-time span (e.g. 30 s), so the\n"
    "    amount of smoothing is independent of the irregular, well-under-1 Hz\n"
    "    reporting rate; with units = Samples it is a fixed count of consecutive\n"
    "    fixes (minimum 3). Partial windows at the ends use whatever samples\n"
    "    exist (min_periods = 1). Good general-purpose smoothing, but sensitive\n"
    "    to any outliers the thresholds missed.\n"
    "\n"
    "• Rolling Median\n"
    "    Identical windowing to Rolling Average but takes the median:\n"
    "        out[i] = median( x[j] within the window around t[i] )\n"
    "    The median ignores a minority of extreme values, so it is far more\n"
    "    robust to residual spikes/outliers, at the cost of slightly blockier\n"
    "    output on smoothly curving paths.\n"
    "\n"
    "• Savitzky-Golay\n"
    "    Fits a low-order polynomial to each centred window by least squares and\n"
    "    evaluates it at the window's centre. Here the window length is\n"
    "    min(31, n) samples (forced odd) with polynomial order 2 — a local\n"
    "    quadratic; the Smoothing Window setting does not apply. Because it\n"
    "    models curvature instead of flattening it, Savitzky-Golay preserves\n"
    "    peaks, turns and the height of sharp features better than averaging.\n"
    "\n"
    "• Forward-Backward Exponentially Weighted Moving Average\n"
    "    A zero-phase exponentially weighted moving average, run as a two-pass\n"
    "    (filtfilt-style) cascade. One pass is the recursive EWMA\n"
    "        y[t] = alpha * x[t] + (1 - alpha) * y[t-1],   alpha = 2 / (span + 1)\n"
    "    which weights the most recent sample most heavily and older samples with\n"
    "    geometrically decaying weight. A single forward pass lags the true path\n"
    "    by about (span - 1) / 2 samples; that forward result is then filtered\n"
    "    again backward, imposing an equal and opposite delay, so the lag cancels\n"
    "    exactly and the output stays centred on the real trajectory. Filtering\n"
    "    twice squares the frequency response, so expect noticeably stronger\n"
    "    smoothing than a single pass at the same span. The span is always in\n"
    "    samples. This is the same filter family the Wiser hardware applies in\n"
    "    real time (where it must stay causal, hence laggy); to match a hardware\n"
    "    filter value F, use span = 2F - 1."
)


class DbQueryWorker(QThread):
    """Run a read-only database query off the GUI thread.

    ``fn`` is a zero-arg callable that opens its OWN connection (via
    ``connect_ro``), runs its queries, and returns a result object. It must
    never touch Qt widgets — the result is delivered back to the main thread
    through the ``done`` signal, where UI updates are safe. Used to keep the
    on-load metadata scans (DISTINCT tags/days, MIN/MAX time bounds) — which are
    slow on a large, not-yet-indexed database over a network drive — from
    freezing the window.
    """
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            result = self._fn()
        except Exception as e:
            self.failed.emit(str(e))
            return
        self.done.emit(result)


class PreviewIndexBuilder(QThread):
    """Build an indexed *copy* of the database for fast preview scrubbing.

    The Wiser database is a primary record, so it is never modified. This
    creates a byte-identical duplicate in the analysis folder and adds one
    index on (shortid, timestamp) — no filtering, no schema changes, no row
    edits. The copy is a derived cache and is safe to delete at any time.

    Worth being precise about the payoff: the index accelerates the preview's
    narrow time-window reads by ~80x, but does **not** speed up export, which
    reads every row for a tag and is a full scan either way.
    """
    progress = pyqtSignal(str)
    # (source_db_path, indexed_copy_path) — the source is carried through so the
    # receiver can tell whether the database it was built for is still current.
    done = pyqtSignal(str, str)
    failed = pyqtSignal(str, str)   # (source_db_path, error)

    INDEX_NAME = "idx_fnt_shortid_ts"

    def __init__(self, src_path, table_name, dst_path, meta_path):
        super().__init__()
        self.src_path = src_path
        self.table_name = table_name
        self.dst_path = dst_path
        self.meta_path = meta_path

    def run(self):
        tmp = self.dst_path + ".partial"
        try:
            self.progress.emit("Copying database…")
            # Ensure the destination folder (the analysis/export subfolder)
            # exists — the caller normally creates it, but this keeps the
            # builder self-contained if invoked another way.
            os.makedirs(os.path.dirname(self.dst_path), exist_ok=True)
            # Write to a temp name first so an interrupted build can never be
            # mistaken for a finished cache.
            if os.path.exists(tmp):
                os.remove(tmp)
            shutil.copy2(self.src_path, tmp)

            self.progress.emit("Building index…")
            conn = sqlite3.connect(tmp)
            conn.execute(f"CREATE INDEX IF NOT EXISTS {self.INDEX_NAME} "
                         f"ON {self.table_name}(shortid, timestamp)")
            conn.commit()
            conn.close()

            os.replace(tmp, self.dst_path)

            st = os.stat(self.src_path)
            from datetime import datetime
            with open(self.meta_path, "w") as f:
                json.dump({
                    "source_path": self.src_path,
                    "source_size": st.st_size,
                    "source_mtime": st.st_mtime,
                    "table_name": self.table_name,
                    "index_name": self.INDEX_NAME,
                    "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "note": ("Derived cache for preview scrubbing. Identical to "
                             "the source database plus one index on "
                             "(shortid, timestamp). No data was filtered or "
                             "altered. Safe to delete; it will be rebuilt on "
                             "demand."),
                }, f, indent=4)
            self.done.emit(self.src_path, self.dst_path)
        except Exception as e:
            for p in (tmp,):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
            self.failed.emit(self.src_path, str(e))


class PreviewChunkLoader(QThread):
    """Reads one bounded time slice off the UI thread.

    Only the SQL read happens here — measured at ~100% of chunk load cost on an
    unindexed 4.7M-row table — so the widget's filter/smoothing methods stay on
    the main thread where they can safely read their own spinboxes.

    All selected tags are fetched in a single query. One combined scan is ~2.6x
    faster than looping per tag, because without an index every query is a full
    table scan and the per-tag loop pays for one scan each.
    """
    loaded = pyqtSignal(int, object)    # chunk_index, DataFrame
    failed = pyqtSignal(int, str)

    def __init__(self, db_path, table_name, tags, start_ms, end_ms, chunk_index):
        super().__init__()
        self.db_path = db_path
        self.table_name = table_name
        self.tags = list(tags)
        self.start_ms = int(start_ms)
        self.end_ms = int(end_ms)
        self.chunk_index = int(chunk_index)

    def run(self):
        try:
            # The connection must be created in this thread; sqlite3 objects
            # cannot be shared across threads.
            conn = connect_ro(self.db_path)
            cols = processing_select_clause(conn, self.table_name)
            placeholders = ",".join(["?"] * len(self.tags))
            df = pd.read_sql_query(
                f"SELECT {cols} FROM {self.table_name} "
                f"WHERE shortid IN ({placeholders}) AND timestamp BETWEEN ? AND ? "
                f"ORDER BY shortid, timestamp",
                conn, params=self.tags + [self.start_ms, self.end_ms])
            conn.close()
            # On a worker thread, so no logging from here - the preview's
            # collapse is reported once by the chunk consumer instead of once
            # per chunk, which would be thousands of identical lines.
            df, _ = collapse_multi_solutions(df)
            self.loaded.emit(self.chunk_index, df)
        except Exception as e:
            self.failed.emit(self.chunk_index, str(e))


class AnimationClipDialog(QDialog):
    """Play a short rendered clip so export settings can be judged before
    committing to a multi-hour render.

    Frames are decoded up front with OpenCV and shown in a QLabel on a timer,
    rather than handed to QtMultimedia. The clip is a handful of megabytes, so
    holding it in memory is cheap, and it sidesteps the two things that make
    embedded playback unreliable on Windows: QtMultimedia is an optional
    install, and the mp4v stream the renderer writes is not always one its
    backend will open. Decoding it ourselves means the dialog can play exactly
    what was written, or say plainly that it could not.
    """

    def __init__(self, video_path, fps, caption="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Animation preview")
        self.video_path = video_path
        self.frames = []
        self._playing = True

        cap = cv2.VideoCapture(video_path)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, _ = rgb.shape
                # copy(): QImage does not own the numpy buffer, and the frame
                # it points at is rebound on the next read.
                self.frames.append(QImage(rgb.data, w, h, 3 * w,
                                          QImage.Format_RGB888).copy())
        finally:
            cap.release()

        layout = QVBoxLayout()
        if caption:
            lbl = QLabel(caption)
            lbl.setStyleSheet("color:#9fb3c8;")
            lbl.setWordWrap(True)
            layout.addWidget(lbl)

        self.view = QLabel()
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(480, 384)
        self.view.setStyleSheet("background:#1e1e1e;")
        layout.addWidget(self.view, 1)

        if not self.frames:
            self.view.setText("The clip rendered but could not be decoded "
                              "for playback.\nOpen the file directly:\n"
                              + video_path)
            self.view.setWordWrap(True)
        else:
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setRange(0, len(self.frames) - 1)
            self.slider.valueChanged.connect(self._show_frame)
            layout.addWidget(self.slider)

            row = QHBoxLayout()
            self.btn_play = QPushButton("Pause")
            self.btn_play.setFixedWidth(80)
            self.btn_play.clicked.connect(self._toggle)
            row.addWidget(self.btn_play)
            self.chk_loop = QCheckBox("Loop")
            self.chk_loop.setChecked(True)
            row.addWidget(self.chk_loop)
            self.lbl_pos = QLabel("")
            self.lbl_pos.setStyleSheet("color:#9fb3c8;")
            row.addWidget(self.lbl_pos)
            row.addStretch(1)
            btn_close = QPushButton("Close")
            btn_close.clicked.connect(self.accept)
            row.addWidget(btn_close)
            layout.addLayout(row)

            self.timer = QTimer(self)
            self.timer.setInterval(max(1, int(1000.0 / max(1.0, float(fps)))))
            self.timer.timeout.connect(self._advance)
            self._show_frame(0)
            self.timer.start()

        self.setLayout(layout)
        first = self.frames[0] if self.frames else None
        if first is not None:
            # Fit the dialog to the clip, capped to the screen.
            w, h = first.width(), first.height()
            try:
                avail = QApplication.primaryScreen().availableGeometry()
                scale = min(1.0, (avail.width() - 80) / w,
                            (avail.height() - 200) / h)
            except Exception:
                scale = 1.0
            self.resize(int(w * scale) + 20, int(h * scale) + 120)

    def _show_frame(self, i):
        if not self.frames:
            return
        i = max(0, min(int(i), len(self.frames) - 1))
        pix = QPixmap.fromImage(self.frames[i])
        self.view.setPixmap(pix.scaled(self.view.size(), Qt.KeepAspectRatio,
                                       Qt.SmoothTransformation))
        self.lbl_pos.setText(f"frame {i + 1} / {len(self.frames)}")

    def _advance(self):
        i = self.slider.value() + 1
        if i >= len(self.frames):
            if not self.chk_loop.isChecked():
                self._toggle()
                return
            i = 0
        self.slider.setValue(i)

    def _toggle(self):
        self._playing = not self._playing
        self.btn_play.setText("Pause" if self._playing else "Play")
        if self._playing:
            self.timer.start()
        else:
            self.timer.stop()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.frames:
            self._show_frame(self.slider.value())

    def done(self, result):
        timer = getattr(self, 'timer', None)
        if timer is not None:
            timer.stop()
        self.frames = []          # a few hundred MB of decoded RGB
        super().done(result)


class InheritSettingsDialog(QDialog):
    """Offer to copy the analysis settings from a trial already in the queue.

    Shown when a database is loaded while the batch queue is non-empty. Copies
    only the settings that describe HOW a trial is analysed; everything that
    describes WHICH trial it is - the tag selection, the identities and their
    active windows, the background image, the arena geometry, the day list -
    stays with the newly loaded database.
    """

    def __init__(self, items, new_db_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Apply Settings From a Queued Trial?")
        self.setMinimumWidth(560)
        layout = QVBoxLayout()

        head = QLabel(
            f"<b>{new_db_name}</b> was just loaded. Copy the tracking and "
            f"export settings from a trial already in the queue?")
        head.setWordWrap(True)
        layout.addWidget(head)

        detail = QLabel(
            "Copies: filtering and smoothing, behaviour-detection thresholds, "
            "the tracking XY range, timezone, every export checkbox, the plot "
            "selection and the animation settings.\n\n"
            "Keeps this trial's own: tag selection, identities and their "
            "start/stop windows, background image and alignment, arena "
            "geometry, and the per-day animation list.")
        detail.setWordWrap(True)
        detail.setStyleSheet("color:#9fb3c8; font-size: 11px;")
        layout.addWidget(detail)

        row = QHBoxLayout()
        row.addWidget(QLabel("Copy from:"))
        self.combo = QComboBox()
        for i, it in enumerate(items, 1):
            cfg = it.get('config') or {}
            bits = []
            sm = (cfg.get('preview') or {}).get('smoothing') or cfg.get('smoothing_method')
            if sm and sm != "None":
                bits.append(sm.replace("Forward-Backward Exponentially Weighted "
                                       "Moving Average", "EWMA"))
            for key, tag in (('export_raw_csv', 'raw'),
                             ('proximity_detection', 'prox'),
                             ('export_roi_bouts', 'ROIbouts'),
                             ('export_roi_daily', 'ROIdaily'),
                             ('export_behavior_events', 'behav'),
                             ('save_plots', 'plots'),
                             ('save_animation', 'anim')):
                if cfg.get(key):
                    bits.append(tag)
            self.combo.addItem(
                f"{i}. {os.path.basename(it['path'])}  ({', '.join(bits) or 'defaults'})",
                i - 1)
        # Most recent first is the usual intent when queueing a run of trials.
        self.combo.setCurrentIndex(self.combo.count() - 1)
        row.addWidget(self.combo, 1)
        layout.addLayout(row)

        buttons = QHBoxLayout()
        buttons.addStretch()
        btn_skip = QPushButton("Keep This Trial's Settings")
        btn_skip.clicked.connect(self.reject)
        buttons.addWidget(btn_skip)
        btn_apply = QPushButton("Apply Settings")
        btn_apply.setDefault(True)
        btn_apply.setStyleSheet(
            "QPushButton { background-color: #2ea043; color: white; "
            "padding: 6px 14px; font-weight: bold; }")
        btn_apply.clicked.connect(self.accept)
        buttons.addWidget(btn_apply)
        layout.addLayout(buttons)

        self.setLayout(layout)

    def chosen_index(self):
        return self.combo.currentData()


class ExportConflictDialog(QDialog):
    """Dialog shown when export would overwrite existing files.
    Shows categorized lists of conflicting and new files."""

    # Non-zero so none of these collide with QDialog.Rejected (0), which is
    # what the Cancel button and the window close (X) return via reject(). If
    # SKIP were 0 a cancel would be indistinguishable from "skip existing" and
    # the export would proceed anyway instead of halting.
    SKIP = 1
    OVERWRITE = 2
    NEW_FOLDER = 3
    # Delete every previous export from the folder, then write the current
    # selection into it. Distinct from OVERWRITE, which only replaces the
    # files this run happens to produce and so leaves a renamed or retired
    # product's file behind looking current.
    CLEAR = 4

    def __init__(self, conflicting_files, new_files, parent=None):
        """
        Args:
            conflicting_files: list of (filename, subfolder) tuples for files that already exist
            new_files: list of (filename, subfolder) tuples for files that will be newly created
        """
        super().__init__(parent)
        self.setWindowTitle("Export Conflict")
        self.setModal(True)
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)

        layout = QVBoxLayout()

        # Summary message
        num_conflicts = len(conflicting_files)
        num_new = len(new_files)
        total = num_conflicts + num_new
        message = QLabel(
            f"<b>{total}</b> file(s) will be produced. "
            f"<b>{num_conflicts}</b> existing file(s) would be overwritten, "
            f"and <b>{num_new}</b> file(s) are new."
        )
        message.setWordWrap(True)
        layout.addWidget(message)

        # File tree
        tree = QTreeWidget()
        tree.setHeaderHidden(True)
        tree.setRootIsDecorated(True)
        tree.setIndentation(20)
        tree.setStyleSheet("""
            QTreeWidget {
                background-color: #1e1e1e;
                border: 1px solid #3f3f3f;
                border-radius: 4px;
                color: #cccccc;
                font-size: 11px;
            }
            QTreeWidget::item {
                padding: 2px 0px;
            }
        """)

        # Conflicting files section
        if conflicting_files:
            conflict_root = QTreeWidgetItem(tree)
            conflict_root.setText(0, f"⚠ Files that would be overwritten ({num_conflicts})")
            conflict_root.setForeground(0, Qt.yellow)
            conflict_root.setExpanded(True)

            # Group by subfolder
            grouped = {}
            for fname, subfolder in conflicting_files:
                grouped.setdefault(subfolder, []).append(fname)

            for subfolder in sorted(grouped.keys()):
                if subfolder:
                    folder_item = QTreeWidgetItem(conflict_root)
                    folder_item.setText(0, f"📁 {subfolder}/")
                    folder_item.setExpanded(True)
                    parent_item = folder_item
                else:
                    parent_item = conflict_root

                for fname in sorted(grouped[subfolder]):
                    file_item = QTreeWidgetItem(parent_item)
                    file_item.setText(0, fname)

        # New files section
        if new_files:
            new_root = QTreeWidgetItem(tree)
            new_root.setText(0, f"✓ New files to be created ({num_new})")
            new_root.setForeground(0, Qt.green)
            new_root.setExpanded(True)

            # Group by subfolder
            grouped = {}
            for fname, subfolder in new_files:
                grouped.setdefault(subfolder, []).append(fname)

            for subfolder in sorted(grouped.keys()):
                if subfolder:
                    folder_item = QTreeWidgetItem(new_root)
                    folder_item.setText(0, f"📁 {subfolder}/")
                    folder_item.setExpanded(True)
                    parent_item = folder_item
                else:
                    parent_item = new_root

                for fname in sorted(grouped[subfolder]):
                    file_item = QTreeWidgetItem(parent_item)
                    file_item.setText(0, fname)

        layout.addWidget(tree)

        # Question
        question = QLabel("What would you like to do?")
        question.setStyleSheet("font-weight: bold; margin-top: 6px;")
        layout.addWidget(question)

        # Buttons
        btn_layout = QHBoxLayout()

        # Overwrite sits first: it is the usual choice when re-running a trial.
        overwrite_btn = QPushButton("Overwrite")
        overwrite_btn.setToolTip("Replace all conflicting files and write new files")
        overwrite_btn.clicked.connect(lambda: self.done(self.OVERWRITE))
        btn_layout.addWidget(overwrite_btn)

        skip_btn = QPushButton("Skip Existing")
        skip_btn.setToolTip("Only write the new files; leave existing files untouched")
        skip_btn.clicked.connect(lambda: self.done(self.SKIP))
        btn_layout.addWidget(skip_btn)

        clear_btn = QPushButton("Clear && Re-export")
        clear_btn.setToolTip(
            "Delete EVERY previous export from this trial's analysis folder, "
            "then write the current selection into the empty folder.\n"
            "\n"
            "Different from Overwrite in one way that matters: Overwrite only "
            "replaces the files this run produces, so a product that has been "
            "renamed or switched off leaves its old file behind, still looking "
            "current. This removes those too.\n"
            "\n"
            "KEPT: fnt_config.json and the fast-index copy (gigabytes, and a "
            "cache of the source database rather than an export).\n"
            "\n"
            "You are shown exactly what will be deleted, and asked again, "
            "before anything is removed.")
        clear_btn.clicked.connect(lambda: self.done(self.CLEAR))
        btn_layout.addWidget(clear_btn)

        new_folder_btn = QPushButton("New Folder")
        new_folder_btn.setToolTip("Create a new timestamped analysis folder instead")
        new_folder_btn.clicked.connect(lambda: self.done(self.NEW_FOLDER))
        btn_layout.addWidget(new_folder_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)


class IdentityAssignmentDialog(QDialog):
    """Assign sex, identity and an optional active time window to each tag.

    The Start/Stop pickers TRIM the data: fixes outside a bound window are
    dropped from the preview and from every export. Each bound therefore has an
    explicit mode:

      * AUTO (default) - the bound is not applied. The picker shows the tag's
        observed first/last fix, read-only: in Auto there is nothing for the
        user to choose, the answer is whatever the data says.
      * MANUAL - the bound is applied, at exactly the value shown.

    That distinction exists because the pickers are pre-filled from the data's
    own extent. Without it, simply opening this dialog and pressing OK armed a
    hard trim at whatever the recording happened to reach that day - so a
    database that grew afterwards (a trial still running, re-copied later) had
    its new days silently discarded.
    """

    # Bounds are shown and stored to the millisecond, because the fixes they
    # trim are. Truncating to the second made a Stop ambiguous by up to one
    # second, which at ~7.6 Hz is several fixes either side of the cut.
    TIME_FMT = "yyyy-MM-dd HH:mm:ss.zzz"
    LEGACY_TIME_FMT = "yyyy-MM-dd HH:mm:ss"     # configs written before ms

    @classmethod
    def parse_time(cls, value):
        """A stored bound as a QDateTime, tolerating the pre-millisecond form."""
        for fmt in (cls.TIME_FMT, cls.LEGACY_TIME_FMT):
            dt = QDateTime.fromString(value or '', fmt)
            if dt.isValid():
                return dt
        return QDateTime()

    def __init__(self, available_tags, existing_identities=None, tag_time_ranges=None,
                 parent=None):
        super().__init__(parent)
        self.available_tags = available_tags
        self.identities = existing_identities if existing_identities else {}
        self.tag_time_ranges = tag_time_ranges if tag_time_ranges else {}
        self.initUI()

    @staticmethod
    def _mode_of(info, which, default='auto'):
        """'manual' or 'auto' for a bound, tolerating pre-flag configs.

        A config written before the modes existed carries only the times. Those
        are treated as MANUAL so an existing trial keeps exporting exactly what
        it exported before - deliberate removals stay honoured. The rows for
        such a tag are flagged in the dialog when the bound is cutting data, so
        a stale one is visible rather than silent.
        """
        if not info:
            return default
        mode = info.get(f'{which}_mode')
        if mode in ('auto', 'manual'):
            return mode
        return 'manual' if info.get(f'{which}_time') else default

    def initUI(self):
        self.setWindowTitle("Configure Tag Identities")
        self.setMinimumWidth(860)

        layout = QVBoxLayout()

        instructions = QLabel(
            "Assign sex (M/F) and IDs. To merge tags (e.g. a lost tag replaced), "
            "give both the same ID.\n"
            "<b>Start / Stop trim the data.</b> Leave a bound on "
            "<i>Auto</i> \u2014 the tag's own first/last fix, shown read-only \u2014 "
            "unless you actually want fixes outside it discarded; choose "
            "<i>Manual</i> only for a real deployment or removal time.")
        instructions.setWordWrap(True)
        instructions.setTextFormat(Qt.RichText)
        layout.addWidget(instructions)

        form_layout = QFormLayout()
        self.sex_combos = {}
        self.identity_edits = {}
        self.start_edits = {}
        self.stop_edits = {}
        self.start_modes = {}
        self.stop_modes = {}
        stale = []

        for tag in sorted(self.available_tags):
            info = self.identities.get(tag) or {}
            tr = self.tag_time_ranges.get(tag) or {}

            sex_combo = QComboBox()
            sex_combo.addItems(["M", "F"])
            identity_edit = QLineEdit()
            identity_edit.setPlaceholderText(f"e.g., {tag}")

            def _picker():
                e = QDateTimeEdit()
                e.setDisplayFormat(self.TIME_FMT)
                e.setCalendarPopup(True)
                return e

            start_edit, stop_edit = _picker(), _picker()

            # Seed from the observed range; a saved time overrides it below.
            obs_start = self.parse_time(tr.get('start', ''))
            obs_stop = self.parse_time(tr.get('end', ''))
            if obs_start.isValid():
                start_edit.setDateTime(obs_start)
            if obs_stop.isValid():
                stop_edit.setDateTime(obs_stop)

            start_mode = self._mode_of(info, 'start')
            stop_mode = self._mode_of(info, 'stop')

            if info:
                sex_combo.setCurrentIndex(0 if info.get('sex', 'M') == 'M' else 1)
                identity_edit.setText(info.get('identity', ''))
                # A manual bound keeps its saved value; an automatic one is
                # re-seeded from the CURRENT data extent, so it can never go
                # stale.
                if start_mode == 'manual':
                    dt = self.parse_time(info.get('start_time', ''))
                    if dt.isValid():
                        start_edit.setDateTime(dt)
                if stop_mode == 'manual':
                    dt = self.parse_time(info.get('stop_time', ''))
                    if dt.isValid():
                        stop_edit.setDateTime(dt)
            else:
                sex_combo.setCurrentIndex(-1)
                identity_edit.setText("")

            def _mode_combo(mode, tip):
                c = QComboBox()
                c.addItem("Auto", "auto")
                c.addItem("Manual", "manual")
                c.setCurrentIndex(1 if mode == 'manual' else 0)
                c.setFixedWidth(96)
                c.setToolTip(tip)
                return c

            start_combo = _mode_combo(
                start_mode,
                "Auto: the tag's own first fix, shown read-only; nothing is "
                "trimmed.\n"
                "Manual: discard everything BEFORE the time you set.")
            stop_combo = _mode_combo(
                stop_mode,
                "Auto: the tag's own last fix, shown read-only; nothing is "
                "trimmed. Choose this if the recording may still grow.\n"
                "Manual: discard everything AFTER the time you set - for an "
                "animal actually removed or dead at that moment.")

            def _wire(combo, edit, observed):
                """Auto shows the data's extent; Manual restores the user's own.

                The manual entry has to survive a round trip through Auto.
                Auto overwrites the picker with the observed extent, so without
                remembering the typed value, switching back would hand the user
                that extent as though they had chosen it - which is exactly the
                stale-trim trap the two modes exist to prevent.
                """
                remembered = {'dt': edit.dateTime()}

                def _remember(dt):
                    if combo.currentData() == 'manual':
                        remembered['dt'] = dt

                def _sync():
                    manual = combo.currentData() == 'manual'
                    if manual:
                        edit.setDateTime(remembered['dt'])
                    elif observed.isValid():
                        edit.setDateTime(observed)
                    # Disabled rather than merely read-only: in Auto the value
                    # is not the user's to set, and it should not look like it.
                    edit.setEnabled(manual)

                edit.dateTimeChanged.connect(_remember)
                combo.currentIndexChanged.connect(_sync)
                _sync()
            _wire(start_combo, start_edit, obs_start)
            _wire(stop_combo, stop_edit, obs_stop)

            tag_widget = QWidget()
            tag_vlayout = QVBoxLayout()
            tag_vlayout.setContentsMargins(0, 0, 0, 0)
            tag_vlayout.setSpacing(2)

            row1 = QHBoxLayout()
            row1.addWidget(QLabel("Sex:"))
            row1.addWidget(sex_combo)
            row1.addWidget(QLabel("ID:"))
            row1.addWidget(identity_edit)
            tag_vlayout.addLayout(row1)

            row2 = QHBoxLayout()
            lbl = QLabel("Start:"); lbl.setFixedWidth(35)
            row2.addWidget(lbl)
            row2.addWidget(start_combo)
            row2.addWidget(start_edit, 1)
            row2.addSpacing(10)
            lbl2 = QLabel("Stop:"); lbl2.setFixedWidth(35)
            row2.addWidget(lbl2)
            row2.addWidget(stop_combo)
            row2.addWidget(stop_edit, 1)
            tag_vlayout.addLayout(row2)

            # Only the WARNING. The plain "data: start \u2192 end" line that used to
            # sit here repeated what the two pickers already show - on an Auto
            # bound, character for character - so it was a row of noise under
            # every tag. A Stop that is throwing days away is the one thing the
            # pickers cannot tell you, so that stays.
            if tr.get('start') and tr.get('end') and stop_mode == 'manual':
                try:
                    cut = ((pd.Timestamp(tr['end'])
                            - pd.Timestamp(stop_edit.dateTime().toString(
                                self.TIME_FMT))).total_seconds() / 86400.0)
                except Exception:
                    cut = 0.0
                if cut > 0.5:
                    hint = QLabel(f"\u26a0 Stop discards the last {cut:.1f} day(s) "
                                  f"\u2014 data runs to {tr['end']}")
                    hint.setStyleSheet(
                        "color:#e5c07b; font-size: 9px; font-weight: bold;")
                    tag_vlayout.addWidget(hint)
                    stale.append(tag)

            tag_widget.setLayout(tag_vlayout)
            hex_id = hex(tag).upper().replace('0X', '')
            form_layout.addRow(f"HexID {hex_id}:", tag_widget)

            self.sex_combos[tag] = sex_combo
            self.identity_edits[tag] = identity_edit
            self.start_edits[tag] = start_edit
            self.stop_edits[tag] = stop_edit
            self.start_modes[tag] = start_combo
            self.stop_modes[tag] = stop_combo

        if stale:
            banner = QLabel(
                f"\u26a0 {len(stale)} tag(s) have a Stop time earlier than their "
                "last recorded fix, so those days are being discarded. If those "
                "Stop times were only ever the pre-filled data extent (not a real "
                "removal), set them back to <i>Auto</i>.")
            banner.setWordWrap(True)
            banner.setTextFormat(Qt.RichText)
            banner.setStyleSheet(
                "color:#e5c07b; background:#3a3226; padding:6px; border-radius:3px;")
            layout.addWidget(banner)

            btn_release = QPushButton("Set every flagged Stop back to \u201cAuto\u201d")
            btn_release.setToolTip(
                "Switch the Stop bound to Auto for every tag flagged above, so "
                "their full recordings are used. Start times are left exactly "
                "as they are.")
            btn_release.clicked.connect(lambda _=False, t=list(stale): self._release_stops(t))
            layout.addWidget(btn_release)

        form_container = QWidget()
        form_container.setLayout(form_layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_container)
        layout.addWidget(scroll, 1)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

        try:
            avail = QApplication.primaryScreen().availableGeometry()
            self.setMaximumHeight(avail.height())
            self.resize(900, min(880, int(avail.height() * 0.9)))
        except Exception:
            self.resize(900, 800)

    def _merged_overlaps(self):
        """Tags sharing an ID whose active windows overlap, as a list of clashes.

        Merging is deliberate - a replaced tag keeps the animal's ID - and it
        only works when the two deployments are consecutive. If they overlap,
        the merged animal has two positions at the same instant, and everything
        built on that (home range, proximity, the 1 Hz ethogram grid, which
        keeps one fix per bin) silently resolves it by picking one. In practice
        that means the earlier tag's Stop, or the later tag's Start, was never
        narrowed to the moment of the swap.

        Compares the EFFECTIVE window: a bound on Auto contributes the tag's
        real data extent, not the value sitting in its (disabled) picker.
        The timestamps are fixed-width TIME_FMT, so ordering them as strings is
        ordering them chronologically - which holds only while both sides carry
        milliseconds, so tag_time_ranges must format to TIME_FMT too.
        """
        windows = {}
        for tag in self.available_tags:
            tr = self.tag_time_ranges.get(tag) or {}
            key = (self.sex_combos[tag].currentText(),
                   self.identity_edits[tag].text().strip() or str(tag))
            start = (self.start_edits[tag].dateTime().toString(self.TIME_FMT)
                     if self.start_modes[tag].currentData() == 'manual'
                     else tr.get('start'))
            stop = (self.stop_edits[tag].dateTime().toString(self.TIME_FMT)
                    if self.stop_modes[tag].currentData() == 'manual'
                    else tr.get('end'))
            if start and stop:
                windows.setdefault(key, []).append((tag, start, stop))

        clashes = []
        for (sex, ident), entries in windows.items():
            if len(entries) < 2:
                continue
            entries.sort(key=lambda e: e[1])
            for (t1, _s1, e1), (t2, s2, _e2) in zip(entries, entries[1:]):
                if s2 <= e1:
                    clashes.append((f"{sex}{ident}", t1, e1, t2, s2))
        return clashes

    def accept(self):
        """Refuse silently-corrupting merges; everything else saves as before."""
        clashes = self._merged_overlaps()
        if clashes:
            def _hex(t):
                return hex(t).upper().replace('0X', '')
            detail = "\n".join(
                f"  {name}: HexID {_hex(t1)} is active until {e1}, "
                f"but HexID {_hex(t2)} starts at {s2}"
                for name, t1, e1, t2, s2 in clashes)
            reply = QMessageBox.warning(
                self, "Merged tags overlap",
                "These tags share an ID, so they are treated as one animal — "
                "but their active windows overlap, which would put that animal "
                "in two places at once:\n\n" + detail +
                "\n\nUsually the earlier tag's Stop, or the later tag's Start, "
                "was never narrowed to the swap. Save anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        super().accept()

    def _release_stops(self, tags):
        """Put the flagged tags' Stop bounds back on Auto.

        Switching the mode is enough: Auto re-reads the tag's observed last
        fix into the picker, so the stale value that triggered the flag is
        replaced rather than merely ignored.
        """
        for tag in tags:
            combo = self.stop_modes.get(tag)
            if combo is not None:
                combo.setCurrentIndex(0)          # Auto

    def get_identities(self):
        """Configured identities, with each bound's mode recorded alongside it."""
        result = {}
        for tag in self.available_tags:
            identity = self.identity_edits[tag].text().strip() or str(tag)
            result[tag] = {
                'sex': self.sex_combos[tag].currentText(),
                'identity': identity,
                'start_time': self.start_edits[tag].dateTime().toString(
                    self.TIME_FMT),
                'stop_time': self.stop_edits[tag].dateTime().toString(
                    self.TIME_FMT),
                'start_mode': self.start_modes[tag].currentData(),
                'stop_mode': self.stop_modes[tag].currentData(),
            }
        return result


class PlotSaverWorker(QThread):
    """Worker thread for saving plots without blocking the UI"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, db_path, table_name, selected_tags, downsample, smoothing_method,
                 # `downsample` is vestigial (the downsampling feature was removed);
                 # kept only to preserve this positional signature.
                 plot_types=None, skip_existing=False, rolling_window=10, timezone='US/Mountain',
                 tag_identities=None, use_identities=False, background_image=None,
                 bg_width_meters=None, bg_height_meters=None, csv_path=None, save_svg=False,
                 output_dir=None, plots_dir=None, rolling_window_units='Seconds',
                 plot_layers=None, xml_zones=None, anchor_positions=None,
                 rois=None, occupancy_sources=None,
                 xml_map_image=None, xml_map_extent=None,
                 bg_offset_x=0.0, bg_offset_y=0.0, behavior_params=None,
                 axis_limits=None):
        super().__init__()
        self.db_path = db_path
        self.table_name = table_name
        self.csv_path = csv_path
        self.selected_tags = selected_tags
        self.downsample = downsample
        self.smoothing_method = smoothing_method
        self.plot_types = plot_types if plot_types is not None else {
            'daily_paths': True,
            'trajectory_overview': True,
            'battery_levels': True
        }
        self.skip_existing = skip_existing
        self.output_dir = output_dir
        self.plots_dir = plots_dir  # Subfolder for plot output (PNGs/SVGs)
        self.rolling_window = rolling_window
        self.rolling_window_units = rolling_window_units
        self.timezone = timezone
        self.tag_identities = tag_identities if tag_identities else {}
        self.use_identities = use_identities
        self.background_image = background_image
        self.bg_width_meters = bg_width_meters
        self.bg_height_meters = bg_height_meters
        self.bg_offset_x = bg_offset_x
        self.bg_offset_y = bg_offset_y
        self.save_svg = save_svg
        # Spatial context layer choice + sources for the trajectory plots.
        self.plot_layers = plot_layers if plot_layers is not None else dict(DEFAULT_PLOT_LAYERS)
        self.xml_zones = xml_zones or []
        # The user's own regions, drawn over the preview (see uwb_roi).
        self.rois = rois or []
        # Which of the two region sets the occupancy plot is allowed to use.
        # The plot has to agree with the occupancy CSV about what was being
        # measured, so this is the SAME choice, passed in rather than
        # re-derived.
        self.occupancy_sources = (('zone', 'roi') if occupancy_sources is None
                                  else tuple(occupancy_sources))
        self.occupancy_zones = self._merge_occupancy_zones()
        self.anchor_positions = anchor_positions or []
        self.xml_map_image = xml_map_image
        self.xml_map_extent = xml_map_extent
        # Detection thresholds for the ethogram, taken from the preview so
        # the plotted states are the ones the user tuned while watching.
        self.behavior_params = behavior_params
        # (x_min, x_max, y_min, y_max) pinned by the user, or None to fit
        # each spatial plot to its data.
        self.axis_limits = axis_limits

    def _bg_source(self):
        """(image, extent) for the background layer, or (None, None).

        Prefers the user-loaded floorplan PNG, then falls back to the
        XML-embedded site map so the map still shows when no PNG was loaded.
        """
        if self.background_image is not None and self.bg_width_meters is not None:
            x0, y0 = self.bg_offset_x, self.bg_offset_y
            return self.background_image, [x0, x0 + self.bg_width_meters,
                                           y0, y0 + self.bg_height_meters]
        if self.xml_map_image is not None and self.xml_map_extent is not None:
            return self.xml_map_image, list(self.xml_map_extent)
        return None, None

    def _apply_axis_limits(self, ax, x_lo, x_hi, y_lo, y_hi, equal=False):
        """Set axis limits, letting a user-pinned XY range override the fit.

        Every spatial plot routes through here so a pinned range means the same
        window in all of them — and in the animation and the live preview.
        """
        if self.axis_limits is not None:
            x_lo, x_hi, y_lo, y_hi = self.axis_limits
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        if equal:
            ax.set_aspect('equal', adjustable='box')

    @staticmethod
    def facet_grid(n_panels, max_cols=5, panel_w=3.4, panel_h=3.4,
                   extra_h=0.0):
        """(rows, cols, figsize) for n panels laid out as a wrapped grid.

        Days used to be laid out in a single row with the figure width capped,
        so a 19-day trial got 19 panels across 20 inches - about one inch each,
        illegible at any DPI. Wrapping keeps every panel a fixed physical size
        and lets the figure grow downwards instead, so a long trial simply
        produces a taller image that stays readable when zoomed.
        """
        n = max(1, int(n_panels))
        cols = min(max_cols, n)
        rows = (n + cols - 1) // cols
        return rows, cols, (panel_w * cols, panel_h * rows + extra_h)

    def save_figure(self, fig, output_path):
        """Save figure as PNG, and optionally also as SVG"""
        # 200 dpi: a faceted plot of a long trial is meant to be zoomed
        # into, and 150 left the per-day panels soft once wrapped.
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        if self.save_svg:
            svg_path = os.path.splitext(output_path)[0] + '.svg'
            fig.savefig(svg_path, format='svg', bbox_inches='tight')
            self.progress.emit(f"Saved SVG: {os.path.basename(svg_path)}")

    def run(self):
        try:
            # Load from CSV if available (much faster and ensures consistency)
            if self.csv_path and os.path.exists(self.csv_path):
                self.progress.emit("Loading data from CSV...")
                data = pd.read_csv(self.csv_path, low_memory=False)
                
                data['Timestamp'] = smoothed_csv_timestamps(data, self.timezone)
                
                self.progress.emit(f"Loaded {len(data)} records from CSV")
                coord = 'smoothed' if 'smoothed_x' in data.columns else 'raw (unsmoothed)'
                self.progress.emit(f"Plots will use {coord} coordinates")
            else:
                # Fallback: Load from database (old behavior)
                self.progress.emit("Loading data from database...")
                
                # Connect to database
                conn = connect_ro(self.db_path)
                query = (f"SELECT {processing_select_clause(conn, self.table_name)} "
                         f"FROM {self.table_name}")
                data = pd.read_sql_query(query, conn)
                conn.close()

                data, _cs = collapse_multi_solutions(
                    data, log=self.progress.emit, label="read")
                self.progress.emit(f"Loaded {len(data)} records")
                
                # Process data
                data['Timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', origin='unix', utc=True)
                tz = pytz.timezone(self.timezone)
                data['Timestamp'] = data['Timestamp'].dt.tz_convert(tz)
                
                # Convert location to meters
                data['location_x'] *= 0.0254
                data['location_y'] *= 0.0254
                
                data = data.sort_values(by=['shortid', 'Timestamp'])
                
                # Filter to selected tags
                if self.selected_tags:
                    data = data[data['shortid'].isin(self.selected_tags)]
                
                # Apply custom sex and identities if configured
                if self.use_identities and self.tag_identities:
                    data['sex'] = data['shortid'].map(lambda x: self.tag_identities.get(x, {}).get('sex', 'M'))
                    data['identity'] = data['shortid'].map(lambda x: self.tag_identities.get(x, {}).get('identity', f'Tag{x}'))
                else:
                    data['sex'] = 'M'
                    data['identity'] = data['shortid'].apply(lambda x: f'Tag{x}')
                
                # Apply smoothing FIRST (on full resolution data)
                if self.smoothing_method != "None":
                    self.progress.emit("Applying smoothing to full resolution data...")
                    data = self.apply_smoothing(data, self.smoothing_method)

            # Get output directory
            if self.output_dir:
                output_dir = self.output_dir
            else:
                db_dir = os.path.dirname(self.db_path)
                db_filename = os.path.basename(self.db_path)
                db_name = os.path.splitext(db_filename)[0]
                output_dir = os.path.join(db_dir, f"{db_name}_FNT_analysis")

            # Use plots subfolder for all plot output
            plots_dir = self.plots_dir if self.plots_dir else os.path.join(output_dir, 'plots')

            db_name = os.path.splitext(os.path.basename(self.db_path))[0]
            ensure_dir(output_dir)
            ensure_dir(plots_dir)
            
            # Generate and save plots based on selection
            generated_count = 0
            skipped_count = 0
            
            # Each plot runs in its own try/except: a failure in one (an
            # unusual data shape, a matplotlib edge case) is reported and the
            # remaining plots still run. One shared try/except meant a single
            # bad plot silently cancelled every plot after it in this list.
            plot_jobs = [
                ('daily_paths', self.save_daily_paths_per_tag),
                ('trajectory_overview', self.save_trajectory_overview),
                ('battery_levels', self.save_battery_levels),
                ('activity_timeline', self.save_activity_timeline),
                ('velocity_distribution', self.save_velocity_distribution),
                ('cumulative_distance', self.save_cumulative_distance),
                ('velocity_timeline', self.save_velocity_timeline),
                ('data_quality', self.save_data_quality),
                ('home_range', self.save_home_range),
                ('mcp_range', self.save_mcp_range),
                ('zone_occupancy', self.save_zone_occupancy),
                ('ethogram', self.save_ethogram),
                ('social_budget', self.save_social_budget),
                ('behavior_counts', self.save_chase_timeline),
                ('behavior_matrix', self.save_chase_matrix),
            ]
            # These write one file per tag and return a count; the rest write
            # a single file and return True/False.
            per_tag = {'daily_paths', 'home_range', 'mcp_range', 'ethogram'}
            failed = []
            # One frame per animal for most plots, the raw per-tag frame for
            # the two that are about the hardware. Built once, not per plot.
            merged_data, remap = self.merge_tag_data(data)
            if remap:
                self.progress.emit(
                    "  merged " + ", ".join(
                        f"HexID {hex(a).upper().replace('0X', '')}"
                        f"\u2192{hex(b).upper().replace('0X', '')}"
                        for a, b in sorted(remap.items()))
                    + " for the per-animal plots (battery and data quality "
                      "stay per tag)")

            for key, fn in plot_jobs:
                if not self.plot_types.get(key, False):
                    continue
                try:
                    result = fn(data if key in self._PER_TAG_PLOTS
                                else merged_data, plots_dir, db_name)
                except Exception as exc:
                    failed.append(f"{key} ({type(exc).__name__}: {exc})")
                    self.progress.emit(f"✗ {key} plot failed: {exc}")
                    continue
                if key in per_tag:
                    generated_count += int(result or 0)
                elif result:
                    generated_count += 1
                else:
                    skipped_count += 1

            msg = f"Generated {generated_count} plot(s)"
            if skipped_count > 0:
                msg += f", skipped {skipped_count}"
            if failed:
                msg += f", FAILED {len(failed)} ({'; '.join(failed)})"
            msg += f" in {output_dir}"
            # Plots are one product among many, and each one is already
            # isolated. Reporting overall failure because a single plot
            # raised marked the whole trial Failed and skipped its
            # animation - even though every CSV had been written and 113 of
            # 114 plots were on disk. Only a run that produced NOTHING is a
            # failure; a partial one is a warning the caller logs.
            ok = bool(generated_count) or not failed
            self.finished.emit(ok, msg)
            
        except Exception as e:
            self.finished.emit(False, f"Error generating plots: {str(e)}")

    def apply_smoothing(self, data, method):
        """Apply smoothing to trajectory data"""
        def apply_savgol_filter(group):
            window_length = min(31, len(group))
            if window_length % 2 == 0:
                window_length -= 1
            polyorder = min(2, window_length - 1)
            if len(group) > polyorder:
                return savgol_filter(group, window_length=window_length, polyorder=polyorder)
            return group
        
        if method == "Savitzky-Golay":
            data['smoothed_x'] = data.groupby('shortid')['location_x'].transform(apply_savgol_filter)
            data['smoothed_y'] = data.groupby('shortid')['location_y'].transform(apply_savgol_filter)
        elif method in ("Rolling Average", "Rolling Median"):
            # Same interpretation as the interactive path: window is seconds
            # (time-based, centred) or a sample count, per the units setting.
            agg = 'mean' if method == "Rolling Average" else 'median'
            time_based = self.rolling_window_units == "Seconds"
            rolling_smooth_xy(data, agg, self.rolling_window, time_based)
        elif method == "Forward-Backward Exponentially Weighted Moving Average":
            # No minimum floor here — a span of 1-2 is legitimate for EWMA
            span = max(1, self.rolling_window)
            data['smoothed_x'] = data.groupby('shortid')['location_x'].transform(
                lambda x: forward_backward_ewma(x, span))
            data['smoothed_y'] = data.groupby('shortid')['location_y'].transform(
                lambda x: forward_backward_ewma(x, span))

        return data
    
    def save_daily_paths_per_tag(self, data, output_dir, db_name):
        """Save daily paths - one PNG per tag with all days
        Returns: number of plots generated"""
        self.progress.emit("Generating daily paths per tag...")
        
        data = data.copy()
        if 'Date' not in data.columns:
            data['Date'] = data['Timestamp'].dt.date
        
        x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'
        
        # Get global min/max coordinates across ALL data
        x_min, x_max = data[x_col].min(), data[x_col].max()
        y_min, y_max = data[y_col].min(), data[y_col].max()
        
        # Background is the loaded PNG, else the XML-embedded site map. If it
        # will be drawn, expand the shared axis limits to include its extent.
        day_bg_image, day_bg_extent = self._bg_source()
        if self.plot_layers.get('background') and day_bg_extent is not None:
            x_min = min(x_min, day_bg_extent[0])
            x_max = max(x_max, day_bg_extent[1])
            y_min = min(y_min, day_bg_extent[2])
            y_max = max(y_max, day_bg_extent[3])

        # Add padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_pad = x_range * 0.05 if x_range > 0 else 1
        y_pad = y_range * 0.05 if y_range > 0 else 1

        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad

        unique_dates = sorted(data['Date'].unique())
        unique_tags = sorted(data['shortid'].unique())
        
        generated = 0
        
        # One plot per animal - which is per tag unless a tag was swapped
        # mid-trial, in which case both of that animal's tags share the file.
        for file_suffix, member_tags in self.animal_groups(unique_tags):
            output_path = os.path.join(output_dir, f'{db_name}_DailyPaths_{file_suffix}.png')
            
            # Check if file exists and overwrite is False
            if self.skip_existing and os.path.exists(output_path):
                self.progress.emit(f"Skipped (exists): {db_name}_DailyPaths_{file_suffix}.png")
                continue
            
            tag_data = data[data['shortid'].isin(member_tags)]
            # Only the days this animal was actually recorded. Faceting over
            # the trial's full day list gave an animal present for 2 of 20
            # days eighteen empty panels, shrinking the two that mattered.
            tag_dates = [d for d in unique_dates
                         if not tag_data[tag_data['Date'] == d].empty]
            if not tag_dates:
                continue
            num_days = len(tag_dates)

            n_rows, n_cols, figsize = self.facet_grid(num_days, max_cols=5,
                                                      extra_h=0.6)
            fig = Figure(figsize=figsize)

            for day_idx, date in enumerate(tag_dates):
                day_data = tag_data[tag_data['Date'] == date]

                ax = fig.add_subplot(n_rows, n_cols, day_idx + 1)

                # Spatial context (background/zones/anchors) per the export choice
                draw_context_layers(
                    ax, self.plot_layers,
                    bg_image=day_bg_image, bg_extent=day_bg_extent,
                    zones_xml=self.xml_zones, anchors=self.anchor_positions,
                    rois=self.rois)

                if not day_data.empty:
                    ax.plot(day_data[x_col], day_data[y_col],
                           linewidth=1.5, alpha=0.8, color='blue')
                    ax.scatter(day_data[x_col].iloc[0], day_data[y_col].iloc[0],
                              c='black', s=50, marker='o', zorder=5)
                    ax.scatter(day_data[x_col].iloc[-1], day_data[y_col].iloc[-1],
                              c='black', s=50, marker='s', zorder=5)
                
                # Global axis limits, or the user's pinned XY range
                self._apply_axis_limits(ax, x_min, x_max, y_min, y_max)
                
                ax.set_xlabel('X (m)', fontsize=9)
                ax.set_ylabel('Y (m)', fontsize=9)
                # Numbered against the TRIAL, not this animal's subset, so
                # 'Day 7' means the same day on every animal's figure.
                ax.set_title(f'Day {unique_dates.index(date) + 1}: {date}',
                             fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
            
            fig.suptitle(f'Daily Paths - {file_suffix}', fontsize=14, fontweight='bold')
            fig.tight_layout()
            
            self.save_figure(fig, output_path)
            plt.close(fig)
            generated += 1
            
            self.progress.emit(f"Saved: {db_name}_DailyPaths_{file_suffix}.png")
        
        return generated
    
    def save_trajectory_overview(self, data, output_dir, db_name):
        """Save trajectory overview
        Returns: True if generated, False if skipped"""
        self.progress.emit("Generating trajectory overview...")
        
        output_path = os.path.join(output_dir, f'{db_name}_TrajectoryOverview.png')
        
        # Check if file exists and overwrite is False
        if self.skip_existing and os.path.exists(output_path):
            self.progress.emit(f"Skipped (exists): {db_name}_TrajectoryOverview.png")
            return False
        
        # Wider than tall: the extra width is the legend column, so the
        # arena itself is not squeezed by a long animal list.
        fig = Figure(figsize=(13, 8))
        ax = fig.add_subplot(111)

        # Spatial context (background/zones/anchors) per the export choice.
        # Background is the loaded PNG, else the XML-embedded site map.
        bg_image, bg_extent = self._bg_source()
        draw_context_layers(
            ax, self.plot_layers,
            bg_image=bg_image, bg_extent=bg_extent,
            zones_xml=self.xml_zones, anchors=self.anchor_positions,
            rois=self.rois)

        x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'
        
        # Plot each tag with sex-based coloring
        for tag in data['shortid'].unique():
            tag_data = data[data['shortid'] == tag]
            
            # Determine label and color based on identity configuration
            if self.use_identities and tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', 'M')
                identity = info.get('identity', str(tag))
                label = f"{sex}-{identity}"
                color = 'blue' if sex == 'M' else 'red'
            else:
                hex_id = hex(tag).upper().replace('0X', '')
                label = f"HexID {hex_id}"
                color = 'blue'  # Default to blue
            
            ax.plot(tag_data[x_col], tag_data[y_col], 
                   linewidth=1, alpha=0.7, color=color, label=label)
        
        ax.set_xlabel('X Position (m)', fontsize=10)
        ax.set_ylabel('Y Position (m)', fontsize=10)
        ax.set_title('Trajectory Overview - All Tags', fontsize=12, fontweight='bold')
        # Outside the axes: overlaid tracks fill the arena, so an in-plot
        # legend always covers data.
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=8,
                  borderaxespad=0.0, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        # Only overrides when a range is pinned; otherwise matplotlib's
        # autoscale (which this plot has always relied on) is left alone.
        if self.axis_limits is not None:
            self._apply_axis_limits(ax, *self.axis_limits)
        fig.tight_layout()
        
        self.save_figure(fig, output_path)
        plt.close(fig)
        
        self.progress.emit(f"Saved: {db_name}_TrajectoryOverview.png")
        return True
    
    def save_battery_levels(self, data, output_dir, db_name):
        """Save battery levels plot — faceted by tag (one subplot per tag).
        Returns: True if generated, False if skipped or no battery data"""
        self.progress.emit("Generating battery levels (faceted by tag)...")

        output_path = os.path.join(output_dir, f'{db_name}_BatteryLevels.png')

        # Check if file exists and skip_existing is True
        if self.skip_existing and os.path.exists(output_path):
            self.progress.emit(f"Skipped (exists): {db_name}_BatteryLevels.png")
            return False

        # Search for battery column
        battery_col = None
        possible_names = ['battery_voltage', 'vbat', 'battery', 'bat', 'voltage']
        for col_name in possible_names:
            if col_name in data.columns:
                battery_col = col_name
                break

        if battery_col is None:
            self.progress.emit("No battery column found, skipping battery plot")
            return False

        # A battery column can exist but hold nothing usable (never populated,
        # or all NaN for a tag that does not report voltage). Non-finite values
        # propagate into the shared y-limits and set_ylim then raises, so
        # drop them here and skip the plot outright if none survive.
        data = data.copy()
        data[battery_col] = pd.to_numeric(data[battery_col], errors='coerce')
        data = data[np.isfinite(data[battery_col])]
        if data.empty:
            self.progress.emit(
                f"Battery column '{battery_col}' has no usable values, "
                "skipping battery plot")
            return False

        tags = sorted(data['shortid'].unique())
        n_tags = len(tags)

        if n_tags == 0:
            self.progress.emit("No tags found, skipping battery plot")
            return False

        # Convert the time axis once for the whole frame rather than letting
        # each ax.plot re-convert its slice (see date_axis).
        data = data.assign(_x=date_axis(data['Timestamp']))

        # Size: generous height per facet so many tags remain readable when zoomed
        subplot_height = 2.0
        fig_height = max(6, n_tags * subplot_height + 1.5)
        fig = Figure(figsize=(14, fig_height), dpi=150)

        axes = fig.subplots(n_tags, 1, sharex=True, squeeze=False)
        axes = axes.flatten()

        # Shared y-limits across all facets for easy comparison
        global_min = data[battery_col].min()
        global_max = data[battery_col].max()
        y_margin = (global_max - global_min) * 0.1 if global_max != global_min else 0.1
        y_lo = global_min - y_margin
        y_hi = global_max + y_margin

        for idx, tag in enumerate(tags):
            ax = axes[idx]
            tag_data = data[data['shortid'] == tag]

            # Determine label from identity info
            if self.use_identities and tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', 'M')
                identity = info.get('identity', str(tag))
                label = f"{sex}-{identity}"
            else:
                hex_id = hex(tag).upper().replace('0X', '')
                label = f"HexID {hex_id}"

            color = 'tab:blue'
            if self.use_identities and tag in self.tag_identities:
                sex = self.tag_identities[tag].get('sex', 'M')
                color = 'tab:blue' if sex == 'M' else 'tab:red'

            # A marker per fix stops meaning anything once there are more
            # fixes than pixels to put them on: the axis is ~2800 px wide, so
            # a 691k-point series drew a solid band the line already covered,
            # and it cost 6 of the 7 seconds spent encoding this figure. Keep
            # them where they still separate - which is also where they earn
            # their place, since a tag with a handful of readings would
            # otherwise be a bare line, and a single reading nothing at all.
            marker_kw = ({} if len(tag_data) > 2000
                         else dict(marker='o', markersize=1))
            ax.xaxis_date(self.timezone)
            ax.plot(tag_data['_x'].to_numpy(),
                    tag_data[battery_col].to_numpy(),
                    linewidth=1.0, color=color, alpha=0.85, **marker_kw)

            ax.set_ylim(y_lo, y_hi)
            ax.set_ylabel(label, fontsize=8, fontweight='bold', rotation=0, labelpad=60, ha='right')
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis='y', labelsize=7)
            ax.tick_params(axis='x', labelsize=7)

            # Only show x-tick labels on the bottom subplot
            if idx < n_tags - 1:
                ax.tick_params(axis='x', labelbottom=False)

        # Bottom axis label
        axes[-1].set_xlabel('Time', fontsize=10)

        fig.suptitle(f'Battery Levels Over Time — {battery_col} (V)', fontsize=13, fontweight='bold', y=1.0)
        fig.autofmt_xdate()
        fig.tight_layout(rect=[0, 0, 1, 0.98])

        self.save_figure(fig, output_path)
        plt.close(fig)

        self.progress.emit(f"Saved: {db_name}_BatteryLevels.png")
        return True
    
    def save_activity_timeline(self, data, output_dir, db_name):
        """Save activity timeline
        Returns: True if generated, False if skipped"""
        self.progress.emit("Generating activity timeline...")
        
        output_path = os.path.join(output_dir, f'{db_name}_ActivityTimeline.png')
        
        if self.skip_existing and os.path.exists(output_path):
            self.progress.emit(f"Skipped (exists): {db_name}_ActivityTimeline.png")
            return False
        
        fig = Figure(figsize=(15, 6))
        ax = fig.add_subplot(111)

        for tag in sorted(data['shortid'].unique()):
            tag_data = data[data['shortid'] == tag]
            hourly_counts = tag_data.set_index('Timestamp').resample('h').size()
            # Hex tag AND the configured animal ID side by side: the hex is
            # what the hardware reports, the ID is what the analysis uses,
            # and this plot is where you check one against the other.
            ax.plot(hourly_counts.index, hourly_counts.values,
                    label=self._tag_legend_label(tag), linewidth=1.5)

        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Data Points per Hour', fontsize=10)
        ax.set_title('Activity Timeline - Data Points Over Time', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=8,
                  borderaxespad=0.0, framealpha=0.9, title='HexID \u2192 animal')
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        self.save_figure(fig, output_path)
        plt.close(fig)
        
        self.progress.emit(f"Saved: {db_name}_ActivityTimeline.png")
        return True
    
    def save_velocity_distribution(self, data, output_dir, db_name):
        """Save velocity distribution
        Returns: True if generated, False if skipped"""
        self.progress.emit("Generating velocity distribution...")
        
        output_path = os.path.join(output_dir, f'{db_name}_VelocityDistribution.png')
        
        if self.skip_existing and os.path.exists(output_path):
            self.progress.emit(f"Skipped (exists): {db_name}_VelocityDistribution.png")
            return False
        
        data = data.copy()
        x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'
        
        # Calculate velocity
        data['time_diff'] = data.groupby('shortid')['Timestamp'].diff().dt.total_seconds()
        data['distance'] = np.sqrt(
            (data[x_col] - data.groupby('shortid')[x_col].shift())**2 +
            (data[y_col] - data.groupby('shortid')[y_col].shift())**2
        )
        data['velocity'] = data['distance'] / data['time_diff']
        
        # Filter out unrealistic velocities
        data = data[(data['velocity'] <= 2) | (data['velocity'].isna())]
        
        # One panel per animal rather than 17 translucent histograms on one
        # axis: overlaid densities stop being readable past about four
        # animals, and the point of this plot is per-animal shape.
        tags = [t for t in sorted(data['shortid'].unique())
                if len(data[data['shortid'] == t]['velocity'].dropna())]
        if not tags:
            self.progress.emit("No usable velocities, skipping distribution")
            return False

        n_rows, n_cols, figsize = self.facet_grid(
            len(tags), max_cols=4, panel_w=3.4, panel_h=2.6, extra_h=0.6)
        fig = Figure(figsize=figsize)
        # A shared x-axis is what makes the panels comparable; y is per-panel
        # because a mostly-still animal has a far taller spike at zero.
        v_max = float(data['velocity'].dropna().max() or 1.0)
        for i, tag in enumerate(tags):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            vals = data[data['shortid'] == tag]['velocity'].dropna()
            ax.hist(vals, bins=50, range=(0, v_max), density=True,
                    color='#4da3ff', alpha=0.85)
            ax.set_title(self._tag_label(tag), fontsize=9)
            ax.set_xlim(0, v_max)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            if i % n_cols == 0:
                ax.set_ylabel('Density', fontsize=8)
            if i >= len(tags) - n_cols:
                ax.set_xlabel('Velocity (m/s)', fontsize=8)
        fig.suptitle('Velocity Distribution by Animal',
                     fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.save_figure(fig, output_path)
        plt.close(fig)
        
        self.progress.emit(f"Saved: {db_name}_VelocityDistribution.png")
        return True
    
    def save_cumulative_distance(self, data, output_dir, db_name):
        """Save cumulative distance plots (reset daily)
        Returns: True if generated, False if skipped"""
        self.progress.emit("Generating cumulative distance plots...")
        
        output_path = os.path.join(output_dir, f'{db_name}_CumulativeDistance.png')
        
        if self.skip_existing and os.path.exists(output_path):
            self.progress.emit(f"Skipped (exists): {db_name}_CumulativeDistance.png")
            return False
        
        data = data.copy()
        if 'Date' not in data.columns:
            data['Date'] = data['Timestamp'].dt.date
        
        x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'
        
        # Calculate distance between consecutive points
        data = data.sort_values(['shortid', 'Timestamp'])
        # Per-tag step distance. Done column-wise rather than with
        # groupby().apply(): with a single tag the apply returned a DataFrame
        # instead of a Series and the assignment raised, which took down
        # every plot queued after this one.
        dx = data.groupby('shortid')[x_col].diff()
        dy = data.groupby('shortid')[y_col].diff()
        data['distance_step'] = np.sqrt(dx ** 2 + dy ** 2).fillna(0)
        
        # Cumulative distance per day (reset each day)
        data['cumulative_distance'] = data.groupby(['shortid', 'Date'])['distance_step'].cumsum()
        data['time_of_day'] = (data['Timestamp'] - data['Timestamp'].dt.normalize()).dt.total_seconds() / 3600
        
        unique_days = sorted(data['Date'].unique())
        num_days = len(unique_days)
        num_rows, num_cols, figsize = self.facet_grid(
            num_days, max_cols=5, panel_w=5.0, panel_h=4.0, extra_h=0.5)
        fig = Figure(figsize=figsize)
        
        for i, day in enumerate(unique_days):
            ax = fig.add_subplot(num_rows, num_cols, i + 1)
            day_data = data[data['Date'] == day]
            
            for tag in day_data['shortid'].unique():
                tag_data = day_data[day_data['shortid'] == tag]
                
                # Generate label
                if self.use_identities and tag in self.tag_identities:
                    info = self.tag_identities[tag]
                    sex = info.get('sex', 'M')
                    identity = info.get('identity', str(tag))
                    label = f"{sex}-{identity}"
                else:
                    hex_id = hex(tag).upper().replace('0X', '')
                    label = f"HexID {hex_id}"
                
                ax.plot(tag_data['time_of_day'], tag_data['cumulative_distance'], label=label, alpha=0.7)
            
            ax.set_xlabel('Hour of Day', fontsize=9)
            ax.set_ylabel('Distance (m)', fontsize=9)
            ax.set_title(f'Day {i+1}: {day}', fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        
        fig.suptitle('Cumulative Distance Traveled by Day', fontsize=12, fontweight='bold')
        fig.tight_layout()
        self.save_figure(fig, output_path)
        plt.close(fig)
        
        self.progress.emit(f"Saved: {db_name}_CumulativeDistance.png")
        return True
    
    def save_velocity_timeline(self, data, output_dir, db_name):
        """One faceted figure of speed over time, a panel per animal.

        Was a separate file per animal, which made comparing them a matter of
        opening 17 images side by side. Returns True/False like the other
        single-file plots.
        """
        self.progress.emit("Generating velocity timeline...")

        output_path = os.path.join(output_dir, f'{db_name}_VelocityTimeline.png')
        if self.skip_existing and os.path.exists(output_path):
            self.progress.emit(f"Skipped (exists): {db_name}_VelocityTimeline.png")
            return False

        data = data.copy()
        x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'

        data['time_diff'] = data.groupby('shortid')['Timestamp'].diff().dt.total_seconds()
        data['distance'] = np.sqrt(
            (data[x_col] - data.groupby('shortid')[x_col].shift()) ** 2 +
            (data[y_col] - data.groupby('shortid')[y_col].shift()) ** 2
        )
        data['velocity'] = data['distance'] / data['time_diff']
        data = data[(data['velocity'] <= 2) | (data['velocity'].isna())]

        tags = [t for t in sorted(data['shortid'].unique())
                if len(data[data['shortid'] == t]['velocity'].dropna())]
        if not tags:
            self.progress.emit("No usable velocities, skipping velocity timeline")
            return False

        # The threshold drawn here is the SAME one the behaviour detector uses
        # to call an animal inactive, so the line shows exactly where the
        # inactive/moving split falls on this trial rather than a hard-coded
        # 0.1 m/s that matched nothing.

        n_rows, n_cols, figsize = self.facet_grid(
            len(tags), max_cols=3, panel_w=5.5, panel_h=2.4, extra_h=0.7)
        fig = Figure(figsize=figsize)
        # One conversion of the time axis for the whole frame (see date_axis).
        data = data.assign(_x=date_axis(data['Timestamp']))
        t0, t1 = data['_x'].min(), data['_x'].max()
        for i, tag in enumerate(tags):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            ax.xaxis_date(self.timezone)
            d = data[data['shortid'] == tag]
            ax.plot(d['_x'].to_numpy(), d['velocity'].to_numpy(),
                    alpha=0.6, linewidth=0.4, color='#4da3ff')
            ax.set_title(self._tag_label(tag), fontsize=9)
            # Shared time axis so the panels line up as a single timeline.
            ax.set_xlim(t0, t1)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            if i % n_cols == 0:
                ax.set_ylabel('m/s', fontsize=8)

        fig.suptitle('Velocity Timeline by Animal',
                     fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.save_figure(fig, output_path)
        plt.close(fig)
        self.progress.emit(f"Saved: {db_name}_VelocityTimeline.png")
        return True

    def save_data_quality(self, data, output_dir, db_name):
        """Save data quality metrics table
        Returns: True if generated, False if skipped"""
        self.progress.emit("Generating data quality metrics...")
        
        output_path = os.path.join(output_dir, f'{db_name}_DataQuality.png')
        
        if self.skip_existing and os.path.exists(output_path):
            self.progress.emit(f"Skipped (exists): {db_name}_DataQuality.png")
            return False
        
        quality_data = []
        for tag in sorted(data['shortid'].unique()):
            tag_data = data[data['shortid'] == tag].sort_values('Timestamp')
            gaps = tag_data['Timestamp'].diff().dt.total_seconds()
            
            # Generate label
            if self.use_identities and tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', 'M')
                identity = info.get('identity', str(tag))
                label = f"{sex}-{identity}"
            else:
                hex_id = hex(tag).upper().replace('0X', '')
                label = f"HexID {hex_id}"
            
            median_gap = gaps.median()
            max_gap = gaps.max()
            large_gaps = (gaps > 60).sum()
            
            quality_data.append([
                label,
                f"{median_gap:.2f}s" if pd.notna(median_gap) else "N/A",
                f"{max_gap:.2f}s" if pd.notna(max_gap) else "N/A",
                str(large_gaps)
            ])
        
        fig = Figure(figsize=(10, max(4, len(quality_data) * 0.5)))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        table = ax.table(cellText=quality_data,
                        colLabels=['Tag', 'Median Gap', 'Max Gap', 'Gaps >60s'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.25, 0.25, 0.25, 0.25])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Style header
        for i in range(4):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Data Quality Metrics', fontsize=12, fontweight='bold', pad=20)
        fig.tight_layout()
        self.save_figure(fig, output_path)
        plt.close(fig)
        
        self.progress.emit(f"Saved: {db_name}_DataQuality.png")
        return True

    # ── Home range ───────────────────────────────────────────────────────────

    def _tag_label(self, tag):
        """Display label for a tag: sex-identity when configured, else hex ID."""
        if self.use_identities and tag in self.tag_identities:
            info = self.tag_identities[tag]
            return f"{info.get('sex', 'M')}-{info.get('identity', str(tag))}"
        return f"HexID {hex(tag).upper().replace('0X', '')}"

    def _tag_legend_label(self, tag):
        """'HexID 30 - F9701' when identities are configured, else the hex alone."""
        hex_id = hex(tag).upper().replace('0X', '')
        if self.use_identities and tag in self.tag_identities:
            info = self.tag_identities[tag]
            return (f"HexID {hex_id} \u2014 "
                    f"{info.get('sex', 'M')}{info.get('identity', str(tag))}")
        return f"HexID {hex_id}"

    def _tag_file_suffix(self, tag):
        if self.use_identities and tag in self.tag_identities:
            info = self.tag_identities[tag]
            return f"{info.get('sex', 'M')}-{info.get('identity', str(tag))}"
        return f"HexID{hex(tag).upper().replace('0X', '')}"

    def _animal_label(self, member_tags):
        """Plot label for an animal, naming every tag it wore.

        One tag is the ordinary case and reads exactly as before. Two means the
        tag was swapped mid-trial, and both HexIDs are named so the figure says
        which hardware produced it.
        """
        if len(member_tags) == 1:
            return self._tag_label(member_tags[0])
        hexes = ', '.join(hex(t).upper().replace('0X', '') for t in member_tags)
        info = self.tag_identities.get(member_tags[0], {}) if self.use_identities else {}
        name = f"{info.get('sex', '')}{info.get('identity', '')}" or None
        return f"HexIDs {hexes}" + (f" — {name}" if name else "")

    # Plots that describe an ANIMAL take the merged frame; plots that
    # describe a TAG do not. A replaced tag is still the same animal, so its
    # ethogram row, its speed and its tracks belong on one line - but its
    # battery is the battery of one piece of hardware, and the coverage plot
    # is about how well each tag reported, so those two keep the tags apart.
    _PER_TAG_PLOTS = frozenset({'battery_levels', 'data_quality',
                                'activity_timeline'})

    def merge_tag_data(self, data):
        """Rewrite the shortid of a replaced tag to the animal's first tag.

        When one animal wore two tags, everything downstream keys off
        `shortid` and therefore treated it as two animals: two ethogram rows,
        two velocity panels, two entries in the trajectory legend. Collapsing
        the ids here fixes all of them at once and keeps the label lookups
        working, because the surviving id is a real tag of that animal.

        Rows are re-sorted by (shortid, Timestamp) afterwards: the per-tag
        diffs that several plots take - speed, cumulative distance - run down
        the frame in order, and two tags concatenated out of order would put a
        spurious backwards step at the join.
        """
        if 'shortid' not in data.columns:
            return data, {}
        groups = self.animal_groups(sorted(data['shortid'].unique()))
        remap = {}
        for _suffix, members in groups:
            for extra in members[1:]:
                remap[extra] = members[0]
        if not remap:
            return data, {}
        merged = data.copy()
        merged['shortid'] = merged['shortid'].map(lambda t: remap.get(t, t))
        merged = merged.sort_values(['shortid', 'Timestamp'], kind='stable')
        return merged.reset_index(drop=True), remap

    def animal_groups(self, tags):
        """[(file suffix, [shortid, ...]), ...] - one entry per ANIMAL.

        Usually one tag per animal, so this is just the tag list. But a tag
        replaced mid-trial gives one animal two shortids, and both resolve to
        the same 'sex-identity' filename: the per-animal plots were written
        per TAG, so the second file silently overwrote the first and every day
        the original tag recorded was lost from the figure. Grouping by the
        filename gives one file per animal holding all of its data, which also
        matches what proximity, the GBI and the edge list already do - they
        key on the identity label, which is why a 17-tag trial reports 16
        animals.

        Order follows the first tag of each animal, so a one-tag-per-animal
        trial produces exactly the file order it always did.
        """
        groups = {}
        for tag in tags:
            groups.setdefault(self._tag_file_suffix(tag), []).append(tag)
        told = getattr(self, '_told_about_groups', None)
        if told is None:
            told = self._told_about_groups = set()
        for suffix, members in groups.items():
            if len(members) > 1 and suffix not in told:
                told.add(suffix)
                hexes = ', '.join(hex(t).upper().replace('0X', '') for t in members)
                self.progress.emit(
                    f"  {suffix}: HexIDs {hexes} share this identity — "
                    f"plotted together as one animal")
        return list(groups.items())

    # ── Home range: KDE and MCP as separate, comparable products ────────────
    # Kept as one renderer with a `method` switch rather than two near-copies:
    # the framing, faceting, shared axis limits and area annotation must stay
    # identical or the two plots are not comparable, which is the entire point
    # of splitting them.

    _HR_STYLE = {
        'kde': dict(
            tag_prefix='HomeRange', title='Home Range (KDE)',
            legend=(('KDE 50% (core)', '#ff4d4d', '-', 1.8),
                    ('KDE 95% (home range)', '#4da3ff', '-', 1.4)),
            subtitle=('filled = KDE 50% core · line = KDE 95%')),
        'mcp': dict(
            tag_prefix='MCP', title='Home Range (MCP)',
            legend=(('MCP 95% (peeled)', '#2ea043', '-', 1.6),
                    ('MCP 100% (all fixes)', '#f0c419', '--', 1.3)),
            subtitle=('solid = 95% MCP · dashed = 100% MCP')),
    }

    def _hr_estimate(self, ax, sub_x, sub_y, method):
        """Draw one panel's estimator(s); returns {label: area_m2}."""
        from fnt.uwb import spatial_metrics as SM

        areas = {}
        if method == 'kde':
            xg, yg, dens, cell = SM.kde_utilization(sub_x, sub_y)
            for pct, colour, lw in ((95, '#4da3ff', 1.4), (50, '#ff4d4d', 1.8)):
                level = SM.kde_isopleth(dens, cell, pct)
                if level is None:
                    areas[f'KDE {pct}%'] = float('nan')
                    continue
                ax.contour(xg, yg, dens, levels=[level], colors=[colour],
                           linewidths=lw, zorder=4)
                areas[f'KDE {pct}%'] = SM.kde_isopleth_area(dens, cell, level)
        else:
            for pct, colour, style, lw in ((100, '#f0c419', '--', 1.3),
                                           (95, '#2ea043', '-', 1.6)):
                verts, area = SM.mcp(sub_x, sub_y, percent=pct)
                areas[f'MCP {pct}%'] = area
                if verts is None:
                    continue
                ring = np.vstack([verts, verts[:1]])
                ax.plot(ring[:, 0], ring[:, 1], style, color=colour,
                        linewidth=lw, alpha=0.9, zorder=3)
        return areas

    def _save_home_range(self, data, output_dir, db_name, method):
        """Per-tag panels faceted by day, plus a whole-trial overview."""
        from fnt.uwb import spatial_metrics as SM

        style = self._HR_STYLE[method]
        self.progress.emit(f"Generating {style['title']} plots...")

        data = data.copy()
        if 'Date' not in data.columns:
            data['Date'] = data['Timestamp'].dt.date
        x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'

        unique_dates = sorted(data['Date'].unique())
        unique_tags = sorted(data['shortid'].unique())
        bg_image, bg_extent = self._bg_source()
        generated = 0

        # Shared limits so every panel and every tag is on the same scale -
        # comparing range SIZE across animals is the point of the plot.
        gx_min, gx_max = data[x_col].min(), data[x_col].max()
        gy_min, gy_max = data[y_col].min(), data[y_col].max()
        if self.plot_layers.get('background') and bg_extent is not None:
            gx_min = min(gx_min, bg_extent[0]); gx_max = max(gx_max, bg_extent[1])
            gy_min = min(gy_min, bg_extent[2]); gy_max = max(gy_max, bg_extent[3])
        pad = 0.05 * max(gx_max - gx_min, gy_max - gy_min, 1e-6)

        def _draw(ax, sub_x, sub_y, title, show_legend):
            draw_context_layers(
                ax, self.plot_layers, bg_image=bg_image, bg_extent=bg_extent,
                zones_xml=self.xml_zones, anchors=self.anchor_positions,
                rois=self.rois)
            ax.plot(sub_x, sub_y, '.', ms=0.6, color='#999999', alpha=0.35, zorder=2)
            areas = self._hr_estimate(ax, sub_x, sub_y, method)
            self._apply_axis_limits(ax, gx_min - pad, gx_max + pad,
                                    gy_min - pad, gy_max + pad, equal=True)
            ax.set_title(title, fontsize=9)
            ax.tick_params(labelsize=7)
            txt = "\\n".join(
                f"{k}: {v:.2f} m²" if np.isfinite(v) else f"{k}: n/a"
                for k, v in areas.items())
            ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', ha='left',
                    fontsize=6.5, family='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.75,
                              ec='none'),
                    zorder=6)
            if show_legend:
                handles = [Line2D([], [], color=c, lw=lw, ls=ls, label=lab)
                           for lab, c, ls, lw in style['legend']]
                ax.legend(handles=handles, loc='lower right', fontsize=6,
                          framealpha=0.8)
            return areas

        # ---- one file per animal, faceted by day ----------------------------
        for suffix, member_tags in self.animal_groups(unique_tags):
            output_path = os.path.join(
                output_dir, f"{db_name}_{style['tag_prefix']}_{suffix}.png")
            if self.skip_existing and os.path.exists(output_path):
                self.progress.emit(f"Skipped (exists): {os.path.basename(output_path)}")
                continue

            tag_data = data[data['shortid'].isin(member_tags)]
            if tag_data.empty:
                continue
            # Days this animal was actually recorded; an absent day adds an
            # empty panel and shrinks the ones that have data.
            tag_days = [d for d in unique_dates
                        if not tag_data[tag_data['Date'] == d].empty]
            if not tag_days:
                continue
            n_rows, n_cols, figsize = self.facet_grid(
                len(tag_days), max_cols=5, panel_w=4.2, panel_h=4.4, extra_h=0.5)
            fig = Figure(figsize=figsize)
            for i, day in enumerate(tag_days):
                ax = fig.add_subplot(n_rows, n_cols, i + 1)
                d = tag_data[tag_data['Date'] == day]
                # Trial-wide day number, so 'Day 7' is the same day on every
                # animal's figure even though the panel sets differ.
                day_no = unique_dates.index(day) + 1
                _draw(ax, d[x_col].values, d[y_col].values,
                      f'Day {day_no}: {day}  (n={len(d):,})', show_legend=(i == 0))
            fig.suptitle(
                f"{style['title']} by Day — {self._animal_label(member_tags)}",
                fontsize=13, fontweight='bold')
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            self.save_figure(fig, output_path)
            plt.close(fig)
            generated += 1
            self.progress.emit(f"Saved: {os.path.basename(output_path)}")

        # ---- one overview: whole trial, every animal --------------------------
        overview_path = os.path.join(
            output_dir, f"{db_name}_{style['tag_prefix']}_Overview.png")
        if self.skip_existing and os.path.exists(overview_path):
            self.progress.emit(f"Skipped (exists): {os.path.basename(overview_path)}")
            return generated

        fig = Figure(figsize=(11, 9))
        ax = fig.add_subplot(111)
        draw_context_layers(ax, self.plot_layers, bg_image=bg_image,
                            bg_extent=bg_extent, zones_xml=self.xml_zones,
                            anchors=self.anchor_positions,
                            rois=self.rois)
        cmap = plt.get_cmap('tab10')
        handles, rows = [], []
        for i, (_suffix, member_tags) in enumerate(self.animal_groups(unique_tags)):
            colour = cmap(i % 10)
            d = data[data['shortid'].isin(member_tags)]
            xs, ys = d[x_col].values, d[y_col].values
            label = self._animal_label(member_tags)
            if method == 'kde':
                xg, yg, dens, cell = SM.kde_utilization(xs, ys)
                lvl95 = SM.kde_isopleth(dens, cell, 95)
                lvl50 = SM.kde_isopleth(dens, cell, 50)
                if lvl95 is not None:
                    ax.contour(xg, yg, dens, levels=[lvl95], colors=[colour],
                               linewidths=1.6, zorder=4)
                if lvl50 is not None:
                    ax.contourf(xg, yg, dens, levels=[lvl50, dens.max()],
                                colors=[colour], alpha=0.22, zorder=3)
                rows.append((label,
                             SM.kde_isopleth_area(dens, cell, lvl50),
                             SM.kde_isopleth_area(dens, cell, lvl95)))
            else:
                v100, a100 = SM.mcp(xs, ys, percent=100)
                v95, a95 = SM.mcp(xs, ys, percent=95)
                if v100 is not None:
                    ring = np.vstack([v100, v100[:1]])
                    ax.plot(ring[:, 0], ring[:, 1], '--', color=colour, lw=1.0,
                            alpha=0.7, zorder=3)
                if v95 is not None:
                    ring = np.vstack([v95, v95[:1]])
                    ax.plot(ring[:, 0], ring[:, 1], '-', color=colour, lw=1.8,
                            alpha=0.9, zorder=4)
                    ax.fill(ring[:, 0], ring[:, 1], color=colour, alpha=0.12,
                            zorder=2)
                rows.append((label, a95, a100))
            handles.append(Line2D([], [], color=colour, lw=2, label=label))
        self._apply_axis_limits(ax, gx_min - pad, gx_max + pad,
                                gy_min - pad, gy_max + pad, equal=True)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.85)
        if method == 'kde':
            summary = "  ".join(f"{n}: core {a:.1f} / range {b:.1f} m²"
                                for n, a, b in rows)
        else:
            summary = "  ".join(f"{n}: 95% {a:.1f} / 100% {b:.1f} m²"
                                for n, a, b in rows)
        ax.set_title(f"{style['title']} — whole trial\\n{style['subtitle']}",
                     fontsize=12, fontweight='bold')
        fig.text(0.01, 0.005, summary, fontsize=6.5, family='monospace', wrap=True)
        fig.tight_layout(rect=[0, 0.03, 1, 1])
        self.save_figure(fig, overview_path)
        plt.close(fig)
        generated += 1
        self.progress.emit(f"Saved: {os.path.basename(overview_path)}")
        return generated

    def save_home_range(self, data, output_dir, db_name):
        """Kernel-density home range: 50% core and 95% range."""
        return self._save_home_range(data, output_dir, db_name, 'kde')

    def save_mcp_range(self, data, output_dir, db_name):
        """Minimum convex polygon home range: 95% peeled and 100%."""
        return self._save_home_range(data, output_dir, db_name, 'mcp')

    def _merge_occupancy_zones(self):
        """The regions this plot measures: the drawn ones, in list order.

        DRAWN REGIONS ONLY, matching the occupancy CSVs. Site-XML zones used
        to be a second measurement source merged in ahead of these; they are
        now imported into the ROI list instead, so a zone that has not been
        imported is drawn (if the zones layer is on) but not measured. One
        source means precedence is just list order, first match wins - the
        same rule the CSVs and the preview use.
        """
        zones = [dict(r) for r in (self.rois or [])
                 if 'roi' in self.occupancy_sources]
        return [z for z in zones if _polygon_len(z) >= 3]

    def save_zone_occupancy(self, data, output_dir, db_name):
        """Time spent in each region, per animal, over the recording.

        Regions are the ones drawn over the preview; returns False if the
        trial has none. Independent of whether they are being DRAWN on the
        other figures - a region hidden for legibility is still measured here.
        """
        from fnt.uwb import spatial_metrics as SM

        self.progress.emit("Generating zone occupancy...")

        zones = list(self.occupancy_zones or [])
        if not zones:
            self.progress.emit(
                "No regions drawn for this trial, skipping zone occupancy")
            return False

        output_path = os.path.join(output_dir, f'{db_name}_ZoneOccupancy.png')
        if self.skip_existing and os.path.exists(output_path):
            self.progress.emit(f"Skipped (exists): {db_name}_ZoneOccupancy.png")
            return False

        data = data.copy()
        if 'Date' not in data.columns:
            data['Date'] = data['Timestamp'].dt.date
        x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'

        data['zone_idx'] = SM.assign_zones(data[x_col].values, data[y_col].values, zones)
        zone_names = [z.get('name', f'Zone {i+1}') for i, z in enumerate(zones)]
        labels = zone_names + ['outside']
        n = len(zones)

        unique_tags = sorted(data['shortid'].unique())
        unique_dates = sorted(data['Date'].unique())

        # Zone colours come from the XML where authored, so the bars match the
        # arena drawing the user already knows.
        cmap = plt.get_cmap('tab20')
        colours = []
        for i, z in enumerate(zones):
            c = z.get('color') or ''
            colours.append(c if isinstance(c, str) and c.startswith('#') else cmap(i % 20))
        colours.append('#bbbbbb')   # outside

        fig = Figure(figsize=(15, 3.0 + 2.6 * len(unique_tags)))
        gs = fig.add_gridspec(len(unique_tags) + 1, 2, width_ratios=[3, 1],
                              hspace=0.55, wspace=0.25)

        # --- per animal: hourly stacked occupancy over the whole recording ---
        for r, tag in enumerate(unique_tags):
            ax = fig.add_subplot(gs[r, 0])
            d = data[data['shortid'] == tag].sort_values('Timestamp')
            if d.empty:
                ax.set_ylabel(self._tag_label(tag), fontsize=8)
                continue
            hours = d['Timestamp'].dt.floor('h')
            frac = np.zeros((n + 1, 0))
            centres = []
            for h, g in d.groupby(hours, sort=True):
                occ = SM.zone_occupancy(g['zone_idx'].values, g['Timestamp'].values, n)
                tot = occ.sum()
                frac = np.column_stack([frac, occ / tot if tot > 0 else occ])
                centres.append(h)
            if not centres:
                continue
            centres = pd.to_datetime(pd.Series(centres))
            bottom = np.zeros(frac.shape[1])
            for i in range(n + 1):
                ax.bar(centres, frac[i], width=pd.Timedelta(hours=1),
                       bottom=bottom, color=colours[i], edgecolor='none',
                       label=labels[i] if r == 0 else None)
                bottom = bottom + frac[i]
            ax.set_ylim(0, 1)
            ax.set_ylabel(self._tag_label(tag), fontsize=8, fontweight='bold')
            ax.tick_params(labelsize=7)
            ax.margins(x=0)
            if r == 0:
                ax.legend(fontsize=6.5, ncol=min(6, n + 1), loc='upper center',
                          bbox_to_anchor=(0.5, 1.55), framealpha=0.9)
            if r < len(unique_tags) - 1:
                ax.tick_params(axis='x', labelbottom=False)

        # --- right column: total time per zone per animal, as a matrix -------
        ax = fig.add_subplot(gs[:, 1])
        mat = np.zeros((len(unique_tags), n + 1))
        for r, tag in enumerate(unique_tags):
            d = data[data['shortid'] == tag].sort_values('Timestamp')
            if d.empty:
                continue
            occ = SM.zone_occupancy(d['zone_idx'].values, d['Timestamp'].values, n)
            tot = occ.sum()
            mat[r] = occ / tot * 100.0 if tot > 0 else occ
        im = ax.imshow(mat, aspect='auto', cmap='viridis', vmin=0,
                       vmax=max(1.0, mat.max()))
        ax.set_xticks(range(n + 1))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(len(unique_tags)))
        ax.set_yticklabels([self._tag_label(t) for t in unique_tags], fontsize=7)
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                if mat[r, c] >= 1.0:
                    ax.text(c, r, f'{mat[r, c]:.0f}', ha='center', va='center',
                            fontsize=6, color='white')
        ax.set_title('% of tracked time', fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03).ax.tick_params(labelsize=6)

        fig.suptitle(
            f'Zone Occupancy — {len(zones)} drawn region(s), '
            f'{len(unique_dates)} day(s)',
            fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.save_figure(fig, output_path)
        plt.close(fig)
        self.progress.emit(f"Saved: {db_name}_ZoneOccupancy.png")
        return True

    # ── Behaviour ethogram ───────────────────────────────────────────────────

    def _ethogram_states(self, data, tags, hz=1.0, collect_links=False):
        """Classify every tag on a regular grid, in overlapping chunks.

        Returns ``(grid_times, loc, soc, events, links)`` with loc/soc as
        (frames, tags) code arrays. The classifier builds (frames, tags, tags) pairwise arrays, so a
        multi-day trial is processed in blocks with an overlap margin wide
        enough to warm the lagged terms (heading lag, displacement window,
        staleness limit); the margin is then trimmed off.
        """
        from fnt.uwb import behavior_detection as BD

        params = self.behavior_params or BD.BehaviorParams()
        x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'

        step_ns = int(1e9 / hz)
        t = data['Timestamp'].dt.as_unit('ns').astype('int64')
        t0, t1 = int(t.min()), int(t.max())
        grid = np.arange(t0, t1 + step_ns, step_ns, dtype='int64')
        n_frames, n_tags = len(grid), len(tags)
        if n_frames == 0 or n_tags == 0:
            return None, None, None, None, None, None

        # Bin fixes onto the grid, one column per tag.
        binned = data.copy()
        binned['fi'] = ((t - t0) // step_ns).clip(0, n_frames - 1)
        col = {tag: i for i, tag in enumerate(tags)}
        X = np.full((n_frames, n_tags), np.nan)
        Y = np.full((n_frames, n_tags), np.nan)
        for tag, g in binned.groupby('shortid', sort=False):
            if tag not in col:
                continue
            j = col[tag]
            # At typical ping rates ~20 fixes land in the same 1 s bin, and a
            # plain assignment keeps whichever pandas hands over LAST - so the
            # grid depended on the row order of the input frame. Two callers
            # pass the same data in different orders (the events export reads
            # the CSV as written, the plot run sorts it), which could give two
            # different ethograms from one file. Sort by time so a bin always
            # keeps its final fix.
            g = g.sort_values('Timestamp', kind='stable')
            fi = g['fi'].to_numpy()
            X[fi, j] = g[x_col].to_numpy()
            Y[fi, j] = g[y_col].to_numpy()

        # True millisecond observation window per frame: the earliest and latest
        # real fix (any tag) that landed in each 1 Hz slot. The grid itself is a
        # regular lattice anchored at t0, so a lattice time carries only t0's
        # constant sub-second offset - exported as-is it reads as millisecond
        # precision while every row shares the same fraction. Bout edges are
        # reported from these instead, so a behaviour's start/stop is a time an
        # animal was actually observed. -1 marks a frame with no fixes.
        _fr = pd.DataFrame({'fi': binned['fi'].to_numpy(), 'ts': t.to_numpy()})
        _agg = _fr.groupby('fi')['ts'].agg(['min', 'max'])
        frame_first = np.full(n_frames, -1, dtype='int64')
        frame_last = np.full(n_frames, -1, dtype='int64')
        _idx = _agg.index.to_numpy()
        frame_first[_idx] = _agg['min'].to_numpy()
        frame_last[_idx] = _agg['max'].to_numpy()

        # classify() derives dt by reading these as datetime64[ns] (that is
        # what the live preview hands it). Passing float SECONDS made every
        # value read as a nanosecond, so dt came out 1e-9 s: speeds were
        # ~1e9x too large and the heading lag spanned ~3e9 frames, leaving
        # every heading NaN. grid is already int64 nanoseconds since the
        # epoch, so hand it over unchanged.
        times_ns = grid.astype("datetime64[ns]")
        margin = int(np.ceil(max(params.heading_lag_s, params.displace_window_s,
                                 params.max_stale_s, 1.0) * hz)) + 2
        # ~2 h of 1 Hz frames per block keeps the (frames, tags, tags) pairwise
        # arrays comfortably small even for a full cohort.
        block = max(margin * 4, int(7200 * hz))

        # Has-a-fix, which is all the locomotor array was still needed
        # for once the inactive/moving split went: the social raster has
        # to tell 'alone' from 'not tracked'.
        have = np.zeros((n_frames, n_tags), dtype=bool)
        soc = np.full((n_frames, n_tags), BD.SOC_NONE, dtype=np.int8)
        events = []
        links = []
        start = 0
        while start < n_frames:
            stop = min(start + block, n_frames)
            lo = max(0, start - margin)
            hi = min(n_frames, stop + margin)
            res = BD.classify(times_ns[lo:hi], X[lo:hi], Y[lo:hi], params)
            have[start:stop] = res['present'][start - lo:stop - lo]
            soc[start:stop] = res['social'][start - lo:stop - lo]
            if not DISPLACEMENT_ENABLED:
                # classify() still resolves displacement and it outranks
                # contact in the social precedence, so leaving it here would
                # paint a parked detector's output straight into the exported
                # ethogram. Fall back to what the animal was otherwise doing:
                # contact if it was touching someone, else no social state.
                blk = soc[start:stop]
                hit = (blk == BD.SOC_DISPLACING) | (blk == BD.SOC_DISPLACED)
                if hit.any():
                    touching = res['contact'][start - lo:stop - lo].any(axis=2)
                    blk[hit] = np.where(touching[hit],
                                        BD.SOC_CONTACT, BD.SOC_NONE)
            # Directed bouts. Extracted over the block INCLUDING its warm-up
            # margin, then kept only when the bout STARTS inside this
            # block's own span: a bout straddling a boundary is therefore
            # counted exactly once, by the block that saw its onset with the
            # lagged terms already warm.
            for ev in BD.extract_events(res, offset=lo):
                if start <= ev['start_frame'] < stop:
                    events.append(ev)
            # Who was joined to whom, per frame, for the animation overlay.
            # The pairwise arrays are (frames, tags, tags) and far too big to
            # keep, but the TRUE entries are sparse - a handful per frame at
            # most - so they are collected as (frame, actor, target, kind)
            # rows and the arrays themselves are dropped with the block.
            if collect_links:
                sl = slice(start - lo, stop - lo)
                for field, kind in (('contact', BD.SOC_CONTACT),
                                    ('displacing', BD.SOC_DISPLACING),
                                    ('chase', BD.SOC_CHASING)):
                    arr = res.get(field)
                    if arr is None:
                        continue
                    fr, a, b = np.nonzero(arr[sl])
                    if kind == BD.SOC_CONTACT and len(fr):
                        # Symmetric: keep one row per pair, not both halves.
                        keep = a < b
                        fr, a, b = fr[keep], a[keep], b[keep]
                    if len(fr):
                        links.append(np.column_stack([
                            fr + start, a, b,
                            np.full(len(fr), kind)]).astype(np.int32))
            del res
            start = stop
            self.progress.emit(f"  behaviour classified {min(stop, n_frames):,}/{n_frames:,} frames")
        link_arr = (np.concatenate(links) if links
                    else np.empty((0, 4), dtype=np.int32))
        if len(link_arr):
            # Sorted by frame so a renderer can slice one frame's links with
            # two searchsorted calls instead of scanning.
            link_arr = link_arr[np.argsort(link_arr[:, 0], kind='stable')]
        return grid, have, soc, events, link_arr, (frame_first, frame_last)

    # (key, result) for the last classification, shared across every worker in
    # this process. An export classifies the same data twice - once to write
    # behavior_events.csv, then again for the ethogram - in two different
    # PlotSaverWorker instances, so the old per-instance cache never caught the
    # second one. On a 19-day, 17-tag trial that was 82 s of pairwise geometry
    # recomputed for an identical answer. Class-level because separate
    # instances are the only thing that ever separated the two calls.
    _BEH_CACHE = None

    def _behavior_classification(self, data, tags, hz=1.0, collect_links=False):
        """Classify once and reuse: the ethogram and both chase plots need it.

        Classification is the expensive part (pairwise geometry over every
        frame), so running it per plot would triple the cost of ticking all
        three, and running it again for the events CSV doubled that.

        The key covers everything the result depends on: which tags, how many
        rows, the detection thresholds, and the time span. Row order is not in
        the key because the binning above is now order-independent.
        """
        t = data['Timestamp']
        key = (tuple(tags), len(data), repr(self.behavior_params), hz,
               bool(collect_links),
               None if not len(t) else (t.min().value, t.max().value))
        cached = PlotSaverWorker._BEH_CACHE
        if cached is not None and cached[0] == key:
            self.progress.emit(
                "  reusing the behaviour classification from earlier in this run")
            return cached[1]
        result = self._ethogram_states(data, tags, hz=hz,
                                       collect_links=collect_links)
        PlotSaverWorker._BEH_CACHE = (key, result)
        return result

    def behavior_events_frame(self, data, tags, tz=None):
        """Directed chase/displacement bouts as a tidy DataFrame.

        One row per bout: who did it, to whom, when, and for how long. This is
        the dyadic record the ethogram raster cannot show — the raster says an
        animal was chasing, this says whom it was chasing.
        """
        cols = ['behavior', 'actor', 'target', 'actor_sex', 'target_sex',
                'dyad_type', 'Day', 'Date', 'bout_start', 'bout_stop',
                'duration_s']
        grid, _have, soc, events, _links, ftimes = self._behavior_classification(data, tags)
        if grid is None or not events:
            return pd.DataFrame(columns=cols)

        times = pd.to_datetime(grid, utc=True)
        if tz is not None:
            times = times.tz_convert(tz)

        def _as_local(ns):
            ts = pd.Timestamp(int(ns), unit='ns', tz='UTC')
            return ts.tz_convert(tz) if tz is not None else ts
        dt_s = (grid[1] - grid[0]) / 1e9 if len(grid) > 1 else 1.0

        def _label(i):
            tag = tags[i]
            if self.use_identities and tag in self.tag_identities:
                info = self.tag_identities[tag]
                return f"{info.get('sex', 'M')}{info.get('identity', str(tag))}"
            return f"HexID{hex(tag).upper().replace('0X', '')}"

        def _sex(i):
            tag = tags[i]
            if self.use_identities and tag in self.tag_identities:
                return self.tag_identities[tag].get('sex', 'M')
            return 'U'

        rows = []
        for ev in events:
            # Displacement is parked (see DISPLACEMENT_ENABLED). classify()
            # still computes it and extract_events still recovers the bouts,
            # but an unvalidated behaviour must not appear in an exported data
            # product where it reads as a finding.
            if not DISPLACEMENT_ENABLED and ev['behavior'] == 'displacement':
                continue
            i, j = ev['start_frame'], ev['stop_frame']
            # Prefer the true observation times bounding this bout over the
            # lattice times, so start/stop are moments an animal was actually
            # seen. Falls back to the lattice for a frame with no fixes.
            t_i = ftimes[0][i] if ftimes is not None else -1
            t_j = ftimes[1][j] if ftimes is not None else -1
            ev_start = _as_local(t_i) if t_i > 0 else times[i]
            ev_stop = _as_local(t_j) if t_j > 0 else times[j]
            sa, sb = _sex(ev['actor']), _sex(ev['target'])
            rows.append({
                'behavior': ev['behavior'],
                'actor': _label(ev['actor']),
                'target': _label(ev['target']),
                'actor_sex': sa, 'target_sex': sb,
                # MM / FF / MF / NA. Mixed is always 'MF', never 'FM':
                # sorting the two letters would give 'FM', which is not the
                # convention the rest of the work uses and would make a
                # downstream filter on "MF" match nothing at all.
                'dyad_type': _dyad_type(sa, sb),
                'bout_start': ev_start, 'bout_stop': ev_stop,
                # Observed span between the true edge fixes. A single-frame
                # bout is credited one sampling interval (the classifier's
                # resolution) rather than the sub-second gap between two fixes,
                # since that is genuinely how long the state was resolved for.
                'duration_s': float(max((ev_stop - ev_start).total_seconds(),
                                        (j - i + 1) * dt_s)),
            })
        if not rows:
            # Every bout was filtered out - all of them displacement,
            # which is parked. The empty frame's 'start' column is not
            # datetimelike, so the .dt work below cannot run on it.
            return pd.DataFrame(columns=cols)
        out = pd.DataFrame(rows, columns=cols)
        # Day is the day the bout STARTED. Nothing is split at a boundary,
        # here or in the overlap bouts: a split row is indistinguishable from
        # two real ones, and downstream can always divide what it needs.
        day0 = out['bout_start'].dt.normalize()
        out['Date'] = day0.dt.date
        out['Day'] = (day0 - day0.min()).dt.days + 1
        return out.sort_values('bout_start').reset_index(drop=True)

    def save_chase_timeline(self, data, output_dir, db_name):
        """Chase and displacement counts per day, one line per animal.

        Two panels per behaviour: how often each animal was the ACTOR, and how
        often it was the TARGET. Separating them is the point — an animal that
        chases a lot and an animal that gets chased a lot look identical in a
        combined count.
        """
        self.progress.emit("Generating chase timeline...")

        output_path = os.path.join(output_dir, f'{db_name}_BehaviorCounts.png')
        if self.skip_existing and os.path.exists(output_path):
            self.progress.emit(f"Skipped (exists): {db_name}_BehaviorCounts.png")
            return False

        tags = sorted(data['shortid'].unique())
        ev = self.behavior_events_frame(data, tags,
                                        tz=data['Timestamp'].dt.tz)
        if ev.empty:
            self.progress.emit(
                "No chases detected — skipping behaviour counts. "
                "Check the thresholds against the speed summary.")
            return False

        behaviours = [b for b in ('chase', 'displacement')
                      if (ev['behavior'] == b).any()]
        animals = sorted(set(ev['actor']) | set(ev['target']))
        days = sorted(ev['Day'].unique())
        cmap = plt.get_cmap('tab10')
        colour = {a: cmap(i % 10) for i, a in enumerate(animals)}

        fig = Figure(figsize=(6.5 * len(behaviours), 8.5))
        for bi, beh in enumerate(behaviours):
            sub = ev[ev['behavior'] == beh]
            for ri, (role, label) in enumerate((('actor', 'as actor'),
                                                ('target', 'as target'))):
                ax = fig.add_subplot(2, len(behaviours), ri * len(behaviours) + bi + 1)
                counts = (sub.groupby([role, 'Day']).size()
                            .unstack('Day').reindex(columns=days).fillna(0))
                for animal in animals:
                    y = (counts.loc[animal].to_numpy()
                         if animal in counts.index else np.zeros(len(days)))
                    ax.plot(days, y, marker='o', ms=4, lw=1.6,
                            color=colour[animal], label=animal, alpha=0.9)
                ax.set_xlabel('Day', fontsize=9)
                ax.set_ylabel(f'{beh}s {label}', fontsize=9)
                ax.set_title(f'{beh.capitalize()} — {label}', fontsize=10)
                ax.set_xticks(days)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)
                ax.set_ylim(bottom=0)
                if bi == 0 and ri == 0:
                    ax.legend(fontsize=7, ncol=2, framealpha=0.85)

        total = ", ".join(f"{len(ev[ev['behavior'] == b]):,} {b}s"
                          for b in behaviours)
        fig.suptitle(f'Behaviour Counts by Day — {total}',
                     fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        self.save_figure(fig, output_path)
        plt.close(fig)
        self.progress.emit(f"Saved: {db_name}_BehaviorCounts.png")
        return True

    def save_chase_matrix(self, data, output_dir, db_name):
        """Directed actor x target matrices for chase and displacement.

        Rows are the actor, columns the target, so cell (i, j) counts the times
        i chased (or displaced) j. Deliberately NOT symmetric: the asymmetry
        between (i, j) and (j, i) is the dominance signal.
        """
        self.progress.emit("Generating chase matrices...")

        output_path = os.path.join(output_dir, f'{db_name}_BehaviorMatrix.png')
        if self.skip_existing and os.path.exists(output_path):
            self.progress.emit(f"Skipped (exists): {db_name}_BehaviorMatrix.png")
            return False

        tags = sorted(data['shortid'].unique())
        ev = self.behavior_events_frame(data, tags,
                                        tz=data['Timestamp'].dt.tz)
        if ev.empty:
            self.progress.emit(
                "No chases detected — skipping behaviour matrix. "
                "Check the thresholds against the speed summary.")
            return False

        behaviours = [b for b in ('chase', 'displacement')
                      if (ev['behavior'] == b).any()]
        animals = sorted(set(ev['actor']) | set(ev['target']))
        n = len(animals)
        idx = {a: i for i, a in enumerate(animals)}

        # Counts and total seconds side by side: a pair with many brief chases
        # and a pair with one long one are different findings.
        fig = Figure(figsize=(1.6 + 5.2 * len(behaviours), 5.6))
        for bi, beh in enumerate(behaviours):
            sub = ev[ev['behavior'] == beh]
            mat = np.zeros((n, n))
            dur = np.zeros((n, n))
            for _, r in sub.iterrows():
                mat[idx[r['actor']], idx[r['target']]] += 1
                dur[idx[r['actor']], idx[r['target']]] += r['duration_s']

            ax = fig.add_subplot(1, len(behaviours), bi + 1)
            im = ax.imshow(mat, cmap='magma', vmin=0,
                           vmax=max(1.0, mat.max()))
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(animals, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(animals, fontsize=8)
            ax.set_xlabel(f'target (was {beh}d)', fontsize=9)
            ax.set_ylabel(f'actor (did the {beh[:5]})', fontsize=9)
            ax.set_title(f'{beh.capitalize()}: {int(mat.sum()):,} events',
                         fontsize=11, fontweight='bold')
            # The diagonal is structurally empty (no self-directed bouts).
            for i in range(n):
                ax.add_patch(MplRectangle((i - 0.5, i - 0.5), 1, 1,
                                          facecolor='#444444', edgecolor='none'))
            for i in range(n):
                for j in range(n):
                    if i == j or mat[i, j] == 0:
                        continue
                    # Label with count and total duration; white on the dark end
                    # of magma, black on the light end.
                    frac = mat[i, j] / max(1.0, mat.max())
                    ax.text(j, i, f"{int(mat[i, j])}\n{dur[i, j]:.0f}s",
                            ha='center', va='center', fontsize=6.5,
                            color='white' if frac < 0.6 else 'black')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label='events').ax.tick_params(labelsize=7)

        fig.suptitle('Directed Behaviour Matrix — row acts on column '
                     '(count above, total seconds below)',
                     fontsize=12, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        self.save_figure(fig, output_path)
        plt.close(fig)
        self.progress.emit(f"Saved: {db_name}_BehaviorMatrix.png")
        return True

    # Present, but in no social state at all. The classifier has no code for
    # this: SOC_NONE means "nothing social happened", which in the old combined
    # raster was just the locomotor colour showing through, so "alone" was not
    # readable anywhere. Splitting the raster makes it a state in its own right
    # and it needs its own colour. Dark reads as absent, light as present but
    # alone, saturated as an event.
    ALONE_COLOR = '#cfd8e3'

    def _behaviour_rasters(self, have, soc):
        """Social display codes and their colour keys.

        Codes ascend by how much they are worth seeing, because a pixel covers
        many frames and the rasteriser keeps the maximum in each bin.

        ``have`` is the has-a-fix mask, which is what separates an animal that
        was alone from one that was not being tracked at all.
        """
        from fnt.uwb import behavior_detection as BD

        soc_spec = [('no data', None, BD.NODATA_COLOR),
                    ('alone', 'alone', self.ALONE_COLOR),
                    ('contact', BD.SOC_CONTACT, BD.SOCIAL_COLORS[BD.SOC_CONTACT])]
        if DISPLACEMENT_ENABLED:
            soc_spec += [
                ('displaced', BD.SOC_DISPLACED, BD.SOCIAL_COLORS[BD.SOC_DISPLACED]),
                ('displacing', BD.SOC_DISPLACING, BD.SOCIAL_COLORS[BD.SOC_DISPLACING])]
        soc_spec += [
            ('chased', BD.SOC_CHASED, BD.SOCIAL_COLORS[BD.SOC_CHASED]),
            ('chasing', BD.SOC_CHASING, BD.SOCIAL_COLORS[BD.SOC_CHASING])]

        soc_disp = np.zeros(soc.shape, dtype=np.int8)
        for k, (_lab, code, _c) in enumerate(soc_spec):
            if code == 'alone':
                # Has a fix, but nothing social - the state the combined
                # raster could not show.
                soc_disp[have & (soc == BD.SOC_NONE)] = k
            elif code is not None:
                soc_disp[soc == code] = k

        soc_key = [(lab, colour) for lab, _code, colour in soc_spec]
        return soc_disp, soc_key

    @staticmethod
    def _rasterise(disp, n_tags, max_cols=3000):
        """Collapse frames to one column per output pixel, keeping the max.

        A pixel covers many frames; taking the maximum means a brief chase
        survives a bin full of contact rather than being averaged away. That
        over-represents rare states by design - the time budget is where
        proportions should be read.
        """
        n_frames = disp.shape[0]
        stride = max(1, int(np.ceil(n_frames / max_cols)))
        n_cols = int(np.ceil(n_frames / stride))
        raster = np.zeros((n_tags, n_cols), dtype=np.int8)
        for c in range(n_cols):
            sl = disp[c * stride:(c + 1) * stride]
            if sl.size:
                raster[:, c] = sl.max(axis=0)
        return raster, stride

    def save_ethogram(self, data, output_dir, db_name):
        """Social state as one raster per animal.

        Returns the number of plots generated.
        """
        self.progress.emit("Generating behaviour ethogram...")

        output_path = os.path.join(output_dir, f'{db_name}_Ethogram.png')
        if self.skip_existing and os.path.exists(output_path):
            self.progress.emit(f"Skipped (exists): {db_name}_Ethogram.png")
            return 0

        data = data.copy()
        tags = sorted(data['shortid'].unique())
        grid, have, soc, _events, _links, _ft = self._behavior_classification(data, tags)
        if grid is None:
            self.progress.emit("No data to classify, skipping ethogram")
            return 0

        soc_disp, soc_key = self._behaviour_rasters(have, soc)
        soc_raster, stride = self._rasterise(soc_disp, len(tags))

        times = pd.to_datetime(grid, utc=True).tz_convert(
            data['Timestamp'].dt.tz or 'UTC')
        extent = [date_axis(pd.Series([times[0]]))[0],
                  date_axis(pd.Series([times[-1]]))[0],
                  len(tags) - 0.5, -0.5]
        labels = [self._tag_label(t) for t in tags]

        row_h = 0.55 * len(tags)
        fig = Figure(figsize=(16, 2.0 + (row_h + 1.4)))
        panels = ((soc_raster, soc_key, 'Social state'),)
        for i, (raster, key, name) in enumerate(panels):
            ax = fig.add_subplot(1, 1, i + 1)
            ax.imshow(raster, aspect='auto', interpolation='nearest',
                      cmap=ListedColormap([c for _l, c in key]),
                      vmin=0, vmax=len(key) - 1, extent=extent)
            ax.set_yticks(range(len(tags)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(30); lbl.set_ha('right'); lbl.set_fontsize(7)
            ax.set_title(name, fontsize=11, fontweight='bold')
            ax.legend(handles=[Patch(facecolor=c, label=l) for l, c in key],
                      ncol=len(key), fontsize=7, loc='upper center',
                      bbox_to_anchor=(0.5, -0.18), framealpha=0.9,
                      borderaxespad=0.0)
            if i == 1:
                ax.set_xlabel('Time', fontsize=9)

        secs = (grid[-1] - grid[0]) / 1e9
        fig.suptitle(
            f'Behaviour Ethogram — 1 Hz classification, {secs / 3600:.1f} h, '
            f'{stride} s per pixel (each pixel shows the most notable state '
            f'it covers)',
            fontsize=12, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        self.save_figure(fig, output_path)
        plt.close(fig)
        self.progress.emit(f"Saved: {db_name}_Ethogram.png")
        return 1

    def save_social_budget(self, data, output_dir, db_name):
        """One pie per animal: how its whole tracked recording was spent.

        Faceted by animal rather than by day, and over the whole recording
        rather than per day, so the panels answer "who is social and who is
        not" at a glance - a question a per-day breakdown makes you assemble
        yourself across twenty panels.

        The counterpart to the ethogram raster, not a repeat of it: the raster
        deliberately over-represents brief events so they stay visible at a
        pixel per several minutes, so proportions have to come from the frame
        counts instead.

        Returns True if generated, False if skipped.
        """
        self.progress.emit("Generating social time budget...")

        output_path = os.path.join(output_dir, f'{db_name}_SocialBudget.png')
        if self.skip_existing and os.path.exists(output_path):
            self.progress.emit(f"Skipped (exists): {db_name}_SocialBudget.png")
            return False

        data = data.copy()
        tags = sorted(data['shortid'].unique())
        grid, have, soc, _events, _links, _ft = self._behavior_classification(data, tags)
        if grid is None:
            self.progress.emit("No data to classify, skipping social budget")
            return False

        soc_disp, soc_key = self._behaviour_rasters(have, soc)
        # Code 0 is 'no data'; a budget of TRACKED time excludes it.
        slices = list(soc_key[1:])
        dt_s = (grid[1] - grid[0]) / 1e9 if len(grid) > 1 else 1.0

        budgets = []
        for col, tag in enumerate(tags):
            counts = [int((soc_disp[:, col] == k + 1).sum())
                      for k in range(len(slices))]
            if sum(counts) > 0:
                budgets.append((tag, counts))
        if not budgets:
            self.progress.emit(
                "No tracked time to summarise, skipping social budget")
            return False

        n_rows, n_cols, figsize = self.facet_grid(
            len(budgets), max_cols=5, panel_w=3.0, panel_h=3.4, extra_h=0.9)
        fig = Figure(figsize=figsize)
        for i, (tag, counts) in enumerate(budgets):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            total = sum(counts)
            keep = [(colour, c) for (_lab, colour), c in zip(slices, counts)
                    if c > 0]
            ax.pie([c for _c, c in keep], colors=[c for c, _n in keep],
                   startangle=90, counterclock=False,
                   wedgeprops=dict(edgecolor='white', linewidth=0.6),
                   autopct=lambda p: f'{p:.0f}%' if p >= 4 else '',
                   textprops=dict(fontsize=7))
            social = total - counts[0]
            ax.set_title(
                f'{self._tag_label(tag)}\n'
                f'{total * dt_s / 3600:.1f} h tracked, '
                f'{100 * social / total:.0f}% with others',
                fontsize=9)
        fig.legend(handles=[Patch(facecolor=c, label=l) for l, c in slices],
                   ncol=len(slices), fontsize=9, loc='lower center',
                   framealpha=0.9)
        fig.suptitle('Social Time Budget \u2014 whole recording, per animal',
                     fontsize=13, fontweight='bold')
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        self.save_figure(fig, output_path)
        plt.close(fig)
        self.progress.emit(f"Saved: {db_name}_SocialBudget.png")
        return True


class FigurePopup(QDialog):
    """A resizable, zoom/pan-able window wrapping a matplotlib Figure.

    Used for the occupancy heatmaps: the matplotlib navigation toolbar gives
    scroll/box zoom and pan over the raster panels, and Save exports the view.
    """
    def __init__(self, fig, title="Figure", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(820, 640)
        # Non-modal so the user can keep working with the tool behind it.
        self.setModal(False)
        layout = QVBoxLayout()
        self.canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas, 1)
        self.setLayout(layout)


# ── Progress ticks ─────────────────────────────────────────────────────────
# A long export emits thousands of "x of y" counting lines: one per 2 h block of
# behaviour classification (227 for a 19-day trial, and it runs twice), one per
# 50 rendered frames (~16 000 for a 20-day animation). They are worth watching
# live and worthless afterwards - they buried the real messages and turned Copy
# Session Logs into a 969-line paste that was almost entirely counting.
#
# A tick now OVERWRITES the previous tick of the same kind rather than stacking
# below it, so a phase occupies one line that counts up and then stops at its
# final value. Only CONSECUTIVE ticks collapse: any real message in between ends
# the run and is appended normally, which is what keeps the per-tag export
# detail intact. The phase's own start line above the tick holds the start time
# and the tick's own timestamp is the end time, so elapsed cost stays readable.
_PROGRESS_PREFIXES = (
    ('Rendering frame', 'frames'),
    ('behaviour classified', 'behaviour'),
    ('Verifying the smoothed CSV', 'verify'),
    ('Building preview index', 'index'),
)


def progress_line_key(message):
    """Name of the progress run a log line belongs to, or None if it is not one.

    Classifies by prefix rather than by a marker in the line because the same
    text arrives two ways: directly from an in-process export, and relayed
    through a batch worker's stdout with a ``[worker]`` prefix bolted on. One
    classifier keeps both paths collapsing identically.
    """
    text = str(message).strip()
    if text.startswith('[worker]'):
        text = text[len('[worker]'):].strip()
    for prefix, key in _PROGRESS_PREFIXES:
        if text.startswith(prefix):
            return key
    return None


class UWBQuickVisualizationWindow(QWidget):
    # Preview view modes
    VIEW_XML = "XML (site map)"
    VIEW_2D = "2D Idealized"
    VIEW_3D = "3D Idealized"

    def __init__(self):
        super().__init__()
        self.db_path = None
        self.table_name = None
        self.available_tags = []
        self.data = None
        self.worker = None

        # Export control flags
        self.export_cancelled = False
        self.exporting = False
        self._last_export_failed = False

        # Batch queue: preprocess many trials sequentially (bounded memory).
        self._batch_items = []          # [{'path','table','config','conflict_choice','status'}]
        self._batch_attempts = {}       # (job index, phase) -> attempts this run
        self._batch_active = False
        self._batch_index = 0
        self._batch_stop_requested = False
        self._batch_proc = None         # isolated worker process for the running trial
        self._batch_log_path = None
        self._batch_log_pos = 0
        self._batch_plan = []           # [(job_index, 'data'|'animation')] — data pass first
        self._batch_plan_pos = 0
        # One launch per plan position: see _batch_step.
        self._batch_step_running = False
        self._batch_phase = 'data'
        self._batch_reuse_smoothed = False

        # Tag identity and sex mapping
        self.tag_identities = {}  # {tag_id: {'sex': 'M', 'identity': 'Animal1'}}
        
        # XML configuration and background image
        self.xml_config_path = None
        self.background_image_path = None
        self.background_image = None  # Loaded matplotlib image
        self.xml_scale = None  # Scale from XML in inches/pixel (first Map element)
        self.bg_width_meters = None  # Background image width in meters
        self.bg_height_meters = None  # Background image height in meters
        # Manual background transform (option (b)): the background is placed by
        # pixels * bg_scale(in/px), then shifted by (bg_offset_x, bg_offset_y)
        # metres. Defaults come from the XML but the user can nudge them live to
        # match the Wiser server frame — needed when the image is a different
        # render (resolution/decoration) than the one the XML scale was
        # calibrated against, and when the XML's own site map does not land on
        # the surveyed zones. It governs whichever background is ACTIVE: the
        # loaded floorplan if there is one, otherwise the XML site map.
        self.bg_scale = None          # effective in/px for the active background
        self.bg_offset_x = 0.0        # metres: world X of the image's left edge
        self.bg_offset_y = 0.0        # metres: world Y of the image's bottom edge
        self.arena_zones = None  # DataFrame with zone coordinates from XML
        self.anchor_positions = []  # List of dicts: {'shortid': int, 'x': float, 'y': float, 'z': float}
        self.xml_zones = []         # [{name, color, points:(N,2) m}] from the site XML
        # User-drawn regions of interest, same shape as an XML zone. These
        # belong to the trial, not to the analysis settings: they are drawn
        # against THIS arena, so they are never inherited from another one.
        self.rois = []              # [{name, color, linewidth, points:(N,2) m}]
        self.xml_map_image = None   # site map decoded from the XML
        self.xml_map_extent = None  # (x0, x1, y0, y1) in metres, AS PLACED
        self.xml_map_px = None      # (w_px, h_px) of that map, for re-placing it
        self.xml_map_scale = None   # in/px the XML authored for that map
        # Every embedded <Map>/<Image> render, each with its own pixel dims and
        # scale: Wiser writes several (e.g. 'default' + a decorated 'Dark2d').
        # Used to pick the correct default scale for a loaded external image by
        # matching its resolution.
        self.xml_maps = []          # [{name, scale, w_px, h_px}]

        # Spatial context layers drawn on exported plots/animation, applied to
        # every spatial output (trajectory plots, animation, occupancy,
        # last-known). NOT an independent setting: plot_layers_from_preview
        # copies the Preview Options 'Show ...' toggles into it at export /
        # add-to-queue time, so the export draws the arena the preview drew.
        # Persisted in the config so a queued or re-run job replays the layers
        # it was queued with.
        self.plot_layers = dict(DEFAULT_PLOT_LAYERS)
        self._exported_sitemap = None  # {filename, extent_m} once an export writes it
        self._animation_tags = None    # None = all; else a subset of shortids

        # Preview state. The timeline spans the whole recording but memory
        # holds at most MAX_CACHED_CHUNKS slices — see the streaming chunk
        # engine below.
        self.preview_timer = None
        # Playback state. The playhead is derived from the WALL CLOCK rather
        # than accumulated per tick, so a slow render drops frames instead of
        # making the trial clock run slow. See _playback_tick.
        self._playing = False
        self._playback_seeking = False   # True while playback moves the slider
        self._play_wall0 = 0.0           # perf_counter at the last anchor
        self._play_media0 = 0.0          # playhead (s from t0) at that anchor
        self._play_frames = 0
        self._play_fps_t0 = 0.0
        self.preview_arena = None
        self.preview_tags = []
        self.preview_times = None
        self.preview_hz = 1
        self.preview_x = None           # (frames, tags) smoothed positions
        self.preview_y = None
        self.preview_xf = None          # forward-filled marker positions
        self.preview_yf = None
        self.preview_raw_x = None       # same grid, pre-smoothing
        self.preview_raw_y = None
        self.preview_batt = None        # battery voltage per tag, same grid
        self.preview_step = None        # raw step distance (m) per tag, same grid
        self.preview_speed = None       # raw step speed (m/s) per tag, same grid
        self._preview_step_pool = None  # (step, speed) over the chunk, PRE-filter
        # (tags, points) an export would process, cached against the tag
        # configuration that determines it. See export_scope_points.
        self._export_scope_key = None
        self._export_scope_value = None
        self.preview_colors = None

        self.preview_cache = OrderedDict()   # chunk_index -> frame arrays (LRU)
        self.preview_inflight = {}           # chunk_index -> running loader
        self.preview_current_chunk = None
        self.preview_pending_current = None  # chunk to display once it lands
        self.preview_t0 = None               # first ping, epoch ms
        self.preview_t1 = None               # last ping, epoch ms
        self.preview_playhead_ms = 0
        self._timeline_guard = False         # suppress slider feedback loops
        self._tag_selection_guard = False    # suppress bulk checkbox churn
        self._preview_active = False         # streaming begins once data exists
        self.preview_db_path = None          # indexed copy, or the original
        self.preview_index_builder = None

        self.initUI()
        
    def log_message(self, message):
        """Add a message to the messages window.

        Consecutive progress ticks of the same kind overwrite each other
        instead of accumulating - see ``_PROGRESS_PREFIXES``. Anything that is
        not a tick, including a tick of a DIFFERENT kind, ends the run and is
        appended normally, so the last value a phase reached is always left
        standing in the log rather than being swallowed by the next phase.
        """
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        # Section dividers are printed by the block that ends and again by the
        # block that starts, so they arrive in pairs. One rule is enough.
        text = str(message).strip()
        if (text and len(text) >= 10 and len(set(text)) == 1
                and text == getattr(self, '_last_log_text', None)):
            return
        self._last_log_text = text
        key = progress_line_key(message)
        if (key is not None and key == getattr(self, '_progress_key', None)
                and self.txt_messages.document().blockCount() > 1):
            # BlockUnderCursor takes the block AND the separator before it, so
            # removing it and appending leaves exactly one line where there
            # were two. Guarded on blockCount because the selection is a no-op
            # on the first block, which would double the line instead.
            cursor = self.txt_messages.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.select(QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
        self._progress_key = key
        self.txt_messages.append(f"[{timestamp}] {message}")
        # Auto-scroll to bottom
        self.txt_messages.verticalScrollBar().setValue(
            self.txt_messages.verticalScrollBar().maximum()
        )
        # Also update legacy status label for compatibility
        self.lbl_status.setText(message)

    def copy_session_logs(self):
        """Copy the full session log to the clipboard.

        The confirmation is logged BEFORE the copy so it ends up in what
        gets copied. Otherwise a pasted log always stops one line short of
        the action that produced it, and there is nothing in the paste
        saying when it was taken.
        """
        if not self.txt_messages.toPlainText().strip():
            self.log_message("Session logs are empty — nothing to copy")
            return
        # +2: one for the trailing line, one for the confirmation appended
        # on the next statement, which is copied along with everything else.
        lines = self.txt_messages.toPlainText().count("\n") + 2
        self.log_message(f"Copied session logs to clipboard ({lines} lines)")
        QApplication.clipboard().setText(self.txt_messages.toPlainText())

    def save_message_log(self, output_dir):
        """Save the message log to a text file"""
        try:
            db_name = os.path.splitext(os.path.basename(self.db_path))[0]
            log_path = os.path.join(output_dir, f"{db_name}_messageLog.txt")
            
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(self.txt_messages.toPlainText())
            
            self.log_message(f"✓ Message log saved: {os.path.basename(log_path)}")
        except Exception as e:
            self.log_message(f"Warning: Could not save message log: {str(e)}")
    
    def initUI(self):
        self.setWindowTitle("UWB PreProcessing Tool")
        self.setGeometry(50, 50, 1500, 950)
        
        # Set dark theme style
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #cccccc;
                font-family: Arial;
            }
            /* Tooltips need their own rule: otherwise they inherit whatever
               colours the hovered widget carries (a button styled
               'color: white' rendered white text on a pale tooltip, and the
               inherited padding/min-width squashed the box). Matches the
               tooltip styling used by FNT's other tools. */
            QToolTip {
                background-color: #1e1e1e;
                color: #dddddd;
                border: 1px solid #0078d4;
                border-radius: 3px;
                padding: 4px 6px;
                font-weight: normal;
                font-size: 11px;
            }
            QLabel {
                color: #cccccc;
                background-color: transparent;
            }
            /* Trimmed from 8px/16px padding and a 100px floor, which made
               every button a 33px slab - heavy against the spinboxes and
               checkboxes it sits between, and the floor kept even one-word
               buttons full width. */
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 5px 12px;
                border-radius: 4px;
                font-weight: bold;
                min-width: 84px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #3f3f3f;
                color: #666666;
            }
            QGroupBox {
                background-color: #1e1e1e;
                border: 1px solid #3f3f3f;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: bold;
                color: #cccccc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 8px;
                background-color: #2b2b2b;
                color: #0078d4;
            }
            QCheckBox {
                color: #cccccc;
                spacing: 8px;
            }
            QComboBox {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3f3f3f;
                border-radius: 4px;
                padding: 4px 8px;
                min-width: 100px;
            }
            QScrollArea {
                border: none;
            }
            QScrollBar:vertical {
                background-color: #2b2b2b;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #0078d4;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #106ebe;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            QScrollBar:horizontal {
                background-color: #2b2b2b;
                height: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background-color: #0078d4;
                border-radius: 4px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #106ebe;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
            }
        """)

        # Settings column on the left, live preview on the right — the same
        # left-controls / right-display layout as the other FNT tools. The
        # preview is always present and simply sits empty until a database is
        # loaded; no data is read until tags exist and the user scrubs/plays.
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(5)
        # Build the preview pane (canvases) first: the left column's Preview
        # Options group wires signals whose slots touch the canvases, so the
        # canvases must exist before that group is constructed.
        self.preview_panel = self.create_preview_panel()
        settings_panel = self.create_settings_panel()
        # Pristine values for every preview-owned setting, captured before
        # any database is loaded. reset_to_defaults replays this, so a
        # setting added to _PREVIEW_CONFIG_WIDGETS is reset automatically
        # instead of having to be remembered in two places.
        self._preview_defaults = self.get_preview_config()
        # Processing settings live in the preview now; the matching export
        # rows are hidden and mirrored from it.
        self._hide_inherited_export_rows()
        self.main_splitter.addWidget(settings_panel)
        self.main_splitter.addWidget(self.preview_panel)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([520, 980])

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.main_splitter)
        self.setLayout(main_layout)

        # The preview canvas exists now, so its ROI signals can be hooked up
        # and any regions restored from a config painted onto it.
        self._wire_roi_canvas()
        self._refresh_roi_list(keep=-1)
        # Both occupancy sources exist by now, so the sub-choices can be
        # shown or greyed out against what this trial actually has.
        self.on_zone_occupancy_toggled()

        # First pass over the animation "Show" rows: the preview toggles they
        # follow exist by now, so hide the ones the preview is not offering.
        self._sync_anim_show_options()

        # Scrubbing fires a continuous stream of slider values; only fetch
        # where the user actually stops.
        self.preview_scrub_timer = QTimer(self)
        self.preview_scrub_timer.setSingleShot(True)
        self.preview_scrub_timer.timeout.connect(self._on_scrub_settled)

        # Changing the trail length redraws instantly from cache, but the chunks
        # also carry a lead-in overlap sized to the trail; reloading them keeps
        # the trail continuous across chunk boundaries. Debounced so holding the
        # spinbox arrow doesn't reload on every step.
        self.preview_trail_timer = QTimer(self)
        self.preview_trail_timer.setSingleShot(True)
        self.preview_trail_timer.timeout.connect(self.invalidate_preview_cache)

        # ← / → arrow scrubbing (1 second per step). App-level filter so the
        # keys work regardless of which widget has focus.
        QApplication.instance().installEventFilter(self)

        # Select All/None and config loading toggle every tag checkbox in a
        # burst; coalesce them into one preview rebuild.
        self.preview_tag_timer = QTimer(self)
        self.preview_tag_timer.setSingleShot(True)
        self.preview_tag_timer.timeout.connect(self._apply_tag_selection_change)

        # Anything upstream of the rendered frames invalidates cached chunks,
        # otherwise the pane would keep showing data processed under the old
        # settings. The preview has its OWN thresholds/smoothing/window controls
        # (wired where they are built), so the Export Options equivalents are
        # deliberately NOT connected here — they no longer feed _process_chunk,
        # and invalidating on them would refetch every chunk for nothing.
        # Timezone and the time-gap grouping are still shared by both paths.
        self.combo_timezone.currentTextChanged.connect(self.invalidate_preview_cache)
        self.spin_time_gap.valueChanged.connect(self.invalidate_preview_cache)
        
    def create_settings_panel(self):
        """Create the settings panel as a single scrollable column"""
        panel = QWidget()
        layout = QVBoxLayout()

        # Database selection
        db_group = QGroupBox("Database Selection")
        db_layout = QVBoxLayout()

        btn_select = QPushButton("Select SQLite Database")
        btn_select.setToolTip("Open a UWB SQLite database file (.db) for preprocessing")
        btn_select.clicked.connect(self.select_database)
        db_layout.addWidget(btn_select)

        self.lbl_db = QLabel("No database selected")
        self.lbl_db.setStyleSheet("color: #666666; font-style: italic;")
        self.lbl_db.setWordWrap(True)
        db_layout.addWidget(self.lbl_db)

        self.lbl_config_status = QLabel("")
        self.lbl_config_status.setStyleSheet("color: #ff4444; font-style: italic; font-size: 9px;")
        self.lbl_config_status.setVisible(False)
        db_layout.addWidget(self.lbl_config_status)

        # Background-work feedback, right under the Select button so the user
        # sees that work is happening instead of a frozen window. Used for the
        # on-load metadata reads (tags/days/time-bounds) AND the fast-index
        # build. The bar is indeterminate (busy) because these durations are
        # unknown; it animates via the event loop while the work runs on a
        # background thread. Shown/hidden through a ref-counted message stack
        # (_show_busy/_hide_busy) so overlapping tasks don't hide each other.
        self.lbl_busy = QLabel("")
        self.lbl_busy.setStyleSheet("color: #cc9900; font-style: italic; font-size: 9px;")
        self.lbl_busy.setVisible(False)
        db_layout.addWidget(self.lbl_busy)
        self.busy_bar = QProgressBar()
        self.busy_bar.setRange(0, 0)  # indeterminate / busy marquee
        self.busy_bar.setTextVisible(False)
        self.busy_bar.setFixedHeight(8)
        self.busy_bar.setVisible(False)
        db_layout.addWidget(self.busy_bar)
        self._busy_msgs = []  # LIFO stack of active busy messages

        table_layout = QHBoxLayout()
        table_layout.addWidget(QLabel("Table:"))
        self.combo_table = QComboBox()
        self.combo_table.setEnabled(False)
        self.combo_table.setToolTip("Select which table in the database contains the UWB tracking data")
        self.combo_table.currentTextChanged.connect(self.on_table_selected)
        table_layout.addWidget(self.combo_table)
        db_layout.addLayout(table_layout)

        self.btn_preview_table = QPushButton("Preview Table")
        self.btn_preview_table.setToolTip("Show the first rows of the selected table to verify correct data")
        self.btn_preview_table.clicked.connect(self.preview_table)
        self.btn_preview_table.setEnabled(False)
        db_layout.addWidget(self.btn_preview_table)

        db_group.setLayout(db_layout)
        layout.addWidget(db_group)

        # Timezone
        tz_group = QGroupBox("Timezone")
        tz_group_layout = QVBoxLayout()
        tz_layout = QHBoxLayout()
        tz_layout.addWidget(QLabel("Timezone:"))
        self.combo_timezone = QComboBox()
        self.combo_timezone.setToolTip(
            "Timestamps in the database are stored as UTC epoch milliseconds. "
            "This setting converts them to your local timezone for all outputs, "
            "plots, and the identity dialog time pickers."
        )
        common_timezones = [
            "US/Mountain", "US/Pacific", "US/Central", "US/Eastern",
            "UTC", "Europe/London", "Europe/Paris", "Asia/Tokyo"
        ]
        self.combo_timezone.addItems(common_timezones)
        self.combo_timezone.setCurrentText("US/Mountain")
        tz_layout.addWidget(self.combo_timezone)
        tz_group_layout.addLayout(tz_layout)
        tz_group.setLayout(tz_group_layout)
        layout.addWidget(tz_group)

        # Tag selection
        self.tag_group = QGroupBox("Tag Selection")
        self.tag_layout = QVBoxLayout()
        self.tag_checkboxes = {}

        self.lbl_no_tags = QLabel("Load a database to see available tags")
        self.lbl_no_tags.setStyleSheet("color: #666666; font-style: italic;")
        self.tag_layout.addWidget(self.lbl_no_tags)

        self.tag_group.setLayout(self.tag_layout)
        layout.addWidget(self.tag_group)

        # Preview Options group now sits directly under Tag Selection (above the
        # export controls). It is a checkable "Enable Preview and Playback"
        # group, off by default, and also hosts the background/anchor toggles.
        layout.addWidget(self._build_preview_options_group())

        # Export Options. The velocity/jump thresholds and the smoothing
        # method/window controls lead this group; the CSV / plot / animation
        # options follow below. (These threshold/smoothing controls formerly
        # lived in their own "Smoothing & Filtering Options" group.)
        export_group = QGroupBox("Export Options")
        export_layout = QVBoxLayout()
        # Rows duplicated from the preview. The preview is the single place
        # these are set; these widgets stay alive (a lot of code reads them,
        # and the saved config records what was actually used) but are hidden
        # and mirrored from the preview by _sync_export_from_preview().
        self._inherited_rows = []

        velocity_filter_layout = QHBoxLayout()
        self.chk_velocity_filter = QCheckBox("Velocity threshold (remove >")
        self.chk_velocity_filter.setChecked(True)
        self.chk_velocity_filter.setToolTip(
            "Discard fixes whose computed velocity exceeds the threshold. "
            "Catches teleportation artifacts from multipath/reflection errors.\n"
            "\n"
            "Applied BEFORE smoothing, on the raw fixes — velocity is measured "
            "between consecutive unsmoothed positions, and offending fixes are "
            "removed before the smoothing pass runs."
        )
        velocity_filter_layout.addWidget(self.chk_velocity_filter)
        self.spin_velocity_threshold = QDoubleSpinBox()
        self.spin_velocity_threshold.setRange(0.1, 10.0)
        self.spin_velocity_threshold.setValue(2.0)
        self.spin_velocity_threshold.setSuffix(" m/s)")
        self.spin_velocity_threshold.setDecimals(1)
        self.spin_velocity_threshold.setSingleStep(0.1)
        self.spin_velocity_threshold.setToolTip(
            "Points moving faster than this are removed. "
            "2 m/s is a good default for rodents in an arena."
        )
        velocity_filter_layout.addWidget(self.spin_velocity_threshold)
        velocity_filter_layout.addStretch()
        export_layout.addLayout(velocity_filter_layout)
        self._inherited_rows.append(velocity_filter_layout)

        jump_filter_layout = QHBoxLayout()
        self.chk_jump_filter = QCheckBox("Jump threshold (remove >")
        self.chk_jump_filter.setChecked(True)
        self.chk_jump_filter.setToolTip(
            "Discard fixes where the spatial jump between consecutive samples "
            "exceeds the threshold. Catches single-sample position glitches.\n"
            "\n"
            "Applied BEFORE smoothing, on the raw fixes — the jump distance is "
            "measured between consecutive unsmoothed positions, and offending "
            "fixes are removed before the smoothing pass runs."
        )
        jump_filter_layout.addWidget(self.chk_jump_filter)
        self.spin_jump_threshold = QDoubleSpinBox()
        self.spin_jump_threshold.setRange(0.1, 10.0)
        self.spin_jump_threshold.setValue(2.0)
        self.spin_jump_threshold.setSuffix(" m)")
        self.spin_jump_threshold.setDecimals(1)
        self.spin_jump_threshold.setSingleStep(0.1)
        self.spin_jump_threshold.setToolTip(
            "Consecutive points farther apart than this distance are removed. "
            "2 m is a good default for typical arena sizes."
        )
        jump_filter_layout.addWidget(self.spin_jump_threshold)
        jump_filter_layout.addStretch()
        export_layout.addLayout(jump_filter_layout)
        self._inherited_rows.append(jump_filter_layout)


        smoothing_label_layout = QHBoxLayout()
        smoothing_label_layout.addWidget(QLabel("Smoothing method:"))
        self.combo_smoothing = QComboBox()
        self.combo_smoothing.setToolTip(SMOOTHING_METHODS_TOOLTIP)
        self.combo_smoothing.addItems([
            "None (default)", "Forward-Backward Exponentially Weighted Moving Average", "Savitzky-Golay",
            "Rolling Median", "Rolling Average"
        ])
        self.combo_smoothing.setCurrentIndex(0)
        self.combo_smoothing.currentTextChanged.connect(self.on_smoothing_changed)
        smoothing_label_layout.addWidget(self.combo_smoothing)
        export_layout.addLayout(smoothing_label_layout)
        self._inherited_rows.append(smoothing_label_layout)

        self.rolling_window_layout = QHBoxLayout()
        self.lbl_rolling_window = QLabel("Smoothing Window:")
        self.rolling_window_layout.addWidget(self.lbl_rolling_window)
        self.spin_rolling_window = QSpinBox()
        # Range covers both interpretations: up to 600 s (10 min) of time-based
        # window, or up to 600 samples.
        self.spin_rolling_window.setRange(1, 600)
        self.spin_rolling_window.setValue(30)
        self.spin_rolling_window.setEnabled(False)
        self.spin_rolling_window.setToolTip(
            "Size of the smoothing window. How it is read depends on the units "
            "selector to the right and on the smoothing method:\n"
            "\n"
            "• Rolling Average / Rolling Median, units = Seconds: a real-time "
            "window (e.g. 30 = a 30-second window centred on each fix). The "
            "amount of smoothing is independent of the reporting rate.\n"
            "• Rolling Average / Rolling Median, units = Samples: a fixed count "
            "of consecutive fixes (e.g. 30 = 30 fixes). Its wall-clock span "
            "varies with the reporting rate.\n"
            "• Forward-Backward Exponentially Weighted Moving Average: the span (always in samples), which sets "
            "the decay rate (alpha = 2 / (span + 1)). To match a Wiser hardware "
            "filter value F, use span = 2F - 1.\n"
            "\n"
            "Larger values produce smoother trajectories but lose fine detail. "
            "The window is centred in every mode, so smoothing adds no lag."
        )
        self.rolling_window_layout.addWidget(self.spin_rolling_window)

        # Units selector: does the number above mean seconds or samples? Only
        # meaningful for the rolling methods (EWMA/Savitzky-Golay ignore it), so
        # on_smoothing_changed shows/hides it with the method.
        self.combo_window_units = QComboBox()
        self.combo_window_units.addItems(["Seconds", "Samples"])
        self.combo_window_units.setCurrentText("Seconds")
        self.combo_window_units.setToolTip(
            "How to interpret the Smoothing Window value (Rolling Average / "
            "Rolling Median only):\n"
            "\n"
            "• Seconds (default): a real-time window evaluated on each tag's "
            "timestamps. A value of 30 averages all fixes within a 30-second "
            "window centred on each point, so the smoothing is the same amount "
            "of real time everywhere regardless of how many fixes were "
            "recorded. Robust to the tags' irregular, sub-1 Hz reporting.\n"
            "• Samples: the window is a fixed number of consecutive fixes. A "
            "value of 30 averages 30 fixes; the real-time span it covers grows "
            "and shrinks with the reporting rate, so identical settings smooth "
            "different amounts of time on sparsely- vs densely-sampled tags."
        )
        self.combo_window_units.currentTextChanged.connect(self.invalidate_preview_cache)
        self.rolling_window_layout.addWidget(self.combo_window_units)
        self.rolling_window_layout.addStretch()
        export_layout.addLayout(self.rolling_window_layout)
        self._inherited_rows.append(self.rolling_window_layout)
        # Sync the window control's visibility to the default method now.
        # setCurrentIndex above ran before the signal was connected, so
        # on_smoothing_changed never fired — sync the Smoothing Window's
        # visibility to the initial method (hidden for the None default,
        # shown once a rolling method is chosen).
        self.on_smoothing_changed(self.combo_smoothing.currentText())

        # ---- CSV / plot / animation export options (continue the same group) --
        # Spatial-outlier threshold. Lives beside the other two thresholds so
        # the export record and the run summary carry it; the visible controls
        # are the preview's.
        self.chk_outlier_filter = QCheckBox("Hampel threshold")
        self.chk_outlier_filter.setChecked(True)
        self.chk_outlier_filter.setToolTip(OUTLIER_FILTER_TOOLTIP)
        self.spin_outlier_window = QSpinBox()
        self.spin_outlier_window.setRange(5, 501)
        self.spin_outlier_window.setSingleStep(2)
        self.spin_outlier_window.setValue(31)
        self.spin_outlier_sigma = QDoubleSpinBox()
        self.spin_outlier_sigma.setRange(1.0, 20.0)
        self.spin_outlier_sigma.setDecimals(1)
        self.spin_outlier_sigma.setSingleStep(0.5)
        self.spin_outlier_sigma.setValue(6.0)
        self.chk_adjacency_filter = QCheckBox("Excursion threshold")
        self.chk_adjacency_filter.setChecked(True)
        self.chk_adjacency_filter.setToolTip(ADJACENCY_FILTER_TOOLTIP)
        self.spin_excursion_s = QDoubleSpinBox()
        self.spin_excursion_s.setRange(0.5, 120.0)
        self.spin_excursion_s.setDecimals(1)
        self.spin_excursion_s.setValue(5.0)
        self.spin_velocity_min_dt = QDoubleSpinBox()
        self.spin_velocity_min_dt.setRange(0.0, 10.0)
        self.spin_velocity_min_dt.setDecimals(2)
        self.spin_velocity_min_dt.setSingleStep(0.05)
        self.spin_velocity_min_dt.setValue(0.50)
        for _w in (self.chk_outlier_filter, self.spin_outlier_window,
                   self.spin_outlier_sigma, self.spin_velocity_min_dt,
                   self.chk_adjacency_filter, self.spin_excursion_s):
            _w.setVisible(False)

        # No standalone "Clear Previous Exports" button: clearing is offered by
        # the export's conflict dialog, and a second red button at the top of
        # the panel was one more destructive thing to mis-click.
        #
        # The trade, so it is not rediscovered as a bug: that dialog only opens
        # when a file the CURRENT run produces already exists. Switch a product
        # off and its old file becomes an orphan the dialog never mentions. To
        # sweep those, tick the product back on so the dialog appears, then
        # choose Clear & Re-export. clear_previous_exports() is still here and
        # still tested if a route back to it is ever wanted.
        self.chk_export_raw_csv = QCheckBox("Export Raw CSV")
        self.chk_export_raw_csv.setChecked(False)
        self.chk_export_raw_csv.setToolTip(
            "Dump the raw database table to CSV with no filtering, smoothing, or unit conversion. "
            "Useful as a reference baseline."
        )
        export_layout.addWidget(self.chk_export_raw_csv)

        # The filtered + smoothed CSV is ALWAYS exported (it's the canonical
        # analysis product and the source for plots/animations/proximity), so it
        # has no toggle. A hidden always-checked box keeps the old references
        # (config, summary) working without a UI control.
        self.chk_export_smoothed_csv = QCheckBox("Export Smoothed CSV")
        self.chk_export_smoothed_csv.setChecked(True)
        self.chk_export_smoothed_csv.setVisible(False)

        # -- Social / proximity CSV products ---------------------------------
        # Three independent files, ordered raw -> most derived. All three share
        # the same contact criterion: two animals touch when their centres are
        # within 2x the social radius set in the preview. Ticking either social
        # product runs the proximity computation whether or not the bouts CSV
        # itself is written.
        self.chk_proximity_detection = QCheckBox("Export Social Overlap Bouts CSV")
        self.chk_proximity_detection.setChecked(True)
        self.chk_proximity_detection.setToolTip(SOCIAL_BOUTS_TOOLTIP)
        self.chk_proximity_detection.stateChanged.connect(self.on_proximity_detection_toggled)
        export_layout.addWidget(self.chk_proximity_detection)

        self.proximity_threshold_widget = QWidget()
        prox_layout = QHBoxLayout()
        prox_layout.setContentsMargins(30, 0, 0, 0)
        prox_layout.addWidget(QLabel("Proximity threshold:"))
        self.spin_proximity_threshold = QDoubleSpinBox()
        self.spin_proximity_threshold.setRange(0.01, 10.0)
        self.spin_proximity_threshold.setValue(0.5)
        self.spin_proximity_threshold.setSingleStep(0.05)
        self.spin_proximity_threshold.setDecimals(2)
        self.spin_proximity_threshold.setSuffix(" m")
        self.spin_proximity_threshold.setToolTip(
            "Two tags closer than this distance are considered in sociospatial proximity. "
            "0.5 m is a typical threshold for rodent social contact.")
        self.spin_proximity_threshold.setFixedWidth(100)
        prox_layout.addWidget(self.spin_proximity_threshold)
        prox_layout.addStretch()
        self.proximity_threshold_widget.setLayout(prox_layout)
        self.proximity_threshold_widget.setVisible(False)   # inherited from the preview
        export_layout.addWidget(self.proximity_threshold_widget)

        # The GBI and the windowed edge list were REMOVED. Both were derived
        # from the bouts above and neither survived contact with continuously
        # tracked data: the edge list is a pivot table of the bouts, and the
        # GBI is an asnipe input whose row-counting association indices assume
        # discrete sampling periods of equal weight, which a 6 h huddle and a
        # 2 s brush are not. Groups are recoverable from the dyadic bouts at
        # full fidelity - see SOCIAL_OVERLAP_ANALYSIS.md for the recipe.

        self.chk_export_behavior = QCheckBox("Export Behavior Events CSV")
        self.chk_export_behavior.setChecked(False)
        self.chk_export_behavior.setToolTip(
            "FILE: {db}_behavior_events.csv\n"
            "\n"
            "One row per DIRECTED behaviour bout - who did it, to whom, when, "
            "and for how long. Columns: behavior (chase), actor, target, "
            "actor_sex, target_sex, dyad_type, Day, Date, bout_start, "
            "bout_stop, duration_s.\n"
            "\n"
            "This is the only place the direction is recorded. The ethogram "
            "raster shows that an animal WAS chasing; this shows WHOM it was "
            "chasing. The social overlap bouts are undirected contact and "
            "cannot express it.\n"
            "\n"
            "HOW IT IS BUILT. The smoothed track is binned to a regular 1 Hz "
            "grid and classified in blocks with an overlap margin, exactly as "
            "the ethogram is. A bout is attributed to the block its onset falls "
            "in, so one straddling a block boundary is counted exactly once. "
            "actor is the chaser; target is the animal chased.\n"
            "\n"
            "Chase bouts carry the minimum-duration filter from the Behaviour "
            "Detection settings, and inherit every threshold from those "
            "preview controls, so tune them against the live overlay first.\n"
            "\n"
            "Contact is NOT included here - it is undirected and already "
            "exported, in full, as the proximity bouts file.")
        export_layout.addWidget(self.chk_export_behavior)

        # Occupancy is an ANALYSIS, not a drawing. It is here rather than
        # beside the preview's "Show regions of interest" checkbox because
        # that one only decides what gets drawn: turning a region invisible
        # must not silently stop measuring it, and measuring it must not force
        # it onto every figure. The two questions are answered in different
        # places on purpose.
        #
        # DRAWN REGIONS ONLY. Site-XML zones are no longer a second, parallel
        # region source; import them into the ROI tool ("Import XML" beside
        # the region list) and they are measured like anything else. One
        # region pipeline, one set of outputs.
        self.chk_export_roi_bouts = QCheckBox("Export ROI Bout Occupancy CSV")
        self.chk_export_roi_bouts.setChecked(False)
        self.chk_export_roi_bouts.setToolTip(ROI_BOUTS_TOOLTIP)
        self.chk_export_roi_bouts.stateChanged.connect(
            self.on_zone_occupancy_toggled)
        export_layout.addWidget(self.chk_export_roi_bouts)

        self.chk_export_roi_daily = QCheckBox("Export ROI Daily Summary CSV")
        self.chk_export_roi_daily.setChecked(False)
        self.chk_export_roi_daily.setToolTip(ROI_DAILY_TOOLTIP)
        self.chk_export_roi_daily.stateChanged.connect(
            self.on_zone_occupancy_toggled)
        export_layout.addWidget(self.chk_export_roi_daily)

        self.lbl_occ_sources = QLabel("")
        self.lbl_occ_sources.setWordWrap(True)
        self.lbl_occ_sources.setStyleSheet("color: #9aa0a6; font-size: 10px;")
        self.occ_source_widget = self.lbl_occ_sources
        self.occ_source_widget.setVisible(False)
        export_layout.addWidget(self.occ_source_widget)


        self.chk_save_plots = QCheckBox("Export Quick Plots")
        self.chk_save_plots.setChecked(False)  # off by default; opt in per run
        self.chk_save_plots.stateChanged.connect(self.on_save_plots_toggled)
        self.chk_save_plots.setToolTip(
            "Generate and save visualization plots to the plots/ subfolder. "
            "Built from the filtered + smoothed data. "
            "PNG is always produced; SVG is optional below."
        )
        export_layout.addWidget(self.chk_save_plots)

        self.plot_types_widget = QWidget()
        plot_types_layout = QVBoxLayout()
        plot_types_layout.setContentsMargins(30, 0, 0, 0)
        self.plot_type_checkboxes = {}
        plot_types = [
            ("daily_paths", "Daily Paths per Tag",
             "One plot per tag showing XY trajectory for each day, color-coded by date"),
            ("trajectory_overview", "Trajectory Overview",
             "All selected tags overlaid on a single plot with optional background image"),
            ("battery_levels", "Battery Levels",
             "Battery voltage over time for each tag — useful for detecting low-power dropouts"),
            ("activity_timeline", "Activity Timeline",
             "Data points per hour across the recording — reveals active vs. inactive periods"),
            ("velocity_distribution", "Velocity Distribution",
             "Histogram of movement speeds per tag to characterize locomotion patterns"),
            ("cumulative_distance", "Cumulative Distance",
             "Total distance traveled over time per tag, reset at midnight each day"),
            ("velocity_timeline", "Velocity Timeline",
             "Velocity over time per tag with an activity threshold line overlay"),
            ("data_quality", "Data Quality Metrics",
             "Summary table of data gaps, sample counts, and coverage statistics per tag"),
            ("home_range", "Home Range \u2014 KDE (50% core / 95%)",
            "Where each animal actually lived, drawn four ways so the "
            "estimators can be compared directly. One file per animal "
            "faceted by day, plus a whole-trial overview with every animal "
            "overlaid.\n"
            "\n"
            "KDE 50% (red) and KDE 95% (blue) - KERNEL DENSITY ISOPLETHS.\n"
            "A 2-D Gaussian kernel density estimate is fitted to that "
            "animal's fixes and evaluated on a 180x180 grid padded 15% "
            "beyond the data on each side so the kernel tails are not "
            "clipped. Bandwidth is Scott's rule (n^(-1/6) scaled by the "
            "data covariance) - the standard reference bandwidth, "
            "equivalent to 'href' in most home-range software. Grid cells "
            "are then sorted by density, high to low, and probability mass "
            "(density x cell area) is accumulated until it reaches 50% or "
            "95% of the total; the density at that point is the contour "
            "level drawn. So the 50% isopleth encloses the smallest area "
            "accounting for half of all space use (the CORE AREA), and the "
            "95% isopleth the area accounting for 95% (the HOME RANGE, with "
            "the outer 5% of the distribution treated as excursions). "
            "Because these are density-based they may enclose several "
            "disjoint patches - the honest answer for an animal shuttling "
            "between a nest and a feeder.\n"
            "\n"
            "MCP 100% (yellow dashed) - MINIMUM CONVEX POLYGON.\n"
            "The convex hull of every fix: the smallest convex shape "
            "containing them all. Purely geometric, with no notion of how "
            "intensively space is used. Reported because it is the oldest "
            "and most comparable metric in the literature, but note it is "
            "set by the outermost fixes alone, so a single excursion can "
            "double it, and being convex it necessarily includes ground the "
            "animal never crossed.\n"
            "\n"
            "MCP 95% (green) - PEELED MINIMUM CONVEX POLYGON.\n"
            "The arithmetic centroid of all fixes is computed, the 5% of "
            "fixes furthest from it are discarded, and the hull is taken "
            "over the rest. The centroid is computed once, from the full "
            "set, and not recomputed after peeling. This is the "
            "conventional guard against one excursion dominating the "
            "estimate, which is why MCP 95% is often far smaller than MCP "
            "100%.\n"
            "\n"
            "Areas in m^2 are printed on every panel. KDE areas are "
            "measured by counting grid cells at or above the contour level "
            "rather than by measuring the drawn line, so disjoint patches "
            "are summed correctly. MCP areas use the shoelace formula on "
            "the hull vertices.\n"
            "\n"
            "All panels share one set of axis limits, so range SIZE is "
            "directly comparable across days and across animals. Computed "
            "from the filtered + smoothed coordinates. Needs at least 10 "
            "fixes, and fails gracefully to 'n/a' for an animal that never "
            "moved (a degenerate cloud has no estimable density).\n"),
            ("mcp_range", "Home Range \u2014 MCP (95% / 100%)",
             "Home range as a MINIMUM CONVEX POLYGON, kept separate from "
             "the KDE plot so the two estimators can be compared side by "
             "side on the same animals and the same axes.\n"
             "\n"
             "FILES: {db}_MCP_{animal}.png (per animal, faceted by day) "
             "and {db}_MCP_Overview.png (whole trial, every animal).\n"
             "\n"
             "MCP 100% (yellow dashed) - the convex hull of every fix: the "
             "smallest convex shape containing them all. Purely geometric, "
             "with no notion of how intensively space is used. It is the "
             "oldest and most comparable metric in the literature, but it "
             "is set by the outermost fixes alone, so a single excursion "
             "can double it, and being convex it necessarily includes "
             "ground the animal never crossed.\n"
             "\n"
             "MCP 95% (green solid) - the peeled hull. The arithmetic "
             "centroid of all fixes is computed, the 5% of fixes furthest "
             "from it are discarded, and the hull is taken over the rest. "
             "The centroid is computed once, from the full set, and not "
             "recomputed after peeling. This is the conventional guard "
             "against one excursion dominating the estimate, which is why "
             "95% is often far smaller than 100%.\n"
             "\n"
             "Areas use the shoelace formula on the hull vertices and are "
             "printed on every panel, in m^2.\n"
             "\n"
             "WHAT TO COMPARE. Against the KDE plot: a convex hull cannot "
             "express multiple activity centres, so an animal shuttling "
             "between a nest and a feeder gets one polygon spanning both "
             "plus all the ground between, while the KDE splits into "
             "disjoint patches. Where MCP is much larger than the KDE 95%, "
             "that gap IS the ground the animal never used.\n"
             "\n"
             "Axis limits are shared with every other spatial plot, so "
             "ranges are directly comparable across days and animals.\n"),
            ("zone_occupancy", "Zone Occupancy",
            "How each animal divided its time between the arena zones "
            "defined in the site XML. Skipped with a message if no XML "
            "zones are loaded.\n"
            "\n"
            "ZONE ASSIGNMENT. Every fix is tested against each zone polygon "
            "with an even-odd point-in-polygon test, so concave zones "
            "behave as authored. Zones are tested in XML order and the "
            "FIRST match wins: where two zones overlap, a fix is credited "
            "to the earlier one rather than counted twice. Fixes inside no "
            "zone are collected as 'outside'.\n"
            "\n"
            "TIME, NOT FIX COUNTS. Each fix is credited with the real "
            "interval to the NEXT fix, so a zone where the tag happened to "
            "report more often does not appear to hold the animal longer. "
            "The last fix of a series is credited with the median interval, "
            "since its true dwell is unknown. Intervals are capped at 10x "
            "the median before summing, so a dropout is not billed to "
            "whichever zone the tag was last seen in.\n"
            "\n"
            "LEFT PANELS: one row per animal, stacked bars showing the "
            "FRACTION of each clock hour spent in each zone. Bars fill to "
            "1.0 wherever the animal was tracked at all, so the shape shows "
            "composition, not coverage - check the Activity Timeline or "
            "Data Quality plot for coverage.\n"
            "\n"
            "RIGHT PANEL: animal x zone matrix of the percentage of that "
            "animal's total tracked time spent in each zone, over the whole "
            "recording. Rows sum to 100%. Values below 1% are left "
            "unlabelled.\n"
            "\n"
            "Bar colours are taken from each zone's authored XML colour "
            "where present, so they match the arena drawing; zones without "
            "one fall back to a categorical palette. Computed from the "
            "filtered + smoothed coordinates.\n"),
            ("behavior_counts", "Behaviour Counts by Day",
             "How often each animal chased, and how often it was on the receiving end, per day.\n\nFILE: {db}_BehaviorCounts.png\n\nTwo panels: AS ACTOR (this animal did the chasing) and AS TARGET (it was the one chased). One line per animal, x = day, y = number of bouts that day. The split is the point - an animal that chases constantly and one that is constantly chased have the same combined total and opposite meanings.\n\nA bout is counted on the day it STARTED, and carries the minimum-duration filter from the detection settings.\n\nThresholds are inherited from the preview's Behaviour Detection controls. If nothing is detected at all, the plot is skipped with a message - check the speed summary under those controls, since a chase speed above nearly all observed movement can never fire.\n\nSkipped entirely if no chases are detected."),
            ("behavior_matrix", "Behaviour Matrix (who \u2192 whom)",
             "Who does it to whom: a directed actor x target matrix.\n\nFILE: {db}_BehaviorMatrix.png\n\nROWS are the actor, COLUMNS the target, so cell (i, j) is the number of times i chased j. Each cell is labelled with the event count and the total seconds those bouts lasted, because many brief chases and one long chase are different findings.\n\nThe matrix is deliberately NOT symmetric, and that is what it is for: the difference between cell (i, j) and cell (j, i) is the dominance signal for that pair. A symmetric contact matrix - which is what the proximity bouts and the GBI give you - cannot show it. The diagonal is greyed out; there are no self-directed bouts.\n\nThresholds are inherited from the preview's Behaviour Detection controls. Skipped if no chases are detected."),
            ("social_budget", "Social Time Budget (per animal)",
             "How much of its tracked time each animal spent alone versus "
             "with others, over the whole recording.\n\n"
             "FILE: {db}_SocialBudget.png - one pie per animal, faceted so "
             "the cohort can be compared side by side.\n\n"
             "Slices are alone, contact, chased and chasing - the same social "
             "states as the ethogram's lower panel, from the same 1 Hz "
             "classification, so one second of tracked time is one unit of "
             "pie. Untracked time is excluded rather than drawn as a slice: "
             "the question is how an animal spent the time it WAS tracked "
             "for, so an animal that reported for three days and one that "
             "reported for twenty stay comparable. Each panel is captioned "
             "with the hours behind it and the share spent with others.\n\n"
             "This is the counterpart to the ethogram raster, not a repeat "
             "of it. The raster deliberately over-represents brief events - "
             "a pixel spans minutes and shows the most notable state in it - "
             "so proportions cannot be read off it. They come from the frame "
             "counts here.\n\n"
             "Slices under 4% are left unlabelled to keep the panels "
             "legible; they are still drawn. An animal with no tracked fixes "
             "is left out rather than drawn as an empty circle.\n\n"
             "Thresholds are inherited from the preview's Behaviour "
             "Detection controls."),
            ("ethogram", "Behaviour Ethogram",
            "The behaviour states from the preview, classified across the "
            "ENTIRE recording and written out - the detections otherwise "
            "exist only live in the preview window.\n"
            "\n"
            "FILE: {db}_Ethogram.png - TWO stacked rasters over a shared "
            "time axis, one row per animal in each.\n"
            "\n"
            "SOCIAL: no data / alone / contact / chased / chasing.\n"
            "\n"
            "ALONE is a state in its own right — an animal that has a fix "
            "and is doing nothing social — so time spent apart is readable "
            "rather than being the absence of a colour.\n"
            "\n"
            "THRESHOLDS ARE INHERITED FROM THE PREVIEW. Social radius, chase "
            "distance/speed/angle/duration and heading lag are taken from "
            "the Behaviour Detection controls exactly as set there. Tune "
            "them in the preview until the overlay looks right, then "
            "export.\n"
            "\n"
            "HOW IT IS COMPUTED. Fixes are binned onto a regular 1 Hz grid, "
            "one column per animal, and each animal's last known position "
            "is held forward (up to the staleness limit) - the same held "
            "track the preview draws, so the plotted states match the "
            "overlay you tuned against. Speed and heading are measured "
            "between real fixes using true elapsed time, never between grid "
            "slots. Classification runs in ~2 h blocks with an overlap "
            "margin wide enough to warm the lagged terms (heading lag, "
            "staleness limit); the margin is trimmed "
            "off, so block boundaries do not appear in the output. Blocking "
            "is what keeps the pairwise (frames x animals x animals) "
            "geometry from exhausting memory on a multi-day trial.\n"
            "\n"
            "STATES. Contact (social circles overlapping, i.e. centres "
            "within 2x the social radius), chasing/chased, or alone. An "
            "animal carries at most one social state per frame; where "
            "several qualify, the higher-precedence one wins (being chased "
            "outranks merely touching).\n"
            "\n"
            "RASTER. One row per animal, time along x. Each pixel covers "
            "several seconds (printed in the title); within a pixel the "
            "stronger state wins, so brief chases stay visible rather than "
            "being averaged away. This means the raster over-represents "
            "rare social events by design - read the time budget for true "
            "proportions.\n"
            "\n"
            "TIME BUDGET. One panel per day, horizontal bars of the "
            "percentage of that day's frames in each state, per animal. "
            "Unlike the raster this is a straight frame count with no "
            "precedence weighting, so the bars are quantitatively honest. "
            "'No data' frames are excluded from the bars but still count "
            "toward the day's denominator, so bars falling short of 100% "
            "show tracking gaps.\n"),
        ]
        for key, plot_name, plot_desc in plot_types:
            cb = QCheckBox(plot_name)
            cb.setChecked(True)
            cb.setToolTip(plot_desc)
            self.plot_type_checkboxes[key] = cb
            plot_types_layout.addWidget(cb)
        self.plot_types_widget.setLayout(plot_types_layout)
        self.plot_types_widget.setVisible(False)  # hidden until Export Quick Plots is on
        export_layout.addWidget(self.plot_types_widget)

        self.svg_option_widget = QWidget()
        svg_option_layout = QHBoxLayout()
        svg_option_layout.setContentsMargins(30, 0, 0, 0)
        self.chk_save_svg = QCheckBox("Also save as SVG")
        self.chk_save_svg.setChecked(False)
        self.chk_save_svg.setToolTip(
            "Save each plot as a scalable vector graphic in addition to PNG. "
            "Useful for publication-quality figures that can be edited in Illustrator."
        )
        svg_option_layout.addWidget(self.chk_save_svg)
        svg_option_layout.addStretch()
        self.svg_option_widget.setLayout(svg_option_layout)
        self.svg_option_widget.setVisible(False)  # hidden until Export Quick Plots is on
        export_layout.addWidget(self.svg_option_widget)

        self.chk_save_animation = QCheckBox("Export Tracking Animation")
        self.chk_save_animation.setChecked(False)
        self.chk_save_animation.stateChanged.connect(self.on_save_animation_toggled)
        self.chk_save_animation.setToolTip(
            "Render an MP4 video of tag TRAJECTORIES over time - the arena "
            "view with moving markers - from the filtered + smoothed data. "
            "Trail length, colour-by and tag icon size are inherited from "
            "the preview, so the video looks like what you tuned there.\n"
            "\n"
            "Much slower than the plots: a multi-day recording at high "
            "quality can take hours. In a batch run every trial's data "
            "products are finished before any animation starts, so a long "
            "render never holds up the CSVs."
        )
        export_layout.addWidget(self.chk_save_animation)

        self.animation_options_widget = QWidget()
        animation_options_layout = QVBoxLayout()
        animation_options_layout.setContentsMargins(30, 0, 0, 0)

        trail_layout = QHBoxLayout()
        trail_layout.addWidget(QLabel("Trail length (seconds):"))
        self.spin_animation_trail = QSpinBox()
        self.spin_animation_trail.setRange(1, 1000)
        self.spin_animation_trail.setValue(500)
        self.spin_animation_trail.setToolTip(
            "How many seconds of trailing path to draw behind each tag in the animation. "
            "Higher values show more historical movement context."
        )
        trail_layout.addWidget(self.spin_animation_trail)
        animation_options_layout.addLayout(trail_layout)
        self._inherited_rows.append(trail_layout)

        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Color by:"))
        self.combo_color_by = QComboBox()
        self.combo_color_by.addItems(["None", "ID", "sex"])
        self.combo_color_by.setCurrentText("None")
        self.combo_color_by.setToolTip(
            "How trajectories are coloured. None: all the same. ID: a unique "
            "colour per tag. sex: blue = male, red = female (needs identities).")
        color_layout.addWidget(self.combo_color_by, 1)
        animation_options_layout.addLayout(color_layout)
        self._inherited_rows.append(color_layout)

        # Which animals to render (default: all configured tags).
        inctag_layout = QHBoxLayout()
        inctag_layout.addWidget(QLabel("Include tags:"))
        self.lbl_anim_tags = QLabel("All configured tags")
        self.lbl_anim_tags.setStyleSheet("color:#aaaaaa; font-style:italic;")
        inctag_layout.addWidget(self.lbl_anim_tags, 1)
        self.btn_anim_tags = QPushButton("Select…")
        # This label needs more room than the pane's shared floor allows, and
        # has been rendering as "Selec\u2026". A stylesheet min-width outranks
        # both setMinimumWidth and the size policy, so the only thing that
        # moves it is another stylesheet rule.
        self.btn_anim_tags.setStyleSheet("QPushButton { min-width: 120px; }")
        self.btn_anim_tags.setToolTip("Choose a subset of tags to include in the animation")
        self.btn_anim_tags.clicked.connect(self.select_animation_tags)
        inctag_layout.addWidget(self.btn_anim_tags)
        animation_options_layout.addLayout(inctag_layout)

        # Marker diameter (points) for the rendered video. Mirrors the preview's
        # 'Tag Icon Size' so a size dialled in during preview matches the export.
        anim_tagsize_layout = QHBoxLayout()
        anim_tagsize_layout.addWidget(QLabel("Tag Icon Size:"))
        self.spin_anim_tag_size = QSpinBox()
        self.spin_anim_tag_size.setRange(2, 40)
        self.spin_anim_tag_size.setValue(10)
        self.spin_anim_tag_size.setSuffix(" pts")
        self.spin_anim_tag_size.setToolTip(
            "Diameter (in points) of each tag's marker in the exported animation. "
            "Match this to the preview's 'Tag Icon Size' so the video looks "
            "like what you tuned in the preview window.")
        anim_tagsize_layout.addWidget(self.spin_anim_tag_size)
        anim_tagsize_layout.addStretch(1)
        animation_options_layout.addLayout(anim_tagsize_layout)
        self._inherited_rows.append(anim_tagsize_layout)

        # Battery display is inherited from the preview's "Display Tag
        # Battery Levels". The widget stays alive (the config and run summary
        # read it, and _sync_export_from_preview writes it) but is not shown.
        self.chk_show_battery_export = QCheckBox("Display Tag Battery Levels")
        self.chk_show_battery_export.setChecked(False)
        self.chk_show_battery_export.setVisible(False)

        # ── which preview layers make it into the video ──────────────────
        # The preview is where these are tuned, so the video should show what
        # the preview showed. Each row here only appears when the matching
        # preview toggle is on (see _sync_anim_show_options): the export can
        # subtract from the preview, never add to it.
        animation_options_layout.addWidget(QLabel("Show in animation:"))

        def _anim_show_box(attr, text, tip, indent=0):
            cb = QCheckBox(text)
            cb.setChecked(True)
            cb.setToolTip(tip)
            if indent:
                cb.setStyleSheet(f"margin-left:{indent}px;")
            # Remember a deliberate clear, so re-syncing from the preview does
            # not silently switch a layer the user turned off back on.
            cb.clicked.connect(
                lambda checked, c=cb: setattr(c, '_user_cleared', not checked))
            setattr(self, attr, cb)
            animation_options_layout.addWidget(cb)
            return cb

        _anim_show_box(
            'chk_anim_data_view', "Data View (live ROI counters)",
            DATA_VIEW_TOOLTIP)
        # Hidden until a region exists, and set HERE as well as in
        # on_rois_changed: a freshly opened window has not had a region change
        # yet, so without this the option would show on a trial with none.
        self.chk_anim_data_view.setVisible(bool(getattr(self, 'rois', None)))

        _anim_show_box(
            'chk_anim_show_trail', "Show trail",
            "Draw each animal's recent path behind its marker. Length is the "
            "Trail Length above, inherited from the preview.\n\n"
            "Off: markers only, which is easier to read for a full cohort in "
            "a small arena.")
        _anim_show_box(
            'chk_anim_show_tag_id', "Show tag ID",
            "Label each marker with the animal's ID, in the same form the "
            "preview uses (Display ID / HexID / ShortID).")
        _anim_show_box(
            'chk_anim_show_behavior', "Show behavior detection",
            "Draw the behaviour overlays from the preview into the video: "
            "social radius circles, the connections between interacting "
            "animals, and the per-animal state text.\n\n"
            "Classified with the thresholds set in the preview, at the "
            "preview's own rate — so the video shows the same behaviour at "
            "the same moments however fast it is rendered.\n\n"
            "This adds a classification pass over the animation's data before "
            "rendering starts.")
        _anim_show_box(
            'chk_anim_beh_social', "Social overlap",
            "Draw the dashed social-radius circle around each animal and join "
            "the pairs whose circles overlap. Radius and colour come from the "
            "preview's Social Overlap settings.", indent=18)
        _anim_show_box(
            'chk_anim_beh_chase', "Chasing",
            "Join chaser to chased while a chase is firing, and name it in "
            "both animals' state text.", indent=18)
        # Parked with the rest of the displacement feature; built so the
        # layer dict and the saved config keep their shape.
        _anim_show_box(
            'chk_anim_beh_displace', "Displacement (supplant)",
            "Join winner to loser while a displacement is firing, and name it "
            "in both animals' state text.", indent=18)
        self.chk_anim_beh_displace.setChecked(DISPLACEMENT_ENABLED)
        self.chk_anim_beh_displace.setVisible(DISPLACEMENT_ENABLED)

        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Animation Speed:"))
        self.combo_animation_speed = QComboBox()
        self.combo_animation_speed.addItems(["1x", "5x", "15x", "30x", "60x", "120x", "240x"])
        self.combo_animation_speed.setCurrentText("60x")
        self.combo_animation_speed.setToolTip(
            "How fast real time plays in the video. "
            "80x means 80 seconds of tracking data per 1 second of video."
        )
        speed_layout.addWidget(self.combo_animation_speed)
        animation_options_layout.addLayout(speed_layout)

        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("FPS:"))
        self.combo_animation_fps = QComboBox()
        self.combo_animation_fps.addItems(["1", "5", "10", "20", "30"])
        self.combo_animation_fps.setCurrentText("30")
        self.combo_animation_fps.setToolTip(
            "Video frames per second. Higher values are smoother but "
            "produce larger files and take longer to render."
        )
        fps_layout.addWidget(self.combo_animation_fps)
        animation_options_layout.addLayout(fps_layout)

        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Video Quality:"))
        self.combo_video_quality = QComboBox()
        self.combo_video_quality.addItems(["Draft (Fast)", "Standard", "High Quality"])
        self.combo_video_quality.setCurrentText("High Quality")
        self.combo_video_quality.setToolTip(
            "Render resolution: Draft (75 dpi, ~4x faster), "
            "Standard (100 dpi), High Quality (150 dpi). "
            "Draft is good for quick previews."
        )
        quality_layout.addWidget(self.combo_video_quality)
        animation_options_layout.addLayout(quality_layout)

        self.lbl_estimated_frames = QLabel("Estimated frames: -- (load data first)")
        self.lbl_estimated_frames.setStyleSheet("color: #888; font-size: 10pt; font-style: italic;")
        animation_options_layout.addWidget(self.lbl_estimated_frames)

        self.combo_animation_speed.currentTextChanged.connect(self.update_frame_estimate)
        self.combo_animation_fps.currentTextChanged.connect(self.update_frame_estimate)

        # A few seconds of the real thing, before committing to hours of it.
        # Sits directly under speed / FPS / quality and the frame estimate,
        # because those are what it is standing in for - the clip is rendered
        # with whatever they are set to.
        clip_row = QHBoxLayout()
        self.btn_preview_clip = QPushButton("Preview")
        # The pane-wide QPushButton rule carries padding: 8px 16px and
        # min-width: 100px, which is why every button here is the same 132 px
        # slab. Override both so this one sizes to its own text - it is a small
        # aside next to the settings it previews, not another primary action.
        self.btn_preview_clip.setStyleSheet(
            "QPushButton { padding: 3px 10px; min-width: 0px; "
            "font-weight: normal; }")
        # The font goes on the widget, not in the stylesheet: sizeHint is
        # computed from the widget's QFont, so a stylesheet-only font-size
        # shrinks the drawn text while the button keeps its old width.
        _clip_font = self.btn_preview_clip.font()
        _clip_font.setPointSize(8)
        _clip_font.setBold(False)
        self.btn_preview_clip.setFont(_clip_font)
        self.btn_preview_clip.setToolTip(
            "Render a short clip at the speed, frame rate and quality set "
            "above, and play it in a window.\n\n"
            "The clip starts at the preview playhead, so scrub to a moment "
            "worth looking at first. It is rendered through the same code as "
            "the full export, from the same rows the preview is drawing, so "
            "trail length, marker size, colouring, context layers and "
            "behaviour overlays all appear exactly as they will in the "
            "finished video.\n\n"
            "Length is fixed in SECONDS OF VIDEO, so it costs the same "
            "whatever the speed multiplier - at a higher speed it simply "
            "covers more of the recording.")
        self.btn_preview_clip.clicked.connect(self.preview_animation_clip)
        clip_row.addWidget(self.btn_preview_clip)
        lbl_clip = QLabel(f"{self.ANIM_CLIP_SECONDS}s from the playhead")
        lbl_clip.setStyleSheet("color: #888; font-size: 9pt; font-style: italic;")
        clip_row.addWidget(lbl_clip)
        clip_row.addStretch(1)
        animation_options_layout.addLayout(clip_row)

        # A clip is not instant - it builds the Data View's trial-wide counters
        # on the first run, classifies behaviour, then renders every frame -
        # and until this appeared the button simply sat there looking dead.
        self.lbl_clip_progress = QLabel("")
        self.lbl_clip_progress.setStyleSheet(
            "color: #6fa8dc; font-size: 9pt; margin-left: 4px;")
        self.lbl_clip_progress.setVisible(False)
        animation_options_layout.addWidget(self.lbl_clip_progress)

        # Full first, then the per-day breakdown beneath it: the whole-recording
        # video is the default output, and the daily option (with its day list)
        # reads as a subdivision of it.
        self.chk_full_animation = QCheckBox("Generate full animation (all days)")
        self.chk_full_animation.setChecked(True)   # on by default (the original behaviour)
        self.chk_full_animation.setToolTip(
            "Render one video spanning the whole recording. If daily animations "
            "are also on AND every day is selected, this is built by simply "
            "concatenating the daily videos (fast, no re-render); otherwise it "
            "is rendered from scratch.\n"
            "\n"
            "Either route gives the same picture: every clip, daily or full, "
            "is drawn in one shared arena frame taken from the whole "
            "recording, so the view never shifts between days.")
        animation_options_layout.addWidget(self.chk_full_animation)

        self.chk_daily_animations = QCheckBox("Generate daily animations (one per day)")
        self.chk_daily_animations.setChecked(False)
        self.chk_daily_animations.stateChanged.connect(self.on_daily_animations_toggled)
        self.chk_daily_animations.setToolTip(
            "Also create a separate MP4 for each calendar day (midnight to "
            "midnight), on top of - or instead of - the full video above. "
            "Tick the days you want below.\n"
            "\n"
            "These render first; the full video then follows, reusing them "
            "when every day is selected.")
        animation_options_layout.addWidget(self.chk_daily_animations)

        self.daily_animation_days_widget = QWidget()
        daily_days_layout = QVBoxLayout()
        daily_days_layout.setContentsMargins(20, 5, 0, 5)
        self.daily_animation_day_checkboxes = {}

        # Master toggle. Tristate so it can SHOW a partial selection, but a
        # click is always all-or-nothing - cycling a user through a
        # third 'partial' state they cannot act on would be noise.
        self.chk_all_days = QCheckBox("All days")
        self.chk_all_days.setTristate(True)
        self.chk_all_days.setChecked(True)
        self.chk_all_days.setStyleSheet("font-weight: bold;")
        self.chk_all_days.setToolTip(
            "Select or clear every day at once.\n"
            "\n"
            "Shows a partial mark when only some days are ticked; clicking "
            "then selects all of them. Clicking again clears all.\n"
            "\n"
            "Note that the full animation is only built by concatenating the "
            "daily videos when EVERY day is selected; with a subset it is "
            "rendered separately from scratch.")
        self.chk_all_days.clicked.connect(self.on_all_days_clicked)
        daily_days_layout.addWidget(self.chk_all_days)

        self.daily_days_layout_inner = QVBoxLayout()
        daily_days_layout.addLayout(self.daily_days_layout_inner)
        self.daily_animation_days_widget.setLayout(daily_days_layout)
        self.daily_animation_days_widget.setVisible(False)
        animation_options_layout.addWidget(self.daily_animation_days_widget)

        self.animation_options_widget.setLayout(animation_options_layout)
        self.animation_options_widget.setVisible(False)  # hidden until Save Animation is on
        export_layout.addWidget(self.animation_options_widget)

        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        # Progress bar (hidden by default)
        self.progress_widget = QWidget()
        progress_layout = QVBoxLayout()
        progress_layout.setContentsMargins(0, 5, 0, 5)
        self.lbl_export_progress = QLabel("")
        self.lbl_export_progress.setStyleSheet("color: #00aa00; font-weight: bold;")
        progress_layout.addWidget(self.lbl_export_progress)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 3px;
                text-align: center;
                background-color: #1e1e1e;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        self.progress_widget.setLayout(progress_layout)
        self.progress_widget.setVisible(False)
        layout.addWidget(self.progress_widget)

        # The standalone Export button is retired: the single export path is now
        # the Batch Queue's "Export Batch" (queue one DB or many, then export).
        # These button objects are still created — hidden — so the shared
        # export_data / stop_export code can keep toggling their enabled/visible
        # state without being special-cased.
        self.btn_export = QPushButton("Export", panel)
        self.btn_export.clicked.connect(self.export_data)
        self.btn_export.setEnabled(False)
        self.btn_export.setVisible(False)
        self.btn_stop_export = QPushButton("Stop Export", panel)
        self.btn_stop_export.clicked.connect(self.stop_export)
        self.btn_stop_export.setVisible(False)

        # --- Batch queue: preprocess several trials unattended --------------
        batch_group = QGroupBox("Export Queue")
        batch_group.setToolTip(
            "Queue several trials and preprocess them one at a time, unattended. "
            "Workflow: load a database up top, set its options, then 'Add current "
            "to Queue' — that captures THIS database with THESE settings as a job. "
            "Repeat for other databases (each can have its own settings), then Run "
            "Batch. One trial at a time keeps memory bounded, so a batch won't "
            "exhaust RAM the way several windows at once can.")
        batch_layout = QVBoxLayout()
        # Two rows so the four buttons never force the left column to scroll
        # horizontally: queue-management on top, the export/stop action below.
        batch_btns_row1 = QHBoxLayout()
        self.btn_add_batch = QPushButton("Add to Queue")
        self.btn_add_batch.setToolTip(
            "Add current to Queue: snapshot the currently loaded database and "
            "the settings/export options shown now as one queued job. Then load "
            "another database up top, adjust its settings, and add it too.")
        self.btn_add_batch.clicked.connect(self.add_current_to_batch)
        batch_btns_row1.addWidget(self.btn_add_batch)
        self.btn_clear_batch = QPushButton("Clear")
        self.btn_clear_batch.setToolTip("Remove all databases from the queue")
        self.btn_clear_batch.clicked.connect(self.clear_batch)
        self.btn_clear_batch.setEnabled(False)
        batch_btns_row1.addWidget(self.btn_clear_batch)
        batch_layout.addLayout(batch_btns_row1)

        batch_btns_row2 = QHBoxLayout()
        self.btn_run_batch = QPushButton("Export Batch")
        self.btn_run_batch.setToolTip(
            "Export every queued job sequentially, each with its own captured "
            "settings. This is the export button — queue a single database or "
            "several, then export them here.")
        self.btn_run_batch.clicked.connect(self.run_batch)
        self.btn_run_batch.setEnabled(False)
        self.btn_run_batch.setStyleSheet(
            "QPushButton { padding: 5px 10px; font-size: 11px; font-weight: bold; "
            "background-color: #2ea043; color: white; }")
        batch_btns_row2.addWidget(self.btn_run_batch)
        self.btn_stop_batch = QPushButton("Stop Batch")
        self.btn_stop_batch.setToolTip("Stop after tearing down the current trial")
        self.btn_stop_batch.clicked.connect(self.stop_batch)
        self.btn_stop_batch.setVisible(False)
        self.btn_stop_batch.setStyleSheet(
            "QPushButton { padding: 5px 10px; font-size: 11px; font-weight: bold; "
            "background-color: #d41100; color: white; }")
        batch_btns_row2.addWidget(self.btn_stop_batch)
        batch_layout.addLayout(batch_btns_row2)
        retry_row = QHBoxLayout()
        retry_row.setContentsMargins(0, 2, 0, 2)
        lbl_retry = QLabel("Retry a crashed trial:")
        lbl_retry.setStyleSheet("font-size: 10px;")
        retry_row.addWidget(lbl_retry)
        self.spin_batch_retries = QSpinBox()
        self.spin_batch_retries.setRange(0, 5)
        self.spin_batch_retries.setValue(2)
        self.spin_batch_retries.setSuffix("x")
        self.spin_batch_retries.setFixedWidth(60)
        self.spin_batch_retries.setToolTip(
            "How many times to re-run a trial whose worker process DIED - "
            "exited on a native fault such as an access violation "
            "(0xC0000005) or heap corruption (0xC0000374) rather than "
            "finishing with an error.\n"
            "\n"
            "Those faults are non-deterministic: the same trial with the same "
            "settings often succeeds on the next attempt. Retrying lets an "
            "unattended queue finish instead of losing a trial to a fault that "
            "had nothing to do with the data.\n"
            "\n"
            "A trial that fails CLEANLY - the worker ran and reported an error, "
            "such as a missing file or an unreadable table - is NOT retried. "
            "It would fail the same way every time, and retrying would just "
            "delay the rest of the queue.\n"
            "\n"
            "Each attempt restarts the trial from the beginning in a fresh "
            "process. Set to 0 to disable.")
        retry_row.addWidget(self.spin_batch_retries)
        retry_row.addStretch()
        batch_layout.addLayout(retry_row)

        self.batch_list = QListWidget()
        self.batch_list.setToolTip("Queued databases and their status")
        self.batch_list.setMaximumHeight(120)
        batch_layout.addWidget(self.batch_list)
        batch_group.setLayout(batch_layout)
        layout.addWidget(batch_group)

        # --- Batch progress ------------------------------------------------
        # The queue list says WHICH trial is running; this says how far along
        # the run is as a whole, and (during a render) how far through the
        # frames — the part that takes hours. Hidden until a batch starts.
        self.batch_progress_widget = QWidget()
        _bp = QVBoxLayout()
        _bp.setContentsMargins(0, 4, 0, 0)
        self.lbl_batch_progress = QLabel("")
        self.lbl_batch_progress.setStyleSheet(
            "color: #2ea043; font-weight: bold; font-size: 10px;")
        self.lbl_batch_progress.setWordWrap(True)
        _bp.addWidget(self.lbl_batch_progress)
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setRange(0, 100)
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setTextVisible(True)
        self.batch_progress_bar.setFormat("%p% of queue")
        self.batch_progress_bar.setFixedHeight(16)
        _bp.addWidget(self.batch_progress_bar)
        self.lbl_batch_eta = QLabel("")
        self.lbl_batch_eta.setStyleSheet("color: #888888; font-size: 9px;")
        self.lbl_batch_eta.setWordWrap(True)
        _bp.addWidget(self.lbl_batch_eta)
        self.batch_progress_widget.setLayout(_bp)
        self.batch_progress_widget.setVisible(False)
        layout.addWidget(self.batch_progress_widget)

        # Session Logs window
        messages_label = QLabel("Session Logs:")
        messages_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(messages_label)

        self.txt_messages = QTextEdit()
        self.txt_messages.setReadOnly(True)
        self.txt_messages.setMaximumHeight(150)
        self.txt_messages.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #555555;
                padding: 5px;
                font-family: Consolas, monospace;
                font-size: 9px;
            }
        """)
        layout.addWidget(self.txt_messages)

        self.btn_copy_logs = QPushButton("Copy Session Logs to Clipboard")
        self.btn_copy_logs.setToolTip("Copy the full session log above to the clipboard for pasting into a bug report or notes")
        self.btn_copy_logs.setStyleSheet("QPushButton { padding: 4px 8px; font-size: 10px; }")
        self.btn_copy_logs.clicked.connect(self.copy_session_logs)
        layout.addWidget(self.btn_copy_logs)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #666666; font-style: italic; font-size: 10px;")
        self.lbl_status.setVisible(False)
        layout.addWidget(self.lbl_status)

        layout.addStretch()
        panel.setLayout(layout)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)

        return scroll

    # ------------------------------------------------------------------ #
    # Preview pane
    # ------------------------------------------------------------------ #
    def create_preview_panel(self):
        """Right-hand display: the arena map on top, transport controls below.

        Every configuration control lives in the left column's 'Preview
        Options' group (see _build_preview_options_group); this pane is kept
        deliberately spare so the map gets the room.
        """
        panel = QWidget()
        layout = QVBoxLayout()

        # --- render surface (the map) ---
        self.preview_stack = QWidget()
        stack_layout = QVBoxLayout()
        stack_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_canvas_2d = UWBPreview2D(self)
        self.preview_canvas_2d.set_theme("light")   # light is the default view
        stack_layout.addWidget(self.preview_canvas_2d)
        self.preview_canvas_3d = None
        if PREVIEW_HAVE_GL:
            try:
                self.preview_canvas_3d = UWBPreview3D(self)
                self.preview_canvas_3d.setVisible(False)
                stack_layout.addWidget(self.preview_canvas_3d)
            except Exception as e:
                self.preview_canvas_3d = None
                print(f"3D preview unavailable: {e}")
        self.preview_stack.setLayout(stack_layout)
        layout.addWidget(self.preview_stack, 1)

        # --- transport, directly under the map ---
        self.slider_timeline = QSlider(Qt.Horizontal)
        self.slider_timeline.setRange(0, 0)
        self.slider_timeline.setToolTip(
            "Scrub the whole recording. Drag, or use the \u2190 / \u2192 arrow keys "
            "(hold to scan) to step back and forth. Nothing is read while you "
            "move \u2014 the chunk under the playhead loads once you stop.\n"
            "\n"
            "Or press Play below (Space) to run it forward at a chosen speed. "
            "Dragging this slider, or an arrow key, takes the playhead back "
            "and pauses playback.")
        self.slider_timeline.valueChanged.connect(self.on_timeline_moved)
        layout.addWidget(self.slider_timeline)

        transport = QHBoxLayout()
        transport.setContentsMargins(0, 0, 0, 0)
        self.btn_preview_play = QPushButton("\u25b6 Play")
        self.btn_preview_play.setFixedWidth(84)
        self.btn_preview_play.setToolTip(
            "Play the preview forward from the playhead. Space also "
            "toggles it.\n"
            "\n"
            "Playback is driven by the CLOCK, not by the frame count: the "
            "playhead is placed where it should be for the wall time that has "
            "passed, and frames that could not be drawn in between are simply "
            "skipped. Elapsed trial time is therefore correct at any speed - "
            "a slow machine shows a choppier picture, not a slower one. That "
            "is why the speed list is not capped.")
        self.btn_preview_play.clicked.connect(self.toggle_preview_playback)
        transport.addWidget(self.btn_preview_play)

        transport.addWidget(QLabel("Speed:"))
        self.combo_preview_speed = QComboBox()
        for _mult in (0.5, 1, 2, 4, 8, 16, 30, 60, 120, 240, 480):
            self.combo_preview_speed.addItem(
                f"{_mult:g}x", float(_mult))
        self.combo_preview_speed.setCurrentIndex(1)          # 1x
        self.combo_preview_speed.setFixedWidth(76)
        self.combo_preview_speed.setToolTip(
            "How much trial time passes per second of watching. 1x is real "
            "time; 60x covers an hour a minute.\n"
            "\n"
            "Changing this mid-playback re-anchors the clock at the current "
            "playhead, so the speed changes from where you are rather than "
            "jumping.\n"
            "\n"
            "High speeds outrun the chunk loader when the playhead leaves "
            "what is cached; the picture holds until the next chunk arrives "
            "and then catches up. The clock never drifts.")
        self.combo_preview_speed.currentIndexChanged.connect(
            self._on_preview_speed_changed)
        transport.addWidget(self.combo_preview_speed)

        self.lbl_preview_fps = QLabel("")
        self.lbl_preview_fps.setStyleSheet(
            "color: #9aa7b4; font-family: Consolas, monospace; font-size: 10px;")
        self.lbl_preview_fps.setToolTip(
            "Frames actually drawn per second, measured over the last second "
            "of playback. This is what your machine sustains at the chosen "
            "speed; below about 10 the motion reads as a slideshow even "
            "though the elapsed time is still right.")
        transport.addWidget(self.lbl_preview_fps)
        transport.addStretch(1)
        layout.addLayout(transport)

        self.lbl_preview_time = QLabel("--")
        self.lbl_preview_time.setStyleSheet("color: #cccccc; font-family: Consolas, monospace; font-size: 10px;")
        self.lbl_preview_time.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_preview_time)

        panel.setLayout(layout)
        panel.setMinimumWidth(520)
        return panel

    def _build_preview_options_group(self):
        """All preview configuration, as a group for the left settings column.

        The controls formerly crowded under the map now live here so the right
        pane can be just the map plus its transport bar.

        The group is checkable ("Enable Preview") and off by default: loading a
        database does not start streaming until the user opts in. Qt disables
        every child of an unchecked checkable group, so the preview controls
        (and the background/zone/anchor toggles hosted here) stay greyed out
        until the preview is enabled.

        The preview is a streamlined scrubber: drag the timeline to move through
        the recording as a clean 2D top-down view, with optional zone/background/
        anchor overlays and identity labels.
        """
        group = QGroupBox("Preview")
        v = QVBoxLayout()

        # The view/arena-registration widgets are retained but hidden: the
        # streamlined preview is fixed to the 2D idealized top-down view. They
        # stay instantiated (2D / Auto / zero offset) so the streaming + arena
        # engine keeps working unchanged.
        self._legacy_view_holder = QWidget()
        legacy = QVBoxLayout()
        legacy.setContentsMargins(0, 0, 0, 0)
        self.combo_view_mode = QComboBox()
        self.combo_view_mode.addItems([self.VIEW_XML, self.VIEW_2D, self.VIEW_3D])
        self.combo_view_mode.setCurrentText(self.VIEW_2D)
        self.combo_view_mode.currentTextChanged.connect(self.on_preview_backend_changed)
        legacy.addWidget(self.combo_view_mode)
        self.combo_arena = QComboBox()
        self.combo_arena.addItem("Auto (fit to data)")
        for name in BUILTIN_ARENAS:
            self.combo_arena.addItem(name)
        self.combo_arena.currentTextChanged.connect(self.refresh_preview_arena)
        legacy.addWidget(self.combo_arena)
        self.spin_arena_dx = QDoubleSpinBox(); self.spin_arena_dx.setRange(-500.0, 500.0)
        self.spin_arena_dx.valueChanged.connect(self.refresh_preview_arena)
        legacy.addWidget(self.spin_arena_dx)
        self.spin_arena_dy = QDoubleSpinBox(); self.spin_arena_dy.setRange(-500.0, 500.0)
        self.spin_arena_dy.valueChanged.connect(self.refresh_preview_arena)
        legacy.addWidget(self.spin_arena_dy)
        self._legacy_view_holder.setLayout(legacy)
        self._legacy_view_holder.setVisible(False)
        v.addWidget(self._legacy_view_holder)

        # Background image + anchor display (relocated here from the old
        # Smoothing & Filtering group; these are display concerns for the
        # preview and saved plots).
        bg_buttons_layout = QHBoxLayout()
        self.btn_load_background = QPushButton("Load Background")
        self.btn_load_background.clicked.connect(self.select_background_image)
        self.btn_load_background.setEnabled(False)
        self.btn_load_background.setToolTip(
            "Load a floorplan or arena image to overlay under trajectory plots. "
            "Requires an XML config in the database folder for spatial scaling."
        )
        self.btn_load_background.setStyleSheet("QPushButton { padding: 5px 10px; font-size: 11px; }")
        bg_buttons_layout.addWidget(self.btn_load_background)
        self.btn_remove_background = QPushButton("Remove Background")
        self.btn_remove_background.clicked.connect(self.remove_background)
        self.btn_remove_background.setEnabled(False)
        self.btn_remove_background.setToolTip("Clear the loaded background image from all visualizations")
        self.btn_remove_background.setStyleSheet("QPushButton { padding: 5px 10px; font-size: 11px; }")
        bg_buttons_layout.addWidget(self.btn_remove_background)
        v.addLayout(bg_buttons_layout)

        self.lbl_background_status = QLabel("No background image loaded")
        self.lbl_background_status.setStyleSheet("color: #666666; font-style: italic; font-size: 9px;")
        self.lbl_background_status.setWordWrap(True)
        v.addWidget(self.lbl_background_status)

        self.chk_show_background = QCheckBox("Show background image")
        # OFF by default. The XML's embedded site map is a fallback background
        # nobody asked for, and it is often a bare render that does not land on
        # the surveyed frame — drawing it uninvited put a stray rectangle under
        # every figure. Loading a floorplan yourself ticks this on.
        self.chk_show_background.setChecked(False)
        self.chk_show_background.setToolTip(
            "Draw the site map under the tracks: the loaded floorplan image if "
            "there is one, otherwise the map embedded in the site XML. Turning "
            "it off hides it without unloading it.\n\n"
            "Off by default; loading a floorplan turns it on. Use Background "
            "alignment to register the image onto the anchors and tracks.\n\n"
            "Exported plots and animations inherit this — what the preview "
            "shows is what they draw. Greyed out when there is no image.")
        self.chk_show_background.stateChanged.connect(self.on_show_background_toggled)
        v.addWidget(self.chk_show_background)

        # Manual background transform (option (b)): scale + X/Y origin offset so
        # the loaded image can be nudged live onto the anchor/track frame — the
        # way the Wiser server lets you place a floorplan. Needed when the loaded
        # image is a different render (resolution / decorated margins) than the
        # one the XML scale was calibrated for, so pixels*scale is wrong or the
        # arena content sits inset from the image edges. Defaults seed from the
        # XML; live values feed the preview AND exported plots/animation.
        self._bg_transform_box = QGroupBox("Background alignment")
        self._bg_transform_box.setToolTip(
            "Rescale and reposition the background so it lines up with the "
            "anchors and tracks. Applies to the loaded floorplan, or to the "
            "site map embedded in the XML when no floorplan is loaded. Scale is "
            "inches per pixel (larger = bigger image); offsets shift the "
            "image's bottom-left corner in metres. Defaults come from the site "
            "XML. Anchors/tracks use absolute coordinates and never move, and "
            "there is no rotation control — a rotated render can only be "
            "approximated.")
        bg_tf = QGridLayout()
        bg_tf.setContentsMargins(6, 2, 6, 2)
        bg_tf.addWidget(QLabel("Scale (in/px):"), 0, 0)
        self.spin_bg_scale = QDoubleSpinBox()
        self.spin_bg_scale.setRange(0.0001, 1000.0)
        self.spin_bg_scale.setDecimals(4)
        self.spin_bg_scale.setSingleStep(0.005)
        self.spin_bg_scale.setToolTip(
            "Inches per pixel. The image's physical size = pixels x this x "
            "0.0254 m. Increase to enlarge the image, decrease to shrink it.")
        self.spin_bg_scale.valueChanged.connect(self.on_bg_transform_changed)
        bg_tf.addWidget(self.spin_bg_scale, 0, 1)
        bg_tf.addWidget(QLabel("Offset X (m):"), 1, 0)
        self.spin_bg_offx = QDoubleSpinBox()
        self.spin_bg_offx.setRange(-1000.0, 1000.0)
        self.spin_bg_offx.setDecimals(2)
        self.spin_bg_offx.setSingleStep(0.10)
        self.spin_bg_offx.setToolTip("Shift the image right (+) / left (-), in metres.")
        self.spin_bg_offx.valueChanged.connect(self.on_bg_transform_changed)
        bg_tf.addWidget(self.spin_bg_offx, 1, 1)
        bg_tf.addWidget(QLabel("Offset Y (m):"), 2, 0)
        self.spin_bg_offy = QDoubleSpinBox()
        self.spin_bg_offy.setRange(-1000.0, 1000.0)
        self.spin_bg_offy.setDecimals(2)
        self.spin_bg_offy.setSingleStep(0.10)
        self.spin_bg_offy.setToolTip("Shift the image up (+) / down (-), in metres.")
        self.spin_bg_offy.valueChanged.connect(self.on_bg_transform_changed)
        bg_tf.addWidget(self.spin_bg_offy, 2, 1)
        self.btn_bg_reset = QPushButton("Reset to XML")
        self.btn_bg_reset.setToolTip(
            "Restore the scale to the XML value (matched to this image's "
            "resolution) and the offset to (0, 0).")
        self.btn_bg_reset.clicked.connect(self.reset_bg_transform)
        bg_tf.addWidget(self.btn_bg_reset, 3, 0, 1, 2)
        self._bg_transform_box.setLayout(bg_tf)
        self._bg_transform_box.setEnabled(False)   # enabled once an image loads
        v.addWidget(self._bg_transform_box)

        # ── Tracking XY range ────────────────────────────────────────────
        # The auto fit frames whatever the loaded preview data covers, which is
        # now post-trim. This box pins the frame instead, so every preview and
        # every export share one coordinate window regardless of which animals
        # were configured or which days were rendered.
        self._xy_range_box = QGroupBox("Tracking XY range")
        self._xy_range_box.setToolTip(
            "Fix the coordinate window used to draw tracks, instead of fitting "
            "it to whatever data is loaded.\n"
            "\n"
            "Off (default): the view is fitted to the data — after per-tag "
            "active-window trimming, so removing a tag's early fixes narrows "
            "the frame accordingly — plus the background image, XML zones and "
            "anchors, with a 5% margin.\n"
            "\n"
            "On: the four values below are used verbatim, everywhere — the live "
            "preview, exported plots and the exported animation. That makes "
            "frames comparable ACROSS trials, which an auto fit never is: two "
            "trials in the same arena otherwise get different windows purely "
            "because their animals ranged differently.\n"
            "\n"
            "Values are metres in the Wiser world frame, the same units as the "
            "background offsets above. Saved with the trial's config.")
        xy = QGridLayout()
        xy.setContentsMargins(6, 2, 6, 2)

        self.chk_xy_range = QCheckBox("Set range manually")
        self.chk_xy_range.setChecked(False)
        self.chk_xy_range.setToolTip(
            "Use the fixed range below instead of fitting to the data.")
        self.chk_xy_range.stateChanged.connect(self.on_xy_range_toggled)
        xy.addWidget(self.chk_xy_range, 0, 0, 1, 4)

        def _xy_spin(tip):
            sp = QDoubleSpinBox()
            sp.setRange(-1000.0, 1000.0)
            sp.setDecimals(2)
            sp.setSingleStep(0.25)
            sp.setSuffix(" m")
            sp.setFixedWidth(88)
            sp.setToolTip(tip)
            sp.valueChanged.connect(self.on_xy_range_changed)
            return sp

        xy.addWidget(QLabel("X:"), 1, 0)
        self.spin_xy_xmin = _xy_spin("Left edge of the drawn area, in metres.")
        xy.addWidget(self.spin_xy_xmin, 1, 1)
        xy.addWidget(QLabel("to"), 1, 2)
        self.spin_xy_xmax = _xy_spin("Right edge of the drawn area, in metres.")
        xy.addWidget(self.spin_xy_xmax, 1, 3)

        xy.addWidget(QLabel("Y:"), 2, 0)
        self.spin_xy_ymin = _xy_spin("Bottom edge of the drawn area, in metres.")
        xy.addWidget(self.spin_xy_ymin, 2, 1)
        xy.addWidget(QLabel("to"), 2, 2)
        self.spin_xy_ymax = _xy_spin("Top edge of the drawn area, in metres.")
        xy.addWidget(self.spin_xy_ymax, 2, 3)

        self.btn_xy_fit = QPushButton("Fit to loaded data")
        self.btn_xy_fit.setToolTip(
            "Fill the four values from the data currently loaded in the "
            "preview (after filtering, smoothing and active-window trimming), "
            "including the background image and zones, with the same 5% margin "
            "the auto fit uses. A starting point you can then round off to "
            "tidy numbers and reuse across trials.")
        self.btn_xy_fit.clicked.connect(self.fit_xy_range_to_data)
        xy.addWidget(self.btn_xy_fit, 3, 0, 1, 4)

        self._xy_range_widgets = [self.spin_xy_xmin, self.spin_xy_xmax,
                                  self.spin_xy_ymin, self.spin_xy_ymax,
                                  self.btn_xy_fit]
        self._xy_range_box.setLayout(xy)
        v.addWidget(self._xy_range_box)
        # Enabled state only: refreshing the arena here would run before the
        # rest of the preview pane is built.
        self._sync_xy_range_enabled()

        self.chk_preview_dark = QCheckBox("Dark mode")
        self.chk_preview_dark.setChecked(False)   # light by default
        self.chk_preview_dark.setToolTip(
            "Dark background for the preview. Off by default — the light theme "
            "(white background, tan arena floor) is better for figures and "
            "screenshots.")
        self.chk_preview_dark.stateChanged.connect(self.on_preview_theme_changed)
        v.addWidget(self.chk_preview_dark)

        self.chk_show_anchors = QCheckBox("Show anchor positions")
        self.chk_show_anchors.setChecked(True)
        self.chk_show_anchors.setEnabled(False)   # enabled once anchors are parsed
        self.chk_show_anchors.stateChanged.connect(self.on_show_background_toggled)
        self.chk_show_anchors.setToolTip(
            "Draw the UWB anchor/antenna positions as triangles. Anchors are "
            "parsed from the XML config.\n\n"
            "Exported plots and animations inherit this — what the preview "
            "shows is what they draw. Greyed out when the XML has no anchors."
        )
        v.addWidget(self.chk_show_anchors)

        self.chk_show_zones = QCheckBox("Show zone coordinates")
        self.chk_show_zones.setChecked(True)
        self.chk_show_zones.setEnabled(False)   # enabled once XML zones are parsed
        self.chk_show_zones.setToolTip(
            "Draw the XML-derived zone polygons in their authored colours.\n\n"
            "Exported plots and animations inherit this — what the preview "
            "shows is what they draw. Greyed out when the site XML defines no "
            "zones.")
        self.chk_show_zones.stateChanged.connect(self.on_show_background_toggled)
        v.addWidget(self.chk_show_zones)

        # ── Regions of interest ──────────────────────────────────────
        # Drawn here rather than in Wiser because Wiser makes you place a zone
        # against the raw fix stream, which is far too noisy to trace a real
        # boundary from. Over the preview the smoothed track and the site map
        # are both visible, and the view can be zoomed into the corner being
        # placed.
        self.chk_show_rois = QCheckBox("Show regions of interest")
        self.chk_show_rois.setChecked(True)
        self.chk_show_rois.setToolTip(
            "Draw the regions you have marked out below.\n\n"
            "These are yours, not the site XML's - they are stored in this "
            "trial's fnt_config.json and never written back to the XML.\n\n"
            "Exported plots and animations inherit this, so what the preview "
            "shows is what they draw.")
        self.chk_show_rois.stateChanged.connect(self.on_rois_changed)
        v.addWidget(self.chk_show_rois)

        self.roi_list = QListWidget()
        self.roi_list.setToolTip(
            "Every region drawn for this trial, with its corner count and "
            "area. Select one to recolour it, change its line width, rename "
            "it or delete it.\n"
            "\n"
            "Right-clicking a region in the preview itself offers the same "
            "options plus Move, Modify points and Copy.")
        self.roi_list.setMaximumHeight(96)
        self.roi_list.currentRowChanged.connect(self._on_roi_selected)
        self.roi_list.itemDoubleClicked.connect(lambda _i: self.rename_roi())
        v.addWidget(self.roi_list)

        roi_btns = QHBoxLayout()
        self.btn_roi_draw = QPushButton("Draw new")
        self.btn_roi_draw.setToolTip(
            "Start a new region, then click in the preview to place its "
            "corners.\n\n"
            "WHILE DRAWING:\n"
            "  left click - place a corner\n"
            "  right click / Backspace - take the last one back\n"
            "  scroll wheel - zoom about the cursor, so a corner can be "
            "placed exactly\n"
            "  middle drag - pan (the left button is placing corners)\n"
            "  double click or Enter - close the polygon\n"
            "  Esc - abandon it\n\n"
            "Needs at least three corners; anything less is discarded.\n"
            "\n"
            "ONCE DRAWN: right-click a region in the preview for Move, Modify "
            "points, Copy, Rename, Colour and Delete.")
        self.btn_roi_draw.clicked.connect(self.start_roi_draw)
        roi_btns.addWidget(self.btn_roi_draw)
        self.btn_roi_rename = QPushButton("Rename")
        self.btn_roi_rename.setToolTip(
            "Rename the selected region. The name is what labels it in the "
            "preview and in exported figures.")
        self.btn_roi_rename.clicked.connect(self.rename_roi)
        roi_btns.addWidget(self.btn_roi_rename)
        self.btn_roi_delete = QPushButton("Delete")
        self.btn_roi_delete.setToolTip(
            "Remove the selected region from this trial.\n"
            "\n"
            "It goes from the preview, from exported plots and animations, "
            "and from the occupancy outputs - a region that is not here is "
            "not measured. The change is written to this trial's "
            "fnt_config.json straight away, so it outlives the window; "
            "re-drawing it is the only way back.\n"
            "\n"
            "Only this trial is affected. Regions are never shared between "
            "trials, so deleting one here cannot touch another recording.")
        self.btn_roi_delete.clicked.connect(self.delete_roi)
        roi_btns.addWidget(self.btn_roi_delete)
        self.btn_roi_import_xml = QPushButton("Import XML")
        self.btn_roi_import_xml.setToolTip(ROI_IMPORT_XML_TOOLTIP)
        # No trial is loaded yet, so there is nothing to import; the layer
        # sync turns it on when an XML with zones arrives.
        self.btn_roi_import_xml.setEnabled(bool(self.xml_zones))
        self.btn_roi_import_xml.clicked.connect(self.import_xml_zones)
        roi_btns.addWidget(self.btn_roi_import_xml)
        for _b in (self.btn_roi_draw, self.btn_roi_rename, self.btn_roi_delete,
                   self.btn_roi_import_xml):
            _b.setStyleSheet("QPushButton { padding: 4px 8px; min-width: 0px; "
                             "font-weight: normal; }")
            _f = _b.font(); _f.setPointSize(8); _f.setBold(False); _b.setFont(_f)
        v.addLayout(roi_btns)

        roi_style = QHBoxLayout()
        roi_style.addWidget(QLabel("Colour:"))
        self.btn_roi_color = QPushButton("")
        self.btn_roi_color.setFixedSize(34, 18)
        self.btn_roi_color.setToolTip(
            "Colour of the selected region - its outline, its label and its "
            "wash. With none selected this sets the colour the next region "
            "will be drawn in.")
        self.btn_roi_color.clicked.connect(self.pick_roi_color)
        roi_style.addWidget(self.btn_roi_color)
        roi_style.addSpacing(8)
        roi_style.addWidget(QLabel("Line:"))
        self.spin_roi_width = QDoubleSpinBox()
        self.spin_roi_width.setRange(0.5, 10.0)
        self.spin_roi_width.setSingleStep(0.5)
        self.spin_roi_width.setDecimals(1)
        self.spin_roi_width.setValue(2.0)
        self.spin_roi_width.setFixedWidth(64)
        self.spin_roi_width.setToolTip(
            "Outline width in points for the selected region, or for the next "
            "one drawn when none is selected.")
        self.spin_roi_width.valueChanged.connect(self._on_roi_width_changed)
        roi_style.addWidget(self.spin_roi_width)
        roi_style.addStretch(1)
        v.addLayout(roi_style)

        self.lbl_roi_hint = QLabel("")
        self.lbl_roi_hint.setWordWrap(True)
        self.lbl_roi_hint.setStyleSheet("color:#e5c07b; font-size: 9px;")
        self.lbl_roi_hint.setVisible(False)
        v.addWidget(self.lbl_roi_hint)

        # Show the tag's ID label above each marker, with a choice of ID format.
        tagid_row = QHBoxLayout()
        self.chk_show_tag_id = QCheckBox("Show Tag ID")
        self.chk_show_tag_id.setChecked(False)
        self.chk_show_tag_id.setToolTip(
            "Label each tag in the preview with its ID (above the marker). The "
            "label is tinted to the marker colour under 'Color by: Sex/ID'.")
        self.chk_show_tag_id.stateChanged.connect(self.on_marker_style_changed)
        self.chk_show_tag_id.stateChanged.connect(self._sync_anim_show_options)
        tagid_row.addWidget(self.chk_show_tag_id)
        self.combo_tag_id_type = QComboBox()
        self.combo_tag_id_type.addItems(["Display ID", "HexID", "ShortID"])
        self.combo_tag_id_type.setCurrentText("Display ID")
        self.combo_tag_id_type.setToolTip(
            "Which ID to show:\n"
            "• Display ID: sex + configured identity (e.g. M9627) — the label "
            "used throughout the tool.\n"
            "• HexID: the tag's short address in hexadecimal (e.g. 2A).\n"
            "• ShortID: the decimal tag id as stored in the SQL database (e.g. 42).")
        self.combo_tag_id_type.currentTextChanged.connect(self.on_marker_style_changed)
        tagid_row.addWidget(self.combo_tag_id_type, 1)
        v.addLayout(tagid_row)

        self.chk_show_battery = QCheckBox("Display Tag Battery Levels")
        self.chk_show_battery.setChecked(False)
        self.chk_show_battery.setToolTip(
            "Show each tag's current battery voltage in small black text under "
            "its marker. Independent of 'Show Tag ID'. Off by default.")
        self.chk_show_battery.stateChanged.connect(self.on_marker_style_changed)
        v.addWidget(self.chk_show_battery)

        # Per-fix kinematics under the marker, measured on the track AS DRAWN:
        # they follow the smoothing method, and "None" makes them the unsmoothed
        # fixes. This means they are not what the velocity and jump thresholds
        # see (those run before smoothing) — the summary line under those
        # thresholds is the readout for that question.
        self.chk_show_speed = QCheckBox("Display Speed (m/s)")
        self.chk_show_speed.setChecked(False)
        self.chk_show_speed.setToolTip(
            "Show each tag's speed from its previous fix, under the marker, "
            "over the real elapsed time between them.\n\n"
            "Measured on the track as drawn, so it follows the smoothing "
            "method — set that to None to read the unsmoothed fixes. Heavy "
            "smoothing depresses these numbers substantially.\n\n"
            "Blank at the first fix after a Time gap, which has no predecessor.")
        self.chk_show_speed.stateChanged.connect(self.on_marker_style_changed)
        v.addWidget(self.chk_show_speed)

        self.chk_show_step = QCheckBox("Display Step Distance (m)")
        self.chk_show_step.setChecked(False)
        self.chk_show_step.setToolTip(
            "Show how far each tag moved from its previous fix, under the "
            "marker.\n\n"
            "Measured on the track as drawn, so it follows the smoothing "
            "method — set that to None to read the unsmoothed fixes.\n\n"
            "Blank at the first fix after a Time gap, which has no predecessor.")
        self.chk_show_step.stateChanged.connect(self.on_marker_style_changed)
        v.addWidget(self.chk_show_step)

        # Text colour for the per-tag readouts. The battery line used to be
        # hard-coded black and the speed/step line followed the THEME - both of
        # which assume the figure background is what sits behind the text. It
        # is not: a floorplan image does, and a dark arena over a light theme
        # put black text on a black background.
        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("Label colour:"))
        self.combo_label_color = QComboBox()
        self.combo_label_color.addItem("Auto", None)
        for _name, _hex in (("White", "#ffffff"), ("Black", "#000000"),
                            ("Yellow", "#ffe066"), ("Cyan", "#4dd2ff"),
                            ("Magenta", "#ff6ec7"), ("Green", "#2ea043"),
                            ("Orange", "#ff9f40"), ("Grey", "#9fb3c8")):
            self.combo_label_color.addItem(_name, _hex)
        self.combo_label_color.setCurrentIndex(0)
        self.combo_label_color.setFixedWidth(90)
        self.combo_label_color.setToolTip(
            "Colour of the per-tag readouts drawn under each marker: battery "
            "voltage, speed and step distance.\n"
            "\n"
            "AUTO follows the light/dark theme, which is right over a plain "
            "figure but not necessarily over a background image. Pick a "
            "colour that reads against YOUR floorplan.\n"
            "\n"
            "The tag ID and the behaviour state keep their own colours - "
            "those carry meaning (ID is tinted by 'Color by', state by which "
            "behaviour fired) and overriding them would throw it away. Use "
            "the outline below to keep them legible instead.\n"
            "\n"
            "Exported animations inherit this, so the video reads like the "
            "preview.")
        self.combo_label_color.currentIndexChanged.connect(
            self.on_marker_style_changed)
        label_row.addWidget(self.combo_label_color)
        label_row.addSpacing(8)
        self.chk_label_outline = QCheckBox("Outline")
        self.chk_label_outline.setChecked(True)
        self.chk_label_outline.setToolTip(
            "Draw a thin contrasting halo around every per-tag label.\n"
            "\n"
            "This is what actually makes text survive an arbitrary background "
            "image: the halo is the opposite of the text colour, so light "
            "text gets a dark edge and dark text a light one, and neither can "
            "vanish into whatever it happens to be drawn over.\n"
            "\n"
            "Applies to the tag ID and behaviour state as well as the "
            "readouts, so the colour-coded text stays readable without losing "
            "its colour coding.")
        self.chk_label_outline.stateChanged.connect(
            self.on_marker_style_changed)
        label_row.addWidget(self.chk_label_outline)
        label_row.addStretch(1)
        v.addLayout(label_row)

        self.chk_show_tracking = QCheckBox("Show Tag Tracking Data")
        self.chk_show_tracking.setChecked(True)
        self.chk_show_tracking.setToolTip(
            "Draw the tag position markers and trailing tracks in the LIVE "
            "PREVIEW. Turn off to inspect just the arena / background / anchors "
            "without any tracking data on top.")
        self.chk_show_tracking.stateChanged.connect(self.on_marker_style_changed)
        self.chk_show_tracking.stateChanged.connect(self._sync_anim_show_options)
        v.addWidget(self.chk_show_tracking)

        # Marker diameter in points. Matches the animation export's Tag Icon
        # Size so a size dialled in here reads the same in the rendered video;
        # the preview marker was previously smaller than the export.
        tagsize_row = QHBoxLayout()
        tagsize_row.addWidget(QLabel("Tag Icon Size:"))
        self.spin_tag_size = QSpinBox()
        self.spin_tag_size.setRange(2, 40)
        self.spin_tag_size.setValue(10)
        self.spin_tag_size.setSuffix(" pts")
        self.spin_tag_size.setToolTip(
            "Diameter (in points) of each tag's position marker in the preview. "
            "Set the same value in the animation export's 'Tag Icon Size' to "
            "make the video markers match what you see here.")
        self.spin_tag_size.valueChanged.connect(self.on_marker_style_changed)
        tagsize_row.addWidget(self.spin_tag_size)
        tagsize_row.addStretch(1)
        v.addLayout(tagsize_row)

        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Color by:"))
        self.combo_preview_color = QComboBox()
        self.combo_preview_color.addItems(["None", "ID", "Sex"])
        self.combo_preview_color.setCurrentText("None")
        self.combo_preview_color.setToolTip(
            "How preview tags are coloured. None: all the same. ID: a unique "
            "colour per tag. Sex: blue = male, red = female (from identities). "
            "With Sex/ID the tag's ID label is tinted to match its marker.")
        self.combo_preview_color.currentTextChanged.connect(self.on_preview_color_changed)
        color_row.addWidget(self.combo_preview_color, 1)
        v.addLayout(color_row)

        disp_row = QHBoxLayout()
        disp_row.addWidget(QLabel("Trail length (seconds):"))
        self.spin_preview_trail = QSpinBox()
        self.spin_preview_trail.setRange(0, 3600)     # up to one hour of path
        self.spin_preview_trail.setValue(60)
        self.spin_preview_trail.setSuffix(" s")
        self.spin_preview_trail.setSingleStep(30)
        self.spin_preview_trail.setToolTip(
            "Seconds of trailing track drawn behind each tag's current "
            "position. The export animation has its own 'Trail length' "
            "setting; set them alike for the video to match the preview.")
        self.spin_preview_trail.valueChanged.connect(self.on_preview_trail_changed)
        disp_row.addWidget(self.spin_preview_trail)
        disp_row.addStretch()
        v.addLayout(disp_row)

        self.chk_preview_trail_points = QCheckBox("Show each fix along the trail")
        self.chk_preview_trail_points.setStyleSheet("margin-left:18px;")
        self.chk_preview_trail_points.setToolTip(
            "Mark every surviving fix on the trail with a dot, instead of "
            "drawing the path as a bare line.\n"
            "\n"
            "This is the honest view of what the thresholds left: the line "
            "between two dots is drawn, not measured, so a long straight "
            "stretch with no dots on it is a stretch where the fixes were "
            "removed — a gap the eye otherwise reads as smooth travel.\n"
            "\n"
            "Covers the same span as Trail length above.")
        self.chk_preview_trail_points.stateChanged.connect(
            self.on_preview_trail_points_changed)
        v.addWidget(self.chk_preview_trail_points)


        # Preview thresholding — independent of the Export thresholds, so you can
        # see the track with/without each applied AND tune the cutoff live. Each
        # toggle carries its own value spinbox (defaults match Export: 2.0).
        time_gap_layout = QHBoxLayout()
        time_gap_layout.addWidget(QLabel("Time gap grouping:"))
        self.spin_time_gap = QSpinBox()
        self.spin_time_gap.setRange(5, 300)
        self.spin_time_gap.setValue(30)
        self.spin_time_gap.setSuffix(" sec")
        self.spin_time_gap.setToolTip(
            "Splits data into segments when gaps exceed this duration. "
            "Prevents the velocity/jump thresholds from comparing points across "
            "battery restarts or signal dropouts."
        )
        time_gap_layout.addWidget(self.spin_time_gap)
        time_gap_layout.addStretch()
        v.addLayout(time_gap_layout)

        # The three thresholds, laid out in the order apply_filters_to_data
        # APPLIES them: Hampel first (the only one that survives a run of
        # consecutive ghosts), then velocity, then jump. Reading the panel
        # top to bottom is reading the pipeline in order, and the live
        # figures underneath are counted the same way.
        #
        # Velocity and jump both test arrays computed before either runs, so
        # swapping those two changes which line gets the credit, not what
        # survives. Hampel genuinely must be first: it clears ghosts before
        # the step tests can condemn the innocent fix that follows one.

        pout_row = QHBoxLayout()
        self.chk_preview_outlier = QCheckBox("Hampel threshold (remove >")
        self.chk_preview_outlier.setChecked(True)
        self.chk_preview_outlier.setToolTip(OUTLIER_FILTER_TOOLTIP)
        self.chk_preview_outlier.stateChanged.connect(self.invalidate_preview_cache)
        pout_row.addWidget(self.chk_preview_outlier)
        self.spin_preview_outlier_sigma = QDoubleSpinBox()
        self.spin_preview_outlier_sigma.setRange(1.0, 20.0)
        self.spin_preview_outlier_sigma.setValue(6.0)
        self.spin_preview_outlier_sigma.setDecimals(1)
        self.spin_preview_outlier_sigma.setSingleStep(0.5)
        self.spin_preview_outlier_sigma.setSuffix(" MAD)")
        self.spin_preview_outlier_sigma.setToolTip(OUTLIER_SIGMA_TOOLTIP)
        self.spin_preview_outlier_sigma.valueChanged.connect(
            self.invalidate_preview_cache)
        pout_row.addWidget(self.spin_preview_outlier_sigma)
        pout_row.addWidget(QLabel("over"))
        self.spin_preview_outlier_window = QSpinBox()
        self.spin_preview_outlier_window.setRange(5, 501)
        self.spin_preview_outlier_window.setSingleStep(2)
        self.spin_preview_outlier_window.setValue(31)
        self.spin_preview_outlier_window.setSuffix(" points")
        self.spin_preview_outlier_window.setFixedWidth(90)
        self.spin_preview_outlier_window.setToolTip(OUTLIER_WINDOW_TOOLTIP)
        self.spin_preview_outlier_window.valueChanged.connect(
            self.invalidate_preview_cache)
        pout_row.addWidget(self.spin_preview_outlier_window)
        pout_row.addStretch()
        v.addLayout(pout_row)

        pvel_row = QHBoxLayout()
        self.chk_preview_velocity = QCheckBox("Velocity threshold (remove >")
        self.chk_preview_velocity.setChecked(True)
        self.chk_preview_velocity.setToolTip(
            "Apply a velocity threshold to the PREVIEW track. Toggle to compare "
            "with/without; adjust the value to see the cutoff's effect live.\n"
            "\n"
            "RUNS SECOND, after the outlier threshold and before the jump "
            "threshold, and only ever sees the fixes the one before it left. "
            "The live figures underneath are counted the same way, so the "
            "three percentages add up instead of each claiming the same wild "
            "fix.")
        self.chk_preview_velocity.stateChanged.connect(self.invalidate_preview_cache)
        pvel_row.addWidget(self.chk_preview_velocity)
        self.spin_preview_velocity = QDoubleSpinBox()
        self.spin_preview_velocity.setRange(0.1, 10.0)
        self.spin_preview_velocity.setValue(2.0)
        self.spin_preview_velocity.setSuffix(" m/s)")
        self.spin_preview_velocity.setDecimals(1)
        self.spin_preview_velocity.setSingleStep(0.1)
        self.spin_preview_velocity.setToolTip(
            "Remove preview fixes whose speed exceeds this.\n"
            "\n"
            "Speed is the distance from the previous fix divided by the "
            "interval between them - but NOT the raw interval: it is floored "
            "by the box below, because this system can report a millisecond "
            "apart and dividing a few centimetres of noise by a millisecond "
            "gives tens of m/s.\n"
            "\n"
            "Watch the live line underneath while you set this. If the fixes "
            "it is cutting have a small median STEP, the threshold is "
            "removing tightly spaced reports rather than fast movement, and "
            "the time base below is the control to reach for.")
        self.spin_preview_velocity.valueChanged.connect(self.invalidate_preview_cache)
        pvel_row.addWidget(self.spin_preview_velocity)
        pvel_row.addStretch()
        v.addLayout(pvel_row)

        # Minimum time base for the velocity threshold, right under it so the
        # pair reads as one setting: "faster than X, measured over at least Y".
        pvmin_row = QHBoxLayout()
        pvmin_row.setContentsMargins(22, 0, 0, 0)
        pvmin_row.addWidget(QLabel("measured over at least"))
        self.spin_preview_velocity_min_dt = QDoubleSpinBox()
        self.spin_preview_velocity_min_dt.setRange(0.0, 10.0)
        self.spin_preview_velocity_min_dt.setDecimals(2)
        self.spin_preview_velocity_min_dt.setSingleStep(0.05)
        self.spin_preview_velocity_min_dt.setValue(0.50)
        self.spin_preview_velocity_min_dt.setSuffix(" s")
        self.spin_preview_velocity_min_dt.setFixedWidth(84)
        self.spin_preview_velocity_min_dt.setToolTip(VELOCITY_MIN_DT_TOOLTIP)
        self.spin_preview_velocity_min_dt.valueChanged.connect(
            self.invalidate_preview_cache)
        pvmin_row.addWidget(self.spin_preview_velocity_min_dt)
        pvmin_row.addStretch()
        v.addLayout(pvmin_row)

        pjump_row = QHBoxLayout()
        self.chk_preview_jump = QCheckBox("Jump threshold (remove >")
        self.chk_preview_jump.setChecked(True)
        self.chk_preview_jump.setToolTip(
            "Apply a distance-jump threshold to the PREVIEW track. Toggle to "
            "compare with/without; adjust the value to see the cutoff's "
            "effect live.\n"
            "\n"
            "RUNS LAST, on whatever the outlier and velocity thresholds left. "
            "On its own it cannot contain a RUN of consecutive ghosts - it "
            "measures the distance INTO each fix, so the second ghost looks "
            "like a small step from the first and survives, while the honest "
            "fix after the run is deleted in its place. That is what the "
            "outlier threshold is for; this one is the backstop for a single "
            "wild fix.")
        self.chk_preview_jump.stateChanged.connect(self.invalidate_preview_cache)
        pjump_row.addWidget(self.chk_preview_jump)
        self.spin_preview_jump = QDoubleSpinBox()
        self.spin_preview_jump.setRange(0.1, 10.0)
        self.spin_preview_jump.setValue(2.0)
        self.spin_preview_jump.setSuffix(" m)")
        self.spin_preview_jump.setDecimals(1)
        self.spin_preview_jump.setSingleStep(0.1)
        self.spin_preview_jump.setToolTip(
            "Remove preview fixes that leap more than this distance from the "
            "previous fix.\n"
            "\n"
            "A pure distance test, with no division by time, so unlike the "
            "velocity threshold it is unaffected by how closely spaced the "
            "reports are. Set it to the furthest an animal could actually "
            "travel between two fixes.\n"
            "\n"
            "A fix after a long gap legitimately lands far away, so the "
            "Time gap setting above splits the track first and no step is "
            "measured across a dropout.")
        self.spin_preview_jump.valueChanged.connect(self.invalidate_preview_cache)
        pjump_row.addWidget(self.spin_preview_jump)
        pjump_row.addStretch()
        v.addLayout(pjump_row)

        # Last in the column because it runs last: it can only test the
        # joins that the three thresholds above create by removing points.
        # is in execution order.
        padj_row = QHBoxLayout()
        self.chk_preview_adjacency = QCheckBox("Excursion threshold (drop runs <")
        self.chk_preview_adjacency.setChecked(True)
        self.chk_preview_adjacency.setToolTip(ADJACENCY_FILTER_TOOLTIP)
        self.chk_preview_adjacency.stateChanged.connect(self.invalidate_preview_cache)
        padj_row.addWidget(self.chk_preview_adjacency)
        self.spin_preview_excursion = QDoubleSpinBox()
        self.spin_preview_excursion.setRange(0.5, 120.0)
        self.spin_preview_excursion.setDecimals(1)
        self.spin_preview_excursion.setSingleStep(0.5)
        self.spin_preview_excursion.setValue(5.0)
        self.spin_preview_excursion.setSuffix(" s)")
        self.spin_preview_excursion.setFixedWidth(84)
        self.spin_preview_excursion.setToolTip(ADJACENCY_FILTER_TOOLTIP)
        self.spin_preview_excursion.valueChanged.connect(
            self.invalidate_preview_cache)
        padj_row.addWidget(self.spin_preview_excursion)
        padj_row.addStretch()
        v.addLayout(padj_row)

        # What the thresholds above would do, as a small read-only report.
        # A QLabel wrapped this into an unreadable paragraph; a monospace box
        # keeps the columns aligned so the three can be compared at a glance.
        #
        # NoFocus matters: a focusable text box would sit in the tab order and
        # swallow the space bar, which is the preview's play/pause key.
        # Computed BEFORE thresholding, so it can count the fixes that get
        # removed — the surviving track can no longer show them.
        self.lbl_step_stats = QPlainTextEdit()
        self.lbl_step_stats.setReadOnly(True)
        self.lbl_step_stats.setFocusPolicy(Qt.NoFocus)
        self.lbl_step_stats.setFrameShape(QFrame.StyledPanel)
        self.lbl_step_stats.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.lbl_step_stats.setFixedHeight(122)
        self.lbl_step_stats.setStyleSheet(
            "QPlainTextEdit { color: #9fb3c8; background-color: #232323; "
            "border: 1px solid #3f3f3f; border-radius: 4px; "
            "font-family: Consolas, 'DejaVu Sans Mono', monospace; "
            "font-size: 9px; }")
        self.lbl_step_stats.setToolTip(
            "The distribution of step distances and speeds across the chunk in "
            "view, and how many fixes each threshold removes from it.\n\n"
            "Measured on the PRE-SMOOTHING coordinates and before any "
            "thresholding, because that is what the three thresholds are "
            "applied to — so it counts the fixes that get filtered out, which "
            "the surviving track can no longer show. It therefore does not "
            "follow the smoothing method, unlike the per-tag readouts.\n\n"
            "ONE LINE PER THRESHOLD, in the order they are applied - outlier, "
            "then velocity, then jump - each counting only the fixes the ones "
            "before it left. So the three counts add up to the total removed "
            "rather than each claiming the same wild fix, and moving one "
            "threshold changes what the ones after it have left to find. Each "
            "line also says what the fixes it cut actually looked like, which "
            "is how you tell a threshold that is catching ghosts from one "
            "that is eating real movement.\n"
            "\n"
            "THE LAST LINE is the one to judge a setting by: how many fixes "
            "survive, and the largest step into a surviving fix against what "
            "it was before. That second number is what the smoother would "
            "otherwise have had to chase.")
        v.addWidget(self.lbl_step_stats)

        # Smoothing method: how the live track is drawn — None (actual fixes) or
        # a smoothing method, so you can compare before committing to export.
        smooth_row = QHBoxLayout()
        smooth_row.addWidget(QLabel("Smoothing method:"))
        self.combo_preview_smoothing = QComboBox()
        self.combo_preview_smoothing.addItems([
            "None", "Forward-Backward Exponentially Weighted Moving Average", "Savitzky-Golay",
            "Rolling Median", "Rolling Average"])
        self.combo_preview_smoothing.setCurrentText("None")
        self.combo_preview_smoothing.setToolTip(
            "Preview only: see how each method looks before committing to one for\n"
            "export. It reads the same Smoothing Window / units set in Export\n"
            "Options, so the preview matches the exported result.\n"
            "\n" + SMOOTHING_METHODS_TOOLTIP)
        self.combo_preview_smoothing.currentTextChanged.connect(self.on_preview_smoothing_changed)
        smooth_row.addWidget(self.combo_preview_smoothing, 1)
        v.addLayout(smooth_row)

        # Smoothing Window for the preview — its own controls (independent of
        # Export Options) so you can see live how the window size and the
        # seconds/samples interpretation change the track. on_preview_smoothing_
        # changed shows/hides these per method (None/Savitzky-Golay expose
        # nothing, EWMA hides the units since its span is always in samples).
        pwin_row = QHBoxLayout()
        self.lbl_preview_window = QLabel("Smoothing Window:")
        pwin_row.addWidget(self.lbl_preview_window)
        self.spin_preview_window = QSpinBox()
        self.spin_preview_window.setRange(1, 600)
        self.spin_preview_window.setValue(30)
        self.spin_preview_window.setToolTip(
            "Size of the preview smoothing window. Read as seconds or samples "
            "per the units selector; for Forward-Backward Exponentially Weighted Moving Average it is the span "
            "(alpha = 2 / (span + 1)). Larger = smoother, less detail.")
        self.spin_preview_window.valueChanged.connect(self.invalidate_preview_cache)
        pwin_row.addWidget(self.spin_preview_window)
        self.combo_preview_window_units = QComboBox()
        self.combo_preview_window_units.addItems(["Seconds", "Samples"])
        self.combo_preview_window_units.setCurrentText("Seconds")
        self.combo_preview_window_units.setToolTip(
            "How to read the preview Smoothing Window (Rolling Average / Median "
            "only): Seconds = a real-time window, independent of the reporting "
            "rate; Samples = a fixed count of consecutive fixes.")
        self.combo_preview_window_units.currentTextChanged.connect(self.invalidate_preview_cache)
        pwin_row.addWidget(self.combo_preview_window_units)
        pwin_row.addStretch()
        v.addLayout(pwin_row)
        # Set initial visibility for the default method (None → hidden).
        self.on_preview_smoothing_changed()


        # --- Behaviour detection --------------------------------------------
        # Detection runs on exactly the track shown in the preview, so changing
        # the smoothing above immediately changes what is detected. That makes
        # the smoothing/detectability trade-off visible rather than assumed.
        self.chk_show_behavior = QCheckBox("Show Behavior Detection")
        self.chk_show_behavior.setChecked(False)
        self.chk_show_behavior.setToolTip(
            "Classify each tag in the current frame and draw the result. "
            "Detection uses the preview track as smoothed above — change the "
            "smoothing and the detections change with it.")
        self.chk_show_behavior.stateChanged.connect(self.on_show_behavior_toggled)
        self.chk_show_behavior.stateChanged.connect(self._sync_anim_show_options)
        v.addWidget(self.chk_show_behavior)

        self.behavior_options_widget = QWidget()
        bl = QVBoxLayout()
        bl.setContentsMargins(18, 0, 0, 0)
        bl.setSpacing(2)

        def _beh_spin(label, lo, hi, val, step, decimals, suffix, tip):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            lab = QLabel(label)
            lab.setStyleSheet("font-size: 9px;")
            row.addWidget(lab)
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setValue(val)
            sp.setSingleStep(step)
            sp.setDecimals(decimals)
            sp.setSuffix(suffix)
            sp.setToolTip(tip)
            sp.setFixedWidth(90)
            sp.valueChanged.connect(self.render_preview_frame)
            row.addWidget(sp)
            row.addStretch()
            return row, sp

        self._beh_groups = []

        def _beh_group(checkbox):
            """Container holding one detector's settings, shown only when it is on.

            A detector that is switched off cannot affect anything drawn, so
            its thresholds are noise — and with four detectors' worth of rows
            always expanded they push the rest of the pane out of view.
            """
            holder = QWidget()
            hl = QVBoxLayout()
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(2)
            holder.setLayout(hl)
            bl.addWidget(holder)
            checkbox.toggled.connect(holder.setVisible)
            # Initial visibility is applied once the pane is assembled:
            # bl has no owning widget yet, so the holder is still an
            # orphan and QWidget.setParent() would clear any shown state
            # set here, leaving the group stuck hidden until toggled.
            self._beh_groups.append((checkbox, holder))
            return hl

        # Social overlap
        self.chk_beh_social = QCheckBox("Social overlap")
        self.chk_beh_social.setChecked(True)
        self.chk_beh_social.setToolTip(
            "Each tag carries a circle of the radius below; an overlap is when "
            "two circles intersect, i.e. centres within twice the radius. "
            "0.25 m reproduces the existing 0.50 m proximity exactly.")
        self.chk_beh_social.stateChanged.connect(self.render_preview_frame)
        self.chk_beh_social.stateChanged.connect(self._sync_anim_show_options)
        bl.addWidget(self.chk_beh_social)
        gl = _beh_group(self.chk_beh_social)
        row, self.spin_social_radius = _beh_spin(
            "Social radius:", 0.01, 5.0, 0.25, 0.05, 2, " m",
            "Radius of each animal's social circle. Overlap (contact) occurs "
            "when two circles intersect, i.e. centre-to-centre <= 2x this.\n"
            "\n"
            "This is the single source of truth for contact across every social "
            "product: the Social Overlap Bouts CSV, the GBI and the edge list "
            "are all computed at centre-to-centre <= 2x this radius.")
        # Keep the hidden export mirror (centre-to-centre threshold) in step as
        # soon as the radius changes, so the UI, fnt_config.json and the
        # exported data can never disagree about the contact criterion.
        self.spin_social_radius.valueChanged.connect(self._sync_export_from_preview)
        lab_col = QLabel("colour:")
        lab_col.setStyleSheet("font-size: 9px;")
        row.insertWidget(2, lab_col)
        self.combo_circle_color = QComboBox()
        for name, hexv in (("White", "#ffffff"), ("Yellow", "#ffe066"),
                           ("Cyan", "#4dd2ff"), ("Magenta", "#ff6ec7"),
                           ("Black", "#000000"), ("Grey", "#9fb3c8"),
                           ("Green", "#2ea043")):
            self.combo_circle_color.addItem(name, hexv)
        self.combo_circle_color.setCurrentIndex(0)
        self.combo_circle_color.setFixedWidth(90)
        self.combo_circle_color.setToolTip(
            "Colour of the dotted social-radius circle — pick one that reads "
            "against your background image.")
        self.combo_circle_color.currentIndexChanged.connect(self.render_preview_frame)
        row.insertWidget(3, self.combo_circle_color)
        gl.addLayout(row)

        # Chase
        self.chk_beh_chase = QCheckBox("Chasing")
        self.chk_beh_chase.setChecked(True)
        self.chk_beh_chase.setToolTip(
            "Directional pursuit: the pair is close, both are moving, the "
            "chaser is heading at the target and the target is heading away.")
        self.chk_beh_chase.stateChanged.connect(self.render_preview_frame)
        self.chk_beh_chase.stateChanged.connect(self._sync_anim_show_options)
        bl.addWidget(self.chk_beh_chase)
        gl = _beh_group(self.chk_beh_chase)
        row, self.spin_chase_distance = _beh_spin(
            "Within:", 0.05, 5.0, 0.50, 0.05, 2, " m",
            "Maximum centre-to-centre separation for a chase.")
        gl.addLayout(row)
        row, self.spin_chase_speed = _beh_spin(
            "Both faster than:", 0.0, 10.0, 0.20, 0.05, 3, " m/s",
            "Both animals must exceed this speed.")
        gl.addLayout(row)
        row, self.spin_chase_angle = _beh_spin(
            "Heading within:", 5.0, 180.0, 45.0, 5.0, 0, " deg",
            "How closely the chaser must head at the target (and the target "
            "away from the chaser). Larger is more permissive.")
        gl.addLayout(row)
        row, self.spin_min_chase = _beh_spin(
            "Lasting at least:", 0.0, 60.0, 1.0, 0.5, 1, " s",
            "Minimum bout duration; shorter flickers are discarded.")
        gl.addLayout(row)
        row, self.spin_heading_lag = _beh_spin(
            "Heading over:", 0.5, 30.0, 3.0, 0.5, 1, " s",
            "The window over which each animal's DIRECTION OF TRAVEL is "
            "measured. Only chase detection uses it.\n"
            "\n"
            "WHAT IT COMPUTES\n"
            "An animal's heading at any moment is the direction of the "
            "straight line from where it was this many seconds ago to where "
            "it is now - the net direction over the window, not the "
            "frame-to-frame direction. 'Heading within' above is then "
            "applied to that: the chaser's heading must point at the "
            "target, and the target's heading must point away.\n"
            "\n"
            "Nothing else reads it. Speed is measured between consecutive "
            "real fixes and ignores this setting; contact is pure "
            "distance.\n"
            "\n"
            "WHY IT IS ADJUSTABLE\n"
            "Over a short window an animal barely moves, so the measured "
            "direction is mostly UWB position noise, the angle test fails "
            "at random, and real chases are missed. A longer window lets "
            "genuine travel dominate the jitter.\n"
            "\n"
            "Rule of thumb: the animal should cover several times the "
            "position noise within the window. Travel = the chase speed "
            "threshold x this setting - at 0.20 m/s and 3 s that is 0.60 m. "
            "If your tracks wobble by ~0.1 m between fixes, sub-second "
            "windows are useless and 3-5 s work.\n"
            "\n"
            "Measured on a straight 300 s chase at 0.35 m/s: with clean "
            "data every setting finds ~297 of 300 frames; with 0.10 m of "
            "per-fix noise, 0.5 s finds 159 and 3 s finds 198; with 0.20 m "
            "noise, 0.5 s finds 47 and 5 s finds 87.\n"
            "\n"
            "COST OF A LONG WINDOW\n"
            "No heading exists until this much time has elapsed - at the "
            "start of a recording and after every tracking gap - so no "
            "chase can be detected there. A 20 s setting blinds the first "
            "20 s of every segment. Long windows also average through "
            "turns, so a chase that curves reads as its mean direction "
            "rather than its current one.\n"
            "\n"
            "Set in real seconds and converted using the actual sampling "
            "interval, so it means the same thing at any preview rate.\n")
        gl.addLayout(row)

        # Displacement — PARKED as a future feature; see DISPLACEMENT_ENABLED
        # at the top of this module for why and for how to switch it back on.
        # The widgets are still built so the thresholds keep round-tripping
        # through fnt_config.json and nothing downstream has to special-case a
        # missing attribute; they are simply never shown, and start unchecked
        # so the detector contributes nothing anyone can see.
        self.chk_beh_displace = QCheckBox("Displacement (supplant)")
        self.chk_beh_displace.setChecked(DISPLACEMENT_ENABLED)
        self.chk_beh_displace.setToolTip(
            "One animal moves in on a settled neighbour and the neighbour is "
            "the one that leaves. Needs no sprint from either animal, so it "
            "survives heavy smoothing where chase detection does not.")
        self.chk_beh_displace.stateChanged.connect(self.render_preview_frame)
        self.chk_beh_displace.stateChanged.connect(self._sync_anim_show_options)
        self.chk_beh_displace.setVisible(DISPLACEMENT_ENABLED)
        bl.addWidget(self.chk_beh_displace)
        gl = _beh_group(self.chk_beh_displace)
        row, self.spin_displace_distance = _beh_spin(
            "Arrives within:", 0.05, 5.0, 0.30, 0.05, 2, " m",
            "How close the arriving animal must get for it to count as an arrival.")
        gl.addLayout(row)
        row, self.spin_displace_loser_speed = _beh_spin(
            "Settled below:", 0.0, 5.0, 0.10, 0.01, 3, " m/s",
            "The resident must be at or below this speed when the other arrives.")
        gl.addLayout(row)
        row, self.spin_displace_winner_speed = _beh_spin(
            "Arriving above:", 0.0, 5.0, 0.15, 0.01, 3, " m/s",
            "The arriving animal must be moving at least this fast.")
        gl.addLayout(row)
        row, self.spin_displace_leave = _beh_spin(
            "Resolved past:", 0.1, 10.0, 0.75, 0.05, 2, " m",
            "Separation that counts as the encounter being resolved.")
        gl.addLayout(row)
        row, self.spin_displace_window = _beh_spin(
            "Within:", 0.5, 60.0, 5.0, 0.5, 1, " s",
            "How long to wait after the arrival for that resolution.")
        gl.addLayout(row)

        # One plain-language line instead of raw percentiles: it says whether
        # the track being classified can actually support the thresholds set
        # above, which is the question the numbers were there to answer.
        self.lbl_speed_pcts = QLabel("")
        self.lbl_speed_pcts.setStyleSheet("color: #9fb3c8; font-size: 9px;")
        self.lbl_speed_pcts.setWordWrap(True)
        self.lbl_speed_pcts.setToolTip(
            "How fast the tags actually move on the track being classified, "
            "and whether your speed thresholds sit inside that range. A "
            "threshold above nearly all observed movement can never fire.")
        bl.addWidget(self.lbl_speed_pcts)

        self.behavior_options_widget.setLayout(bl)
        # Now that every holder has a parent, apply the initial states.
        for _cb, _holder in self._beh_groups:
            _holder.setVisible(_cb.isChecked())
        if not DISPLACEMENT_ENABLED:
            # A parked detector's thresholds are not a setting anyone can act
            # on. Unbind the toggle first, or re-checking the (hidden) box
            # from a saved config would put the group back on screen.
            try:
                self.chk_beh_displace.toggled.disconnect()
            except TypeError:
                pass
            for _cb, _holder in self._beh_groups:
                if _cb is self.chk_beh_displace:
                    _holder.setVisible(False)
        self.behavior_options_widget.setVisible(False)
        v.addWidget(self.behavior_options_widget)

        # Chunk length + binning rate are streaming internals: kept at sensible
        # defaults but hidden so the streamlined preview stays simple.
        self._preview_stream_holder = QWidget()
        stream_l = QHBoxLayout(); stream_l.setContentsMargins(0, 0, 0, 0)
        self.spin_preview_minutes = QSpinBox()
        self.spin_preview_minutes.setRange(1, 60)
        self.spin_preview_minutes.setValue(10)
        self.spin_preview_minutes.valueChanged.connect(self.invalidate_preview_cache)
        stream_l.addWidget(self.spin_preview_minutes)
        self.combo_preview_hz = QComboBox()
        self.combo_preview_hz.addItems(["1 Hz", "2 Hz", "5 Hz"])
        self.combo_preview_hz.currentTextChanged.connect(self.invalidate_preview_cache)
        stream_l.addWidget(self.combo_preview_hz)
        self._preview_stream_holder.setLayout(stream_l)
        self._preview_stream_holder.setVisible(False)
        v.addWidget(self._preview_stream_holder)

        self.lbl_preview_status = QLabel("Load a database and select tags to preview")
        self.lbl_preview_status.setStyleSheet("color: #888888; font-style: italic; font-size: 9px;")
        self.lbl_preview_status.setWordWrap(True)
        v.addWidget(self.lbl_preview_status)

        self.lbl_cache_status = QLabel("")
        self.lbl_cache_status.setStyleSheet("color: #666666; font-size: 9px;")
        v.addWidget(self.lbl_cache_status)

        # The fast index builds/adopts automatically when the preview loads
        # (see ensure_preview_index_db) — no manual button needed. _set_index_status
        # writes into lbl_cache_status when a build is running.
        self.preview_options_body = QWidget()
        self.preview_options_body.setLayout(v)
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.preview_options_body)
        group.setLayout(outer)

        # No enable toggle — the preview loads automatically when a database and
        # tags are available (see activate_preview, driven by the tag-selection
        # and load paths). The body is always visible.
        self.grp_preview = group
        return group

    def activate_preview(self):
        """Bring the always-visible preview to life once data is available.

        The pane is shown from startup (empty), mirroring the other FNT tools.
        This is called when a database, table and at least one tag exist.
        Enabling the preview now builds the indexed copy automatically (for the
        fastest possible scrubbing); reads stream from the original meanwhile
        and switch over transparently once the copy is ready.

        The preview loads automatically once a database, table and at least one
        tag are available — there is no enable toggle.
        """
        if not (self.db_path and self.table_name and self.selected_preview_tags()):
            return
        # During a batch run we only export — never scrub — so skip bringing the
        # live preview up (and, crucially, skip building the multi-GB fast-scrub
        # index copy for every trial).
        if getattr(self, '_batch_active', False):
            return
        self._preview_active = True
        # Build/adopt the fast index automatically for instant scrubbing.
        self.ensure_preview_index_db()
        self.refresh_preview_arena()
        # Size the timeline + seed the first chunk on a background thread — the
        # first-load MIN/MAX over the unindexed network DB is the slow part that
        # used to freeze the window (see _load_timeline_async / _on_timeline_loaded).
        self._load_timeline_async()

    def eventFilter(self, obj, event):
        """App-level ← / → scrubbing so the arrows work regardless of focus.

        One second per step — including while held (auto-repeat just steps again
        at the OS repeat rate); no acceleration. The timeline slider's unit is
        one second. Arrows keep their normal behaviour only while a value editor
        (spinbox / text field) is focused, or a combo's dropdown is actually
        open — a merely focused-but-closed combo (e.g. right after picking a
        smoothing option) does NOT block scrubbing.
        """
        from PyQt5.QtCore import QEvent
        from PyQt5.QtWidgets import QAbstractSpinBox, QComboBox, QLineEdit, QWidget

        # A click anywhere in the preview pane hands focus back to the timeline
        # so the arrow keys resume scrubbing. Without this, focus left in a
        # spinbox in the settings column keeps swallowing the arrows, and the
        # only way to get scrubbing back is to click the slider itself.
        #
        # EXCEPT while a region is being drawn or edited. Placing a corner is a
        # click in the preview pane, so this rule moved focus to the slider on
        # the very first corner and Enter never reached the canvas again - the
        # canvas key handler was correct all along, it simply never got the
        # key.
        if event.type() == QEvent.MouseButtonPress and not self._roi_busy():
            panel = getattr(self, 'preview_panel', None)
            if panel is not None and isinstance(obj, QWidget):
                node = obj
                while node is not None:
                    if node is panel:
                        self.slider_timeline.setFocus(Qt.MouseFocusReason)
                        break
                    node = node.parentWidget()
            # never consume the click - panning and slider drags still need it

        # Closing a polygon must not depend on where focus happens to sit, so
        # the keys are read here rather than only on the focused canvas.
        if event.type() == QEvent.KeyPress and self._roi_busy():
            canvas = self.preview_canvas_2d
            key = event.key()
            if key in (Qt.Key_Return, Qt.Key_Enter):
                if canvas._roi_mode:
                    canvas.finish_roi()     # close from the last corner to the first
                else:
                    canvas.end_roi_edit()
                return True
            if key == Qt.Key_Escape:
                if canvas._roi_mode:
                    canvas.cancel_roi()
                else:
                    canvas.end_roi_edit()
                return True
            if key in (Qt.Key_Backspace, Qt.Key_Delete) and canvas._roi_mode:
                canvas.undo_roi_point()
                return True

        # Space is play/pause, on the same terms as the arrows: not while a
        # value editor has focus, and not while a button does either - there
        # Space is that button's own activation and stealing it would make the
        # keyboard unusable.
        if (event.type() == QEvent.KeyPress
                and event.key() == Qt.Key_Space
                and getattr(self, '_preview_active', False)
                and self.slider_timeline.maximum() > 0):
            from PyQt5.QtWidgets import QAbstractButton
            fw = QApplication.focusWidget()
            if not isinstance(fw, (QAbstractSpinBox, QLineEdit, QComboBox,
                                   QAbstractButton)):
                self.toggle_preview_playback()
                return True

        if (event.type() == QEvent.KeyPress
                and event.key() in (Qt.Key_Left, Qt.Key_Right)
                and getattr(self, '_preview_active', False)
                and self.slider_timeline.maximum() > 0):
            fw = QApplication.focusWidget()
            editing = isinstance(fw, (QAbstractSpinBox, QLineEdit))
            combo_open = isinstance(fw, QComboBox) and fw.view().isVisible()
            if not editing and not combo_open:
                # Stepping by hand takes the playhead back from playback.
                self.stop_preview_playback()
                delta = -1 if event.key() == Qt.Key_Left else 1
                self.slider_timeline.setValue(int(np.clip(
                    self.slider_timeline.value() + delta,
                    self.slider_timeline.minimum(), self.slider_timeline.maximum())))
                return True
        return super().eventFilter(obj, event)

    def on_preview_enabled_toggled(self, enabled):
        """Enable/disable the live preview when the group checkbox is toggled.

        Checking it brings the preview to life if a database, table and tags are
        already loaded; otherwise it waits and the normal tag-selection path
        activates it later. Unchecking it tears the stream down and drops the
        pane back to its empty placeholder.

        The controls also collapse when disabled so the group folds down to just
        its title/checkbox instead of showing a greyed-out wall of options.
        """
        if hasattr(self, 'preview_options_body'):
            self.preview_options_body.setVisible(enabled)
        if enabled:
            if self.db_path and self.table_name and self.selected_preview_tags():
                self.activate_preview()
            else:
                self.lbl_preview_status.setText("Load a database and select tags to preview")
        else:
            self._preview_active = False
            self.stop_preview_playback()
            self.preview_canvas_2d.show_placeholder()
            if self.preview_canvas_3d is not None:
                self.preview_canvas_3d.clear()
            self.lbl_preview_status.setText("Preview disabled — tick 'Enable Preview and Playback' to view data")

    def _preview_backend(self):
        """The canvas currently on screen."""
        if (self.combo_view_mode.currentText() == self.VIEW_3D
                and self.preview_canvas_3d is not None):
            return self.preview_canvas_3d
        return self.preview_canvas_2d

    def on_preview_backend_changed(self):
        mode = self.combo_view_mode.currentText()
        if mode == self.VIEW_3D and self.preview_canvas_3d is None:
            self.log_message(
                f"3D view unavailable ({PREVIEW_GL_ERROR or 'pyqtgraph.opengl missing'}) "
                f"— staying in {self.VIEW_2D}")
            self.combo_view_mode.setCurrentText(self.VIEW_2D)
            return

        use_3d = mode == self.VIEW_3D and self.preview_canvas_3d is not None
        self.preview_canvas_2d.setVisible(not use_3d)
        if self.preview_canvas_3d is not None:
            self.preview_canvas_3d.setVisible(use_3d)

        # Arena source only applies to the idealized views; the XML view takes
        # its geometry entirely from the site config.
        is_xml = mode == self.VIEW_XML
        self.combo_arena.setEnabled(not is_xml)
        self.spin_arena_dx.setEnabled(not is_xml)
        self.spin_arena_dy.setEnabled(not is_xml)

        self.refresh_preview_arena()

    def build_xml_arena(self):
        """Arena built from the site XML: surveyed zones, map image, anchors."""
        pts = [z["points"] for z in self.xml_zones if len(z["points"])]
        if pts:
            allp = np.vstack(pts)
            x0, y0 = float(allp[:, 0].min()), float(allp[:, 1].min())
            x1, y1 = float(allp[:, 0].max()), float(allp[:, 1].max())
        elif self.xml_map_extent:
            x0, x1, y0, y1 = self.xml_map_extent
        else:
            return None

        if self.xml_map_extent:   # keep the whole map in frame
            mx0, mx1, my0, my1 = self.xml_map_extent
            x0, y0 = min(x0, mx0), min(y0, my0)
            x1, y1 = max(x1, mx1), max(y1, my1)

        pad = max(x1 - x0, y1 - y0) * 0.03
        return PreviewArena(
            width=(x1 - x0) + 2 * pad, height=(y1 - y0) + 2 * pad,
            origin_x=x0 - pad, origin_y=y0 - pad,
            anchors=list(self.anchor_positions or []),
            zones=list(self.xml_zones),
            map_image=self.xml_map_image, map_extent=self.xml_map_extent,
            label="XML (site map)")

    def xy_range(self):
        """Fixed drawing window as (x_min, x_max, y_min, y_max), or None.

        None means "fit to the data", the default. Returned in metres. An
        inverted or zero-width range is treated as unset rather than being
        silently corrected, so a half-typed value cannot collapse the view.
        """
        if not getattr(self, 'chk_xy_range', None) or not self.chk_xy_range.isChecked():
            return None
        x0, x1 = self.spin_xy_xmin.value(), self.spin_xy_xmax.value()
        y0, y1 = self.spin_xy_ymin.value(), self.spin_xy_ymax.value()
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, x1, y0, y1)

    def _sync_xy_range_enabled(self):
        """Grey the range inputs out unless the manual box is ticked."""
        manual = self.chk_xy_range.isChecked()
        for w in getattr(self, '_xy_range_widgets', []):
            w.setEnabled(manual)
        return manual

    def on_xy_range_toggled(self):
        manual = self._sync_xy_range_enabled()
        # Seed the boxes from the current auto fit the first time it is turned
        # on, so the view does not jump to an arbitrary default.
        if manual and self.spin_xy_xmax.value() <= self.spin_xy_xmin.value():
            self.fit_xy_range_to_data(announce=False)
        self.refresh_preview_arena()

    def on_xy_range_changed(self):
        if getattr(self, 'chk_xy_range', None) and self.chk_xy_range.isChecked():
            self.refresh_preview_arena()

    def _auto_xy_extent(self):
        """(x0, x1, y0, y1) the auto fit would use, or None if nothing to fit.

        Same union the arena builds from: the loaded (trimmed, filtered) track
        data plus the XML zones, map extent and background image.
        """
        xs, ys = [], []
        if self.preview_x is not None and len(self.preview_x):
            xs.append(np.asarray(self.preview_x, float).ravel())
            ys.append(np.asarray(self.preview_y, float).ravel())
        for z in (self.xml_zones or []):
            pts = z.get('points')
            if pts is not None and len(pts):
                pts = np.asarray(pts, float)
                xs.append(pts[:, 0]); ys.append(pts[:, 1])
        if self.xml_map_extent:
            mx0, mx1, my0, my1 = self.xml_map_extent
            xs.append(np.array([mx0, mx1])); ys.append(np.array([my0, my1]))
        bg = self._bg_placed_extent() if self.background_image is not None else None
        if bg:
            xs.append(np.array([bg[0], bg[1]])); ys.append(np.array([bg[2], bg[3]]))
        if not xs:
            return None
        xa = np.concatenate(xs); ya = np.concatenate(ys)
        xa = xa[np.isfinite(xa)]; ya = ya[np.isfinite(ya)]
        if not len(xa) or not len(ya):
            return None
        x0, x1 = float(xa.min()), float(xa.max())
        y0, y1 = float(ya.min()), float(ya.max())
        margin = 0.05 * max(x1 - x0, y1 - y0, 1e-6)
        return (x0 - margin, x1 + margin, y0 - margin, y1 + margin)

    def fit_xy_range_to_data(self, announce=True):
        """Fill the manual range from what the preview currently covers."""
        ext = self._auto_xy_extent()
        if ext is None:
            if announce:
                QMessageBox.information(
                    self, "No Data",
                    "Load a database and let the preview populate first — the "
                    "fit is taken from the data currently on screen.")
            return
        x0, x1, y0, y1 = ext
        for sp, val in ((self.spin_xy_xmin, x0), (self.spin_xy_xmax, x1),
                        (self.spin_xy_ymin, y0), (self.spin_xy_ymax, y1)):
            sp.blockSignals(True); sp.setValue(val); sp.blockSignals(False)
        if announce:
            self.log_message(
                f"Tracking XY range fitted to loaded data: "
                f"x {x0:.2f}-{x1:.2f} m, y {y0:.2f}-{y1:.2f} m")
        self.refresh_preview_arena()

    def refresh_preview_arena(self):
        """Rebuild arena geometry for the selected view mode and offsets."""
        if not hasattr(self, "combo_arena") or not hasattr(self, "chk_show_anchors"):
            return
        self._sync_layer_toggles()
        mode = self.combo_view_mode.currentText()
        dx = self.spin_arena_dx.value()
        dy = self.spin_arena_dy.value()
        arena = None

        fixed = self.xy_range()
        if fixed is not None:
            # A pinned window wins over every fitted/builtin arena: the
            # point is that the frame does not move.
            fx0, fx1, fy0, fy1 = fixed
            arena = PreviewArena(width=fx1 - fx0, height=fy1 - fy0,
                                 origin_x=fx0, origin_y=fy0,
                                 label="Fixed XY range")
            arena.anchors = list(self.anchor_positions or [])

        if mode == self.VIEW_XML:
            arena = self.build_xml_arena()
            if arena is None:
                self.log_message(
                    "No zones or map in the site XML — falling back to 2D Idealized")
                self.combo_view_mode.setCurrentText(self.VIEW_2D)
                return

        if arena is None:
            name = self.combo_arena.currentText()
            if name in BUILTIN_ARENAS:
                arena = BUILTIN_ARENAS[name](origin_x=dx, origin_y=dy)
                arena.anchors = list(self.anchor_positions or [])
            else:
                # Frame to the union of everything on screen so the site is
                # never cropped: data, plus the XML zones and map extent when
                # present. (fit_arena_to_data already folds in the anchors.)
                xs_parts, ys_parts = [], []
                if self.preview_x is not None and len(self.preview_x):
                    xs_parts.append(self.preview_x.ravel())
                    ys_parts.append(self.preview_y.ravel())
                for z in (self.xml_zones or []):
                    pts = z.get('points')
                    if pts is not None and len(pts):
                        pts = np.asarray(pts, float)
                        xs_parts.append(pts[:, 0]); ys_parts.append(pts[:, 1])
                if self.xml_map_extent:
                    mx0, mx1, my0, my1 = self.xml_map_extent
                    xs_parts.append(np.array([mx0, mx1])); ys_parts.append(np.array([my0, my1]))
                xs = np.concatenate(xs_parts) if xs_parts else None
                ys = np.concatenate(ys_parts) if ys_parts else None
                arena = fit_arena_to_data(xs, ys, self.anchor_positions)
                arena.origin_x += dx
                arena.origin_y += dy

        if arena is None:
            return
        if not self.chk_show_anchors.isChecked():
            arena.anchors = []

        # Zone overlay for the streamlined 2D preview: the data-fit arena carries
        # no zones, so inject the XML zones when the user wants them shown. (In
        # the retained XML view the arena already has them; re-setting is a
        # harmless no-op.)
        if getattr(self, 'chk_show_zones', None) is not None and self.chk_show_zones.isChecked() and self.xml_zones:
            arena.zones = list(self.xml_zones)
        elif getattr(self, 'chk_show_zones', None) is not None and not self.chk_show_zones.isChecked():
            arena.zones = []

        self.preview_arena = arena
        self.preview_canvas_2d.set_arena(arena)
        self.preview_canvas_2d.show_anchors = self.chk_show_anchors.isChecked()
        # Background-image mapping (UWB world coords, metres):
        #   • data location_x/y are inches -> metres via *0.0254
        #   • image footprint = pixels * xml_scale(in/px) * 0.0254
        #   • placed at extent [0, W, 0, H] so the image's bottom-left corner
        #     is world (0, 0) — the Wiser reference frame
        #   • drawn origin="upper" (see UWBPreview2D.update_frame): the image is
        #     shown UPRIGHT, north at top, exactly as the file looks
        #   • the canvas expands the view to include the whole image so it is
        #     never clipped by the data-fit arena bounds
        # ONE source for the background, the same one the exports use
        # (_context_bg_source): the loaded floorplan if there is one, else the
        # XML's embedded site map. Routed through the canvas's background_image
        # rather than the arena's own map_image so that (a) the site map shows
        # in every view mode, not only "XML (site map)", and (b) the show/hide
        # toggle governs it — an export must never draw a map the preview
        # silently withheld.
        arena.map_image = None
        arena.map_extent = None
        bg_image, bg_extent = self._context_bg_source()
        show_bg = bg_image is not None and self.chk_show_background.isChecked()
        self.preview_canvas_2d.background_image = bg_image if show_bg else None
        self.preview_canvas_2d.bg_extent = list(bg_extent) if show_bg else None
        if self.preview_canvas_3d is not None:
            self.preview_canvas_3d.set_arena(arena)
        self.render_preview_frame()

    def auto_register_arena(self):
        """Centre the selected enclosure on the loaded data's bounding box."""
        if self.preview_x is None or not len(self.preview_x):
            QMessageBox.information(self, "No Preview Data",
                                    "Load a preview window first so there is data to register against.")
            return
        name = self.combo_arena.currentText()
        if name not in BUILTIN_ARENAS:
            QMessageBox.information(self, "Auto-Fit",
                                    "Auto-Fit applies to a named enclosure. "
                                    "'Auto (fit to data)' already tracks the data.")
            return

        xs = self.preview_x[np.isfinite(self.preview_x)]
        ys = self.preview_y[np.isfinite(self.preview_y)]
        if not len(xs) or not len(ys):
            return
        data_cx = (float(xs.min()) + float(xs.max())) / 2.0
        data_cy = (float(ys.min()) + float(ys.max())) / 2.0

        base = BUILTIN_ARENAS[name](0.0, 0.0)
        self.spin_arena_dx.setValue(data_cx - base.width / 2.0)
        self.spin_arena_dy.setValue(data_cy - base.height / 2.0)
        self.log_message(f"Arena registered: offset ({self.spin_arena_dx.value():.2f}, "
                         f"{self.spin_arena_dy.value():.2f}) m")

    # -- streaming chunk engine ------------------------------------------- #
    # The timeline spans the whole recording, but memory holds at most
    # MAX_CACHED_CHUNKS slices. Scrubbing is debounced so dragging costs
    # nothing; only where you stop is fetched, on a background thread.
    MAX_CACHED_CHUNKS = 6
    SCRUB_DEBOUNCE_MS = 70

    def preview_time_bounds(self, selected_tags):
        """(min_ts, max_ts) epoch-ms across the selected tags, or None."""
        try:
            conn = connect_ro(self.preview_db_path or self.db_path)
            placeholders = ",".join(["?"] * len(selected_tags))
            row = conn.execute(
                f"SELECT MIN(timestamp), MAX(timestamp) FROM {self.table_name} "
                f"WHERE shortid IN ({placeholders})", selected_tags).fetchone()
            conn.close()
            if row and row[0] is not None:
                return int(row[0]), int(row[1])
        except Exception as e:
            if is_corruption_error(e):
                self.report_corrupt_database(e)
            else:
                self.log_message(f"Could not read time bounds: {e}")
        return None

    def selected_preview_tags(self):
        return [tag for tag, cb in self.tag_checkboxes.items() if cb.isChecked()]

    def on_tag_selection_changed(self, _state=None):
        """Tag selection changed — refresh the preview for the new tag set.

        Coalesced through a timer because Select All/None and config loading
        toggle every checkbox in a burst, and each one emits separately.
        """
        if self._tag_selection_guard:
            return
        # During a batch run the preview is never shown; skip the refresh timer
        # entirely so nothing races with the sequential trial stepping.
        if getattr(self, '_batch_active', False):
            return
        # First tags for a freshly-loaded database bring the preview to life;
        # later changes just refresh it.
        if not self._preview_active:
            if self.db_path and self.table_name and self.selected_preview_tags():
                self.preview_tag_timer.start(150)
            return
        self.preview_tag_timer.start(150)

    def _apply_tag_selection_change(self):
        """Rebuild the timeline and cached chunks for the current tag set."""
        if not (self.db_path and self.table_name):
            return
        if not self._preview_active:
            self.activate_preview()
            return
        tags = self.selected_preview_tags()
        if not tags:
            self.stop_preview_playback()
            self.preview_cache.clear()
            self.preview_x = None
            self._update_cache_label()
            self.lbl_preview_status.setText("No tags selected")
            self.preview_canvas_2d.clear()
            if self.preview_canvas_3d is not None:
                self.preview_canvas_3d.clear()
            return

        # Switching databases with the preview already open lands here (config
        # load re-checks the boxes), and that path never runs the enable
        # handler, so make sure the new database gets its indexed copy.
        if self.preview_db_path is None:
            self.ensure_preview_index_db()

        # The recording bounds are per-tag, so adding or removing a tag can
        # change the span the timeline has to cover.
        keep = self.preview_playhead_ms
        if self.init_preview_timeline():
            if self.preview_t0 <= keep <= self.preview_t1:
                self.preview_playhead_ms = keep     # stay where the user was
            else:
                self.preview_playhead_ms = self.preview_t0
            self._sync_timeline_to_playhead()
        self.invalidate_preview_cache()

    def invalidate_preview_cache(self):
        # Keep the (hidden) export settings identical to the preview's.
        self._sync_export_from_preview()
        """Drop cached chunks. Called whenever anything upstream of the frames
        changes — filtering, smoothing, timezone, tag selection, chunk size —
        so the pane can never show data processed with stale settings."""
        self.preview_cache.clear()
        self.preview_x = None
        self._update_cache_label()
        if self._preview_active and self.preview_t0 is not None:
            # Re-fetch whatever the playhead is sitting on under the new settings
            self._request_chunk(self._chunk_index_for(self.preview_playhead_ms),
                                make_current=True)

    def _windowed_bounds(self, bounds):
        """Narrow (min_ts, max_ts) to the tags' configured active windows.

        Fixes outside a tag's window are trimmed away, so a timeline that
        still spanned the raw table would open on a stretch where the
        trimmed tags simply are not there. A tag with no window configured
        contributes the full data range, so mixing configured and
        unconfigured tags never hides anything.

        The result is always clamped back inside the data, and falls back
        to the original bounds if the windows leave nothing.
        """
        lo, hi = bounds
        tags = self.selected_preview_tags()
        if not tags or not self.tag_identities:
            return lo, hi
        try:
            tz = pytz.timezone(self.combo_timezone.currentText())
        except Exception:
            return lo, hi

        starts, stops = [], []
        for tag in tags:
            w_start, w_stop = self._tag_window(tag, tz)
            # No window on this tag -> it can span the whole recording.
            starts.append(lo if w_start is None
                          else int(w_start.timestamp() * 1000))
            stops.append(hi if w_stop is None
                         else int(w_stop.timestamp() * 1000))
        if not starts:
            return lo, hi
        new_lo = max(lo, min(starts))
        new_hi = min(hi, max(stops))
        if new_hi <= new_lo:
            self.log_message(
                "Tag active windows leave no overlap with the recording; "
                "showing the full timeline instead.")
            return lo, hi
        return new_lo, new_hi

    def _apply_timeline_bounds(self, bounds):
        """Size the timeline slider to (min_ts, max_ts) epoch-ms. Returns success."""
        if not bounds:
            self.lbl_preview_status.setText("No records for the selected tags")
            return False
        raw_lo, raw_hi = bounds
        # Keep the raw span: widening a window later must be able to grow
        # the timeline back out, which needs the untrimmed extent.
        self._preview_raw_bounds = (raw_lo, raw_hi)
        self.preview_t0, self.preview_t1 = self._windowed_bounds(bounds)
        if (self.preview_t0, self.preview_t1) != (raw_lo, raw_hi):
            self.log_message(
                "Timeline narrowed to the tags' configured active windows.")
        total_s = max(1, (self.preview_t1 - self.preview_t0) // 1000)
        self._timeline_guard = True
        self.slider_timeline.setRange(0, int(total_s))
        # Arrow-key / hold scrubbing: ~1000 steps across the recording per
        # single press, ~50 per Page. Fine for fine control, fast when held.
        self.slider_timeline.setSingleStep(max(1, int(total_s) // 1000))
        self.slider_timeline.setPageStep(max(1, int(total_s) // 50))
        self._timeline_guard = False
        days = total_s / 86400.0
        self.log_message(
            f"Preview timeline: {days:.1f} days "
            f"({pd.Timestamp(self.preview_t0, unit='ms', tz='UTC').tz_convert(self.combo_timezone.currentText()):%Y-%m-%d %H:%M} "
            f"to {pd.Timestamp(self.preview_t1, unit='ms', tz='UTC').tz_convert(self.combo_timezone.currentText()):%Y-%m-%d %H:%M})")
        return True

    def init_preview_timeline(self):
        """Size the timeline slider synchronously (fast path, e.g. tag changes).

        Used once the fast index already exists, where MIN/MAX is quick. The
        first-load path uses _load_timeline_async instead, because that MIN/MAX
        runs against the not-yet-indexed original and would freeze the window.
        """
        tags = self.selected_preview_tags()
        if not self.db_path or not self.table_name or not tags:
            return False
        return self._apply_timeline_bounds(self.preview_time_bounds(tags))

    def _load_timeline_async(self):
        """Size the timeline off the GUI thread, then seed the first chunk.

        The first-load MIN/MAX runs against the unindexed original over the
        network — slow enough to freeze the UI — so it is computed on a
        DbQueryWorker. On completion (_on_timeline_loaded) the slider is sized
        and the initial chunk requested.
        """
        tags = self.selected_preview_tags()
        if not (self.db_path and self.table_name and tags):
            return
        db = self.current_indexed_db() or self.preview_db_path or self.db_path
        table = self.table_name
        tags_snap = list(tags)

        def _query():
            conn = connect_ro(db)
            try:
                ph = ",".join("?" * len(tags_snap))
                row = conn.execute(
                    f"SELECT MIN(timestamp), MAX(timestamp) FROM {table} "
                    f"WHERE shortid IN ({ph})", tags_snap).fetchone()
            finally:
                conn.close()
            bounds = (int(row[0]), int(row[1])) if row and row[0] is not None else None
            return {'table': table, 'tags': tags_snap, 'bounds': bounds}

        self._show_busy("Reading recording time range…")
        self._start_db_query(_query, self._on_timeline_loaded, self._on_meta_load_failed)

    def _on_timeline_loaded(self, res):
        self._hide_busy()
        # Stale if the table changed or the tag selection moved on while loading;
        # a newer selection will have launched its own load.
        if res['table'] != self.table_name or res['tags'] != self.selected_preview_tags():
            return
        if self._apply_timeline_bounds(res['bounds']):
            self.preview_playhead_ms = self.preview_t0
            self._sync_timeline_to_playhead()
            self._request_chunk(0, make_current=True)

    def _chunk_ms(self):
        return self.spin_preview_minutes.value() * 60 * 1000

    def _chunk_index_for(self, ts_ms):
        if self.preview_t0 is None:
            return 0
        return max(0, int((ts_ms - self.preview_t0) // self._chunk_ms()))

    def _chunk_bounds(self, idx):
        c = self._chunk_ms()
        start = self.preview_t0 + idx * c
        return start, start + c

    def _resident_chunk_for(self, ts_ms):
        """Index of the cached chunk whose span contains ``ts_ms``, else None.

        Chunks are fixed-width and index-aligned, so the only candidate is
        ``_chunk_index_for(ts_ms)``; return it only when it is actually cached.
        """
        idx = self._chunk_index_for(ts_ms)
        c = self.preview_cache.get(idx)
        if c is not None and c["t_start"] <= ts_ms <= c["t_end"]:
            return idx
        return None

    def _prefetch_neighbors(self, idx):
        """Warm the chunks on either side so a continuous scrub stays in cache.

        ``_request_chunk`` no-ops for anything already cached or in flight, so
        calling this on every chunk switch is cheap and self-limiting.
        """
        for nb in (idx + 1, idx - 1):
            if (nb >= 0 and nb not in self.preview_cache
                    and nb not in self.preview_inflight):
                self._request_chunk(nb, make_current=False)

    def _update_cache_label(self):
        if not hasattr(self, "lbl_cache_status"):
            return
        n = len(self.preview_cache)
        kb = sum(c["nbytes"] for c in self.preview_cache.values()) / 1024.0
        self.lbl_cache_status.setText(
            f"{n}/{self.MAX_CACHED_CHUNKS} chunks · {kb:.0f} KB" if n else "")

    # -- request / receive ------------------------------------------------- #
    def _request_chunk(self, idx, make_current=False):
        """Fetch a chunk in the background unless it is cached or in flight."""
        # Never start a read once the preview is torn down (window closed) or
        # disabled: a late-arriving result can otherwise re-enter here — e.g.
        # _handle_empty_chunk skipping a recording gap — and spawn a thread
        # after teardown already drained them.
        if not getattr(self, '_preview_active', False):
            return
        if self.preview_t0 is None or idx < 0:
            return
        if idx in self.preview_cache:
            self.preview_cache.move_to_end(idx)
            if make_current:
                self._show_chunk(idx)
            return
        if idx in self.preview_inflight:
            # Only a still-RUNNING loader means "already being fetched".
            # Retirement is deferred by one event-loop turn (see _retire_loader,
            # which avoids freeing a QThread inside its own finished signal), so
            # a chunk that just completed lingers here briefly. Treating that as
            # in-flight would silently drop a re-request issued in that window —
            # e.g. changing a preview setting right after a chunk lands — and
            # the pane would stay blank forever waiting on a load that already
            # happened. A finished loader is safe to release now.
            loader = self.preview_inflight[idx]
            if loader.isRunning():
                if make_current:
                    self.preview_pending_current = idx
                return
            self.preview_inflight.pop(idx, None)

        tags = self.selected_preview_tags()
        if not tags:
            return
        start, end = self._chunk_bounds(idx)
        if start > self.preview_t1:
            return
        # Load a lead-in before the chunk so smoothing and forward-fill are
        # warmed and the trailing track is continuous across the boundary — a
        # tag's position and its tail no longer reset when you scrub from one
        # chunk into the next. Bounded by the chunk width: a trail longer than a
        # chunk can't be shown in full anyway (frames are chunk-local).
        lead = min(self.spin_preview_trail.value() * 1000, self._chunk_ms())
        load_start = max(self.preview_t0, start - lead)

        if make_current:
            self.preview_pending_current = idx
            self.lbl_preview_status.setText("Loading…")

        loader = PreviewChunkLoader(self.preview_db_path or self.db_path,
                                    self.table_name, tags, load_start, end, idx)
        loader.loaded.connect(self._on_chunk_loaded)
        loader.failed.connect(self._on_chunk_failed)
        # The inflight dict holds the only strong reference to the thread.
        # Dropping it inside the finished handler can free the QThread during
        # signal emission ("Destroyed while thread is still running"), so defer
        # the drop to the next event-loop turn, once emission has unwound.
        loader.finished.connect(lambda i=idx: QTimer.singleShot(
            0, lambda: self._retire_loader(i)))
        self.preview_inflight[idx] = loader
        loader.start()

    def _retire_loader(self, idx):
        """Release a finished loader. Safe: runs outside the finished signal."""
        loader = self.preview_inflight.get(idx)
        if loader is None:
            return
        if not loader.isFinished():
            # Slot raced ahead of the thread actually stopping; try again.
            QTimer.singleShot(50, lambda: self._retire_loader(idx))
            return
        self.preview_inflight.pop(idx, None)

    def _on_chunk_failed(self, idx, err):
        if is_corruption_error(err):
            # Stop streaming: every further chunk would fail the same way.
            self._preview_active = False
            self.stop_preview_playback()
            self.preview_pending_current = None
            self.lbl_preview_status.setText("Database file is damaged — see the message log.")
            self.report_corrupt_database(err)
            return
        self.log_message(f"Preview chunk {idx} failed: {err}")
        if self.preview_pending_current == idx:
            self.preview_pending_current = None
            self.lbl_preview_status.setText(f"Load failed: {err}")

    def _on_chunk_loaded(self, idx, df):
        """Process a delivered slice on the main thread and cache it.

        Filtering and smoothing run here rather than in the worker because they
        read the settings widgets directly, and they are a negligible share of
        the cost (the SQL read dominates).
        """
        try:
            frames = self._process_chunk(df, idx) if df is not None and len(df) else None
        except Exception as e:
            self.log_message(f"Preview chunk {idx} processing failed: {e}")
            frames = None

        if frames is not None:
            self.preview_cache[idx] = frames
            self.preview_cache.move_to_end(idx)
            while len(self.preview_cache) > self.MAX_CACHED_CHUNKS:
                self.preview_cache.popitem(last=False)   # evict least-recent
            self._update_cache_label()

        if self.preview_pending_current == idx:
            self.preview_pending_current = None
            if frames is None:
                self._handle_empty_chunk(idx)
            else:
                self._show_chunk(idx)

    def _handle_empty_chunk(self, idx):
        """Chunk had no samples — jump to the next real data.

        Real recordings contain outages (Echo T001 has a ~14 h gap), and the
        timeline is linear in wall-clock time, so a position can land in dead
        air. Rather than showing an empty scene, find the next sample and move
        the playhead there.
        """
        tags = self.selected_preview_tags()
        start, _ = self._chunk_bounds(idx)
        try:
            conn = connect_ro(self.preview_db_path or self.db_path)
            nxt = self._nearest_sample_ts(conn, tags, start)
            conn.close()
        except Exception:
            nxt = None

        if nxt is None or self._chunk_index_for(nxt) == idx:
            self.lbl_preview_status.setText("No samples here — try elsewhere on the timeline.")
            return

        gap_min = abs(nxt - start) / 60000.0
        self.log_message(f"No data at that position — skipped {gap_min:.0f} min to the next samples.")
        self.lbl_preview_status.setText(f"Gap in recording — skipped {gap_min:.0f} min forward.")
        self.preview_playhead_ms = nxt
        self._sync_timeline_to_playhead()
        self._request_chunk(self._chunk_index_for(nxt), make_current=True)

    def get_preview_smoothing_method(self):
        """Smoothing method for the LIVE preview (its own 'Smoothing method'
        selector, independent of the export method). 'None' shows raw fixes."""
        return (self.combo_preview_smoothing.currentText()
                if hasattr(self, 'combo_preview_smoothing') else 'None')

    def _tag_window(self, tag, tz):
        """(start, stop) timestamps for a tag's configured active window.

        Returns (None, None) when the tag has no window set. Local wall-clock
        strings are localized defensively: a DST spring-forward gap or
        fall-back repeat would otherwise raise and take the whole preview
        chunk or export with it.
        """
        info = (self.tag_identities or {}).get(tag)
        if not info:
            return None, None
        # A bound on Auto is not applied at all; only an
        # explicitly limited (Manual) one trims.
        start_manual = IdentityAssignmentDialog._mode_of(info, 'start') == 'manual'
        stop_manual = IdentityAssignmentDialog._mode_of(info, 'stop') == 'manual'
        if not start_manual and not stop_manual:
            return None, None

        def _loc(value):
            try:
                ts = pd.Timestamp(value)
            except (ValueError, TypeError):
                return None
            if ts is pd.NaT:
                return None
            if ts.tzinfo is not None:
                return ts.tz_convert(tz)
            try:
                return ts.tz_localize(tz)
            except Exception:
                # Nonexistent (spring forward) or ambiguous (fall back) local
                # time: take the first valid reading rather than failing.
                return ts.tz_localize(tz, ambiguous=True, nonexistent='shift_forward')

        return (_loc(info['start_time']) if start_manual and 'start_time' in info else None,
                _loc(info['stop_time']) if stop_manual and 'stop_time' in info else None)

    def _trim_to_tag_window(self, data, tag, tz):
        """Drop fixes outside the tag's active window set in Configure Identities.

        Those Start/Stop pickers are a data-trimming control, not just
        metadata: a tag deployed late, recovered early, or swapped mid-trial
        must contribute nothing outside its window. Applied identically to the
        preview, the occupancy heatmap and every export, so what the preview
        shows is what the exported files contain.

        ``data`` must already carry a tz-aware 'Timestamp' column. Returns
        (trimmed_data, n_removed).
        """
        start, stop = self._tag_window(tag, tz)
        if start is None and stop is None:
            return data, 0
        before = len(data)
        keep = pd.Series(True, index=data.index)
        if start is not None:
            keep &= data['Timestamp'] >= start
        if stop is not None:
            keep &= data['Timestamp'] <= stop
        return data[keep], before - int(keep.sum())

    def _process_chunk(self, df, idx=None):
        """Filter, smooth and bin one slice into frame arrays, or None if empty.

        ``idx`` is the chunk index; when given, the frames' nominal span is set
        from it (not from the data), so the lead-in overlap the loader prepends
        stays outside the scrubbable/containment range.
        """
        merged = self._processed_slice(df)
        if merged is None:
            return None
        frames = self._build_preview_frames(merged, self.selected_preview_tags(), idx)
        if frames is not None:
            # Travels WITH the chunk: a prefetched neighbour also runs through
            # _processed_slice, so a pool left on the window would describe some
            # other chunk than the one on screen.
            frames['step_pool'] = self._preview_step_pool
        return frames

    def analysis_dir_for(self, db_path=None):
        """The <db>_FNT_analysis folder for a database, or None."""
        db_path = db_path or self.db_path
        if not db_path:
            return None
        d = os.path.dirname(db_path)
        name = os.path.splitext(os.path.basename(db_path))[0]
        return os.path.join(d, f"{name}_FNT_analysis")

    def confirm_clear_analysis_folder(self, db_path=None, parent=None):
        """Show what a clear would delete and ask. True if the user agrees.

        Deliberately a second prompt even when the conflict dialog already
        offered the choice: that dialog lists what the EXPORT will write, which
        is a different set from what a clear will REMOVE, and the difference is
        exactly where the surprises live.
        """
        folder = self.analysis_dir_for(db_path)
        if not folder or not os.path.isdir(folder):
            QMessageBox.information(
                parent or self, "Nothing to clear",
                "This trial has no analysis folder yet.")
            return False
        removable, kept, unknown = classify_analysis_folder(folder)
        if not removable:
            QMessageBox.information(
                parent or self, "Nothing to clear",
                f"{os.path.basename(folder)} holds no previous exports.")
            return False

        total = sum(sz for _n, sz, _d in removable)
        lines = [f"Delete {len(removable)} item(s) from "
                 f"{os.path.basename(folder)}, freeing {human_bytes(total)}?",
                 ""]
        for name, sz, is_dir in removable[:14]:
            lines.append(f"   {'[folder] ' if is_dir else ''}{name}"
                         f"   ({human_bytes(sz)})")
        if len(removable) > 14:
            lines.append(f"   ...and {len(removable) - 14} more")
        if kept:
            lines.append("")
            lines.append("KEPT: " + ", ".join(n for n, _s, _d in kept))
        if unknown:
            # The one case worth a person's attention: files FNT did not write.
            lines.append("")
            lines.append("NOT WRITTEN BY FNT - these would be deleted too:")
            for name, _sz, _d in unknown[:8]:
                lines.append(f"   {name}")
        lines.append("")
        lines.append("This cannot be undone, and the smoothed CSV will have to "
                     "be re-derived from the database.")

        box = QMessageBox(parent or self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Clear & Re-export")
        box.setText("\n".join(lines))
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        box.setDefaultButton(QMessageBox.Cancel)
        return box.exec_() == QMessageBox.Yes

    def clear_previous_exports(self):
        """The standalone Clear & Re-export action (not the export path).

        Exists separately because the conflict dialog only opens when a file
        the CURRENT export produces already exists. Change which products you
        export and the old ones become orphans that dialog never mentions - so
        there has to be a way to ask for a clean folder directly.
        """
        if not self.db_path:
            QMessageBox.warning(self, "No Database",
                                "Load a database first.")
            return
        if not self.confirm_clear_analysis_folder():
            return
        folder = self.analysis_dir_for()
        removed, freed, failures = clear_analysis_folder(
            folder, log=self.log_message)
        # The fast index survives, but the preview may hold file handles into
        # products that no longer exist.
        self.invalidate_preview_cache()
        self._export_scope_key = None
        if failures:
            QMessageBox.warning(
                self, "Clear & Re-export",
                f"Removed {removed} item(s), {human_bytes(freed)} freed.\n\n"
                f"{len(failures)} could not be removed (open in another "
                f"program?):\n" + "\n".join(failures[:6]))
        else:
            QMessageBox.information(
                self, "Clear & Re-export",
                f"Removed {removed} item(s) and freed {human_bytes(freed)}.\n\n"
                f"Run an export to rebuild the folder.")

    def export_scope_points(self):
        """(tags, points) that an export would actually process, or None.

        The denominator the threshold report should be judged against. NOT the
        row count of the raw database: it is the number of points left after
        the tag configuration is applied - only the ticked tags, and only the
        rows inside each tag's active start/stop window. That is the population
        the export will see, so it is the one a removal percentage means
        something against.

        Counted on the INDEXED preview copy, whose (shortid, timestamp) index
        makes each per-tag window a covering-index range scan: about a second
        for 17 tags over 24.7 M rows, against ~35 s on the raw database, which
        has no indexes at all. Returns None rather than blocking when that copy
        is not built yet.

        Cached against the tags and their windows, because that is the only
        thing that changes it - moving a threshold does not.
        """
        tags = sorted(t for t, cb in self.tag_checkboxes.items() if cb.isChecked())
        if not tags or not self.table_name:
            return None
        tz = pytz.timezone(self.combo_timezone.currentText())
        windows = {}
        for t in tags:
            try:
                start, stop = self._tag_window(t, tz)
            except Exception:
                start, stop = None, None
            windows[t] = (
                int(start.timestamp() * 1000) if start is not None else None,
                int(stop.timestamp() * 1000) if stop is not None else None)
        key = (self.db_path, self.table_name, tuple(tags),
               tuple(sorted(windows.items())))
        if getattr(self, '_export_scope_key', None) == key:
            return self._export_scope_value

        path = self.current_indexed_db()
        if not path:
            return None            # no fast index yet; do not stall the GUI
        total = 0
        try:
            conn = connect_ro(path)
            try:
                for t in tags:
                    lo, hi = windows[t]
                    q = f"SELECT COUNT(*) FROM {self.table_name} WHERE shortid = ?"
                    args = [t]
                    if lo is not None:
                        q += " AND timestamp >= ?"; args.append(lo)
                    if hi is not None:
                        q += " AND timestamp <= ?"; args.append(hi)
                    total += conn.execute(q, args).fetchone()[0]
            finally:
                conn.close()
        except Exception as e:
            self.log_message(f"Could not count the export scope: {e}")
            return None

        self._export_scope_key = key
        self._export_scope_value = (len(tags), total)
        return self._export_scope_value

    def _update_step_stats_label(self):
        """Say what the velocity/jump thresholds do to the chunk in view.

        Counts are reproduced the way apply_filters_to_data applies them: both
        quantities are measured once on the raw track, the velocity cut runs
        first, and the jump cut then runs on what is left.
        """
        if not hasattr(self, 'lbl_step_stats'):
            return
        pool = getattr(self, '_preview_step_pool', None)
        if not pool:
            self.lbl_step_stats.setPlainText("")
            return
        if len(pool) < 7:                  # a chunk cached before the rewrite
            self.lbl_step_stats.setPlainText("")
            return
        step, speed, gap, dev, px, py, pgrp = pool
        ok = np.isfinite(step)
        step, speed = step[ok], speed[ok]
        gap, dev = gap[ok], dev[ok]
        px, py, pgrp = px[ok], py[ok], pgrp[ok]
        if not len(step):
            self.lbl_step_stats.setPlainText("")
            return
        n = len(step)
        pct = lambda k: 100.0 * k / n if n else 0.0
        min_dt = self.spin_preview_velocity_min_dt.value()
        # Recompute the speed at the CURRENT time base, so moving the floor
        # re-prices the velocity cut in front of the user.
        with np.errstate(divide='ignore', invalid='ignore'):
            eff_speed = step / np.maximum(gap, min_dt if min_dt else 1e-9)

        def _median(a, mask=None):
            v = a if mask is None else a[mask]
            v = v[np.isfinite(v)]
            return float(np.median(v)) if v.size else float('nan')

        def _si(v):
            """Big counts, readably: 18,243,551 -> 18.2 M."""
            if v >= 1e6:
                return f"{v / 1e6:.1f} M"
            if v >= 1e4:
                return f"{v / 1e3:.0f} k"
            return f"{v:,}"

        # Applied in the order apply_filters_to_data applies them, each seeing
        # only what the one before it left: the counts add up to the total
        # removed instead of triple-counting the same wild point.
        on_o = self.chk_preview_outlier.isChecked()
        on_v = self.chk_preview_velocity.isChecked()
        on_j = self.chk_preview_jump.isChecked()
        sig = self.spin_preview_outlier_sigma.value()
        win = int(self.spin_preview_outlier_window.value())
        vthr = self.spin_preview_velocity.value()
        jthr = self.spin_preview_jump.value()
        cut_o = (dev > sig) & np.isfinite(dev) if on_o else np.zeros(n, bool)
        cut_v = (eff_speed > vthr) & ~cut_o if on_v else np.zeros(n, bool)
        cut_j = (step > jthr) & ~cut_o & ~cut_v if on_j else np.zeros(n, bool)
        # The excursion pass, run here for real rather than described. It is
        # the LAST filter and it works on what the other three left behind, so
        # reporting the first three alone understates the export badly - on a
        # tag with a long run of bad fixes this pass removes more than all of
        # them together.
        first_pass = ~(cut_o | cut_v | cut_j)
        # Gated exactly as the EXPORT gates it (see apply_filters_to_data):
        # excursion re-tests the joins that velocity and jump removals create,
        # so with both of those off it has no test to apply and never runs.
        # A row here in that case would promise a pass the export skips.
        on_a = self.chk_preview_adjacency.isChecked() and (on_v or on_j)
        if on_a and n > 2:
            # A seconds clock rebuilt from the intervals the pool carries.
            # resolve_adjacency only ever differences times WITHIN a group, so
            # a single running clock is enough - the group boundaries it must
            # not cross are handed to it separately.
            _newgrp = np.empty(n, bool)
            _newgrp[0] = True
            _newgrp[1:] = pgrp[1:] != pgrp[:-1]
            pt = np.cumsum(np.where(_newgrp, 0.0, np.nan_to_num(gap, nan=0.0)))
            adj_keep, _n_adj, adj_converged = resolve_adjacency(
                pt, px, py, first_pass, group=pgrp,
                v_thresh=vthr if on_v else None, min_dt=min_dt,
                j_thresh=jthr if on_j else None,
                max_excursion_s=self.spin_preview_excursion.value())
            cut_a = first_pass & ~adj_keep
        else:
            cut_a = np.zeros(n, bool)
            adj_converged = True
        kept = first_pass & ~cut_a
        # What the surviving track is supposed to respect. With the jump
        # threshold off there is no claim to check, so nothing is flagged.
        worst_step_limit = jthr if on_j else float('inf')

        lines = []
        # The population an export would actually process. Absent until the
        # fast index exists, and the report says so rather than quietly
        # falling back to the window count and mislabelling it.
        scope = self.export_scope_points()
        if scope:
            n_tags, n_pts = scope
            lines.append("EXPORT SCOPE  after tag configuration")
            lines.append("  {} tags \u00b7 {:,} points inside their active windows"
                         .format(n_tags, n_pts))
        else:
            lines.append("EXPORT SCOPE  unavailable until the fast index is built")
        lines.append("")

        lines.append("THRESHOLDS  measured on the {:,} points in view".format(n))
        rows = []
        if on_o:
            rows.append(("Hampel", "> {:g} MAD over {:,} pts".format(sig, win),
                         cut_o,
                         "median {:.1f} MAD out, {:.2f} m off track".format(
                             _median(dev, cut_o), _median(step, cut_o))))
        if on_v:
            # Without this tail a large percentage reads as "the animal
            # teleported a lot", when at this reporting rate it usually means
            # the interval was short: a 10 cm step in 30 ms is 3 m/s.
            rows.append(("Velocity",
                         "> {:g} m/s over \u2265{:g} s".format(vthr, min_dt),
                         cut_v,
                         "median {:.2f} m in {:.3f} s".format(
                             _median(step, cut_v), _median(gap, cut_v))))
        if on_j:
            rows.append(("Jump", "> {:g} m".format(jthr), cut_j,
                         "median {:.2f} m".format(_median(step, cut_j))))
        if on_a:
            # Says what it DOES, not what it is named: these points survived
            # the three tests above and only became impossible once their
            # neighbours were removed and the track closed up around them.
            rows.append(("Excursion",
                         "runs < {:g} s stranded by the above".format(
                             self.spin_preview_excursion.value()),
                         cut_a,
                         "re-tested after removal closed the gaps"))
        if not rows:
            lines.append("  none enabled \u2014 every point is kept")
        else:
            for name, spec, mask, tail in rows:
                lines.append("  {:<10} {:<34} {:>7,} removed ({:>5.1f}%)   {}"
                             .format(name, spec, int(mask.sum()),
                                     pct(mask.sum()),
                                     tail if mask.any() else ""))
            lines.append("  " + "-" * 64)
            keep_pct = pct(kept.sum())
            tail = ""
            if scope:
                # An estimate, and labelled as one: this is one window's
                # removal rate carried across the whole recording, and a quiet
                # night need not resemble an active afternoon.
                tail = "  \u2192 est. {} of {} points".format(
                    _si(int(round(scope[1] * keep_pct / 100.0))), _si(scope[1]))
            lines.append("  keeps {:,} of {:,} in view ({:.1f}%){}".format(
                int(kept.sum()), n, keep_pct, tail))
            worst_before = float(step.max())
            # Measured on the track that is LEFT, not on the old step into
            # each survivor. Those differ exactly where a run of bad fixes is
            # partly removed: the survivor's original neighbour is gone, so it
            # inherits a distant one, and that stride appears in neither the
            # removed set nor the old figure.
            sx, sy, sg = px[kept], py[kept], pgrp[kept]
            if sx.size > 1:
                same = sg[1:] == sg[:-1]      # never step across two tags
                left = np.hypot(np.diff(sx), np.diff(sy))[same]
                worst_after = float(left.max()) if left.size else float('nan')
            else:
                worst_after = float('nan')
            lines.append("  largest step REMAINING in the track {:.2f} m "
                         "(largest before filtering {:.2f} m)"
                         .format(worst_after, worst_before))
            if np.isfinite(worst_after) and worst_after > worst_step_limit:
                if on_a:
                    # Excursion IS the re-test, so when it is on and a long
                    # stride survives, that is the filter declining to delete
                    # rather than nobody having looked.
                    lines.append(
                        "     ⚠ exceeds the {:.2f} m jump threshold. Excursion "
                        "re-tested this join and did NOT cut it: the run is "
                        "longer than {:g} s, so it is treated as a real "
                        "relocation, not a glitch. Raise the excursion seconds "
                        "to remove longer runs.".format(
                            worst_step_limit,
                            self.spin_preview_excursion.value()))
                else:
                    lines.append(
                        "     ⚠ exceeds the {:.2f} m jump threshold — removing "
                        "a point makes its neighbours adjacent, and with "
                        "Excursion OFF that new step is never re-tested. Tick "
                        "Excursion threshold to test it.".format(
                            worst_step_limit))
            if on_a and not adj_converged:
                lines.append(
                    "     ⚠ excursion did not settle within its pass limit — "
                    "the track may still contain an untested join")
        lines.append("")
        lines.append("IN VIEW  reporting every {:.2f} s (median) \u00b7 steps: "
                     "half below {:.2f} m, 99th {:.2f} m, largest {:.2f} m"
                     .format(_median(gap), float(np.percentile(step, 50)),
                             float(np.percentile(step, 99)), float(step.max())))
        # rstrip: a threshold that removed nothing has no explanatory tail, and
        # the column padding would otherwise leave the line trailing spaces.
        self.lbl_step_stats.setPlainText(
            "\n".join(line.rstrip() for line in lines))

    def _step_columns(self, g, prefix='', x_col=None, y_col=None):
        """Add per-fix step distance / speed to one tag's frame.

        Grouped by the Time gap setting so no step is measured across a battery
        restart, and the first fix of each group stays NaN rather than reading
        zero. ``x_col``/``y_col`` default to the SMOOTHED track when it exists,
        so the marker readout describes the track actually drawn — select
        "None" as the smoothing method and it describes the unsmoothed fixes.
        The threshold summary passes the pre-smoothing columns explicitly,
        because that is what the thresholds are applied to.
        """
        x_col = x_col or ('smoothed_x' if 'smoothed_x' in g.columns else 'location_x')
        y_col = y_col or ('smoothed_y' if 'smoothed_y' in g.columns else 'location_y')
        gap = self.spin_time_gap.value()
        dt = g['Timestamp'].diff().dt.total_seconds()
        grp = (np.ceil(dt.fillna(0)).astype(int) > gap).cumsum()
        x, y = g[x_col], g[y_col]
        step = np.hypot(x - x.groupby(grp).shift(), y - y.groupby(grp).shift())
        g[prefix + 'step_m'] = step
        with np.errstate(divide='ignore', invalid='ignore'):
            g[prefix + 'speed_ms'] = step / dt.where(dt > 0)
        return g

    def _processed_slice(self, df):
        """One raw time slice, trimmed/thresholded/smoothed, in long form.

        The tail end of the preview pipeline, split out because the animation
        clip preview needs the same rows the preview is drawing - same tag
        windows, same thresholds, same smoothing - and reimplementing that
        would let the clip drift away from what the user is looking at.
        Returns None when nothing survives.
        """
        tz = pytz.timezone(self.combo_timezone.currentText())
        smoothing_method = self.get_preview_smoothing_method()
        # Preview thresholding toggles are independent of the export ones, so the
        # user can compare the track with/without each threshold live.
        use_vel = self.chk_preview_velocity.isChecked()
        use_jump = self.chk_preview_jump.isChecked()
        use_out = self.chk_preview_outlier.isChecked()
        do_filter = use_vel or use_jump or use_out

        chunks = []
        # Pre-filter step/speed for every fix in view, kept BEFORE thresholding
        # drops anything: the whole point of the summary is to show the fixes a
        # threshold would remove, which the surviving rows can no longer show.
        pre_step, pre_speed, pre_gap, pre_dev = [], [], [], []
        # Coordinates and a per-tag id travel with them, so the read-out can
        # measure the track thresholding actually LEAVES. Removing a point
        # makes its neighbours adjacent, and that new step is not in pre_step -
        # which is why the old figure could report 0.93 m for a track holding
        # a 7.22 m stride.
        pre_x, pre_y, pre_grp = [], [], []
        for _gi, (tag, g) in enumerate(df.groupby('shortid', sort=False)):
            g = g.copy()
            g['Timestamp'] = pd.to_datetime(
                g['timestamp'], unit='ms', origin='unix', utc=True).dt.tz_convert(tz)
            g['location_x'] *= 0.0254
            g['location_y'] *= 0.0254
            # Keep pre-smoothing fixes so the pane can overlay raw vs smoothed
            g['raw_x'] = g['location_x']
            g['raw_y'] = g['location_y']
            # Trim to the tag's active window BEFORE thresholding and
            # smoothing, exactly as the export does — otherwise the preview
            # shows fixes the exported CSV will not contain.
            g, _dropped = self._trim_to_tag_window(g, tag, tz)
            if not len(g):
                continue
            # Threshold BOTH velocity and jump on the raw coordinates, then
            # smooth (see _filter_and_smooth: thresholding velocity on the
            # smoothed track was tried and produced a jumpy result).
            # collect_stats=False: preview scrubbing must not overwrite the
            # filter_stats an export records in fnt_config.json. Preview uses its own
            # Smoothing Window / units so the effect can be tuned live,
            # independent of the Export Options values.
            g = self._step_columns(g, prefix='pre_', x_col='raw_x', y_col='raw_y')
            pre_step.append(g['pre_step_m'].to_numpy(float))
            pre_speed.append(g['pre_speed_ms'].to_numpy(float))
            # The interval itself, rather than inferring it from step/speed:
            # a fix that did not move has speed 0 and no recoverable interval.
            _dt = g['Timestamp'].diff().dt.total_seconds().to_numpy(float)
            pre_gap.append(_dt)
            pre_dev.append(hampel_deviation(
                g['raw_x'].to_numpy(float), g['raw_y'].to_numpy(float),
                window=self.spin_preview_outlier_window.value()))
            pre_x.append(g['raw_x'].to_numpy(float))
            pre_y.append(g['raw_y'].to_numpy(float))
            pre_grp.append(np.full(len(g), _gi, dtype=np.int32))
            g = self._filter_and_smooth(
                g, smoothing_method, collect_stats=False,
                velocity=use_vel, jump=use_jump,
                velocity_thresh=self.spin_preview_velocity.value(),
                jump_thresh=self.spin_preview_jump.value(),
                window=self.spin_preview_window.value(),
                units=self.combo_preview_window_units.currentText(),
                outlier=use_out,
                outlier_window=self.spin_preview_outlier_window.value(),
                outlier_sigma=self.spin_preview_outlier_sigma.value(),
                velocity_min_dt=self.spin_preview_velocity_min_dt.value(),
                adjacency=self.chk_preview_adjacency.isChecked(),
                excursion_s=self.spin_preview_excursion.value())
            if len(g):
                # Re-measure on the SURVIVING, SMOOTHED fixes: the marker
                # readout describes the track actually drawn, whose steps both
                # span whatever thresholding removed and follow the smoothing.
                chunks.append(self._step_columns(g))
        self._preview_step_pool = (
            (np.concatenate(pre_step), np.concatenate(pre_speed),
             np.concatenate(pre_gap), np.concatenate(pre_dev),
             np.concatenate(pre_x), np.concatenate(pre_y),
             np.concatenate(pre_grp))
            if pre_step else None)
        if not chunks:
            return None
        return pd.concat(chunks, ignore_index=True)

    def _build_preview_frames(self, df, selected_tags, idx=None):
        """Bin a slice onto a fixed time grid: one column per selected tag.

        When ``idx`` is given, t_start/t_end are the chunk's nominal bounds so
        the loader's lead-in overlap is excluded from the resident/scrub range.
        """
        hz = int(self.combo_preview_hz.currentText().split()[0])
        bin_ns = int(1_000_000_000 / hz)
        df = df.copy()
        # as_unit('ns') forces nanoseconds regardless of the datetime resolution
        # (pandas 3.0 defaults to microseconds); a bare astype(int64) would yield
        # microseconds here and make every bin 1000x too wide, collapsing the
        # whole chunk into a single preview frame.
        df['tbin'] = df['Timestamp'].dt.as_unit('ns').astype('int64') // bin_ns

        sx = 'smoothed_x' if 'smoothed_x' in df.columns else 'location_x'
        sy = 'smoothed_y' if 'smoothed_y' in df.columns else 'location_y'

        px = df.pivot_table(index='tbin', columns='shortid', values=sx, aggfunc='first')
        py = df.pivot_table(index='tbin', columns='shortid', values=sy, aggfunc='first')
        rx = df.pivot_table(index='tbin', columns='shortid', values='raw_x', aggfunc='first')
        ry = df.pivot_table(index='tbin', columns='shortid', values='raw_y', aggfunc='first')

        # Reindex onto every selected tag, not just those present in this
        # chunk. A tag that drops out would otherwise lose its column and shift
        # every later tag's colour, so the same animal would change colour
        # between chunks. Absent tags become all-NaN columns, which the
        # canvases hide.
        cols = list(selected_tags)
        px, py = px.reindex(columns=cols), py.reindex(columns=cols)
        rx, ry = rx.reindex(columns=cols), ry.reindex(columns=cols)

        grid = np.arange(px.index.min(), px.index.max() + 1)
        px, py = px.reindex(grid), py.reindex(grid)
        rx, ry = rx.reindex(grid), ry.reindex(grid)

        X = px.to_numpy(float)
        Y = py.to_numpy(float)
        RX = rx.to_numpy(float)
        RY = ry.to_numpy(float)

        # Forward-filled marker positions. UWB tags report asynchronously at
        # well under 1 Hz, so in any single time bin most tags have no fix and
        # would blink out. Carrying each tag's last known position forward keeps
        # a solid, persistent marker; the sparse trail still uses the real
        # fixes. Cells before a tag's first fix stay NaN (nothing to show yet).
        XF = pd.DataFrame(X).ffill().to_numpy()
        YF = pd.DataFrame(Y).ffill().to_numpy()

        # Battery voltage per tag on the same grid, forward-filled so the reading
        # persists between the tag's sparse fixes. None when the table carries no
        # battery column. 'last' per bin (battery varies slowly, so any fix does).
        # Raw step distance / speed per fix, forward-filled like the battery so
        # the label persists on a held marker instead of blinking.
        def _readout(col):
            if col not in df.columns:
                return None
            pv = df.pivot_table(index='tbin', columns='shortid', values=col,
                                aggfunc='last')
            pv = pv.reindex(columns=cols).reindex(grid)
            return pd.DataFrame(pv.to_numpy(float)).ffill().to_numpy()

        STEP = _readout('step_m')
        SPEED = _readout('speed_ms')

        if 'battery_voltage' in df.columns:
            pb = df.pivot_table(index='tbin', columns='shortid',
                                values='battery_voltage', aggfunc='last')
            pb = pb.reindex(columns=cols).reindex(grid)
            BAT = pd.DataFrame(pb.to_numpy(float)).ffill().to_numpy()
        else:
            BAT = None

        times = pd.to_datetime(grid * bin_ns, utc=True).tz_convert(
            pytz.timezone(self.combo_timezone.currentText()))

        # Nominal chunk bounds when we know the index, so the lead-in overlap the
        # loader prepends is not treated as scrubbable/resident; else fall back
        # to the data span.
        if idx is not None:
            t_start, t_end = self._chunk_bounds(idx)
        else:
            t_start = int(times[0].timestamp() * 1000)
            t_end = int(times[-1].timestamp() * 1000)

        return dict(x=X, y=Y, xf=XF, yf=YF, raw_x=RX, raw_y=RY, batt=BAT,
                    step=STEP, speed=SPEED, times=times,
                    tags=cols, hz=hz, t_start=t_start, t_end=t_end,
                    nbytes=X.nbytes + Y.nbytes + XF.nbytes + YF.nbytes
                           + RX.nbytes + RY.nbytes
                           + sum(a.nbytes for a in (BAT, STEP, SPEED)
                                 if a is not None))

    # Default preview marker colours: blue for male, red for female. Tags with
    # no configured identity default to male (matching the codebase's
    # sex.get(..., 'M') convention), so they show as solid blue until the user
    # assigns sex/identity in Configure Identities.
    _SEX_BLUE = (0.20, 0.55, 0.90, 1.0)
    _SEX_RED = (0.90, 0.25, 0.25, 1.0)
    _NEUTRAL = (0.20, 0.55, 0.90, 1.0)   # single colour when "Color by: None"

    def _preview_tag_colors(self, tags):
        """One RGBA per tag, per the preview 'Color by' selector:
        None → all the same; ID → a unique colour per tag; Sex → blue=M, red=F."""
        mode = self.combo_preview_color.currentText() if hasattr(self, 'combo_preview_color') else 'None'
        if mode == 'Sex':
            out = []
            for tag in tags:
                sex = (self.tag_identities.get(tag, {}) or {}).get('sex', 'M')
                out.append(self._SEX_RED if sex == 'F' else self._SEX_BLUE)
            return np.array(out, dtype=float)
        if mode == 'ID':
            cmap = plt.get_cmap('tab20')
            return np.array([cmap(i % 20) for i in range(len(tags))], dtype=float)
        return np.array([self._NEUTRAL] * len(tags), dtype=float)   # None

    def _preview_tag_labels(self, tags, id_type="Display ID"):
        """One label per tag in the requested format.

        • Display ID: sex + identity (e.g. 'M9627'), matching the network /
          proximity labels; falls back to HexID when no identity is configured.
        • HexID: the tag's short address in hex (e.g. '2A').
        • ShortID: the decimal tag id from the SQL database.
        """
        labels = []
        for tag in tags:
            if id_type == "ShortID":
                labels.append(str(tag))
            elif id_type == "HexID":
                labels.append(hex(tag).upper().replace('0X', ''))
            else:  # Display ID (SexID)
                info = self.tag_identities.get(tag, {}) or {}
                sex, ident = info.get('sex', ''), info.get('identity', '')
                labels.append(f"{sex}{ident}" if (sex and ident)
                              else hex(tag).upper().replace('0X', ''))
        return labels

    def on_preview_smoothing_changed(self, *_):
        """Preview smoothing method changed: show/hide the preview Smoothing
        Window controls to match the method, then reload chunks under it.

        Mirrors on_smoothing_changed for the Export combo: None and
        Savitzky-Golay expose no window; the rolling methods expose window +
        units; EWMA exposes the window (as its span) but not the units.
        """
        method = self.combo_preview_smoothing.currentText()
        is_ewma = method == "Forward-Backward Exponentially Weighted Moving Average"
        is_rolling = method in ("Rolling Average", "Rolling Median")
        needs_param = is_ewma or is_rolling
        self.lbl_preview_window.setText("Span (samples):" if is_ewma else "Smoothing Window:")
        self.spin_preview_window.setVisible(needs_param)
        self.lbl_preview_window.setVisible(needs_param)
        self.combo_preview_window_units.setVisible(is_rolling)
        self.invalidate_preview_cache()

    def on_preview_trail_changed(self, *_):
        """Trail length changed: redraw now from cached frames (cheap), and —
        debounced — reload the chunks so their lead-in overlap matches the new
        trail. Without the reload a longer trail would clip at the chunk's
        loaded start; within a chunk the redraw alone already extends the tail."""
        self.render_preview_frame()
        if getattr(self, '_preview_active', False):
            self.preview_trail_timer.start(400)

    def on_preview_trail_points_changed(self, *_):
        """Dots on/off: a pure redraw, with no reason to reload anything.

        The fixes are already in the frame the trail is drawn from; only
        whether they are marked changes.
        """
        canvas = getattr(self, 'preview_canvas_2d', None)
        if canvas is not None:
            canvas.show_trail_points = self.chk_preview_trail_points.isChecked()
        self.render_preview_frame()

    def on_preview_color_changed(self, *_):
        """Recolour the preview without reloading chunks (colour is cheap)."""
        if getattr(self, 'preview_tags', None) is not None and len(self.preview_tags):
            self.preview_colors = self._preview_tag_colors(self.preview_tags)
            self.render_preview_frame()

    def _show_chunk(self, idx):
        """Make a cached chunk the active one and render the playhead frame."""
        c = self.preview_cache.get(idx)
        if c is None:
            return
        self.preview_current_chunk = idx
        self.preview_tags = c["tags"]
        self.preview_x, self.preview_y = c["x"], c["y"]
        self.preview_xf, self.preview_yf = c["xf"], c["yf"]
        self.preview_raw_x, self.preview_raw_y = c["raw_x"], c["raw_y"]
        self.preview_batt = c.get("batt")
        self.preview_step = c.get("step")
        self.preview_speed = c.get("speed")
        self._preview_step_pool = c.get("step_pool")
        self._update_step_stats_label()
        self.preview_times = c["times"]
        self.preview_hz = c["hz"]
        self.preview_colors = self._preview_tag_colors(self.preview_tags)

        # Clamp the playhead into this chunk's actual span
        self.preview_playhead_ms = int(np.clip(
            self.preview_playhead_ms, c["t_start"], c["t_end"]))

        finite = int(np.isfinite(c["x"]).sum())
        self.lbl_preview_status.setText(
            f"{len(c['x'])} frames · {len(c['tags'])} tags · {finite:,} fixes · "
            f"{self.combo_preview_smoothing.currentText()}")
        if self.preview_arena is None:
            self.refresh_preview_arena()
        self.render_preview_frame()
        # Warm the adjacent chunks so the next scrub crossing is already cached.
        self._prefetch_neighbors(idx)

    # -- timeline / playhead ----------------------------------------------- #
    def on_timeline_moved(self, value):
        """Slider moved: update the clock immediately, defer the actual read.

        Dragging emits a continuous stream of values; firing a query on each
        would be exactly the behaviour that made the old preview unusable. The
        debounce means only where you *stop* is ever fetched.
        """
        if self._timeline_guard or self.preview_t0 is None:
            return
        # A drag or an arrow key is the user taking over; playback stops rather
        # than fighting them for the playhead.
        if self._playing and not self._playback_seeking:
            self.stop_preview_playback()
        self.seek_preview_ms(self.preview_t0 + int(value) * 1000)

    def seek_preview_ms(self, ms):
        """Move the playhead to an absolute time and show it.

        Split out of the slider handler because playback needs sub-second
        steps: the preview runs at up to 5 Hz while the slider's unit is one
        second, so driving playback through the slider would quantise it to
        whole seconds and drop four frames in five.
        """
        if self.preview_t0 is None:
            return
        self.preview_playhead_ms = int(ms)
        self._update_time_label()
        # If the playhead is inside ANY resident chunk, redraw instantly — it is
        # just an array index, no DB read. Because we prefetch the neighbouring
        # chunks, a continuous hold stays inside cache and the tags keep moving
        # instead of freezing at the old chunk's edge.
        resident = self._resident_chunk_for(self.preview_playhead_ms)
        if resident is not None:
            if resident != self.preview_current_chunk:
                self._show_chunk(resident)   # also warms the new neighbours
            else:
                self.render_preview_frame()
        elif not self.preview_scrub_timer.isActive():
            # Not resident: throttle a background load. Testing isActive() rather
            # than restarting guarantees the timer still fires while the arrow
            # key auto-repeats, so a sustained hold keeps advancing instead of
            # starving the loader until the key is released.
            self.preview_scrub_timer.start(self.SCRUB_DEBOUNCE_MS)

    def _on_scrub_settled(self):
        self._request_chunk(self._chunk_index_for(self.preview_playhead_ms),
                            make_current=True)

    def _sync_timeline_to_playhead(self):
        if self.preview_t0 is None:
            return
        self._timeline_guard = True
        self.slider_timeline.setValue(
            int((self.preview_playhead_ms - self.preview_t0) // 1000))
        self._timeline_guard = False
        self._update_time_label()

    def _update_time_label(self):
        if self.preview_t0 is None:
            return
        ts = pd.Timestamp(self.preview_playhead_ms, unit='ms', tz='UTC').tz_convert(
            pytz.timezone(self.combo_timezone.currentText()))
        cached = "" if self.preview_current_chunk in self.preview_cache else "  (loading…)"
        self.lbl_preview_time.setText(f"{ts:%Y-%m-%d %H:%M:%S}{cached}")

    def _frame_index(self):
        """Index into the active chunk for the current playhead time."""
        if self.preview_times is None or not len(self.preview_times):
            return 0
        target = pd.Timestamp(self.preview_playhead_ms, unit='ms', tz='UTC')
        i = int(np.searchsorted(self.preview_times.tz_convert('UTC').values,
                                np.datetime64(target.tz_localize(None), 'ns')))
        return int(np.clip(i, 0, len(self.preview_times) - 1))

    # -- rendering --------------------------------------------------------- #
    def on_show_behavior_toggled(self, *_):
        """Reveal the per-behaviour controls and redraw."""
        if hasattr(self, "behavior_options_widget"):
            self.behavior_options_widget.setVisible(self.chk_show_behavior.isChecked())
        self.render_preview_frame()

    def _hide_inherited_export_rows(self):
        """Hide the export rows that now come from the preview.

        The widgets stay alive rather than being deleted: the export pipeline,
        the saved config and the run summary all read them, and keeping them in
        sync means the config still records exactly what a run used.
        """
        for layout in getattr(self, "_inherited_rows", []):
            for i in range(layout.count()):
                item = layout.itemAt(i)
                wdg = item.widget() if item is not None else None
                if wdg is not None:
                    wdg.setVisible(False)
        if hasattr(self, "proximity_threshold_widget"):
            self.proximity_threshold_widget.setVisible(False)
        self._sync_export_from_preview()

    def _sync_export_from_preview(self):
        """Mirror the preview's processing settings onto the export widgets.

        The preview is the single source of truth for how a track is filtered
        and smoothed, so what you tuned while looking at the data is exactly
        what gets exported.
        """
        try:
            self.chk_velocity_filter.setChecked(self.chk_preview_velocity.isChecked())
            self.spin_velocity_threshold.setValue(self.spin_preview_velocity.value())
            self.chk_jump_filter.setChecked(self.chk_preview_jump.isChecked())
            self.spin_jump_threshold.setValue(self.spin_preview_jump.value())
            # The Hampel threshold and the velocity time base belong here too.
            # Leaving them out did not fall back to the preview's values - it
            # silently exported whatever the hidden widgets last held, so a
            # trial previewed with Hampel on and a 0.5 s time base exported
            # with neither. Every threshold the preview owns must be mirrored.
            self.chk_outlier_filter.setChecked(self.chk_preview_outlier.isChecked())
            self.spin_outlier_window.setValue(self.spin_preview_outlier_window.value())
            self.spin_outlier_sigma.setValue(self.spin_preview_outlier_sigma.value())
            self.spin_velocity_min_dt.setValue(
                round(self.spin_preview_velocity_min_dt.value(), 2))
            self.chk_adjacency_filter.setChecked(
                self.chk_preview_adjacency.isChecked())
            self.spin_excursion_s.setValue(
                round(self.spin_preview_excursion.value(), 1))
            self.combo_smoothing.setCurrentText(self.combo_preview_smoothing.currentText())
            self.spin_rolling_window.setValue(self.spin_preview_window.value())
            self.combo_window_units.setCurrentText(
                self.combo_preview_window_units.currentText())
            # Proximity is parameterised in the preview as a per-animal social
            # radius; the exporter still works in centre-to-centre distance.
            if hasattr(self, "spin_social_radius"):
                self.spin_proximity_threshold.setValue(
                    2.0 * self.spin_social_radius.value())
            # The rendered animation should look like the preview it came from.
            self.spin_animation_trail.setValue(self.spin_preview_trail.value())
            self.spin_anim_tag_size.setValue(self.spin_tag_size.value())
            self.chk_show_battery_export.setChecked(
                self.chk_show_battery.isChecked())
            self._sync_anim_show_options()
            # The two combos spell the same option differently ("Sex" vs "sex"),
            # so match case-insensitively rather than failing silently.
            want = self.combo_preview_color.currentText().strip().lower()
            for i in range(self.combo_color_by.count()):
                if self.combo_color_by.itemText(i).strip().lower() == want:
                    self.combo_color_by.setCurrentIndex(i)
                    break
        except Exception:
            pass

    # ── Regions of interest ─────────────────────────────────────────────
    def selected_roi(self):
        """The ROI currently highlighted in the list, or None."""
        i = self.roi_list.currentRow()
        return self.rois[i] if 0 <= i < len(self.rois) else None

    def _refresh_roi_list(self, keep=None):
        """Rebuild the list rows, restoring the selection where possible."""
        from fnt.uwb import uwb_roi as ROI

        row = self.roi_list.currentRow() if keep is None else keep
        # Where regions overlap, the list ORDER decides which one an animal is
        # credited to, so the order stops being cosmetic and has to be shown.
        # With no overlaps it means nothing, and numbering every row would be
        # noise, so the prefix appears exactly when it starts to matter.
        overlaps = ROI.overlapping_pairs(self.rois)
        entangled = {i for pair in overlaps for i in pair}
        self.roi_list.blockSignals(True)
        self.roi_list.clear()
        for i, r in enumerate(self.rois):
            text = ROI.describe_roi(r)
            if overlaps:
                text = f"{i + 1} · {text}"
                if i in entangled:
                    text += "  ⬚ overlaps"
            item = QListWidgetItem(text)
            item.setForeground(QColor(r.get("color", ROI.DEFAULT_ROI_COLOR)))
            if i in entangled:
                item.setToolTip(
                    "This region overlaps another. A fix inside both is "
                    "credited to whichever appears FIRST in this list, and to "
                    "that one only — it is never counted twice.")
            self.roi_list.addItem(item)
        if 0 <= row < len(self.rois):
            self.roi_list.setCurrentRow(row)
        self.roi_list.blockSignals(False)
        self._sync_roi_style_controls()
        self.btn_roi_rename.setEnabled(bool(self.rois))
        self.btn_roi_delete.setEnabled(bool(self.rois))

    def _sync_roi_style_controls(self):
        """Point the colour swatch and width box at the current selection.

        With nothing selected they show the style the NEXT region will be
        drawn in, so the controls always mean something. This only READS
        ``_roi_next_color``: writing the selection back into it would make
        every new region inherit the colour of whichever row happened to be
        highlighted, instead of taking the next free one from the palette.
        """
        from fnt.uwb import uwb_roi as ROI

        roi = self.selected_roi()
        colour = (roi or {}).get("color") or self._pending_roi_color()
        width = (roi or {}).get("linewidth", self.spin_roi_width.value())
        self.btn_roi_color.setStyleSheet(
            f"QPushButton {{ background-color: {colour}; border: 1px solid "
            f"#888888; border-radius: 3px; min-width: 0px; }}")
        self.spin_roi_width.blockSignals(True)
        self.spin_roi_width.setValue(float(width))
        self.spin_roi_width.blockSignals(False)

    def _on_roi_selected(self, _row=None):
        self._sync_roi_style_controls()

    def _on_roi_width_changed(self, value):
        roi = self.selected_roi()
        if roi is not None:
            roi["linewidth"] = float(value)
            self.on_rois_changed()

    def _pending_roi_color(self):
        """The colour the next region will be drawn in.

        A colour the user picked with nothing selected, otherwise the first
        palette entry no existing region is using.
        """
        from fnt.uwb import uwb_roi as ROI

        return getattr(self, "_roi_next_color", None) or ROI.next_roi_color(self.rois)

    def pick_roi_color(self):
        """Recolour the selected region, or choose the colour for the next."""
        roi = self.selected_roi()
        start = QColor((roi or {}).get("color") or self._pending_roi_color())
        picked = QColorDialog.getColor(start, self, "Region colour")
        if not picked.isValid():
            return
        if roi is not None:
            roi["color"] = picked.name()
        else:
            # Nothing selected, so the pick is for the region about to be drawn.
            self._roi_next_color = picked.name()
        self._refresh_roi_list()
        self.on_rois_changed()

    def start_roi_draw(self):
        """Hand the preview's left button over to placing corners."""
        canvas = getattr(self, "preview_canvas_2d", None)
        if canvas is None:
            return
        if getattr(canvas, "_roi_mode", False):
            canvas.cancel_roi()             # the button doubles as Cancel
            return
        canvas.begin_roi(self._pending_roi_color(), self.spin_roi_width.value())
        canvas.setFocus()                   # so Enter and Esc reach the canvas
        self.btn_roi_draw.setText("Cancel")
        self.lbl_roi_hint.setVisible(True)
        self._on_roi_draft(0)

    def _on_roi_draft(self, n):
        """Keep the hint honest about what the next click will do."""
        need = max(0, 3 - n)
        self.lbl_roi_hint.setText(
            f"Placing corners: {n} so far"
            + (f", {need} more needed" if need else " \u2014 ready to close")
            + ".  Left click places \u00b7 right click undoes \u00b7 scroll "
              "zooms \u00b7 middle-drag pans \u00b7 double-click or Enter "
              "closes \u00b7 Esc cancels.")

    def _on_roi_completed(self, points):
        """A finished polygon becomes a named region."""
        from fnt.uwb import uwb_roi as ROI

        self._end_roi_draw_ui()
        name = ROI.unique_roi_name(self.rois)
        self.rois.append(ROI.make_roi(name, points, self._pending_roi_color(),
                                      self.spin_roi_width.value()))
        # The pick is spent. Clearing it sends the next region back to the
        # palette, so drawing two in a row gives two distinguishable colours
        # without a trip to the picker.
        self._roi_next_color = None
        self._refresh_roi_list(keep=len(self.rois) - 1)
        self.roi_list.setCurrentRow(len(self.rois) - 1)
        self.log_message(
            f"Added region '{name}': {len(points)} corners, "
            f"{ROI.polygon_area(points):.2f} m\u00b2")
        self.on_rois_changed()
        self.write_live_config()
        self._resolve_roi_overlaps(len(self.rois) - 1)

    def ask_roi_precedence(self, name_a, name_b):
        """Which of two overlapping regions wins: 'a', 'b', or None to leave it.

        The decision is a method of its own so the resolution logic can be
        driven without a modal in the way. A QMessageBox INSTANCE cannot be
        stubbed the way the static helpers can, and one appearing inside an
        automated pass blocks it rather than failing it.
        """
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Regions overlap")
        box.setText(
            f"'{name_a}' and '{name_b}' share area.\n\n"
            f"An animal inside both is credited to ONE of them — never to "
            f"both. Which takes precedence?")
        btn_a = box.addButton(f"'{name_a}' wins", QMessageBox.AcceptRole)
        btn_b = box.addButton(f"'{name_b}' wins", QMessageBox.AcceptRole)
        box.addButton("Leave as is", QMessageBox.RejectRole)
        box.setDefaultButton(btn_a)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is btn_a:
            return 'a'
        if clicked is btn_b:
            return 'b'
        return None

    def _data_view_signature(self):
        """What the cached Data View series depend on.

        Deliberately narrow: the series are a function of which animals, which
        regions, and how the track was filtered and smoothed. The animation's
        own settings (speed, fps, trail) change none of it, so re-rendering a
        clip at a different speed must not pay for the trial pass again.
        """
        from fnt.uwb import uwb_roi as ROI

        return (
            self.db_path, self.table_name,
            tuple(sorted(self.selected_preview_tags())),
            tuple((r.get('name'), ROI.polygon_area(r.get('points')))
                  for r in (self.rois or [])),
            self.combo_smoothing.currentText(),
            self.spin_rolling_window.value(), self.combo_window_units.currentText(),
            self.chk_velocity_filter.isChecked(), self.spin_velocity_threshold.value(),
            self.spin_velocity_min_dt.value(),
            self.chk_jump_filter.isChecked(), self.spin_jump_threshold.value(),
            self.chk_outlier_filter.isChecked(), self.spin_outlier_window.value(),
            self.spin_outlier_sigma.value(),
            self.spin_proximity_threshold.value(),
            self.combo_timezone.currentText(),
        )

    def invalidate_data_view_cache(self):
        self._data_view_cache = None

    def _data_view_tag_frames(self):
        """Yield (tag, processed frame) one tag at a time.

        ``self.data`` only exists once an EXPORT has built it, so relying on it
        made the panel silently vanish from the interactive 10 s clip - the
        payload raised, the caller downgraded, and the video rendered without
        it. The counters are trial-relative, so the source has to be the whole
        recording either way.

        One tag at a time on purpose: the processed trial is the size of the
        smoothed CSV, and holding it whole is the shape of the out-of-memory
        crashes this pipeline has produced before. Each tag's bouts are taken
        and its frame dropped.
        """
        if self.data is not None and not self.data.empty:
            id_col = 'shortid' if 'shortid' in self.data.columns else 'ID'
            for tag in sorted(self.data[id_col].unique()):
                yield tag, self.data[self.data[id_col] == tag]
            return

        tags = list(self.selected_preview_tags())
        if not tags:
            raise RuntimeError("no tags selected")
        db = self.current_indexed_db() or self.preview_db_path or self.db_path
        if not db:
            raise RuntimeError("no database loaded")
        self.log_message(
            f"  Data View: no export in memory — reading {len(tags)} tag(s) "
            f"from {os.path.basename(db)}.")
        # _processed_slice publishes the threshold read-out's sample as a side
        # effect. That read-out describes the chunk IN VIEW and says so, so a
        # trial-wide pass must not leave its numbers behind.
        saved_pool = getattr(self, '_preview_step_pool', None)
        conn = connect_ro(db)
        try:
            cols = processing_select_clause(conn, self.table_name)
            for tag in tags:
                if self.export_cancelled:
                    raise RuntimeError("cancelled")
                raw = pd.read_sql_query(
                    f"SELECT {cols} FROM {self.table_name} WHERE shortid = ? "
                    f"ORDER BY timestamp", conn, params=(tag,))
                if not len(raw):
                    continue
                raw, _ = collapse_multi_solutions(raw)
                frame = self._processed_slice(raw)
                del raw
                if frame is not None and len(frame):
                    yield tag, frame
                del frame
        finally:
            conn.close()
            self._preview_step_pool = saved_pool

    def build_data_view_payload(self, force=False):
        """Trial-wide cumulative series for the animation's Data View panel.

        Built over the WHOLE trial regardless of which clip is being rendered,
        because the counters are trial-relative: scrubbing to the end and
        rendering ten seconds has to show the trial's real totals, not the
        clip's. That makes the first build expensive on a long recording, so
        it is cached against everything it depends on and reused.
        """
        from fnt.uwb import dataview as DV
        from fnt.uwb import roi_bouts as RB
        from fnt.uwb import spatial_metrics as SM

        sig = self._data_view_signature()
        cached = getattr(self, '_data_view_cache', None)
        if not force and cached is not None and cached[0] == sig:
            return cached[1]

        rois = list(self.rois or [])
        if not rois:
            raise RuntimeError("no regions of interest drawn")

        t_all = time.time()
        self.log_message(
            "Data View: building trial-wide counters — this reads the whole "
            "recording once, then is reused for every clip.")
        QApplication.processEvents()

        names = [r.get('name', f'ROI {i + 1}') for i, r in enumerate(rois)]

        # Bouts are gathered per ANIMAL, not per tag, and the curves are built
        # once at the end. Two tags can carry the same ID - a lost tag replaced
        # mid-trial is deliberately one animal - and building a series per tag
        # would let the second tag's curve REPLACE the first's, silently
        # discarding everything the animal did before the swap.
        animals, seen = [], set()
        by_animal = {}
        n = 0
        for tag, sub in self._data_view_tag_frames():
            n += 1
            if self.export_cancelled:
                raise RuntimeError("cancelled")
            info = (self.tag_identities or {}).get(tag) or {}
            label = f"{info.get('sex', '')}{info.get('identity', tag)}" or str(tag)
            x_col = 'smoothed_x' if 'smoothed_x' in sub.columns else 'location_x'
            y_col = 'smoothed_y' if 'smoothed_y' in sub.columns else 'location_y'
            sub = sub.sort_values('Timestamp')
            ts = sub['Timestamp']
            t_s = ts.values.astype('datetime64[ns]').astype('int64') / 1e9
            zi = SM.assign_zones(sub[x_col].values, sub[y_col].values, rois)
            # ONE definition of a visit, shared with the exported bout CSV.
            # The panel exists to be checked against that file, so it must not
            # hold a second opinion: region_bouts is the same segmentation and
            # the same duration rule (span, with a lone fix credited 1 s), and
            # calling it here is what stops the two drifting apart. Crediting
            # each fix its sampling interval instead - what this used to do -
            # ran exactly one second per visit ahead of the CSV.
            t_ns = ts.values.astype('datetime64[ns]').astype('int64')
            for r_i, t0, t1, secs, _n in RB.region_bouts(zi, t_ns):
                by_animal.setdefault((label, r_i), []).append(
                    (t0 / 1e9, t1 / 1e9, secs))
            if label not in seen:
                seen.add(label)
                animals.append(label)
            self.log_message(f"  Data View: tag {n} — {label} "
                             f"({len(sub):,} fixes)")
            QApplication.processEvents()

        # Rows read top to bottom, so they are ordered for a reader rather
        # than left in tag order - which is the order the shortids happen to
        # sit in and looks jumbled beside the IDs it displays.
        animals.sort(key=DV.natural_key)

        occupancy = {}
        for label in animals:
            for r in range(len(rois)):
                occupancy[(label, r)] = DV.CumulativeSeries.from_bouts(
                    by_animal.get((label, r), []))

        # No social overlap. Recomputing proximity for a preview would give
        # the trial a SECOND opinion on who was together and for how long,
        # competing with the exported bouts - and detecting that kind of
        # disagreement is what this panel is for. _data_view_overlaps() is
        # kept, and DataViewPanel still renders either dyad view, for whenever
        # the exported bouts are wanted here.
        payload = {
            'animals': animals,
            'roi_names': names,
            'occupancy': occupancy,
            'overlaps': {},
            'dyad_mode': 'none',
            'text_color': '#e6e6e6',
            'bg_color': '#1b1b1b',
        }
        self._data_view_cache = (sig, payload)
        self.log_message(
            f"Data View ready: {len(animals)} animals x {len(names)} regions "
            f"({time.time() - t_all:.1f}s).")
        return payload

    def _data_view_overlaps(self):
        """Dyad overlap series, taken from the exported proximity bouts.

        The panel must not hold a second opinion about who was together and
        for how long. So it does not measure proximity at all: it reads the
        very rows that become ``_SocialOverlapBouts.csv``.

        Preferred source is the exported FILE, when this trial has one - then
        the cells are provably the file's own numbers, with no chance of a
        parameter drifting between the two. Failing that (an interactive clip
        rendered before anything has been exported) the same detector is run
        on the same data, which produces the same rows.
        """
        from fnt.uwb import dataview as DV
        from fnt.uwb.proximity_detection import detect_proximity_bouts

        tz = self.combo_timezone.currentText()
        folder = self.analysis_dir_for()
        db_name = (os.path.splitext(os.path.basename(self.db_path))[0]
                   if self.db_path else None)
        if folder and db_name:
            path = os.path.join(folder, f'{db_name}_SocialOverlapBouts.csv')
            if os.path.exists(path):
                try:
                    bouts = pd.read_csv(path)
                    self.log_message(
                        f"  Data View: dyad counters read from "
                        f"{os.path.basename(path)} ({len(bouts)} bouts).")
                    return DV.series_from_bouts(
                        DV.overlap_bouts_from_frame(bouts, tz))
                except Exception as e:
                    self.log_message(
                        f"  Data View: could not read {os.path.basename(path)} "
                        f"({e}); recomputing.")

        if self.data is None or self.data.empty:
            # Proximity is pairwise, so unlike occupancy it cannot be done a
            # tag at a time - it needs every animal's track at once, which is
            # the whole trial in memory. Not worth doing for a 10 s clip, and
            # not worth risking the memory for. The ROI half still works.
            self.log_message(
                "  Data View: dyad counters need the proximity bouts, which "
                "only an export produces — showing ROI occupancy only. Run an "
                "export (or one with Social Overlap Bouts ticked) and the "
                "dyad grid fills in.")
            return {}

        self.log_message(
            "  Data View: no exported bouts yet — running the same proximity "
            "detector the export uses.")
        QApplication.processEvents()
        _events, bouts = detect_proximity_bouts(
            self.data,
            threshold=float(self.spin_proximity_threshold.value()),
            gap_s=5,
            tag_identities=self.tag_identities,
            log_callback=None)
        return DV.series_from_bouts(DV.overlap_bouts_from_frame(bouts, tz))

    def _resolve_roi_overlaps(self, index):
        """Ask which region wins where ``index`` overlaps another, and reorder.

        Overlap is resolved by PRECEDENCE, never by double counting: a fix
        inside two regions is credited to whichever is tested first, so the
        answer is a position in the list. Asking at the moment the overlap is
        created is the point — discovering it later, from an occupancy total
        that quietly went to the wrong region, is exactly what this prevents.

        Returns True if the order changed.
        """
        from fnt.uwb import uwb_roi as ROI

        if getattr(self, '_batch_active', False):
            return False
        changed = False
        # Pairs are re-derived each pass, because resolving one reorders the
        # list and invalidates every index. They are also remembered, by
        # IDENTITY rather than position: choosing a winner does not separate
        # the regions, so an already-answered pair keeps reappearing and would
        # be asked forever. Marking each pair decided is what terminates this.
        seen = set()
        while True:
            nxt = None
            for i, j in ROI.overlapping_pairs(self.rois):
                if index not in (i, j):
                    continue
                key = frozenset((id(self.rois[i]), id(self.rois[j])))
                if key not in seen:
                    nxt = (i, j, key)
                    break
            if nxt is None:
                break
            i, j, key = nxt
            seen.add(key)
            a, b = self.rois[i], self.rois[j]
            name_a = a.get('name', f'#{i + 1}')
            name_b = b.get('name', f'#{j + 1}')
            answer = self.ask_roi_precedence(name_a, name_b)
            if answer == 'a':
                winner, loser = i, j
            elif answer == 'b':
                winner, loser = j, i
            else:
                # Left alone deliberately: say what that means and move on to
                # any other overlap, so a shrug is still an informed one.
                self.log_message(
                    f"  '{name_a}' and '{name_b}' overlap; the shared area "
                    f"goes to '{self.rois[min(i, j)].get('name', '')}' "
                    f"(first in the list).")
                continue
            # Read the winner's name BEFORE reordering: set_precedence moves
            # list positions, so i and j stop pointing at these two regions
            # the moment it returns.
            winner_name = name_a if winner == i else name_b
            if winner != min(i, j):
                keep = self.rois[index]
                self.rois = ROI.set_precedence(self.rois, winner, loser)
                index = self.rois.index(keep)
                changed = True
                self.log_message(
                    f"  '{winner_name}' now takes precedence: shared area is "
                    f"credited to it.")
            else:
                self.log_message(
                    f"  '{winner_name}' already takes precedence; order "
                    f"unchanged.")
        if changed:
            self._refresh_roi_list(keep=index)
            self.on_rois_changed()
            self.write_live_config()
        return changed

    def _on_roi_cancelled(self):
        self._end_roi_draw_ui()

    def _end_roi_draw_ui(self):
        self.btn_roi_draw.setText("Draw new")
        self.lbl_roi_hint.setVisible(False)

    def rename_roi(self):
        roi = self.selected_roi()
        if roi is None:
            return
        name, ok = QInputDialog.getText(self, "Rename region", "Name:",
                                        text=roi.get("name", ""))
        if not ok or not name.strip():
            return
        roi["name"] = name.strip()
        self._refresh_roi_list()
        self.on_rois_changed()
        self.write_live_config()

    def delete_roi(self):
        roi = self.selected_roi()
        if roi is None:
            return
        # Never leave the canvas mid-drag on a region that is about to vanish.
        self.preview_canvas_2d.end_roi_edit()
        if QMessageBox.question(
                self, "Delete region",
                f"Delete '{roi.get('name', '')}'?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        row = self.roi_list.currentRow()
        self.rois.pop(row)
        self.log_message(f"Deleted region '{roi.get('name', '')}'")
        self._refresh_roi_list(keep=min(row, len(self.rois) - 1))
        self.on_rois_changed()
        self.write_live_config()

    # The XML's own fallback when a zone carries no colour of its own. Every
    # zone then arrives the same grey, which is useless for telling eight of
    # them apart in the preview, so those fall back to the ROI palette.
    XML_ZONE_DEFAULT_COLOR = '#888888'

    def import_xml_zones(self):
        """Copy the site XML's zones in as ordinary drawn regions.

        The occupancy products measure drawn regions only, so this is the one
        route from an authored zone to a statistic. It is a COPY, taken once:
        nothing here is written back to the XML, and an imported region is
        thereafter indistinguishable from one drawn by hand — editable,
        reorderable and deletable, and stored in the trial's fnt_config.json.

        Re-importing is safe. A zone whose name is already in the list is
        skipped rather than duplicated or overwritten, so pressing the button
        twice cannot silently discard an edited boundary.
        """
        from fnt.uwb import uwb_roi as ROI

        zones = list(self.xml_zones or [])
        if not zones:
            QMessageBox.information(
                self, "No XML zones",
                "This trial's site XML defines no zones to import.\n\n"
                "Zones come from the XML that sits beside the database. If "
                "you expected some, check that the XML loaded — the preview's "
                "'Show zone coordinates' toggle is greyed out when it did "
                "not.")
            return

        taken = {str(r.get('name', '')).strip().lower() for r in self.rois}
        fresh = [z for z in zones
                 if str(z.get('name', '')).strip().lower() not in taken]
        skipped = len(zones) - len(fresh)
        if not fresh:
            QMessageBox.information(
                self, "Already imported",
                f"All {len(zones)} XML zone(s) are already in the region "
                f"list, so there is nothing to import.\n\n"
                f"Delete a region first if you want to take a fresh copy of "
                f"that zone from the XML.")
            return

        msg = (f"Import {len(fresh)} zone(s) from the site XML as regions of "
               f"interest?")
        if skipped:
            msg += (f"\n\n{skipped} more are already in the list by name and "
                    f"will be left alone.")
        msg += ("\n\nThey become ordinary regions: edit, rename, recolour or "
                "delete them freely. The XML is not modified.")
        if QMessageBox.question(self, "Import XML zones", msg,
                                QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.Yes) != QMessageBox.Yes:
            return

        # Held by identity, not position: resolving an overlap reorders the
        # list, so the indices collected here would not survive the first
        # answer.
        added = []
        for z in fresh:
            try:
                pts = np.asarray(z.get('points'), dtype=float).reshape(-1, 2)
            except (TypeError, ValueError):
                continue
            if len(pts) < ROI.MIN_ROI_POINTS or not np.isfinite(pts).all():
                self.log_message(
                    f"  Skipped zone '{z.get('name', '')}': not a usable "
                    f"polygon.")
                continue
            color = str(z.get('color') or '').strip()
            if not color or color.lower() == self.XML_ZONE_DEFAULT_COLOR:
                color = ROI.next_roi_color(self.rois + added)
            roi = ROI.make_roi(z.get('name') or
                               ROI.unique_roi_name(self.rois + added),
                               pts, color, self.spin_roi_width.value())
            added.append(roi)
        if not added:
            self.log_message("No XML zones were usable as regions.")
            return

        self.rois.extend(added)
        for roi in added:
            self.log_message(
                f"Imported zone '{roi['name']}' as a region: "
                f"{len(roi['points'])} corners, "
                f"{ROI.polygon_area(roi['points']):.2f} m²")
        self.log_message(
            f"✓ Imported {len(added)} XML zone(s) as regions"
            + (f"; {skipped} already present were skipped." if skipped
               else "."))
        self._refresh_roi_list(keep=len(self.rois) - 1)
        self.roi_list.setCurrentRow(len(self.rois) - 1)
        self.on_rois_changed()
        self.write_live_config()
        # Authored zones routinely abut or overlap, and may overlap regions
        # already drawn. Each one is resolved as if it had just been drawn,
        # re-finding its position every time because the previous answer may
        # have moved it.
        for roi in added:
            try:
                self._resolve_roi_overlaps(self.rois.index(roi))
            except ValueError:      # removed mid-loop; nothing to resolve
                continue

    def on_rois_changed(self):
        """Push the current regions to the preview and refresh the list text."""
        canvas = getattr(self, "preview_canvas_2d", None)
        if canvas is not None:
            canvas.set_rois(self.rois, self.chk_show_rois.isChecked())
        # The "Show regions of interest" toggle greys out when there is
        # nothing to show, like the other layer toggles, and the occupancy
        # source list gains or loses the drawn-regions option with it.
        self._sync_layer_toggles()
        # The Data View counts time in regions, so with none drawn it is not
        # an option that happens to be unavailable - it is not an option at
        # all. Hidden rather than greyed, matching the two ROI export boxes,
        # which gate on the same thing.
        if hasattr(self, 'chk_anim_data_view'):
            has_rois = bool(self.rois)
            self.chk_anim_data_view.setVisible(has_rois)
            if not has_rois:
                # Cleared as well as hidden: unlike the ROI export settings
                # this one is not carried in fnt_config.json, so there is no
                # inherited intent to preserve - and a hidden box left ticked
                # would ask the renderer for a panel with nothing to count.
                self.chk_anim_data_view.setChecked(False)
        if hasattr(self, 'chk_export_roi_bouts'):
            self.on_zone_occupancy_toggled()
        # The row text carries the corner count and area, so a width or colour
        # change does not need it rebuilt - but a rename or a new region does.
        if self.roi_list.count() != len(self.rois):
            self._refresh_roi_list()

    def show_roi_menu(self, index):
        """Right-click menu for the region under the cursor."""
        if not (0 <= index < len(self.rois)):
            return
        roi = self.rois[index]
        self.roi_list.setCurrentRow(index)      # so the panel agrees with the menu

        menu = QMenu(self)
        menu.addSection(roi.get("name", "Region"))
        act_move = menu.addAction("Move")
        act_move.setToolTip("Drag the whole region to a new place.")
        act_edit = menu.addAction("Modify points")
        act_edit.setToolTip("Drag individual corners to reshape it.")
        act_copy = menu.addAction("Copy")
        act_copy.setToolTip(
            "Duplicate it just to the right, ready to drag into place.")
        menu.addSeparator()
        act_rename = menu.addAction("Rename...")
        act_colour = menu.addAction("Colour...")
        menu.addSeparator()
        act_delete = menu.addAction("Delete")

        chosen = menu.exec_(QCursor.pos())
        if chosen is None:
            return
        if chosen is act_move:
            self.start_roi_move(index)
        elif chosen is act_edit:
            self.start_roi_modify(index)
        elif chosen is act_copy:
            self.copy_roi(index)
        elif chosen is act_rename:
            self.rename_roi()
        elif chosen is act_colour:
            self.pick_roi_color()
        elif chosen is act_delete:
            self.delete_roi()

    def start_roi_move(self, index=None):
        index = self.roi_list.currentRow() if index is None else index
        if not (0 <= index < len(self.rois)):
            return
        self.preview_canvas_2d.begin_roi_move(index)
        self._show_roi_edit_hint(
            f"Moving '{self.rois[index].get('name', '')}': drag it anywhere "
            "\u00b7 scroll zooms \u00b7 middle-drag pans \u00b7 Enter, Esc or "
            "right-click finishes.")

    def start_roi_modify(self, index=None):
        index = self.roi_list.currentRow() if index is None else index
        if not (0 <= index < len(self.rois)):
            return
        self.preview_canvas_2d.begin_roi_edit(index)
        self._show_roi_edit_hint(
            f"Reshaping '{self.rois[index].get('name', '')}': drag a corner "
            "\u00b7 scroll zooms \u00b7 middle-drag pans \u00b7 Enter, Esc or "
            "right-click finishes.")

    def copy_roi(self, index=None):
        """Duplicate a region just to the right, already in move mode.

        Offset rather than placed on top of the original, because two
        identical polygons at the same coordinates are indistinguishable and
        the copy would look like nothing happened. Move mode starts straight
        away: the reason to copy a region is to put the same shape somewhere
        else, so the next thing wanted is always the drag.
        """
        from fnt.uwb import uwb_roi as ROI

        index = self.roi_list.currentRow() if index is None else index
        if not (0 <= index < len(self.rois)):
            return
        src = self.rois[index]
        pts = np.asarray(src.get("points"), dtype=float).reshape(-1, 2)
        # Copying "Nest 1" should give "Nest 2", not "Nest 1 1", so a trailing
        # index on the source name is dropped before a fresh one is chosen.
        base = re.sub(r"\s+\d+$", "", str(src.get("name", "ROI"))) or "ROI"
        # A quarter of its own width reads as "next to it" for a big region
        # and for a small one alike; the floor keeps a tiny region from
        # landing back on itself.
        offset = max(0.25 * float(pts[:, 0].max() - pts[:, 0].min()), 0.2)
        name = ROI.unique_roi_name(self.rois, base=base)
        copy = ROI.make_roi(name, pts + (offset, 0.0),
                            src.get("color"), src.get("linewidth"))
        self.rois.append(copy)
        self._refresh_roi_list(keep=len(self.rois) - 1)
        self.log_message(
            f"Copied '{src.get('name', '')}' to '{name}' "
            f"({offset:.2f} m to the right) \u2014 drag it into place.")
        self.on_rois_changed()
        self.write_live_config()
        self.start_roi_move(len(self.rois) - 1)

    def on_roi_edited(self, index):
        """A region was dragged or reshaped: keep the panel and file in step."""
        if not (0 <= index < len(self.rois)):
            return
        self._refresh_roi_list(keep=index)
        self.on_rois_changed()
        self.write_live_config()

    def _show_roi_edit_hint(self, text):
        self.lbl_roi_hint.setText(text)
        self.lbl_roi_hint.setVisible(True)

    def _on_roi_edit_ended(self):
        from fnt.uwb import uwb_roi as ROI

        roi = self.selected_roi()
        if roi is not None:
            self.log_message(
                f"Region '{roi.get('name', '')}' now covers "
                f"{ROI.polygon_area(roi['points']):.2f} m\u00b2")
        self.lbl_roi_hint.setVisible(False)
        self._refresh_roi_list()
        self.write_live_config()
        # Only once the drag has FINISHED. A copy is deliberately dropped on
        # top of its source and then dragged clear, so asking mid-move would
        # prompt about an overlap the user is in the middle of removing.
        if roi is not None:
            self._resolve_roi_overlaps(self.roi_list.currentRow())

    def _roi_busy(self):
        """Is the preview mid-region - placing corners, moving, or editing?"""
        canvas = getattr(self, 'preview_canvas_2d', None)
        return bool(canvas is not None
                    and (canvas._roi_mode or canvas._roi_action))

    def _wire_roi_canvas(self):
        """Connect the preview canvas's ROI signals, once."""
        canvas = getattr(self, "preview_canvas_2d", None)
        if canvas is None or getattr(self, "_roi_canvas_wired", False):
            return
        canvas.roi_draft_changed.connect(self._on_roi_draft)
        canvas.roi_completed.connect(self._on_roi_completed)
        canvas.roi_cancelled.connect(self._on_roi_cancelled)
        canvas.roi_context_requested.connect(self.show_roi_menu)
        canvas.roi_edited.connect(self.on_roi_edited)
        canvas.roi_edit_ended.connect(self._on_roi_edit_ended)
        self._roi_canvas_wired = True
        canvas.set_rois(self.rois, self.chk_show_rois.isChecked())

    # -- region occupancy --------------------------------------------------- #
    # Two independent sets of polygons, deliberately never merged into one:
    #   'zone' - authored in the Wiser site XML
    #   'roi'  - drawn by the user over the preview
    # They have different provenance and routinely overlap (an ROI is often
    # drawn to correct a zone), so a single column would have to invent a
    # precedence rule and would silently lose one of the two answers. Within a
    # set, assign_zones tests in order and the first match wins.
    # Each entry is (column, source attribute, the checkbox that selects it).
    # Drawn regions ONLY. Site-XML zones were a second, parallel region
    # source with its own column and its own outputs; they are now imported
    # into the ROI list instead ("Import XML"), so there is one pipeline and
    # one set of files. Within a set, assign_zones tests in order and the
    # first match wins, which is what makes precedence a list position.
    REGION_SETS = (('roi', 'rois', None),)

    def region_sets(self, selected_only=True):
        """The region sets to measure, as (column, list of regions).

        A set appears only if it has polygons AND its source is selected.
        ``selected_only=False`` ignores the checkboxes and reports what the
        trial HAS, which is what the availability messages need.
        """
        out = []
        for column, attr, chk_name in self.REGION_SETS:
            # chk_name is None for a set with no source checkbox of its own:
            # drawn regions are measured whenever the export is ticked, since
            # there is no longer a second source to choose between.
            if selected_only and chk_name:
                chk = getattr(self, chk_name, None)
                if chk is not None and not chk.isChecked():
                    continue
            regions = [r for r in (getattr(self, attr, None) or [])
                       if _polygon_len(r) >= 3]
            if regions:
                out.append((column, regions))
        return out

    def on_zone_occupancy_toggled(self, _state=None):
        """Show the ROI exports only once there are regions to measure.

        HIDDEN, not greyed, when nothing is drawn: these two files measure
        time spent in drawn regions, so with none drawn they are not an option
        that happens to be unavailable, they are not an option at all. They
        appear the moment a region is drawn or imported.

        The CHECKED state is deliberately left alone while hidden. These are
        analysis settings that ride along in fnt_config.json and are inherited
        between trials, so clearing them here would silently turn ROI export
        off for every later trial that DOES have regions.
        """
        regions = self.region_sets()
        have_regions = bool(regions)
        for cb in (self.chk_export_roi_bouts, self.chk_export_roi_daily):
            cb.setVisible(have_regions)

        # With the boxes gone there is nothing for the note to qualify, so it
        # goes too; it only ever says what the visible boxes will measure.
        on = have_regions and (self.chk_export_roi_bouts.isChecked()
                               or self.chk_export_roi_daily.isChecked())
        self.occ_source_widget.setVisible(on)
        if on:
            n = sum(len(r) for _c, r in regions)
            self.lbl_occ_sources.setText(f"Measuring {n} drawn region(s).")

    def annotate_regions(self, frame, sets):
        """Add a name column per region set to ``frame``, in place.

        A fix outside every region in a set gets the empty string rather than
        NaN, so the CSV column reads as a category and not as missing data -
        being outside is an observation, not a gap.

        Returns {column: zone_idx array} so the caller can compute occupancy
        without paying for the point-in-polygon test twice.
        """
        from fnt.uwb import spatial_metrics as SM

        x_col = 'smoothed_x' if 'smoothed_x' in frame.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in frame.columns else 'location_y'
        idx = {}
        for column, regions in sets:
            names = np.array([r.get('name', f'{column} {i + 1}')
                              for i, r in enumerate(regions)] + [''], dtype=object)
            zi = SM.assign_zones(frame[x_col].values, frame[y_col].values, regions)
            idx[column] = zi
            # -1 (outside) indexes the sentinel '' appended above.
            frame[column] = names[zi]
        return idx

    def animal_label(self, tag):
        """The configured animal's SexID, e.g. M9001.

        Every social and region statistic is keyed on this, not on the tag:
        once tags are assigned to animals, two tags mapping to one identity
        are one animal. Matches the label the bouts CSV uses.
        """
        info = self.tag_identities.get(tag) or {}
        sex = str(info.get('sex', 'M') or 'M')[:1].upper()
        identity = info.get('identity') or f'Tag{tag}'
        return f"{sex}{identity}"

    def region_bout_rows(self, frame, tag, sets, idx=None):
        """One row per visit: a run of consecutive fixes inside one region.

        Strict by design. A single fix outside the region ends the bout, and
        the next fix back inside starts a new one — nothing is bridged and no
        duration is rounded. A lone fix inside a region reports identical
        start and stop, credited one second, because its real span cannot be
        observed.

        Bouts are never split at midnight; ``Day`` is the day the bout
        STARTED. The daily summary does its own splitting.
        """
        from fnt.uwb import roi_bouts as RB

        if frame.empty:
            return []
        if idx is None:
            idx = self.annotate_regions(frame, sets)
        ts = frame['Timestamp']
        tz = getattr(ts.dt, 'tz', None)
        # Back to wall-clock in the recording's own timezone, the way every
        # other exported bout time reads. The int64 round trip is what keeps
        # the millisecond edges exact.
        t_ns = ts.values.astype('datetime64[ns]').astype('int64')
        naive = tz is None

        def _stamp(ns):
            t = pd.Timestamp(int(ns), unit='ns')
            return t if naive else t.tz_localize('UTC').tz_convert(tz)

        label = self.animal_label(tag)
        rows = []
        for column, regions in sets:
            names = [r.get('name', f'{column} {i + 1}')
                     for i, r in enumerate(regions)]
            for r_i, t0, t1, secs, n_reads in RB.region_bouts(idx[column], t_ns):
                rows.append({
                    'SexID': label,
                    'ROI': names[r_i],
                    'bout_start': _stamp(t0),
                    'bout_stop': _stamp(t1),
                    'duration_s': secs,
                    'n_reads': int(n_reads),
                })
        return rows


    @staticmethod
    def _tracked_distance_m(x, y, t_ns):
        """Path length in metres, ignoring steps that bridge a dropout.

        A gap is not travel: the animal did not teleport across the minutes a
        tag was silent, and counting the jump would add distance that never
        happened. A step is dropped when the interval it spans exceeds ten
        times the typical one - the same rule ``zone_intervals`` uses to stop
        a dropout being billed as dwell time.

        The test is on the RAW interval, not on the capped one: capped
        intervals are clipped TO the threshold, so comparing against their
        maximum marks every step in an evenly sampled track as a gap and
        returns zero.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) < 2:
            return 0.0
        step = np.hypot(np.diff(x), np.diff(y))
        good = np.isfinite(step)
        gaps = np.diff(np.asarray(t_ns, dtype='int64')) / 1e9
        if len(gaps) == len(step):
            positive = gaps[gaps > 0]
            typical = float(np.median(positive)) if positive.size else 0.0
            good &= gaps <= max(typical * 10.0, 1.0)
        return float(step[good].sum())

    def db_name(self):
        """This trial's name: the database filename without its extension."""
        if not self.db_path:
            return ''
        return os.path.splitext(os.path.basename(self.db_path))[0]

    def log_audit(self, title, results):
        """Log a product's self-check. Returns True if everything held.

        Written so a clean export reads as a short list of ticks and a broken
        one is impossible to skim past: a failure is prefixed, capitalised and
        repeated in a summary line. These files feed statistics run days later,
        where a silent error looks like biology.

        ``results`` may be a callable, which is how a caller hands over the
        checks WITHOUT risking the export on them: a bug in a check must never
        cost the user the file it was verifying. A check that raises is
        reported as an unrunnable check, not propagated.
        """
        if callable(results):
            try:
                results = results()
            except Exception as e:
                self.log_message(
                    f"  ⚠ {title}: the self-check could not run ({e!r}). The "
                    f"file itself is written and unaffected — this is a fault "
                    f"in the CHECK, not evidence about the data.")
                return False
        bad = [m for ok, m in results if not ok]
        self.log_message(f"  {title} — self-check:")
        for ok, msg in results:
            self.log_message(f"    {'✓' if ok else '✗ FAILED:'} {msg}")
        if bad:
            self.log_message(
                f"  ⚠ {title}: {len(bad)} CHECK(S) FAILED — do NOT analyse "
                f"this file until the cause is understood.")
        return not bad

    def write_roi_bout_csv(self, rows, output_dir, db_name,
                           skip_existing=False):
        """Publish the per-visit bout list. Returns the path, or None."""
        from fnt.uwb import roi_bouts as RB

        fname = f'{db_name}_ROI_bout_occupancy.csv'
        path = os.path.join(csv_output_dir(output_dir, create=True), fname)
        if not rows:
            self.log_message(
                "ROI occupancy: no animal entered a drawn region \u2014 "
                "nothing to write.")
            return None
        if skip_existing and os.path.exists(path):
            self.log_message(f"Skipped (exists): {fname}")
            return path

        # One animal may carry two tags. Union their visits so a replacement
        # tag cannot bill the same minute twice.
        rows = RB.merge_animal_bouts(rows)
        df = pd.DataFrame(rows)
        day0 = df['bout_start'].dt.normalize().min()
        df['Day'] = (df['bout_start'].dt.normalize() - day0).dt.days + 1
        df['_k'] = df['SexID'].map(RB.natural_animal_key)
        df = (df.sort_values(['_k', 'Day', 'bout_start'], kind='mergesort')
                .drop(columns='_k').reset_index(drop=True))
        df = df[['SexID', 'Day', 'ROI', 'bout_start', 'bout_stop', 'duration_s',
                 'n_reads']]
        # Audited on the FINISHED frame, not on the rows that build it: the
        # checks cover Day and the row order, which do not exist until here.
        self.log_audit(fname, lambda: RB.audit_bouts(df))
        df.to_csv(path, index=False)
        self.log_message(
            f"\u2713 Exported {fname} ({len(df)} bout(s), "
            f"{df['SexID'].nunique()} animal(s), {df['ROI'].nunique()} region(s))")
        # Per-region time, so a region nobody used \u2014 or one drawn so large it
        # swallowed the trial \u2014 is visible here rather than in R a week later.
        per = (df.groupby('ROI')['duration_s'].agg(['count', 'sum'])
                 .sort_values('sum', ascending=False))
        for roi, r in per.iterrows():
            self.log_message(f"      {roi}: {int(r['count']):,} visit(s), "
                             f"{r['sum'] / 3600:,.2f} h")
        return path

    def write_roi_daily_summary_csv(self, rows, output_dir, db_name,
                                    skip_existing=False):
        """Publish the per-region-per-day ownership table. Path, or None."""
        from fnt.uwb import roi_bouts as RB

        fname = f'{db_name}_ROI_DailySummary.csv'
        path = os.path.join(csv_output_dir(output_dir, create=True), fname)
        if not rows:
            return None
        if skip_existing and os.path.exists(path):
            self.log_message(f"Skipped (exists): {fname}")
            return path

        merged = RB.merge_animal_bouts(rows)
        summary = RB.daily_summary(merged)
        if not summary:
            return None
        # Audited against the very bouts it was derived from, so the
        # reconciliation is checked on this trial's own numbers rather than
        # assumed from the code being correct.
        self.log_audit(fname, lambda: RB.audit_daily(summary, merged))
        df = pd.DataFrame(summary)
        df.to_csv(path, index=False)
        self.log_message(
            f"\u2713 Exported {fname} ({len(df)} row(s), "
            f"{df['ROI'].nunique()} region(s) over {df['Day'].nunique()} day(s))")
        return path

    def region_bouts_from_csv(self, csv_path, sets, selected_tags=None):
        """Re-derive ROI bouts from an existing smoothed CSV.

        Only reached when the per-tag processing loop did not run - a batch
        retry reusing a verified CSV. Reads in chunks and only the columns the
        assignment needs, so a 15 M-row file does not have to fit in memory.
        """
        head = pd.read_csv(csv_path, nrows=0)
        coord = (['smoothed_x', 'smoothed_y']
                 if 'smoothed_x' in head.columns else ['location_x', 'location_y'])
        want = ['shortid', 'timestamp', 'Timestamp'] + coord
        frames = {}
        for chunk in pd.read_csv(csv_path, low_memory=False, chunksize=2_000_000,
                                 usecols=[c for c in want if c in head.columns]):
            chunk['Timestamp'] = smoothed_csv_timestamps(
                chunk, self.combo_timezone.currentText())
            for tag, g in chunk.groupby('shortid', sort=False):
                if selected_tags and tag not in selected_tags:
                    continue
                frames.setdefault(tag, []).append(g)
        rows = []
        for tag in sorted(frames):
            g = pd.concat(frames.pop(tag)).sort_values('Timestamp')
            rows.extend(self.region_bout_rows(g, tag, sets,
                                              idx=self.annotate_regions(g, sets)))
        return rows


    def _behavior_params(self):
        """Current detection thresholds straight from the preview controls."""
        from fnt.uwb.behavior_detection import BehaviorParams
        return BehaviorParams(
            social_radius=self.spin_social_radius.value(),
            chase_distance=self.spin_chase_distance.value(),
            chase_speed=self.spin_chase_speed.value(),
            chase_angle_deg=self.spin_chase_angle.value(),
            min_chase_s=self.spin_min_chase.value(),
            heading_lag_s=self.spin_heading_lag.value(),
            displace_distance=self.spin_displace_distance.value(),
            displace_loser_speed=self.spin_displace_loser_speed.value(),
            displace_winner_speed=self.spin_displace_winner_speed.value(),
            displace_leave_distance=self.spin_displace_leave.value(),
            displace_window_s=self.spin_displace_window.value(),
        )

    def _behavior_results(self):
        """Classification for the loaded preview chunk, recomputed when needed.

        The chunk's geometry is the expensive part, so the result is cached and
        only rebuilt when the data or a threshold actually changes — which keeps
        dragging a threshold spinbox interactive.
        """
        if self.preview_x is None or not len(self.preview_x):
            return None
        from fnt.uwb import behavior_detection as bd
        params = self._behavior_params()
        key = (id(self.preview_x), self.preview_x.shape, tuple(sorted(params.to_dict().items())))
        cached = getattr(self, "_behavior_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            res = bd.classify(self.preview_times, self.preview_x, self.preview_y, params)
        except Exception as e:
            self.log_message(f"Behavior detection failed: {e}")
            return None
        self._behavior_cache = (key, res)
        return res

    def _behavior_overlay(self, idx):
        """Circles / partner links / per-tag state text for one preview frame."""
        if not (getattr(self, "chk_show_behavior", None)
                and self.chk_show_behavior.isChecked()):
            if hasattr(self, "lbl_speed_pcts"):
                self.lbl_speed_pcts.setText("")
            return None
        res = self._behavior_results()
        if res is None or idx >= len(res["social"]):
            return None
        from fnt.uwb import behavior_detection as bd

        show_social = self.chk_beh_social.isChecked()
        show_chase = self.chk_beh_chase.isChecked()
        show_displace = self.chk_beh_displace.isChecked()

        # Derive the displayed social state from the raw masks rather than
        # masking the collapsed array: chase outranks contact in the collapsed
        # state, so blanking chase there would also hide a contact that is
        # genuinely happening underneath it.
        n_tags = res["social"].shape[1]
        chase_f = res["chase"][idx]
        contact_f = res["contact"][idx]
        soc = np.full(n_tags, bd.SOC_NONE, dtype=np.int8)
        partner_f = np.full(n_tags, -1, dtype=int)
        dist_f = res["pairs"]["dist"][idx]

        def _nearest(mask_row):
            cand = np.where(mask_row, np.nan_to_num(dist_f_row, nan=np.inf), np.inf)
            j = int(np.argmin(cand))
            return j if np.isfinite(cand[j]) else -1

        disp_f = res["displacing"][idx]
        for t in range(n_tags):
            dist_f_row = dist_f[t]
            if show_chase and chase_f[t].any():
                soc[t] = bd.SOC_CHASING
                partner_f[t] = _nearest(chase_f[t])
            elif show_chase and chase_f[:, t].any():
                soc[t] = bd.SOC_CHASED
                partner_f[t] = _nearest(chase_f[:, t])
            elif show_displace and disp_f[t].any():
                soc[t] = bd.SOC_DISPLACING
                partner_f[t] = _nearest(disp_f[t])
            elif show_displace and disp_f[:, t].any():
                soc[t] = bd.SOC_DISPLACED
                partner_f[t] = _nearest(disp_f[:, t])
            elif show_social and contact_f[t].any():
                soc[t] = bd.SOC_CONTACT
                partner_f[t] = _nearest(contact_f[t])

        # Undirected links, so a pair is drawn once.
        links = []
        seen = set()
        if show_chase:
            ca, cb = np.nonzero(res["chase"][idx])
            for i, j in zip(ca.tolist(), cb.tolist()):
                if (min(i, j), max(i, j)) not in seen:
                    seen.add((min(i, j), max(i, j)))
                    links.append((i, j, bd.SOCIAL_COLORS[bd.SOC_CHASING]))
        if show_displace:
            da, db = np.nonzero(res["displacing"][idx])
            for i, j in zip(da.tolist(), db.tolist()):
                if (min(i, j), max(i, j)) not in seen:
                    seen.add((min(i, j), max(i, j)))
                    links.append((i, j, bd.SOCIAL_COLORS[bd.SOC_DISPLACING]))
        if show_social:
            # Contact links use the same colour as the social circles, so the
            # radius and the connection it produced read as one thing.
            circle_col = self.combo_circle_color.currentData() or "#9fb3c8"
            sa, sb = np.nonzero(res["contact"][idx])
            for i, j in zip(sa.tolist(), sb.tolist()):
                if (min(i, j), max(i, j)) not in seen:
                    seen.add((min(i, j), max(i, j)))
                    links.append((i, j, circle_col))

        # Social state only. An animal doing nothing social reads as nothing,
        # rather than as the locomotor label that used to fill the gap.
        states = []
        state_parts = []
        for t in range(n_tags):
            soc_txt = bd.SOCIAL_LABELS.get(soc[t], "") if soc[t] != bd.SOC_NONE else ""
            soc_col = bd.SOCIAL_COLORS.get(soc[t], "#cccccc")
            state_parts.append([(soc_txt, soc_col)] if soc_txt else [])
            states.append((soc_txt, soc_col))

        pct, msg = bd.speed_summary(res["velocity"], res["have_fix"],
                                    self._behavior_params())
        self.lbl_speed_pcts.setText(msg)

        return {"radius": self.spin_social_radius.value() if show_social else None,
                "circle_color": self.combo_circle_color.currentData(),
                "links": links, "states": states, "state_parts": state_parts}

    def render_preview_frame(self):
        """Draw the frame the playhead currently points at."""
        if getattr(self, "preview_x", None) is None or not len(self.preview_x):
            return
        idx = self._frame_index()

        # Marker uses the forward-filled position so a tag stays put between its
        # sparse fixes instead of blinking; the trail below uses the real fixes.
        # getattr's default does not apply when the attribute exists but is
        # None (which it is between cache invalidation and the next chunk), so
        # fall back explicitly.
        xf = getattr(self, "preview_xf", None)
        yf = getattr(self, "preview_yf", None)
        if xf is None or yf is None:
            xf, yf = self.preview_x, self.preview_y
        x = xf[idx]
        y = yf[idx]
        colors = self.preview_colors

        backend = self._preview_backend()
        # Live marker size (diameter, points); see UWBPreview2D.update_frame.
        if hasattr(self, "spin_tag_size"):
            backend.tag_size = self.spin_tag_size.value()
        if hasattr(self, "combo_label_color"):
            backend.label_color = self.combo_label_color.currentData()
            backend.label_outline = self.chk_label_outline.isChecked()

        # "Show Tag Tracking Data" off: draw an empty frame so only the arena /
        # background / anchors remain (no markers, no trails).
        if hasattr(self, "chk_show_tracking") and not self.chk_show_tracking.isChecked():
            backend.update_frame(np.full_like(np.asarray(x, float), np.nan),
                                 np.full_like(np.asarray(y, float), np.nan),
                                 colors, tracks=None, raw_pts=None)
            self._update_time_label()
            return

        # Build the trailing track: one connected polyline per tag over the
        # trail window (seconds -> frames). Chunks are loaded with a lead-in
        # overlap sized to the trail, so the tail stays continuous across chunk
        # boundaries; a trail longer than a whole chunk still can't be shown in
        # full. Points are the (smoothed) positions; drawing them as a line is
        # the "track" the user wants instead of a dot cloud.
        trail_frames = int(self.spin_preview_trail.value() * self.preview_hz)
        tracks = []
        if trail_frames > 0:
            lo = max(0, idx - trail_frames)
            seg_x = self.preview_x[lo:idx + 1]
            seg_y = self.preview_y[lo:idx + 1]
            for t in range(len(self.preview_tags)):
                tx, ty = seg_x[:, t], seg_y[:, t]
                ok = np.isfinite(tx) & np.isfinite(ty)
                if ok.sum() >= 1:
                    tracks.append((np.column_stack([tx[ok], ty[ok]]),
                                   tuple(colors[t])))

        # ID label (Show Tag ID) and battery voltage (Display Tag Battery Levels)
        # are independent overlays. The voltage anchors under the marker whether
        # or not the ID label is shown.
        labels = batteries = None
        if getattr(self, "chk_show_tag_id", None) is not None and self.chk_show_tag_id.isChecked():
            labels = self._preview_tag_labels(
                self.preview_tags, self.combo_tag_id_type.currentText())
        if getattr(self, "chk_show_battery", None) is not None and self.chk_show_battery.isChecked():
            batt = getattr(self, "preview_batt", None)
            if batt is not None and idx < len(batt):
                batteries = batt[idx]

        # Speed / step distance share one line under the marker: the stack
        # already carries ID, behaviour state and voltage, and two more rows
        # collide as soon as two animals are in contact.
        readouts = None
        parts = []
        for cb, attr, fmt in ((self.chk_show_speed, "preview_speed", "{:.3f} m/s"),
                              (self.chk_show_step, "preview_step", "{:.2f} m")):
            arr = getattr(self, attr, None)
            if cb.isChecked() and arr is not None and idx < len(arr):
                parts.append((arr[idx], fmt))
        if parts:
            readouts = []
            for i in range(len(x)):
                bits = [fmt.format(a[i]) for a, fmt in parts
                        if i < len(a) and np.isfinite(a[i])]
                readouts.append(" · ".join(bits) if bits else None)

        behavior = self._behavior_overlay(idx)
        backend.update_frame(x, y, colors, tracks=tracks, raw_pts=None,
                             labels=labels, batteries=batteries,
                             readouts=readouts, behavior=behavior)
        self._update_time_label()

    # -- transport --------------------------------------------------------- #
    # ── playback ──────────────────────────────────────────────────────────
    # Redraws per second to AIM for. Drawing faster than this buys nothing on
    # a normal display, and a machine that cannot reach it simply draws fewer
    # frames - the trial clock is unaffected either way.
    PLAYBACK_FPS = 25

    def playback_speed(self):
        """Trial seconds per wall second, from the transport's Speed box."""
        try:
            return float(self.combo_preview_speed.currentData() or 1.0)
        except (TypeError, ValueError):
            return 1.0

    def toggle_preview_playback(self):
        if self._playing:
            self.stop_preview_playback()
        else:
            self.start_preview_playback()

    def start_preview_playback(self):
        """Begin playing forward from the playhead."""
        if not getattr(self, '_preview_active', False):
            return
        if self.preview_t0 is None or self.slider_timeline.maximum() <= 0:
            return
        # Parked at the end, Play means "watch it again" rather than nothing.
        if self.slider_timeline.value() >= self.slider_timeline.maximum():
            self._playback_seeking = True
            self.slider_timeline.setValue(0)
            self._playback_seeking = False
        if self.preview_timer is None:
            self.preview_timer = QTimer(self)
            # Single-shot and re-armed at the END of each tick: the next frame
            # is never queued until this one has been drawn, so a slow render
            # cannot pile up a backlog of timer events and freeze the window.
            self.preview_timer.setSingleShot(True)
            self.preview_timer.timeout.connect(self._playback_tick)
        self._playing = True
        self._anchor_playback()
        self._play_frames = 0
        self._play_fps_t0 = time.perf_counter()
        self.btn_preview_play.setText("\u23f8 Pause")
        self.lbl_preview_fps.setText("")
        self.preview_timer.start(0)

    def stop_preview_playback(self):
        """Pause playback and leave the playhead where it is."""
        self._playing = False
        if self.preview_timer is not None:
            self.preview_timer.stop()
        if hasattr(self, 'btn_preview_play'):
            self.btn_preview_play.setText("\u25b6 Play")
            self.lbl_preview_fps.setText("")

    def _anchor_playback(self):
        """Peg the wall clock to the current playhead.

        Every tick derives the playhead from the time since this anchor, so
        re-anchoring is how a speed change or a seek takes effect without the
        playhead jumping: the new rate applies from here on.
        """
        self._play_wall0 = time.perf_counter()
        self._play_media0 = (self.preview_playhead_ms - self.preview_t0) / 1000.0

    def _on_preview_speed_changed(self, *_):
        if self._playing:
            self._anchor_playback()

    def _playback_tick(self):
        """Place the playhead where the wall clock says it should be.

        Deliberately NOT "advance by one frame": accumulating a fixed step per
        tick makes the trial clock run at whatever rate the machine can render,
        which is what made the old playback useless above about 4x. Deriving
        the position from elapsed wall time instead means a slow render costs a
        DROPPED FRAME, not lost time - 60x is 60x on any machine, it just looks
        choppier on a slower one. That is why the Speed list needs no cap.
        """
        if not self._playing or self.preview_t0 is None:
            return
        started = time.perf_counter()
        elapsed = started - self._play_wall0
        target_s = self._play_media0 + elapsed * self.playback_speed()
        end_s = float(self.slider_timeline.maximum())

        if target_s >= end_s:
            self._playback_seeking = True
            self.slider_timeline.setValue(int(end_s))
            self.seek_preview_ms(self.preview_t0 + int(end_s * 1000))
            self._playback_seeking = False
            self.stop_preview_playback()
            self.log_message("Preview playback reached the end of the recording.")
            return

        # Sub-second precision for the actual frame; the slider follows as a
        # position indicator, guarded so it does not seek back on us.
        self._playback_seeking = True
        self.seek_preview_ms(self.preview_t0 + int(target_s * 1000))
        self._sync_timeline_to_playhead()
        self._playback_seeking = False

        # Achieved rate, refreshed about once a second. This is what the
        # machine actually sustains - the number the user needs to judge a
        # speed by, rather than a cap guessed in advance.
        self._play_frames += 1
        since = started - self._play_fps_t0
        if since >= 1.0:
            self.lbl_preview_fps.setText(f"{self._play_frames / since:.0f} fps")
            self._play_frames = 0
            self._play_fps_t0 = started

        if not self._playing:      # a seek during the render may have paused us
            return
        # Re-arm for the remainder of this frame's budget; 0 when the render
        # already overran it, so a slow machine free-runs instead of sleeping.
        budget_ms = 1000.0 / self.PLAYBACK_FPS
        spent_ms = (time.perf_counter() - started) * 1000.0
        self.preview_timer.start(int(max(0.0, budget_ms - spent_ms)))

    def _context_bg_source(self):
        """(image, extent) for the 'background' layer of main-thread figures.

        Prefers a user-loaded floorplan PNG (scaled via the XML) and falls back
        to the XML-embedded site map. Returns (None, None) if neither exists.
        """
        ext = self._bg_placed_extent()
        if ext is not None:
            return self.background_image, list(ext)
        if self.xml_map_image is not None and self.xml_map_extent is not None:
            return self.xml_map_image, list(self.xml_map_extent)
        return None, None

    def _export_sitemap(self, output_dir, db_name):
        """Write the background/site-map image into the export folder.

        Makes the analysis folder a self-contained data product for downstream
        analysis: it carries the positions (CSV), zones/anchors/identities/scale
        (config) and now the floorplan image too. Prefers a user-loaded PNG
        (copied verbatim), else the XML-embedded site map (decoded to PNG).
        Records filename + meter-extent in self._exported_sitemap for the config;
        leaves it None when there is no background (downstream consumers then
        draw zones only, or bare tracks).
        """
        self._exported_sitemap = None
        bg_image, bg_extent = self._context_bg_source()
        if bg_image is None or bg_extent is None:
            return
        filename = f'{db_name}_sitemap.png'
        dest = os.path.join(output_dir, filename)
        try:
            if self.background_image_path and os.path.exists(self.background_image_path):
                shutil.copyfile(self.background_image_path, dest)
            else:
                plt.imsave(dest, bg_image)
            self._exported_sitemap = {
                'filename': filename,
                'extent_m': [float(v) for v in bg_extent],
            }
            self.log_message(f"✓ Site-map image exported: {filename}")
        except Exception as e:
            self.log_message(f"Warning: could not export site-map image: {e}")

    def _layer_sources(self):
        """Which spatial context layers actually have data behind them.

        'Background' counts a user-loaded floorplan (via Load Background) OR the
        site XML's embedded map, which is the fallback both the preview and the
        exports draw when no image was loaded.
        """
        return {
            'background': (self.background_image is not None
                           or self.xml_map_image is not None),
            'zones': bool(self.xml_zones) or (
                self.arena_zones is not None and not self.arena_zones.empty),
            'anchors': bool(self.anchor_positions),
            'rois': bool(self.rois),
        }

    def _sync_layer_toggles(self):
        """Grey out the preview layer toggles that have no data behind them.

        The checked state is left alone — a toggle greyed out because the XML
        has not been parsed yet must come back the way the user left it, not
        cleared. `plot_layers_from_preview` resolves checked-but-unavailable to
        off, the same way `_anim_show` does for the animation rows.
        """
        have = self._layer_sources()
        for attr, key in (('chk_show_background', 'background'),
                          ('chk_show_zones', 'zones'),
                          ('chk_show_anchors', 'anchors'),
                          ('chk_show_rois', 'rois')):
            cb = getattr(self, attr, None)
            if cb is not None:
                cb.setEnabled(have[key])
        # Importing zones needs zones, and this runs whenever the XML is
        # parsed or cleared, so the button follows the same signal the
        # "Show zone coordinates" toggle does.
        btn = getattr(self, 'btn_roi_import_xml', None)
        if btn is not None:
            btn.setEnabled(bool(self.xml_zones))
            if not self.xml_zones:
                btn.setToolTip(
                    "This trial's site XML defines no zones to import.\n\n"
                    "Draw regions with 'Draw new' instead — the occupancy "
                    "outputs measure drawn regions only.")
            else:
                btn.setToolTip(ROI_IMPORT_XML_TOOLTIP)

    def plot_layers_from_preview(self, announce=True):
        """Take the spatial context layers straight from the preview toggles.

        STRICT INHERITANCE: exported plots and animations draw the background,
        zones and anchors exactly as the preview has them — there is no separate
        per-export choice to get out of step with what you were looking at. A
        layer with no data behind it resolves to off however its toggle sits.

        Sets and returns self.plot_layers, which is what the exports read and
        what get_config_dict captures for a queued job.
        """
        have = self._layer_sources()

        def _on(attr, key):
            cb = getattr(self, attr, None)
            return bool(cb is not None and cb.isChecked() and have[key])

        self.plot_layers = {
            'background': _on('chk_show_background', 'background'),
            'zones': _on('chk_show_zones', 'zones'),
            'anchors': _on('chk_show_anchors', 'anchors'),
            # Whether regions are DRAWN. Whether they are MEASURED is
            # Export Zone/ROI Occupancy CSV, and the two never consult
            # each other.
            'rois': _on('chk_show_rois', 'rois'),
        }
        if announce:
            self.log_message(
                "Plot/animation layers, inherited from the preview — "
                f"background: {'on' if self.plot_layers['background'] else 'off'}, "
                f"zones: {'on' if self.plot_layers['zones'] else 'off'}, "
                f"anchors: {'on' if self.plot_layers['anchors'] else 'off'}, "
                f"regions: {'on' if self.plot_layers['rois'] else 'off'}")
        return self.plot_layers

    # -- quick snapshot plot ---------------------------------------------- #
    def plot_last_known_locations(self):
        """Save a labelled snapshot of each selected tag's most recent fix.

        One cheap query per tag (MAX(timestamp)), so it is fast even on the
        unindexed original — and faster still if the indexed copy exists. The
        figure is drawn over the same arena the preview shows (XML zones/map
        when available), then written to the analysis folder.
        """
        if not self.db_path or not self.table_name:
            QMessageBox.warning(self, "No Database", "Select a database and table first")
            return
        tags = self.selected_preview_tags()
        if not tags:
            QMessageBox.warning(self, "No Tags", "Select at least one tag first")
            return

        # Background/zones/anchors come from the preview, as drawn.
        self.plot_layers_from_preview()

        try:
            tz = pytz.timezone(self.combo_timezone.currentText())
            db = self.current_indexed_db() or self.preview_db_path or self.db_path
            conn = connect_ro(db)
            placeholders = ",".join(["?"] * len(tags))
            # Last fix per tag: join each tag's MAX(timestamp) back to its row.
            rows = conn.execute(
                f"SELECT d.shortid, d.location_x, d.location_y, d.timestamp "
                f"FROM {self.table_name} d "
                f"JOIN (SELECT shortid, MAX(timestamp) mt FROM {self.table_name} "
                f"      WHERE shortid IN ({placeholders}) GROUP BY shortid) m "
                f"ON d.shortid = m.shortid AND d.timestamp = m.mt",
                tags).fetchall()
            conn.close()
        except Exception as e:
            self.log_message(f"Last-locations query failed: {e}")
            QMessageBox.critical(self, "Query Failed", f"Could not read last locations:\n{e}")
            return

        if not rows:
            QMessageBox.information(self, "No Data", "No fixes found for the selected tags.")
            return

        # Deduplicate on shortid (a tie in MAX could yield two rows) and convert
        # inches -> metres to match every other coordinate in the tool.
        latest = {}
        for shortid, lx, ly, ts in rows:
            if shortid not in latest or ts > latest[shortid][2]:
                latest[shortid] = (lx * 0.0254, ly * 0.0254, ts)

        # Busy cursor for the figure build + save (all on the GUI thread).
        # try/finally guarantees the cursor clears even if drawing/saving raises.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            fig = Figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            pal_bg, pal_fg = "#1e1e1e", "#cccccc"
            fig.patch.set_facecolor(pal_bg)
            ax.set_facecolor(pal_bg)

            # Arena context (background/zones/anchors) per the export layer choice.
            xs = [v[0] for v in latest.values()]
            ys = [v[1] for v in latest.values()]
            bg_image, bg_extent = self._context_bg_source()
            draw_context_layers(
                ax, self.plot_layers, bg_image=bg_image, bg_extent=bg_extent,
                zones_xml=self.xml_zones, anchors=self.anchor_positions,
                rois=self.rois)

            cmap = plt.get_cmap('tab20')
            for i, (shortid, (x, y, ts)) in enumerate(sorted(latest.items())):
                label = self._tag_display_label(shortid)
                local = pd.Timestamp(int(ts), unit='ms', tz='UTC').tz_convert(tz)
                ax.scatter([x], [y], s=140, color=cmap(i % 20),
                           edgecolors="white", linewidths=1.0, zorder=5)
                ax.annotate(f"{label}\n{local:%m-%d %H:%M}", (x, y),
                            textcoords="offset points", xytext=(8, 6),
                            color=pal_fg, fontsize=8, zorder=6)

            ax.set_aspect("equal")
            ax.set_xlabel("X (m)", color=pal_fg)
            ax.set_ylabel("Y (m)", color=pal_fg)
            ax.tick_params(colors=pal_fg)
            for sp in ax.spines.values():
                sp.set_color("#555555")
            ax.set_title(f"Last known locations · {len(latest)} tags", color=pal_fg)
            fig.tight_layout()

            # Save into the analysis folder, matching the export naming.
            db_dir = os.path.dirname(self.db_path)
            db_name = os.path.splitext(os.path.basename(self.db_path))[0]
            out_dir = os.path.join(db_dir, f"{db_name}_FNT_analysis")
            os.makedirs(out_dir, exist_ok=True)
            png = os.path.join(out_dir, f"{db_name}_LastKnownLocations.png")
            try:
                fig.savefig(png, dpi=150, facecolor=pal_bg, bbox_inches="tight")
                saved = [os.path.basename(png)]
                if self.chk_save_svg.isChecked():
                    svg = os.path.splitext(png)[0] + ".svg"
                    fig.savefig(svg, facecolor=pal_bg, bbox_inches="tight")
                    saved.append(os.path.basename(svg))
                self.log_message(f"✓ Saved last-known-locations plot: {', '.join(saved)}")
                QMessageBox.information(
                    self, "Plot Saved",
                    f"Saved {', '.join(saved)}\n\nin {os.path.basename(out_dir)}/")
            except Exception as e:
                self.log_message(f"Could not save last-locations plot: {e}")
                QMessageBox.critical(self, "Save Failed", str(e))
        finally:
            QApplication.restoreOverrideCursor()

    def _tag_display_label(self, shortid):
        """Identity label for a tag: 'sex-identity' if configured, else HexID."""
        info = self.tag_identities.get(shortid)
        if info:
            sex, ident = info.get("sex", ""), info.get("identity", "")
            if sex and ident:
                return f"{sex}-{ident}"
            if ident:
                return str(ident)
        return f"HexID {hex(shortid).upper().replace('0X', '')}"

    # -- occupancy heatmaps ------------------------------------------------ #
    def plot_occupancy_heatmaps(self):
        """Build per-animal space-use heatmaps over the whole recording.

        Each tag's fixes are binned into a 2-D histogram of % time per arena
        cell, then shown faceted (one panel per animal) in a zoomable window.
        Memory stays flat: each tag is read, filtered, histogrammed and then
        discarded — only the small bin grids are kept.
        """
        if not self.db_path or not self.table_name:
            QMessageBox.warning(self, "No Database", "Select a database and table first")
            return
        tags = self.selected_preview_tags()
        if not tags:
            QMessageBox.warning(self, "No Tags", "Select at least one tag first")
            return

        # Zones/anchors overlay the heatmap, as the preview has them.
        self.plot_layers_from_preview()

        db = self.current_indexed_db() or self.preview_db_path or self.db_path
        tz = pytz.timezone(self.combo_timezone.currentText())
        do_filter = self.chk_velocity_filter.isChecked() or self.chk_jump_filter.isChecked()

        progress = QProgressDialog("Building occupancy heatmaps…", "Cancel",
                                   0, len(tags) + 1, self)
        progress.setWindowTitle("Occupancy Heatmaps")
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QApplication.processEvents()

        try:
            conn = connect_ro(db)

            # Common spatial extent so every facet shares one grid. Prefer the
            # anchor footprint (the true arena); UWB data carries stray fixes
            # far outside it, so a raw data min/max would blow up the frame.
            placeholders = ",".join(["?"] * len(tags))
            if self.anchor_positions:
                xs_a = [a["x"] for a in self.anchor_positions]
                ys_a = [a["y"] for a in self.anchor_positions]
                x0, x1 = min(xs_a), max(xs_a)
                y0, y1 = min(ys_a), max(ys_a)
            else:
                # No anchors: use robust percentiles to reject outliers. Sample
                # the data rather than pulling every row for the bounds.
                s = pd.read_sql_query(
                    f"SELECT location_x, location_y FROM {self.table_name} "
                    f"WHERE shortid IN ({placeholders}) "
                    f"ORDER BY timestamp LIMIT 200000", conn, params=tags)
                if len(s) == 0:
                    conn.close(); progress.close()
                    QMessageBox.information(self, "No Data", "No fixes for the selected tags.")
                    return
                x0, x1 = np.percentile(s['location_x'] * 0.0254, [1, 99])
                y0, y1 = np.percentile(s['location_y'] * 0.0254, [1, 99])
            pad = 0.5
            x0, x1 = x0 - pad, x1 + pad
            y0, y1 = y0 - pad, y1 + pad
            # histogram2d range clips out-of-arena outliers automatically
            hist_range = [[x0, x1], [y0, y1]]
            # ~0.15 m cells, capped so huge arenas stay a sane grid size
            nx = int(np.clip(round((x1 - x0) / 0.15), 20, 90))
            ny = int(np.clip(round((y1 - y0) / 0.15), 20, 90))
            xedges = np.linspace(x0, x1, nx + 1)
            yedges = np.linspace(y0, y1, ny + 1)

            heatmaps, labels = [], []
            for i, tag in enumerate(tags):
                if progress.wasCanceled():
                    conn.close(); return
                progress.setLabelText(f"Tag {i + 1}/{len(tags)} — reading & binning…")
                progress.setValue(i)
                QApplication.processEvents()

                td = pd.read_sql_query(
                    f"SELECT timestamp, location_x, location_y FROM {self.table_name} "
                    f"WHERE shortid = ? ORDER BY timestamp", conn, params=(tag,))
                if len(td) == 0:
                    heatmaps.append(None); labels.append(self._tag_display_label(tag))
                    continue

                td['shortid'] = tag
                td['location_x'] *= 0.0254
                td['location_y'] *= 0.0254

                # Respect the tag's active window and the current filters, so
                # the heatmap matches what an export would produce.
                td['Timestamp'] = pd.to_datetime(
                    td['timestamp'], unit='ms', origin='unix', utc=True).dt.tz_convert(tz)
                td, _dropped = self._trim_to_tag_window(td, tag, tz)
                if do_filter and len(td) > 0:
                    td['Timestamp'] = pd.to_datetime(
                        td['timestamp'], unit='ms', origin='unix', utc=True).dt.tz_convert(tz)
                    td = self.apply_filters_to_data(td, collect_stats=False)

                labels.append(self._tag_display_label(tag))
                if len(td) == 0:
                    heatmaps.append(None); continue
                # range= clips fixes outside the arena frame
                H, _, _ = np.histogram2d(td['location_x'].to_numpy(),
                                         td['location_y'].to_numpy(),
                                         bins=[xedges, yedges], range=hist_range)
                total = H.sum()
                heatmaps.append((H / total * 100.0) if total else None)
                self.log_message(f"  {labels[-1]}: {int(total):,} fixes binned")
                # Repaint the progress dialog after this tag's heavy read+bin.
                QApplication.processEvents()

            conn.close()
            progress.setValue(len(tags))
            QApplication.processEvents()

            valid = [h for h in heatmaps if h is not None]
            if not valid:
                progress.close()
                QMessageBox.information(self, "No Data",
                                        "No fixes survived filtering for the selected tags.")
                return

            # Shared colour scale: 99th percentile of occupied cells, so a
            # single hot spot doesn't wash out every panel.
            pooled = np.concatenate([h[h > 0].ravel() for h in valid])
            vmax = float(np.percentile(pooled, 99)) if len(pooled) else 1.0

            fig = self._build_heatmap_figure(heatmaps, labels, xedges, yedges, vmax)
            progress.close()

            # Save alongside the other outputs, then show it zoomably.
            db_name = os.path.splitext(os.path.basename(self.db_path))[0]
            out_dir = os.path.join(os.path.dirname(self.db_path), f"{db_name}_FNT_analysis")
            os.makedirs(out_dir, exist_ok=True)
            png = os.path.join(out_dir, f"{db_name}_OccupancyHeatmaps.png")
            try:
                fig.savefig(png, dpi=150, facecolor="white", bbox_inches="tight")
                self.log_message(f"✓ Saved occupancy heatmaps: {os.path.basename(png)}")
            except Exception as e:
                self.log_message(f"Could not save heatmap PNG: {e}")

            self._heatmap_popup = FigurePopup(
                fig, title=f"Occupancy Heatmaps — {db_name}", parent=self)
            self._heatmap_popup.resize(960, 720)
            self._heatmap_popup.show()

        except Exception as e:
            progress.close()
            self.log_message(f"Occupancy heatmap failed: {e}")
            QMessageBox.critical(self, "Heatmap Failed", str(e))

    def _build_heatmap_figure(self, heatmaps, labels, xedges, yedges, vmax):
        """Faceted magma raster of % time per cell, shared colour scale."""
        n = len(heatmaps)
        ncols = 1 if n == 1 else min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig = Figure(figsize=(3.2 * ncols + 1.2, 3.0 * nrows), facecolor="white")
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = None
        for i, (H, lab) in enumerate(zip(heatmaps, labels)):
            ax = fig.add_subplot(nrows, ncols, i + 1)
            ax.set_facecolor("black")
            if H is not None:
                # histogram2d is [x, y]; transpose so rows=y for imshow, and
                # origin="lower" puts +y up.
                im = ax.imshow(H.T, extent=extent, origin="lower", cmap="magma",
                               vmin=0, vmax=vmax, aspect="equal",
                               interpolation="nearest")
            else:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", color="#999")
                ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
            # Overlay chosen context layers. Background is omitted here — the
            # density raster already fills the panel and would hide it.
            draw_context_layers(
                ax, {'background': False,
                     'zones': self.plot_layers.get('zones'),
                     'anchors': self.plot_layers.get('anchors'),
                     'rois': self.plot_layers.get('rois')},
                zones_xml=self.xml_zones, anchors=self.anchor_positions,
                rois=self.rois)
            ax.set_title(lab, fontsize=10)
            ax.set_xlabel("X Position (m)", fontsize=8)
            ax.set_ylabel("Y Position (m)", fontsize=8)
            ax.tick_params(labelsize=7)
        if im is not None:
            cbar = fig.colorbar(im, ax=fig.axes, fraction=0.025, pad=0.02)
            cbar.set_label("% time", fontsize=9)
        fig.suptitle("Occupancy (space use) over full recording", fontsize=12)
        return fig

    # -- indexed preview copy ---------------------------------------------- #
    # The original Wiser database is treated as a read-only primary record and
    # is never written to. Fast scrubbing instead uses a derived, indexed copy
    # kept in the analysis/export folder with the other analysis outputs.
    def _preview_index_paths(self):
        """(index_dir, indexed_copy_path, provenance_json_path).

        The indexed copy + its provenance JSON live in the analysis/export
        folder (``<db>_FNT_analysis``, in the source DB's directory) alongside
        the other analysis outputs, so every derived product for a run is
        collected in one place rather than scattered next to the raw database.
        The folder is created on demand before the index is built
        (see ensure_preview_index_db).
        """
        db_dir = os.path.dirname(self.db_path)
        db_name = os.path.splitext(os.path.basename(self.db_path))[0]
        analysis_dir = os.path.join(db_dir, f"{db_name}_FNT_analysis")
        return (analysis_dir,
                os.path.join(analysis_dir, f"{db_name}_indexed.sqlite"),
                os.path.join(analysis_dir, f"{db_name}_indexed.json"))

    def current_indexed_db(self):
        """Path of an up-to-date indexed copy on disk, or None.

        Deliberately independent of ``preview_db_path``: that attribute is only
        set once the preview activates, which happens AFTER the database load.
        Anything running during the load - the tag/day scan in particular - has
        to look the copy up itself or it will scan the multi-GB original over
        the network instead.
        """
        if not self.db_path or not self.table_name:
            return None
        try:
            _dir, copy_path, meta_path = self._preview_index_paths()
        except Exception:
            return None
        if self._indexed_copy_is_current(copy_path, meta_path):
            return copy_path
        return None

    def _indexed_copy_is_current(self, copy_path, meta_path):
        """Is an existing copy still faithful to the source database?

        Compared on size and mtime. A stale cache is worse than none, so any
        mismatch (or unreadable metadata) fails closed and forces a rebuild.
        """
        if not (os.path.exists(copy_path) and os.path.exists(meta_path)):
            return False
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            st = os.stat(self.db_path)
            return (meta.get("source_size") == st.st_size
                    and abs(meta.get("source_mtime", 0) - st.st_mtime) < 1.0
                    and meta.get("table_name") == self.table_name)
        except Exception:
            return False

    def adopt_preview_index_db(self):
        """Use an existing indexed copy if one is current, else the original.

        Never builds — this runs automatically when the preview activates, and
        writing a multi-GB copy on every database load would be a surprise. The
        'Fast index' button (ensure_preview_index_db) is the explicit opt-in.
        """
        if not self.db_path or not self.table_name:
            return
        self.preview_db_path = self.db_path
        _, copy_path, meta_path = self._preview_index_paths()

        # Sweep any leftover .partial from a build that was interrupted (crash
        # or hard kill) — these can be multi-GB and would otherwise linger. Only
        # safe to remove when no build is currently running.
        if not (self.preview_index_builder is not None
                and self.preview_index_builder.isRunning()):
            for junk in (copy_path + ".partial", copy_path + ".partial-journal"):
                try:
                    if os.path.exists(junk):
                        os.remove(junk)
                        self.log_message(f"Removed stale index build file: {os.path.basename(junk)}")
                except Exception:
                    pass

        if self._indexed_copy_is_current(copy_path, meta_path):
            self.preview_db_path = copy_path
            self._set_index_status("Fast index: ready ✓", "#00aa00")
            self.log_message(f"Using existing indexed copy for preview: {os.path.basename(copy_path)}")
        else:
            self._set_index_status("Fast index: not built (click to speed up scrubbing)", "#cccccc")

    def ensure_preview_index_db(self):
        """Point the preview at an indexed copy, building one if needed.

        Preview stays usable throughout: reads run against the original until
        the copy is ready, then transparently switch over.
        """
        if not self.db_path or not self.table_name:
            return
        self.preview_db_path = self.db_path      # usable immediately

        index_dir, copy_path, meta_path = self._preview_index_paths()
        if self._indexed_copy_is_current(copy_path, meta_path):
            self.preview_db_path = copy_path
            self._set_index_status("Fast index: ready ✓", "#00aa00")
            self.log_message(f"Using indexed copy for preview: {os.path.basename(copy_path)}")
            return

        if self.preview_index_builder is not None and self.preview_index_builder.isRunning():
            return

        if os.path.exists(copy_path):
            self.log_message("Indexed copy is out of date with the source — rebuilding.")

        try:
            os.makedirs(index_dir, exist_ok=True)
        except Exception as e:
            self.log_message(f"Could not create index folder: {e}")
            return

        src_mb = os.path.getsize(self.db_path) / 1e6
        self.log_message(
            f"Building indexed copy for fast scrubbing (~{src_mb * 1.25:.0f} MB "
            f"in {os.path.basename(index_dir)}). The original database is "
            f"not modified.")
        self._set_index_status("Fast index: building…", "#cc9900")
        self._show_busy("Building fast-scrubbing index (original DB untouched)…")

        b = PreviewIndexBuilder(self.db_path, self.table_name, copy_path, meta_path)
        b.progress.connect(self._on_index_progress)
        b.done.connect(self._on_index_ready)
        b.failed.connect(self._on_index_failed)
        self.preview_index_builder = b
        b.start()

    def _index_build_is_current(self, src_path):
        """Is a build result still for the database the tool is showing?

        A build takes seconds; the user can switch databases in that time. Any
        result for a different source must be discarded, or the preview would
        silently read another trial's data.
        """
        return bool(self.db_path) and os.path.abspath(src_path) == os.path.abspath(self.db_path)

    def _on_index_progress(self, msg):
        if self.preview_index_builder is None:
            return
        if not self._index_build_is_current(self.preview_index_builder.src_path):
            return
        self._set_index_status(f"Fast index: {msg}", "#cc9900")
        self._set_busy_msg(f"Building fast-scrubbing index: {msg}")

    def _set_index_status(self, text, color):
        # Shown as a small status line under the cache label (no button anymore).
        if hasattr(self, "lbl_cache_status"):
            self.lbl_cache_status.setText(text)
            self.lbl_cache_status.setStyleSheet(f"color: {color}; font-size: 9px;")

    def _show_busy(self, msg):
        """Reveal the busy bar + label under the Select-DB button and paint once.

        Ref-counted via a LIFO message stack so several overlapping background
        tasks (e.g. the metadata read and the index build) can each show/hide
        it without prematurely hiding the others. The single processEvents()
        forces an immediate repaint so the feedback appears right away.
        """
        if not hasattr(self, "busy_bar"):
            return
        self._busy_msgs.append(msg)
        self.lbl_busy.setText(msg)
        self.lbl_busy.setVisible(True)
        self.busy_bar.setVisible(True)
        QApplication.processEvents()

    def _set_busy_msg(self, msg):
        """Update the visible busy text (top of stack) without changing count."""
        if hasattr(self, "busy_bar") and self._busy_msgs:
            self._busy_msgs[-1] = msg
            self.lbl_busy.setText(msg)

    def _hide_busy(self):
        """Pop one busy task; hide the bar only when none remain."""
        if not hasattr(self, "busy_bar"):
            return
        if self._busy_msgs:
            self._busy_msgs.pop()
        if self._busy_msgs:
            self.lbl_busy.setText(self._busy_msgs[-1])
        else:
            self.busy_bar.setVisible(False)
            self.lbl_busy.setVisible(False)

    def _start_db_query(self, fn, on_done, on_failed):
        """Run ``fn`` on a DbQueryWorker and route its result to the main thread.

        Keeps a reference to each live worker so it isn't garbage-collected
        mid-run, and drops it when finished.
        """
        w = DbQueryWorker(fn)
        if not hasattr(self, "_db_workers"):
            self._db_workers = []
        self._db_workers.append(w)
        w.done.connect(on_done)
        w.failed.connect(on_failed)
        # The list holds the only strong reference to the thread. Dropping it
        # inside the finished handler can free the QThread during signal
        # emission ("Destroyed while thread is still running"), so defer the
        # release to the next event-loop turn — same pattern as the preview
        # chunk loaders (_retire_loader).
        w.finished.connect(lambda: QTimer.singleShot(0, lambda: self._retire_db_worker(w)))
        w.start()

    def _retire_db_worker(self, w):
        """Release a finished metadata worker, outside its finished signal."""
        if w not in getattr(self, "_db_workers", []):
            return
        if not w.isFinished():
            QTimer.singleShot(50, lambda: self._retire_db_worker(w))
            return
        self._db_workers.remove(w)

    def _on_meta_load_failed(self, err):
        self._hide_busy()
        if is_corruption_error(err):
            self.report_corrupt_database(err)
            return
        self.log_message(f"Could not read database metadata: {err}")

    def report_corrupt_database(self, err, path=None, quiet=False):
        """Surface a damaged database once, loudly, with the recovery command.

        Corruption typically appears only when a query reaches the bad page, so
        without this the tool looks merely broken: the file opens, tables list,
        and then the tag list silently stays empty. Shown at most once per
        database so a burst of failing queries cannot stack dialogs.
        """
        path = path or self.db_path
        name = os.path.basename(path) if path else "the database"
        self.log_message(f"✗ {name} appears to be CORRUPTED: {err}")

        if quiet or getattr(self, "_corrupt_reported", None) == path:
            return
        self._corrupt_reported = path

        recover_hint = ""
        if path:
            root, ext = os.path.splitext(path)
            recover_hint = (
                f"\n\nTo attempt recovery into a NEW file (the original is not "
                f"modified):\n\n"
                f'sqlite3 "{path}" ".recover" | sqlite3 "{root}_recovered{ext or ".sqlite"}"')

        QMessageBox.critical(
            self, "Database File Is Damaged",
            f"{name} is readable at the start but SQLite reports it as damaged "
            f"partway through:\n\n    {err}\n\n"
            f"This is a problem with the file itself, not with FNT — usually an "
            f"interrupted copy or a write that was cut off while the recorder "
            f"was running. Preprocessing cannot use it as-is.\n\n"
            f"If the original is still on the recording machine, re-copying it "
            f"is more reliable than recovery."
            f"{recover_hint}")

    def _on_index_ready(self, src_path, path):
        self.preview_index_builder = None
        self._hide_busy()   # balanced with the _show_busy at build start
        if not self._index_build_is_current(src_path):
            # The user moved to another database while this was building. The
            # copy on disk is still valid for its own database and will be
            # picked up if they return to it — it just must not be adopted now.
            self.log_message(
                f"Indexed copy for {os.path.basename(src_path)} finished after "
                f"the database changed — not used for the current selection.")
            return
        self.preview_db_path = path
        mb = os.path.getsize(path) / 1e6
        self._set_index_status("Fast index: ready ✓", "#00aa00")
        self.log_message(
            f"Indexed copy ready ({mb:.0f} MB). Scrubbing is now ~80x faster; "
            f"the original database is untouched.")
        # Cached chunks came from the unindexed original but are byte-identical
        # in content, so they stay valid — only future reads get faster.

    def _on_index_failed(self, src_path, err):
        self.preview_index_builder = None
        self._hide_busy()   # balanced with the _show_busy at build start
        if not self._index_build_is_current(src_path):
            return
        self._set_index_status("Fast index: unavailable (using original)", "#cc5555")
        self.log_message(
            f"Could not build indexed copy ({err}). Preview still works, "
            f"reading directly from the original database.")

    def rebuild_preview_index(self):
        """Manual rebuild, e.g. after the source database has changed."""
        if not self.db_path or not self.table_name:
            QMessageBox.warning(self, "No Database", "Select a database and table first")
            return
        _, copy_path, meta_path = self._preview_index_paths()
        for p in (copy_path, meta_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception as e:
                self.log_message(f"Could not remove {os.path.basename(p)}: {e}")
        self.preview_db_path = self.db_path
        self.invalidate_preview_cache()
        self.ensure_preview_index_db()

    def _nearest_sample_ts(self, conn, selected_tags, ts):
        """Timestamp of the closest sample at or after ``ts``, else the last before it."""
        placeholders = ",".join(["?"] * len(selected_tags))
        try:
            row = conn.execute(
                f"SELECT MIN(timestamp) FROM {self.table_name} "
                f"WHERE shortid IN ({placeholders}) AND timestamp >= ?",
                list(selected_tags) + [ts]).fetchone()
            if row and row[0] is not None:
                return int(row[0])
            row = conn.execute(
                f"SELECT MAX(timestamp) FROM {self.table_name} "
                f"WHERE shortid IN ({placeholders}) AND timestamp <= ?",
                list(selected_tags) + [ts]).fetchone()
            if row and row[0] is not None:
                return int(row[0])
        except Exception as e:
            self.log_message(f"Could not locate nearest samples: {e}")
        return None
    def reset_to_defaults(self):
        """Reset all settings and state to defaults for a clean database switch."""
        # Clear identity config
        self.tag_identities = {}

        # Clear data state
        self.data = None
        self.available_tags = []
        # A new database gets its own corruption warning if it needs one.
        self._corrupt_reported = None

        # Clear background image and XML state
        self.xml_config_path = None
        self.background_image_path = None
        self.background_image = None
        self.xml_scale = None
        self.bg_width_meters = None
        self.bg_height_meters = None
        self.bg_scale = None
        self.bg_offset_x = 0.0
        self.bg_offset_y = 0.0
        self.xml_maps = []
        self.arena_zones = None
        self.rois = []
        # The list on screen still holds the outgoing trial's regions.
        # Clear it here rather than in the config loader, because a
        # trial with no config file never reaches the loader's restore.
        if hasattr(self, 'roi_list'):
            self._refresh_roi_list(keep=-1)
            self.on_rois_changed()
        self.anchor_positions = []

        # Drop all cached preview chunks so nothing leaks across databases. The
        # pane stays visible but goes inert until the new database's tags load.
        self._preview_active = False
        self.stop_preview_playback()
        self.preview_cache.clear()
        # Let in-flight reads finish before releasing them. Clearing the dict
        # outright would drop the last reference to a running QThread and abort
        # the process. These are short windowed reads (~5 ms indexed, ~0.4 s
        # unindexed), so the wait is not user-visible.
        for loader in list(self.preview_inflight.values()):
            if loader.isRunning():
                loader.wait(5000)
        self.preview_inflight.clear()
        self.preview_current_chunk = None
        self.preview_pending_current = None
        self.preview_t0 = self.preview_t1 = None
        self.preview_playhead_ms = 0
        self.preview_db_path = None
        self._tag_range_cache = None   # (key, ranges) for the identity dialog
        self.preview_x = self.preview_y = None
        self.preview_xf = self.preview_yf = None
        self.preview_raw_x = self.preview_raw_y = None
        self.preview_step = self.preview_speed = None
        self._preview_step_pool = None
        self.preview_batt = None
        self.preview_times = None
        self.preview_tags = []
        self.preview_arena = None
        self.xml_zones = []
        self.xml_map_image = None
        self.xml_map_extent = None
        self.xml_map_px = None
        self.xml_map_scale = None
        self._sync_bg_transform_controls()
        # Nothing is parsed yet, so no layer has data behind it; the next XML
        # parse / config load re-offers whatever this database actually has.
        self._sync_layer_toggles()
        if hasattr(self, 'slider_timeline'):
            self._timeline_guard = True
            self.slider_timeline.setRange(0, 0)
            self._timeline_guard = False
            self.lbl_preview_status.setText("Load a database and select tags to preview")
            self.lbl_preview_time.setText("--")
            self._update_cache_label()
            self._set_index_status("Fast index: not built", "#cccccc")
            self.preview_canvas_2d.clear()
            if self.preview_canvas_3d is not None:
                self.preview_canvas_3d.clear()

        # Reset pending table name
        if hasattr(self, 'pending_table_name'):
            delattr(self, 'pending_table_name')

        # Reset GUI widgets to defaults
        self.combo_timezone.setCurrentText("US/Mountain")
        self.combo_smoothing.setCurrentIndex(0)  # None (default)
        self.spin_rolling_window.setValue(30)
        self.combo_window_units.setCurrentText("Seconds")
        # Every preview-owned setting back to its startup value. This used to
        # reset only the smoothing window, units and the two thresholds, so
        # switching databases half-carried the previous trial's analysis over
        # - smoothing method, behaviour thresholds and the XY range persisted
        # while their companions were reset. Carrying settings between trials
        # is now a deliberate choice offered when a database is loaded (see
        # _offer_inherit_settings), not an accident of which widget happened
        # to be on this list.
        self.apply_preview_config(getattr(self, '_preview_defaults', None) or {})
        self.chk_velocity_filter.setChecked(True)
        self.spin_velocity_threshold.setValue(2.0)
        self.chk_jump_filter.setChecked(True)
        self.spin_jump_threshold.setValue(2.0)
        self.spin_time_gap.setValue(30)
        self.combo_color_by.setCurrentIndex(0)  # None

        # Export options
        self.chk_export_raw_csv.setChecked(False)
        self.chk_export_smoothed_csv.setChecked(True)
        self.chk_proximity_detection.setChecked(True)
        self.spin_proximity_threshold.setValue(0.5)
        self.chk_export_roi_bouts.setChecked(False)
        self.chk_export_roi_daily.setChecked(False)
        self.chk_outlier_filter.setChecked(True)
        self.spin_outlier_window.setValue(31)
        self.spin_outlier_sigma.setValue(6.0)
        self.spin_velocity_min_dt.setValue(0.50)
        self.chk_preview_outlier.setChecked(True)
        self.spin_preview_outlier_window.setValue(31)
        self.spin_preview_outlier_sigma.setValue(6.0)
        self.spin_preview_velocity_min_dt.setValue(0.50)
        self.chk_export_behavior.setChecked(False)
        self.on_social_network_toggled()
        self.chk_save_plots.setChecked(False)
        self.chk_save_svg.setChecked(False)
        for cb in self.plot_type_checkboxes.values():
            cb.setChecked(True)
        self.chk_save_animation.setChecked(False)
        self.chk_daily_animations.setChecked(False)
        self.chk_full_animation.setChecked(True)
        self.spin_animation_trail.setValue(500)
        self.spin_anim_tag_size.setValue(10)
        self.chk_show_battery_export.setChecked(False)
        self.combo_animation_speed.setCurrentText("60x")
        self.combo_animation_fps.setCurrentText("30")

        # Preview display defaults
        self.chk_show_tracking.setChecked(True)
        self.chk_show_tag_id.setChecked(False)
        self.combo_tag_id_type.setCurrentText("Display ID")
        self.spin_tag_size.setValue(10)
        self.combo_label_color.setCurrentIndex(0)      # Auto
        self.chk_label_outline.setChecked(True)
        self.chk_show_battery.setChecked(False)

        # Reset background image label
        if hasattr(self, 'lbl_background_status'):
            self.lbl_background_status.setText("No background image loaded")

        # Reset show anchor positions checkbox
        if hasattr(self, 'chk_show_anchors'):
            self.chk_show_anchors.setChecked(False)

    def select_database(self):
        """Select a SQLite database via a file dialog, then load it."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select SQLite Database", "",
            "SQLite Files (*.sqlite *.db *.sql);;All Files (*.*)"
        )
        if not file_path:
            return
        self._load_database_path(file_path)

    # Config keys that describe WHICH trial this is, not HOW it is analysed.
    # Everything else is inherited, so a setting added later is carried across
    # trials by default rather than being silently left behind.
    _TRIAL_SPECIFIC_CONFIG_KEYS = frozenset({
        'database_path', 'database_name', 'table_name',
        'selected_tags', 'tag_identities',
        'background_image_path', 'background_scale_inpx', 'background_offset_m',
        'arena_zones', 'xml_config', 'sitemap',
        # Regions were traced over THIS trial's arena and background; copying
        # them onto another trial would place boundaries that were never drawn
        # there.
        'rois', 'show_rois',
        'fnt_version', 'run_timestamp',
    })
    # Preview keys in the same category (nested under config['preview']).
    _TRIAL_SPECIFIC_PREVIEW_KEYS = frozenset({'arena_source'})

    def _offer_inherit_settings(self):
        """Ask whether to copy analysis settings from a queued trial.

        Called after a database has finished loading, only when there is a
        queue to copy from. Silent during batch runs and whenever dialogs are
        suppressed.
        """
        # _suppress_dialogs only swaps out QMessageBox; this is a QDialog,
        # so it has to check the same flag itself or it would block an
        # unattended run.
        if (getattr(self, '_batch_active', False)
                or getattr(self, '_saved_dialogs', None) is not None):
            return
        # Wait until the tag list on screen is THIS trial's. The scan runs
        # off-thread and is usually still going when this is first posted - on
        # a multi-GB database over a share, for a long while. Answering the
        # prompt before it lands means answering it against the previous
        # trial's widgets, which is how an inherited trial ended up with only
        # the tags the two cohorts had in common. Re-post rather than block, so
        # the window stays live while the scan runs.
        if getattr(self, '_tags_loaded_for', None) != (self.db_path,
                                                       self.table_name):
            waited = getattr(self, '_inherit_offer_waits', 0) + 1
            self._inherit_offer_waits = waited
            if waited * 0.15 < 120:        # give a slow share two minutes
                QTimer.singleShot(150, self._offer_inherit_settings)
            return
        self._inherit_offer_waits = 0
        items = [it for it in getattr(self, '_batch_items', [])
                 if (it.get('config') or {}) and it.get('path') != self.db_path]
        if not items:
            return

        dlg = InheritSettingsDialog(items, os.path.basename(self.db_path), self)
        if dlg.exec_() != QDialog.Accepted:
            return
        idx = dlg.chosen_index()
        if idx is None or not (0 <= idx < len(items)):
            return
        source = items[idx]
        self.apply_inherited_settings(source.get('config') or {})
        self.log_message(
            "Applied tracking/export settings from "
            f"{os.path.basename(source['path'])}; this trial's tags, "
            "identities and arena were kept.")
        self.lbl_config_status.setText(
            f"Settings copied from {os.path.basename(source['path'])}.")
        self.lbl_config_status.setVisible(True)

    def apply_inherited_settings(self, source_config):
        """Overlay another trial's analysis settings onto the loaded database.

        Built as a merge rather than a list of widget assignments: the source's
        config is taken wholesale, the trial-specific keys are replaced with
        this database's own values, and the result is replayed through the
        ordinary config loader. That way there is exactly one place that knows
        how to turn a config into widget state, so inheritance cannot drift out
        of step with loading.
        """
        if not isinstance(source_config, dict) or not source_config:
            return
        # The trial-specific keys are DROPPED, not overwritten with this
        # window's current values. Substituting a snapshot looked equivalent
        # and was not: the config loader skips any key that is absent, so
        # dropping leaves the loaded trial's own tags, identities and arena
        # exactly as its own load left them - whereas a snapshot pins whatever
        # the widgets happened to hold at the moment the prompt was answered,
        # and then FORCES it. When the prompt was answered before this trial's
        # tag scan landed, that snapshot was the PREVIOUS trial's tag list, and
        # forcing it left only the tags the two cohorts had in common.
        merged = {k: v for k, v in source_config.items()
                  if k not in self._TRIAL_SPECIFIC_CONFIG_KEYS}
        merged['preview'] = {
            k: v for k, v in (source_config.get('preview') or {}).items()
            if k not in self._TRIAL_SPECIFIC_PREVIEW_KEYS}
        # The day list belongs to this recording's dates, not the source's.
        merged.pop('daily_animation_days', None)

        self._pending_config_override = merged
        self.load_config_if_exists()

    def _load_database_path(self, file_path):
        """Load a database by path — shared by the file picker and the batch queue.

        Returns True if the database opened and a table was selected. In batch
        mode (``self._batch_active``) the interactive background-image prompt and
        the error/warning dialogs are suppressed so an unattended run never
        blocks on a modal box.
        """
        batch = getattr(self, '_batch_active', False)
        # Holds ~70 MB of classification arrays for the trial being left.
        PlotSaverWorker._BEH_CACHE = None
        # The tag list on screen still belongs to the outgoing trial until the
        # scan for this one lands.
        self._tags_loaded_for = None
        try:
            conn = connect_ro(file_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()

            if not tables:
                if not batch:
                    QMessageBox.warning(self, "No Tables", "No tables found in database")
                else:
                    self.log_message(f"✗ No tables in {os.path.basename(file_path)}")
                return False

            self.db_path = file_path
            self.lbl_db.setText(f"Selected: {os.path.basename(file_path)}")

            # Reset all settings to defaults before loading new config
            self.reset_to_defaults()

            # Load config BEFORE populating combo_table, so that pending_tag_selection
            # is set before on_table_selected() triggers tag checkbox creation
            config_loaded = self.load_config_if_exists()
            if config_loaded:
                self.lbl_config_status.setText("Loaded previous fnt_config.json file.")
                self.lbl_config_status.setVisible(True)
            else:
                self.lbl_config_status.setVisible(False)

            self.combo_table.clear()
            self.combo_table.addItems(tables)
            self.combo_table.setEnabled(True)
            self.btn_preview_table.setEnabled(True)
            self.btn_load_background.setEnabled(True)  # Enable background image loading

            # Apply saved table name from config (if any), otherwise default to first
            if hasattr(self, 'pending_table_name') and self.pending_table_name:
                index = self.combo_table.findText(self.pending_table_name)
                if index >= 0:
                    self.combo_table.setCurrentIndex(index)
                delattr(self, 'pending_table_name')
            elif len(tables) == 1:
                self.combo_table.setCurrentIndex(0)

            # Check for XML configuration file in the database directory
            self.load_xml_config()

            # Auto-load a background image only if one clearly belongs to this
            # database (name match). Otherwise offer to pick one (interactive only).
            self.auto_load_background()
            if self.background_image is None and not batch:
                self.prompt_background_image()

            # Offer to copy analysis settings from a trial already queued.
            # Deferred to the next event-loop turn rather than called here:
            # applying them replays the config loader, and re-entering it
            # from inside this load rebuilds the tag checkboxes that this
            # frame is still holding - which crashed the process.
            QTimer.singleShot(0, self._offer_inherit_settings)

            return True

        except Exception as e:
            if is_corruption_error(e):
                self.report_corrupt_database(e, path=file_path, quiet=batch)
            elif not batch:
                QMessageBox.critical(self, "Error", f"Failed to open database: {str(e)}")
            else:
                self.log_message(f"✗ Failed to open database {os.path.basename(file_path)}: {e}")
            return False

    def is_wiser_database(self):
        """Heuristic: does this look like a Wiser UWB export?

        True if the site XML has a <Wiser> root, or the table carries the
        Wiser column signature (shortid + location_x/y + timestamp).
        """
        if self.xml_config_path and os.path.exists(self.xml_config_path):
            try:
                root = ET.parse(self.xml_config_path).getroot()
                if root.tag.lower() == "wiser":
                    return True
            except Exception:
                pass
        if self.db_path and self.table_name:
            try:
                conn = connect_ro(self.db_path)
                cols = {r[1].lower() for r in conn.execute(f"PRAGMA table_info({self.table_name})")}
                conn.close()
                return {"shortid", "location_x", "location_y", "timestamp"}.issubset(cols)
            except Exception:
                pass
        return False

    def _find_candidate_backgrounds(self):
        """Visible image files in the DB folder, best name-match candidate first.

        Ranks a filename matching the database name ahead of one matching the
        XML config name, ahead of everything else, so the first entry is the
        most likely floorplan.
        """
        if not self.db_path:
            return []
        db_dir = os.path.dirname(self.db_path)
        exts = ('.png', '.jpg', '.jpeg', '.bmp')
        imgs = [f for f in list_visible_files(db_dir) if f.lower().endswith(exts)]
        if not imgs:
            return []
        db_name = os.path.splitext(os.path.basename(self.db_path))[0].lower()
        xml_name = ''
        if getattr(self, 'xml_config_path', None):
            xml_name = os.path.splitext(os.path.basename(self.xml_config_path))[0].lower()

        def rank(f):
            fname = os.path.splitext(f)[0].lower()
            if db_name and (db_name in fname or fname in db_name):
                return 0
            if xml_name and (xml_name in fname or fname in xml_name):
                return 1
            return 2

        return sorted(imgs, key=rank)

    def _confirm_background_preview(self, image_path):
        """Show ``image_path`` in a preview dialog; return True if the user accepts."""
        from PyQt5.QtGui import QPixmap

        dlg = QDialog(self)
        dlg.setWindowTitle("Use This Background Image?")
        layout = QVBoxLayout(dlg)

        intro = QLabel(f"Detected an image in the database folder:\n\n{os.path.basename(image_path)}")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        preview = QLabel()
        preview.setAlignment(Qt.AlignCenter)
        pix = QPixmap(image_path)
        if not pix.isNull():
            pix = pix.scaled(480, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            preview.setPixmap(pix)
        else:
            preview.setText("(Could not render a preview of this image.)")
        layout.addWidget(preview)

        question = QLabel("Use this image as the arena background?")
        question.setWordWrap(True)
        layout.addWidget(question)

        btn_row = QHBoxLayout()
        btn_no = QPushButton("No")
        btn_yes = QPushButton("Yes, use this image")
        btn_yes.setDefault(True)
        btn_no.clicked.connect(dlg.reject)
        btn_yes.clicked.connect(dlg.accept)
        btn_row.addStretch()
        btn_row.addWidget(btn_no)
        btn_row.addWidget(btn_yes)
        layout.addLayout(btn_row)

        return dlg.exec_() == QDialog.Accepted

    def prompt_background_image(self):
        """Detect a floorplan image in the DB folder, preview it, and confirm use.

        If one or more images are present, the best-matching candidate is shown
        in a preview dialog for the user to accept or decline. Declining (or a
        folder with no images) falls back to the manual file picker.
        """
        if not self.db_path:
            return

        db_dir = os.path.dirname(self.db_path)
        candidates = self._find_candidate_backgrounds()

        if candidates:
            candidate = os.path.join(db_dir, candidates[0])
            self.log_message(f"Detected candidate background image: {candidates[0]}")
            if self._confirm_background_preview(candidate):
                self._apply_background_image(candidate, source="loaded")
            else:
                # User declined the detected image — let them browse for another.
                self.select_background_image()
            return

        # Nothing detected — say so, then offer the manual picker.
        self.log_message("No background image (PNG/JPG) detected in the database folder.")
        reply = QMessageBox.question(
            self, "No Background Image Found",
            "No background image (PNG/JPG) was detected in the database folder.\n\n"
            "Would you like to load one?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.select_background_image()
    
    def load_xml_config(self):
        """Look for XML configuration file in the database directory"""
        if not self.db_path:
            return
        
        db_dir = os.path.dirname(self.db_path)
        
        # Look for .xml files in the same directory (skip hidden/AppleDouble
        # sidecars like '._EchoConfiguration.xml' left by macOS on shares).
        xml_files = [f for f in list_visible_files(db_dir) if f.lower().endswith('.xml')]
        
        if not xml_files:
            self.log_message("No XML configuration file found in database directory")
            if not getattr(self, '_batch_active', False):
                QMessageBox.warning(
                    self, "XML Configuration Not Found",
                    "No XML configuration file found in the database folder.\n\n"
                    "Anchor positions and floorplan scale will not be available."
                )
            return
        
        # If multiple XML files, use the first one or one matching config/Config pattern
        xml_file = None
        for f in xml_files:
            if 'config' in f.lower():
                xml_file = f
                break
        if not xml_file:
            xml_file = xml_files[0]
        
        self.xml_config_path = os.path.join(db_dir, xml_file)
        self.log_message(f"Found XML config: {xml_file}")
        
        try:
            self.parse_xml_config()
        except Exception as e:
            self.log_message(f"Warning: Could not parse XML config: {str(e)}")
    
    def parse_xml_config(self):
        """Parse XML configuration file and check for background image.

        Idempotent per file. Restoring a saved config parses the XML to recover
        the background scale, and the normal load path parses it again moments
        later; the second pass reads the same file, sets the same fields, and
        re-logs the whole scale / zones / anchors / site-map block. Keyed on
        (path, mtime, size) so editing the XML mid-session still re-reads it.
        """
        if not self.xml_config_path or not os.path.exists(self.xml_config_path):
            return
        try:
            st = os.stat(self.xml_config_path)
            stamp = (self.xml_config_path, st.st_mtime, st.st_size)
        except OSError:
            stamp = None
        if stamp is not None and getattr(self, '_parsed_xml_stamp', None) == stamp:
            return

        try:
            tree = ET.parse(self.xml_config_path)
            root = tree.getroot()
            
            # Extract scale attribute (inches/pixel)
            for elem in root.iter():
                if 'scale' in elem.attrib:
                    try:
                        self.xml_scale = float(elem.attrib['scale'])
                        self.log_message(f"Found XML scale: {self.xml_scale} inches/pixel")
                        break
                    except:
                        pass
            
            # Parse zone coordinates from Zones section
            zones_element = root.find('Zones')
            if zones_element is not None:
                zone_data = []
                self.xml_zones = []
                for zone in zones_element.findall('Zone'):
                    zone_name = zone.get('name')
                    if zone_name is None:
                        continue

                    shape = zone.find('Shape')
                    if shape is None:
                        continue

                    # Keep the polygon intact (with its authored colour) for the
                    # XML site view; the flat zone_data table below stays as-is
                    # for the existing config/plot code paths.
                    poly = []
                    for point in shape.findall('Point'):
                        try:
                            poly.append((float(point.get('x')) * 0.0254,
                                         float(point.get('y')) * 0.0254))
                        except (TypeError, ValueError):
                            continue
                    if len(poly) >= 3:
                        self.xml_zones.append({
                            'name': zone_name,
                            'color': zone.get('color', '#888888'),
                            'points': np.array(poly, float),
                        })

                    for point in shape.findall('Point'):
                        x_str = point.get('x')
                        y_str = point.get('y')
                        
                        if x_str is not None and y_str is not None:
                            try:
                                # Convert coordinates to meters (inches to meters)
                                x_meters = float(x_str) * 0.0254
                                y_meters = float(y_str) * 0.0254
                                zone_data.append({
                                    'zone': zone_name,
                                    'x': x_meters,
                                    'y': y_meters
                                })
                            except ValueError:
                                continue
                
                if zone_data:
                    self.arena_zones = pd.DataFrame(zone_data)
                    num_zones = len(self.arena_zones['zone'].unique())
                    self.log_message(f"Parsed {num_zones} zones with {len(zone_data)} coordinate points from XML")
                else:
                    self.log_message("No valid zone coordinates found in XML")
            else:
                self.log_message("No Zones section found in XML")

            # Parse anchor positions
            self.anchor_positions = []
            for anchor in root.iter('Anchor'):
                try:
                    shortid = int(anchor.get('shortid', '0'))
                    x_hex = anchor.get('x', '0x0')
                    y_hex = anchor.get('y', '0x0')
                    z_hex = anchor.get('z', '0x0')

                    # Decode IEEE 754 hex-encoded doubles
                    x_inches = struct.unpack('d', struct.pack('Q', int(x_hex, 16)))[0]
                    y_inches = struct.unpack('d', struct.pack('Q', int(y_hex, 16)))[0]
                    z_inches = struct.unpack('d', struct.pack('Q', int(z_hex, 16)))[0]

                    # Convert inches to meters
                    self.anchor_positions.append({
                        'shortid': shortid,
                        'x': x_inches * 0.0254,
                        'y': y_inches * 0.0254,
                        'z': z_inches * 0.0254
                    })
                except (ValueError, struct.error):
                    continue

            if self.anchor_positions:
                self.log_message(f"Parsed {len(self.anchor_positions)} anchor positions from XML")
            else:
                self.log_message("No anchor positions found in XML")

            # Decode the site map. Wiser writes it as a base64 data URI in the
            # Map element's *attribute* (map="data:image/png;base64,..."), not
            # as element text, so pull it from there and fall back to text for
            # any other dialect.
            self.xml_map_image = None
            self.xml_map_extent = None
            self.xml_map_px = None
            self.xml_map_scale = None
            self.xml_maps = []
            for elem in root.iter():
                if elem.tag not in ('Map', 'BackgroundImage', 'Image'):
                    continue
                payload = elem.get('map') or elem.get('image') or (elem.text or '')
                payload = payload.strip()
                if len(payload) < 200:
                    continue
                try:
                    b64 = payload.split(',', 1)[1] if ',' in payload else payload
                    img = plt.imread(io.BytesIO(base64.b64decode(b64)), format='png')
                    px_scale = float(elem.get('scale', self.xml_scale or 1.0))
                    h_px, w_px = img.shape[:2]
                    w_m = w_px * px_scale * 0.0254
                    h_m = h_px * px_scale * 0.0254
                    # Record every render so a loaded external image can be
                    # matched to the map it came from (and its correct scale).
                    self.xml_maps.append({
                        'name': elem.get('name', ''),
                        'scale': px_scale, 'w_px': w_px, 'h_px': h_px})
                    if self.xml_map_image is None:
                        # First decodable map drives the XML site view.
                        self.xml_map_image = img
                        self.xml_map_px = (w_px, h_px)
                        self.xml_map_scale = px_scale
                        # With no floorplan loaded, the map owns the manual
                        # transform: seed it from the map's OWN scale so the
                        # alignment box reads a real number and can nudge the
                        # map onto the surveyed frame. A scale already restored
                        # from the config is a saved nudge — leave it alone.
                        if self.background_image is None and not self.bg_scale:
                            self.bg_scale = px_scale
                        self._recompute_xml_map_extent()
                    map_label = elem.get('name') or ''
                    map_label = f" '{map_label}'" if map_label else ''
                    self.log_message(
                        f"Decoded embedded site map{map_label}: "
                        f"{w_px}x{h_px} px @ {px_scale} in/px -> {w_m:.2f} x {h_m:.2f} m")
                except Exception as e:
                    self.log_message(f"Could not decode embedded map: {e}")

            if self.xml_map_image is None:
                self.log_message("No embedded background image found in XML")

            # One pass over the layer toggles now that zones, anchors and the
            # site map are all known: each is offered only if this XML actually
            # carries it, whatever the previous database had.
            self._sync_layer_toggles()

            # Set last: a parse that threw part-way leaves the marker unset so
            # the next call retries rather than trusting half-filled state.
            self._parsed_xml_stamp = stamp

        except Exception as e:
            self.log_message(f"Error parsing XML: {str(e)}")
    
    def auto_load_background(self):
        """Silently load a background image whose name clearly matches this DB.

        Only fires on a strong name match (DB name or XML config name); an
        arbitrary lone image (e.g. an ABMA model config PNG) is left for the
        preview/confirm prompt so the user picks deliberately.
        """
        if not self.db_path or self.background_image is not None:
            return  # Already loaded (e.g., from config) or no database

        db_dir = os.path.dirname(self.db_path)
        db_name = os.path.splitext(os.path.basename(self.db_path))[0]

        # Candidates ranked by name match; only accept an actual match here.
        candidates = self._find_candidate_backgrounds()
        xml_name = ''
        if getattr(self, 'xml_config_path', None):
            xml_name = os.path.splitext(os.path.basename(self.xml_config_path))[0].lower()
        selected = None
        for f in candidates:
            fname = os.path.splitext(f)[0].lower()
            if (db_name.lower() in fname or fname in db_name.lower() or
                    (xml_name and (xml_name in fname or fname in xml_name))):
                selected = f
                break

        if not selected:
            return

        self._apply_background_image(os.path.join(db_dir, selected), source="auto-loaded")

    @staticmethod
    def _name_tokens(s):
        """Significant (len>=2) alnum tokens of a name, lowercased.

        'Dark2d' -> ['dark'] ('2'/'d' dropped as too short); 'default' ->
        ['default']. Used to loosely match a map name against a filename.
        """
        return [t for t in re.findall(r'[a-z]+|\d+', (s or '').lower()) if len(t) >= 2]

    def _default_bg_scale(self, w_px, h_px, filename=None):
        """Best XML scale (in/px) for an external image about to be placed.

        Wiser writes several embedded renders, each with its own scale (e.g. a
        plain 'default' map and a decorated 'Dark2d' one at a different
        resolution). Identify which render the loaded image is, in order:

        1. exact pixel-resolution match to an embedded map (most reliable);
        2. unambiguous filename<->map-name match (e.g. an FNT ABMA screenshot
           'ABMA_VoleTerra2D_DarkModel...' -> map 'Dark2d'), for a re-exported
           image whose resolution no longer matches the embedded render;

        then fall back to the first XML scale, then None (1 in/px upstream).
        """
        maps = getattr(self, 'xml_maps', None) or []
        for m in maps:
            if m['w_px'] == w_px and m['h_px'] == h_px:
                return m['scale']
        if filename:
            stem = self._name_tokens(os.path.splitext(os.path.basename(filename))[0])
            fname_norm = ''.join(stem)
            hits = []
            for m in maps:
                toks = self._name_tokens(m.get('name', ''))
                if toks and all(t in fname_norm for t in toks):
                    hits.append(m)
            if len(hits) == 1:   # only trust an unambiguous name match
                return hits[0]['scale']
        return self.xml_scale

    def _recompute_bg_size(self):
        """Set bg_width/height_meters from the current image and bg_scale."""
        if self.background_image is None or not self.bg_scale:
            return
        h_px, w_px = self.background_image.shape[:2]
        self.bg_width_meters = w_px * self.bg_scale * 0.0254
        self.bg_height_meters = h_px * self.bg_scale * 0.0254

    def _xml_map_owns_transform(self):
        """True when the manual transform is placing the XML site map.

        A loaded floorplan takes precedence as the drawn background, so it owns
        the transform; the XML map only owns it when there is no loaded image.
        """
        return self.background_image is None and self.xml_map_image is not None

    def _recompute_xml_map_extent(self):
        """Place the XML site map, honouring the manual transform when it owns it.

        The map's authored scale (the <Map scale=...> attribute) is its default,
        but that render need not land on the surveyed zone/anchor frame — the
        arena it draws can be inset and rotated relative to the survey. Scale and
        offset let the drawn map be registered onto that frame; rotation is not
        offered, so a rotated render can only be approximated.

        Writes self.xml_map_extent, which every consumer reads: the preview
        background, the exported figures, the axis fit, and the extent recorded
        in the exported site map's config entry.
        """
        if self.xml_map_image is None or not self.xml_map_px:
            self.xml_map_extent = None
            return
        w_px, h_px = self.xml_map_px
        if self._xml_map_owns_transform():
            scale = self.bg_scale or self.xml_map_scale or self.xml_scale or 1.0
            x0, y0 = self.bg_offset_x, self.bg_offset_y
        else:
            # A loaded image owns the transform, so the map falls back to the
            # placement the XML authored for it.
            scale = self.xml_map_scale or self.xml_scale or 1.0
            x0 = y0 = 0.0
        # origin='upper' when drawn, so y runs top-down from the top edge.
        self.xml_map_extent = (x0, x0 + w_px * scale * 0.0254,
                               y0, y0 + h_px * scale * 0.0254)

    def _bg_placed_extent(self):
        """(x0, x1, y0, y1) metres for the loaded background, honouring offsets.

        The image spans bg_width/height_meters (from pixels * bg_scale) with its
        bottom-left corner placed at (bg_offset_x, bg_offset_y). Returns None
        when there is no sized background image.
        """
        if self.background_image is None or not self.bg_width_meters:
            return None
        x0, y0 = self.bg_offset_x, self.bg_offset_y
        return (x0, x0 + self.bg_width_meters, y0, y0 + self.bg_height_meters)

    def _apply_background_image(self, file_path, source="loaded"):
        """Load ``file_path`` as the arena background and refresh UI/preview.

        Seeds the manual transform (scale + offset) from the XML — matching the
        image resolution to the right embedded map when several exist — then
        computes real-world dimensions, updates the status label and preview,
        and returns True on success. ``source`` is a short verb used in the log
        line ("loaded"/"auto-loaded").
        """
        try:
            self.background_image = plt.imread(file_path)
        except Exception as e:
            self.log_message(f"Error loading background image: {str(e)}")
            self.background_image = None
            self.background_image_path = None
            self.bg_width_meters = None
            self.bg_height_meters = None
            self.lbl_background_status.setText("Error loading background image")
            self.lbl_background_status.setStyleSheet("color: #aa0000; font-style: italic; font-size: 9px;")
            return False

        self.background_image_path = file_path
        name = os.path.basename(file_path)
        img_height_px, img_width_px = self.background_image.shape[:2]

        # Seed the manual transform: default scale from the matching XML map,
        # offsets reset to 0 (bottom-left corner at world origin, the Wiser
        # frame). The user can nudge these live via the Preview controls.
        self.bg_offset_x = 0.0
        self.bg_offset_y = 0.0
        matched = self._default_bg_scale(img_width_px, img_height_px, file_path)
        self.bg_scale = matched if matched else 1.0
        self._recompute_bg_size()
        # This image now owns the transform, so the XML map (no longer drawn)
        # falls back to the placement the XML authored for it.
        self._recompute_xml_map_extent()
        if matched:
            self.log_message(
                f"✓ Background {source}: {name} "
                f"({self.bg_width_meters:.2f}m x {self.bg_height_meters:.2f}m "
                f"@ {self.bg_scale:.4f} in/px)")
        else:
            self.log_message(
                f"✓ Background {source}: {name} (no XML scale — using 1 in/px; "
                "dimensions may be inaccurate)")

        self.lbl_background_status.setText(f"✓ Background: {name}")
        self.lbl_background_status.setStyleSheet("color: #00aa00; font-style: normal; font-size: 9px;")
        self.btn_remove_background.setEnabled(True)
        self._sync_bg_transform_controls()

        # The background layer is off by default, so picking a file deliberately
        # has to turn it on — loading an image and seeing nothing happen is not
        # a defensible interaction. An auto-loaded or config-restored image does
        # NOT flip it: that would overrule a saved 'off' the user chose.
        if source == "loaded" and getattr(self, 'chk_show_background', None) is not None:
            self.chk_show_background.setChecked(True)

        self._log_background_alignment()

        # Redraw so the floorplan appears immediately. Every view mode draws it
        # now (the canvas takes one background from _context_bg_source), so the
        # view no longer has to be forced to 2D to see what was just loaded.
        self._sync_layer_toggles()
        if self._preview_active:
            self.refresh_preview_arena()
        return True

    def _log_background_alignment(self):
        """Log the numbers that expose an image-vs-tracking scale mismatch.

        The background's physical size is ``img_pixels * xml_scale``, but the
        anchors and zones come from absolute inch coordinates, so they are
        image-independent. If the loaded image's resolution differs from the map
        the XML scale was calibrated against (e.g. a re-rendered floorplan), the
        image ends up the wrong physical size even though its corner still sits
        at (0, 0). Comparing the image extent to the anchor bounding box makes
        that visible and suggests the scale that would fit.

        The comparison is only DIAGNOSTIC when the anchors ring the tracked
        area. On a rig whose anchors sit on a wall set back from the arena the
        span exceeds the image by design, so a verdict is offered only when
        both axes are off by the same factor - the signature of a real scale
        error, which is uniform and therefore preserves aspect.
        """
        if self.background_image is None:
            return
        h_px, w_px = self.background_image.shape[:2]
        self.log_message("── Background alignment check ─────────────")
        self.log_message(f"  Image: {w_px} x {h_px} px")
        self.log_message(
            f"  XML scale (first Map): {self.xml_scale} in/px" if self.xml_scale
            else "  XML scale: none — using 1 in/px fallback (likely wrong)")
        if self.bg_scale:
            res_match = any(m['w_px'] == w_px and m['h_px'] == h_px
                            for m in (getattr(self, 'xml_maps', None) or []))
            matched = self._default_bg_scale(w_px, h_px, self.background_image_path)
            note = ""
            if matched and abs(matched - self.bg_scale) < 1e-9 and matched != self.xml_scale:
                note = (" (matched to XML render of same resolution)" if res_match
                        else " (matched to XML render by filename)")
            self.log_message(f"  Effective bg scale: {self.bg_scale:.4f} in/px{note}")
        ext = self._bg_placed_extent()
        if ext:
            self.log_message(
                f"  Image placed extent: x[{ext[0]:.2f}, {ext[1]:.2f}] "
                f"y[{ext[2]:.2f}, {ext[3]:.2f}] m "
                f"(offset {self.bg_offset_x:.2f}, {self.bg_offset_y:.2f})")

        anchors = self.anchor_positions or []
        if anchors:
            xs = [a['x'] for a in anchors]
            ys = [a['y'] for a in anchors]
            ax0, ax1, ay0, ay1 = min(xs), max(xs), min(ys), max(ys)
            span_x, span_y = ax1 - ax0, ay1 - ay0
            self.log_message(
                f"  Anchors: {len(anchors)} — span x[{ax0:.2f}, {ax1:.2f}] "
                f"y[{ay0:.2f}, {ay1:.2f}] m = {span_x:.2f} x {span_y:.2f} m")
            if self.bg_width_meters and span_x > 0 and span_y > 0:
                rx = self.bg_width_meters / span_x
                ry = self.bg_height_meters / span_y
                self.log_message(
                    f"  Image / anchor-span ratio: x={rx:.2f}, y={ry:.2f}")
                # A SCALE error is uniform, so it preserves aspect and moves
                # both ratios together. Only then can one 'Bg scale' value fix
                # it, and only then is suggesting one honest. Ratios that
                # DISAGREE mean the image and the anchor box are different
                # shapes, which a single scale cannot reconcile - and the
                # ordinary cause is anchors mounted off the tracked area
                # (a wall set back from the arena), where the span SHOULD
                # exceed the image and nothing is wrong.
                agree = abs(rx - ry) <= 0.15 * max(rx, ry)
                if agree and not (0.9 <= rx <= 1.4):
                    self.log_message(
                        "  Both axes are off by the same factor, which is what "
                        "a mis-scaled image looks like (a re-rendered floorplan "
                        "at a different resolution).")
                    if self.bg_scale:
                        self.log_message(
                            f"  Scale to fit the anchor span: "
                            f"≈{self.bg_scale / rx:.4f} in/px (current "
                            f"{self.bg_scale:.4f}) — set via the Preview "
                            "'Bg scale' control. The arena is usually a little "
                            "larger than the anchor box, so nudge up from there.")
                elif agree:
                    self.log_message(
                        "  Consistent on both axes and close to 1 — the "
                        "image scale agrees with the anchors.")
                else:
                    # Deliberately NO scale suggestion here: the two axes want
                    # different numbers, so any single one is wrong, and
                    # entering it would break an alignment that may be correct.
                    self.log_message(
                        "  The axes disagree, so this is NOT a uniform scale "
                        "error and no single 'Bg scale' can fix it. Expected "
                        "when the anchors are not mounted around the tracked "
                        "area — a wall set back from the arena makes the "
                        "span exceed the image legitimately. Only treat it as "
                        "a fault if your anchors DO ring the tracked area.")
                if ext:
                    out = []
                    if ax0 < ext[0] or ax1 > ext[1]:
                        out.append("x")
                    if ay0 < ext[2] or ay1 > ext[3]:
                        out.append("y")
                    if out:
                        self.log_message(
                            f"  Anchors fall outside the image on {'/'.join(out)}"
                            f" — expected if they are mounted off the arena.")
        else:
            self.log_message("  No anchors parsed — cannot cross-check the image scale.")

        if getattr(self, 'xml_map_extent', None) is not None:
            e = self.xml_map_extent
            self.log_message(
                f"  Embedded XML site map extent: {e[1]:.2f} x {e[3]:.2f} m "
                "(what the XML scale actually calibrates)")
        self.log_message("──────────────────────────────────────────")

    def select_background_image(self):
        """Allow user to select a background image via the file picker."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Background Image",
            os.path.dirname(self.db_path) if self.db_path else "",
            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*.*)"
        )

        if file_path and os.path.exists(file_path):
            had_scale = bool(self.xml_scale)
            if self._apply_background_image(file_path, source="loaded") and not had_scale:
                QMessageBox.warning(
                    self, "No Scale Available",
                    "No XML scale has been loaded. The background image dimensions "
                    "may be incorrect.\n\nLoad an XML configuration file first for "
                    "accurate floorplan scaling."
                )

    def remove_background(self):
        """Remove the background image from visualizations"""
        self.background_image = None
        self.background_image_path = None
        self.bg_width_meters = None
        self.bg_height_meters = None
        # Hand the transform back to the XML site map, which becomes the active
        # background again: seed its own scale rather than leaving None behind.
        self.bg_scale = (self.xml_map_scale or self.xml_scale
                         if self.xml_map_image is not None else None)
        self.bg_offset_x = 0.0
        self.bg_offset_y = 0.0
        self._recompute_xml_map_extent()
        self.lbl_background_status.setText("No background image loaded")
        self.lbl_background_status.setStyleSheet("color: #666666; font-style: italic; font-size: 9px;")
        self.btn_remove_background.setEnabled(False)
        self._sync_bg_transform_controls()
        self.log_message("Background image removed")
        if self.preview_canvas_2d is not None:
            self.preview_canvas_2d.background_image = None
            self.preview_canvas_2d.bg_extent = None
        # The XML's embedded site map may still stand behind the layer, so let
        # the refresh below decide what the background toggle now offers.
        self._sync_layer_toggles()
        if self._preview_active:
            self.refresh_preview_arena()

    def on_show_background_toggled(self, _state=None):
        """Redraw the preview after a display toggle (background or anchors)."""
        if self._preview_active:
            self.refresh_preview_arena()

    def on_marker_style_changed(self, _value=None):
        """Redraw the tags without rebuilding the scene behind them.

        Marker size, tag IDs, battery/speed/step readouts and the tracking
        toggle all change per-FRAME artists only. Routing them through the
        arena rebuild re-rendered the background, the zones and the anchors to
        move a dot - and, until set_arena learned to recognise an unchanged
        framing, threw away the user's zoom every time the icon size was
        nudged.
        """
        if self._preview_active:
            self.render_preview_frame()

    def _sync_bg_transform_controls(self):
        """Push the current scale/offset onto the spinboxes without re-firing.

        Enabled whenever there is a background to place — a loaded floorplan OR
        the XML's own site map. The scale shown is the one actually in force for
        that image, never a placeholder: a disabled 1.0000 sitting under a map
        drawn at 0.299 in/px reads as a wrong number rather than an absent one.
        Signals are blocked so seeding the widgets does not trigger a redraw
        loop.
        """
        if not hasattr(self, 'spin_bg_scale'):
            return   # controls not built yet (early call during startup)
        have_img = (self.background_image is not None
                    or self.xml_map_image is not None)
        effective = self.bg_scale
        if not effective and self._xml_map_owns_transform():
            effective = self.xml_map_scale or self.xml_scale
        for w in (self.spin_bg_scale, self.spin_bg_offx, self.spin_bg_offy):
            w.blockSignals(True)
        self.spin_bg_scale.setValue(effective if effective else 1.0)
        self.spin_bg_offx.setValue(self.bg_offset_x)
        self.spin_bg_offy.setValue(self.bg_offset_y)
        for w in (self.spin_bg_scale, self.spin_bg_offx, self.spin_bg_offy):
            w.blockSignals(False)
        if hasattr(self, '_bg_transform_box'):
            self._bg_transform_box.setEnabled(have_img)

    def on_bg_transform_changed(self, _value=None):
        """Apply the manual scale + offset from the spinboxes and redraw.

        Re-sizes the background from the new scale, shifts it by the new offset,
        and refreshes the live preview so the user sees the alignment update as
        they nudge. Applies to whichever background is active — the loaded
        floorplan, or the XML site map when no floorplan is loaded.
        """
        if self.background_image is None and self.xml_map_image is None:
            return
        self.bg_scale = self.spin_bg_scale.value()
        self.bg_offset_x = self.spin_bg_offx.value()
        self.bg_offset_y = self.spin_bg_offy.value()
        self._recompute_bg_size()
        self._recompute_xml_map_extent()
        if self._preview_active:
            self.refresh_preview_arena()

    def reset_bg_transform(self):
        """Restore scale to the XML default for the active background, offset to 0.

        For a loaded floorplan that is the scale of the embedded render it
        matches; for the XML site map it is the scale the XML authored for it.
        """
        if self.background_image is not None:
            h_px, w_px = self.background_image.shape[:2]
            matched = self._default_bg_scale(w_px, h_px, self.background_image_path)
        elif self.xml_map_image is not None:
            matched = self.xml_map_scale or self.xml_scale
        else:
            return
        self.bg_scale = matched if matched else 1.0
        self.bg_offset_x = 0.0
        self.bg_offset_y = 0.0
        self._recompute_bg_size()
        self._recompute_xml_map_extent()
        self._sync_bg_transform_controls()
        self.log_message(
            f"Background transform reset to XML: scale {self.bg_scale:.4f} in/px, "
            "offset (0.00, 0.00) m")
        if self._preview_active:
            self.refresh_preview_arena()

    def on_preview_theme_changed(self, _state=None):
        """Switch the preview canvases between the dark and light palettes.
        Light is the default; the checkbox opts into dark."""
        theme = "dark" if self.chk_preview_dark.isChecked() else "light"
        self.preview_canvas_2d.set_theme(theme)
        if self.preview_canvas_3d is not None:
            self.preview_canvas_3d.set_theme(theme)
        if self._preview_active:
            self.refresh_preview_arena()
        else:
            self.preview_canvas_2d.show_placeholder()

    def on_table_selected(self, table_name):
        """Handle table selection"""
        if table_name:
            self.table_name = table_name
            self.load_tags_from_table()
    
    def preview_table(self):
        """Preview table data in a dialog"""
        if not self.db_path or not self.table_name:
            return
        
        try:
            # Load first 100 rows for preview
            conn = connect_ro(self.db_path)
            query = f"SELECT * FROM {self.table_name} LIMIT 100"
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Create preview dialog
            from PyQt5.QtWidgets import QDialog, QTableWidget, QTableWidgetItem, QHeaderView
            
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Preview: {self.table_name}")
            dialog.setGeometry(100, 100, 1000, 600)
            
            layout = QVBoxLayout()
            
            # Info label
            info_label = QLabel(f"Showing first 100 rows of {len(df)} columns")
            info_label.setStyleSheet("color: #cccccc; font-weight: bold; padding: 10px;")
            layout.addWidget(info_label)
            
            # Create table widget
            table = QTableWidget()
            table.setRowCount(len(df))
            table.setColumnCount(len(df.columns))
            table.setHorizontalHeaderLabels(df.columns.tolist())
            
            # Populate table
            for i in range(len(df)):
                for j in range(len(df.columns)):
                    item = QTableWidgetItem(str(df.iloc[i, j]))
                    table.setItem(i, j, item)
            
            # Auto-resize columns
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            
            # Style table
            table.setStyleSheet("""
                QTableWidget {
                    background-color: #1e1e1e;
                    color: #cccccc;
                    gridline-color: #3f3f3f;
                    border: 1px solid #3f3f3f;
                }
                QHeaderView::section {
                    background-color: #2b2b2b;
                    color: #0078d4;
                    font-weight: bold;
                    padding: 4px;
                    border: 1px solid #3f3f3f;
                }
            """)
            
            layout.addWidget(table)
            
            # Close button
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.close)
            layout.addWidget(close_btn)
            
            dialog.setLayout(layout)
            dialog.exec_()
            
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", f"Failed to preview table: {str(e)}")
    
    def on_smoothing_changed(self, method):
        """Handle smoothing method change"""
        clean_method = method.replace(" (default)", "")
        is_ewma = clean_method == "Forward-Backward Exponentially Weighted Moving Average"
        is_rolling = clean_method in ("Rolling Average", "Rolling Median")
        # Savitzky-Golay sizes its own window, so only the rolling methods and
        # EWMA expose the parameter spinbox.
        needs_param = is_ewma or is_rolling

        self.lbl_rolling_window.setText("Span (samples):" if is_ewma else "Smoothing Window:")
        self.spin_rolling_window.setEnabled(needs_param)
        # These export widgets are inherited from the preview and stay hidden;
        # only their values matter. Re-showing them here is what made the span
        # row reappear whenever EWMA was selected.
        self.spin_rolling_window.setVisible(False)
        self.lbl_rolling_window.setVisible(False)
        self.combo_window_units.setVisible(False)

    def on_save_plots_toggled(self):
        """Handle save plots checkbox toggle"""
        enabled = self.chk_save_plots.isChecked()
        self.plot_types_widget.setVisible(enabled)
        self.svg_option_widget.setVisible(enabled)
    
    def on_proximity_detection_toggled(self):
        """Handle proximity detection checkbox toggle"""
        enabled = self.chk_proximity_detection.isChecked()
        self.proximity_threshold_widget.setVisible(False)  # inherited from the preview

    def on_social_network_toggled(self):
        """Keep the proximity threshold row hidden.

        It is inherited from the preview's social radius, never set here. The
        edge-list window control this method also used to manage went with the
        edge list itself.
        """
        if hasattr(self, 'proximity_threshold_widget'):
            self.proximity_threshold_widget.setVisible(False)

    def _recording_span_hours(self):
        """Total recording duration in hours (from the preview timeline, else the
        number of days), for the clip-length estimate."""
        t0 = getattr(self, 'preview_t0', None)
        t1 = getattr(self, 'preview_t1', None)
        if t0 is not None and t1 is not None and t1 > t0:
            return (t1 - t0) / 3_600_000.0
        n = len(getattr(self, 'daily_animation_day_checkboxes', {}) or {})
        return n * 24.0 if n else 0.0

    def on_save_animation_toggled(self):
        """Handle save animation checkbox toggle"""
        enabled = self.chk_save_animation.isChecked()
        self.animation_options_widget.setVisible(enabled)
    
    def update_frame_estimate(self):
        """Update estimated frame count based on animation settings and loaded data"""
        if self.data is None or 'Timestamp' not in self.data.columns:
            self.lbl_estimated_frames.setText("Estimated frames: -- (load data first)")
            return
        source_data = self.data

        try:
            # Get animation parameters
            fps = int(self.combo_animation_fps.currentText())
            speed_text = self.combo_animation_speed.currentText()
            speed_multiplier = int(speed_text.replace('x', ''))

            # Calculate frame interval (real seconds per frame)
            frame_interval = speed_multiplier / fps

            # Get total time span of data
            time_span = (source_data['Timestamp'].max() - source_data['Timestamp'].min()).total_seconds()
            
            # Estimate number of frames
            estimated_frames = int(time_span / frame_interval)
            
            # Format with commas for readability
            frames_formatted = f"{estimated_frames:,}"
            
            # Calculate estimated video duration
            video_duration = estimated_frames / fps
            
            if video_duration >= 60:
                duration_str = f"{video_duration/60:.1f} min"
            else:
                duration_str = f"{video_duration:.1f} sec"
            
            self.lbl_estimated_frames.setText(
                f"Estimated frames: {frames_formatted} (~{duration_str} video @ {fps} FPS)"
            )
        except Exception as e:
            self.lbl_estimated_frames.setText(f"Estimated frames: Error calculating ({str(e)})")
    
    def on_daily_animations_toggled(self):
        """Handle daily animations checkbox toggle"""
        enabled = self.chk_daily_animations.isChecked()
        self.daily_animation_days_widget.setVisible(enabled)
    
    def populate_animation_days_from_list(self, date_strings):
        """Populate day checkboxes from a list of date strings (works without loading full data)"""
        # Clear existing checkboxes
        for cb in self.daily_animation_day_checkboxes.values():
            cb.deleteLater()
        self.daily_animation_day_checkboxes.clear()

        if not date_strings:
            self._sync_all_days_checkbox()
            return
        
        # Create checkboxes for each day
        for i, date_str in enumerate(date_strings):
            cb = QCheckBox(f"Day {i+1}: {date_str}")
            cb.setChecked(True)  # Default to all days selected
            cb.toggled.connect(self._sync_all_days_checkbox)
            self.daily_animation_day_checkboxes[date_str] = cb
            self.daily_days_layout_inner.addWidget(cb)
        self._sync_all_days_checkbox()

    def on_all_days_clicked(self):
        """Master toggle: select every day, or clear every day."""
        boxes = list(self.daily_animation_day_checkboxes.values())
        if not boxes:
            self._sync_all_days_checkbox()
            return
        # Anything short of "all ticked" means the click selects all; only a
        # fully selected list is cleared by it.
        turn_on = not all(cb.isChecked() for cb in boxes)
        for cb in boxes:
            cb.blockSignals(True)
            cb.setChecked(turn_on)
            cb.blockSignals(False)
        self._sync_all_days_checkbox()

    def _sync_all_days_checkbox(self, *_):
        """Reflect the day checkboxes in the master, without re-triggering it."""
        master = getattr(self, 'chk_all_days', None)
        if master is None:
            return
        boxes = list(self.daily_animation_day_checkboxes.values())
        n_on = sum(1 for cb in boxes if cb.isChecked())
        master.blockSignals(True)
        if not boxes or n_on == len(boxes):
            master.setCheckState(Qt.Checked)
        elif n_on == 0:
            master.setCheckState(Qt.Unchecked)
        else:
            master.setCheckState(Qt.PartiallyChecked)
        master.setText(f"All days ({n_on}/{len(boxes)} selected)"
                       if boxes else "All days")
        master.setEnabled(bool(boxes))
        master.blockSignals(False)

    def populate_animation_days(self):
        """Populate day checkboxes based on loaded data (called after preview loads)"""
        # Clear existing checkboxes
        for cb in self.daily_animation_day_checkboxes.values():
            cb.deleteLater()
        self.daily_animation_day_checkboxes.clear()

        if self.data is None or 'Timestamp' not in self.data.columns:
            self._sync_all_days_checkbox()
            return

        # Get unique dates
        dates = pd.to_datetime(self.data['Timestamp']).dt.date.unique()
        dates = sorted(dates)
        date_strings = [date.strftime('%Y-%m-%d') for date in dates]
        
        # Use the list-based function
        self.populate_animation_days_from_list(date_strings)
        
        self.log_message(f"Found {len(dates)} unique days in dataset")
    
    def _tag_time_ranges(self, db_path, table_name, tags, tz_name):
        """Observed first/last fix per tag, as formatted local-time strings.

        Runs on a worker thread (no Qt access here). Strategy depends on
        whether the database carries the preview index on (shortid, timestamp):

        * INDEXED - one bare MIN() and one bare MAX() per tag. SQLite answers
          each from the index endpoints without reading rows, so the whole
          thing is a handful of seeks regardless of table size.
        * NOT INDEXED - a single grouped query over all tags. One full scan is
          the best available; asking per tag would mean one scan EACH, which
          measured ~5x slower than the grouped form on a 6 M-row table.
        """
        tz = pytz.timezone(tz_name)
        conn = connect_ro(db_path)
        try:
            indexed = bool(conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND tbl_name=? "
                "AND name=? LIMIT 1",
                (table_name, PreviewIndexBuilder.INDEX_NAME)).fetchone())
            raw = {}
            if indexed:
                for tag in tags:
                    lo = conn.execute(
                        f"SELECT MIN(timestamp) FROM {table_name} WHERE shortid=?",
                        (tag,)).fetchone()[0]
                    if lo is None:
                        continue
                    hi = conn.execute(
                        f"SELECT MAX(timestamp) FROM {table_name} WHERE shortid=?",
                        (tag,)).fetchone()[0]
                    raw[tag] = (lo, hi)
            else:
                placeholders = ','.join(['?'] * len(tags))
                for tag, lo, hi in conn.execute(
                        f"SELECT shortid, MIN(timestamp), MAX(timestamp) "
                        f"FROM {table_name} WHERE shortid IN ({placeholders}) "
                        f"GROUP BY shortid", list(tags)):
                    if lo is not None:
                        raw[tag] = (lo, hi)
        finally:
            conn.close()

        out = {}

        # To the millisecond, matching IdentityAssignmentDialog.TIME_FMT. The
        # source is epoch ms, so the precision was always there and was simply
        # being truncated away. %f gives microseconds; the last three digits of
        # a value built from ms are always zero, so slicing them is exact.
        def _fmt(ms):
            return (pd.Timestamp(ms, unit='ms', tz='UTC').tz_convert(tz)
                      .strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])

        for tag, (lo, hi) in raw.items():
            out[tag] = {'start': _fmt(lo), 'end': _fmt(hi)}
        return out

    def open_identity_dialog(self):
        """Open the dialog to assign identities to the selected tags.

        The per-tag first/last-fix lookup that seeds the time pickers used to
        run inline on the GUI thread against the ORIGINAL database, which meant
        a full table scan (no index on shortid) with every page pulled over the
        network — tens of seconds of frozen window on a large trial. It now runs
        on a worker thread, against the indexed preview copy when one exists,
        and the result is cached so reopening the dialog is instant.
        """
        if not self.available_tags:
            QMessageBox.warning(self, "No Tags", "Please load a database and table first")
            return

        selected_tags = [tag for tag, cb in self.tag_checkboxes.items() if cb.isChecked()]
        if not selected_tags:
            QMessageBox.warning(self, "No Tags Selected", "Please select at least one tag")
            return

        # Prefer the indexed copy the preview builds; it turns the scan into
        # index seeks. Falls back to the original when no copy exists yet.
        db = self.current_indexed_db() or self.preview_db_path or self.db_path
        tz_name = self.combo_timezone.currentText()
        key = (db, self.table_name, tuple(sorted(selected_tags)), tz_name)
        cached = getattr(self, '_tag_range_cache', None)
        if cached is not None and cached[0] == key:
            self._show_identity_dialog(selected_tags, cached[1])
            return

        if getattr(self, '_identity_query_running', False):
            return                      # a lookup is already in flight
        self._identity_query_running = True
        self.btn_assign_identities.setEnabled(False)
        self.btn_assign_identities.setText("Reading tag time ranges…")
        QApplication.setOverrideCursor(Qt.WaitCursor)

        table_name = self.table_name

        def _query():
            return self._tag_time_ranges(db, table_name, selected_tags, tz_name)

        def _done(ranges):
            self._identity_query_running = False
            self._tag_range_cache = (key, ranges)
            self._restore_identity_button()
            self._show_identity_dialog(selected_tags, ranges)

        def _failed(err):
            self._identity_query_running = False
            self._restore_identity_button()
            self.log_message(f"Warning: Could not query tag time ranges: {err}")
            # The ranges only seed the pickers' defaults, so a failure must not
            # block identity assignment.
            self._show_identity_dialog(selected_tags, {})

        self._start_db_query(_query, _done, _failed)

    def _retime_preview_for_identities(self):
        """Resize the timeline and drop cached chunks after a window change."""
        if getattr(self, '_preview_raw_bounds', None):
            keep = self.preview_playhead_ms
            if self._apply_timeline_bounds(self._preview_raw_bounds):
                # Keep the playhead where it was when it is still in range.
                self.preview_playhead_ms = min(max(keep, self.preview_t0),
                                               self.preview_t1)
                self._sync_timeline_to_playhead()
        self.invalidate_preview_cache()

    def _restore_identity_button(self):
        QApplication.restoreOverrideCursor()
        self.btn_assign_identities.setText("Configure Identities...")
        self.btn_assign_identities.setEnabled(
            any(cb.isChecked() for cb in self.tag_checkboxes.values()))

    def _show_identity_dialog(self, selected_tags, tag_time_ranges):
        """Run the dialog and apply the result. Ranges are already resolved."""
        dialog = IdentityAssignmentDialog(selected_tags, self.tag_identities,
                                          tag_time_ranges, self)
        if dialog.exec_() != QDialog.Accepted:
            return
        self.tag_identities = dialog.get_identities()
        self.log_message(f"Updated identities for {len(self.tag_identities)} tags")
        # The active windows just changed: cached chunks were built under the
        # old ones and the timeline may now cover a different span.
        self._retime_preview_for_identities()
        # Snapshot the export-details JSON to the analysis folder now, so it
        # exists and reflects the just-applied identities before any export.
        self.write_live_config()
        # Update tag checkbox labels to reflect new identities
        self.update_tag_labels()
        # Recolour preview markers by the newly-assigned sex (blue=M, red=F)
        if self._preview_active and self.preview_tags:
            self.preview_colors = self._preview_tag_colors(self.preview_tags)
            self.render_preview_frame()
        self.lbl_status.setText(f"Identity assignments saved for {len(self.tag_identities)} tags")

        # Auto-save configuration to JSON
        if self.db_path:
            db_dir = os.path.dirname(self.db_path)
            db_filename = os.path.basename(self.db_path)
            db_name = os.path.splitext(db_filename)[0]
            output_dir = os.path.join(db_dir, f"{db_name}_FNT_analysis")
            os.makedirs(output_dir, exist_ok=True)
            self.save_config(output_dir)
            self.log_message("✓ Configuration auto-saved to JSON")

    def stop_export(self):
        """Cancel ongoing export operations"""
        self.export_cancelled = True
        self.log_message("⚠ Export cancellation requested...")
        
        # Stop worker thread if running
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.log_message("✗ Plot export cancelled")

        # Drop any temp render CSVs left behind by the cancelled run.
        self._cleanup_plot_working_files()

        # Reset UI
        self.exporting = False
        self.btn_export.setEnabled(True)
        self.btn_stop_export.setVisible(False)
        self.progress_widget.setVisible(False)
        self.progress_bar.setValue(0)
        self.lbl_export_progress.setText("")
    
    def load_tags_from_table(self):
        """Load the tag list + unique days for the selected table, off the GUI thread.

        DISTINCT scans over a large, not-yet-indexed database on a network drive
        can take many seconds; running them on a background DbQueryWorker keeps
        the window responsive (with a busy indicator under the Select button)
        instead of freezing. UI population happens in _on_tags_days_loaded once
        the result comes back on the main thread.
        """
        if not self.db_path or not self.table_name:
            return
        db_path = self.db_path
        table = self.table_name

        # A batch job already carries the tag list in its captured config, and
        # only needs the day list when it is rendering per-day animations.
        # Re-deriving either means a DISTINCT scan of the whole table; on a
        # multi-GB Wiser database over SMB that is tens of seconds of network
        # I/O per trial, and it is where a worker died with an access
        # violation. Skip whatever the config can answer.
        cfg = getattr(self, '_loaded_config', None) or {}
        batch = getattr(self, '_batch_active', False)
        # An indexed copy turns both DISTINCT scans into index reads. Look
        # it up directly rather than via preview_db_path, which is not set
        # yet at this point in the load.
        scan_db = self.current_indexed_db() or self.preview_db_path or db_path
        tz_name = self.combo_timezone.currentText()
        if scan_db != db_path:
            self.log_message(
                "Reading tags/days from the indexed copy — the original is "
                "a full-table scan.")
        known_tags = list(cfg.get('selected_tags') or []) if batch else []
        needs_days = bool(cfg.get('daily_animations')) and bool(cfg.get('save_animation'))

        def _query():
            if known_tags and not needs_days:
                return {'db_path': db_path, 'table': table,
                        'tags': known_tags, 'days': []}
            conn = connect_ro(scan_db)
            try:
                tags = known_tags or [r[0] for r in conn.execute(
                    f"SELECT DISTINCT shortid FROM {table} ORDER BY shortid")]
                # Bucket to 15 minutes and convert in Python rather than
                # calling date(datetime(...)) on every row. Real UTC offsets
                # are all multiples of 15 minutes, so a bucket never straddles
                # two local dates and the answer is exact - but SQLite now
                # does one integer division per row instead of two date
                # conversions, and the covering index supplies the timestamps
                # without touching the wide unused TEXT columns.
                days = []
                if needs_days or not batch:
                    buckets = [r[0] for r in conn.execute(
                        f"SELECT DISTINCT timestamp/900000 FROM {table}")]
                    if buckets:
                        local = pd.to_datetime(
                            pd.Series(buckets, dtype="int64") * 900000,
                            unit="ms", utc=True).dt.tz_convert(tz_name)
                        days = sorted({d.strftime("%Y-%m-%d") for d in local})
            finally:
                conn.close()
            return {'db_path': db_path, 'table': table, 'tags': tags, 'days': days}

        # In an unattended batch there is no UI to keep responsive, and the
        # background worker + preview timers race with the fast sequential trial
        # stepping — so load synchronously and skip the async path entirely.
        if batch:
            if known_tags and not needs_days:
                self.log_message(
                    f"Using the {len(known_tags)} tag(s) captured with this job; "
                    "skipping the full-table tag/day scan.")
            try:
                self._on_tags_days_loaded(_query())
            except Exception as e:
                self.log_message(f"Could not read tags/days: {e}")
            return

        self._show_busy("Reading tags and days from database…")
        self._start_db_query(_query, self._on_tags_days_loaded, self._on_meta_load_failed)

    def _on_tags_days_loaded(self, res):
        self._hide_busy()
        # Ignore a result for a database/table the user has navigated away from.
        if res['db_path'] != self.db_path or res['table'] != self.table_name:
            return
        self.available_tags = res['tags']
        self.update_tag_selection()
        self.btn_export.setEnabled(True)
        if res['days']:
            self.populate_animation_days_from_list(res['days'])
            self.log_message(f"Found {len(res['days'])} unique days in database")
    
    def load_unique_days_from_database(self):
        """Load unique days from database without loading full data"""
        if not self.db_path or not self.table_name:
            return
        
        try:
            conn = connect_ro(self.db_path)
            # Query for distinct dates
            query = f"""
                SELECT DISTINCT date(datetime(timestamp/1000, 'unixepoch'), 'localtime') as date
                FROM {self.table_name}
                ORDER BY date
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if len(df) > 0:
                self.populate_animation_days_from_list(df['date'].tolist())
                self.log_message(f"Found {len(df)} unique days in database")
            
        except Exception as e:
            self.log_message(f"Could not load unique days: {str(e)}")
    
    def update_tag_selection(self):
        """Update tag checkboxes"""
        for cb in self.tag_checkboxes.values():
            cb.deleteLater()
        self.tag_checkboxes.clear()
        
        if self.lbl_no_tags:
            self.lbl_no_tags.deleteLater()
            self.lbl_no_tags = None
        
        # Remove existing buttons if they exist
        if hasattr(self, 'tag_buttons_layout'):
            while self.tag_buttons_layout.count():
                item = self.tag_buttons_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self.tag_layout.removeItem(self.tag_buttons_layout)
        
        if hasattr(self, 'btn_assign_identities'):
            self.btn_assign_identities.deleteLater()
        
        for tag in self.available_tags:
            hex_id = hex(tag).upper().replace('0X', '')
            # Show HexID with identity info only if user has configured it
            if tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', '')
                identity = info.get('identity', '')
                if sex and identity:
                    cb = QCheckBox(f"HexID {hex_id} ({sex}, {identity})")
                else:
                    cb = QCheckBox(f"HexID {hex_id}")
            else:
                cb = QCheckBox(f"HexID {hex_id}")
            cb.setChecked(True)
            cb.stateChanged.connect(self.update_identity_button_state)
            # The preview's frame columns and its timeline range both derive
            # from the tag selection, so a change has to rebuild both.
            cb.stateChanged.connect(self.on_tag_selection_changed)
            self.tag_checkboxes[tag] = cb
            self.tag_layout.addWidget(cb)
        
        # Add Select All/None buttons below checkboxes
        self.tag_buttons_layout = QHBoxLayout()
        btn_select_all = QPushButton("Select All")
        btn_select_all.setToolTip("Check all tags for export")
        btn_select_all.setStyleSheet("QPushButton { padding: 4px 8px; font-size: 10px; }")
        btn_select_all.clicked.connect(self.select_all_tags)
        btn_select_none = QPushButton("Select None")
        btn_select_none.setToolTip("Uncheck all tags")
        btn_select_none.setStyleSheet("QPushButton { padding: 4px 8px; font-size: 10px; }")
        btn_select_none.clicked.connect(self.select_none_tags)
        self.tag_buttons_layout.addWidget(btn_select_all)
        self.tag_buttons_layout.addWidget(btn_select_none)
        self.tag_layout.addLayout(self.tag_buttons_layout)

        # Add Configure Identities button below Select All/None
        self.btn_assign_identities = QPushButton("Configure Identities...")
        self.btn_assign_identities.clicked.connect(self.open_identity_dialog)
        self.btn_assign_identities.setEnabled(False)
        self.btn_assign_identities.setToolTip(
            "Assign sex (M/F), custom IDs, and active time windows to each tag. "
            "Time pickers default to the tag's first/last timestamp in your selected timezone."
        )
        self.btn_assign_identities.setStyleSheet("QPushButton { padding: 4px 8px; font-size: 10px; }")
        self.tag_layout.addWidget(self.btn_assign_identities)
        
        # Whose tags these are. Anything that must not act on the previous
        # trial's widgets can compare this against (db_path, table_name);
        # "no background worker running" is NOT the same test, because the
        # scan for a newly loaded database has not necessarily been posted yet.
        self._tags_loaded_for = (self.db_path, self.table_name)

        # Apply pending tag selection from loaded config
        self.apply_pending_tag_selection()

        # Update identity button state
        self.update_identity_button_state()

        # Tags now exist for this database — bring the live preview to life.
        # Checkbox creation above sets states before their signals are wired,
        # so no on_tag_selection_changed fired; activate explicitly.
        if not self._preview_active and self.selected_preview_tags():
            self.activate_preview()

    def update_identity_button_state(self):
        """Enable Configure Identities button if any tag is selected"""
        any_selected = any(cb.isChecked() for cb in self.tag_checkboxes.values())
        self.btn_assign_identities.setEnabled(any_selected)
    
    def update_tag_labels(self):
        """Update tag checkbox labels to reflect sex and ID information"""
        for tag, cb in self.tag_checkboxes.items():
            hex_id = hex(tag).upper().replace('0X', '')
            if tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', '')
                identity = info.get('identity', '')
                if sex and identity:
                    cb.setText(f"HexID {hex_id} ({sex}, {identity})")
                else:
                    cb.setText(f"HexID {hex_id}")
            else:
                cb.setText(f"HexID {hex_id}")
    
    def select_all_tags(self):
        """Select all tags"""
        for cb in self.tag_checkboxes.values():
            cb.setChecked(True)

    def select_none_tags(self):
        """Deselect all tags"""
        for cb in self.tag_checkboxes.values():
            cb.setChecked(False)
    
    def get_smoothing_method(self):
        """Get the current smoothing method name (stripped of UI hints like '(default)')"""
        text = self.combo_smoothing.currentText()
        return text.replace(" (default)", "")

    def apply_smoothing_to_data(self, data, method, window=None, units=None):
        """Apply smoothing to a dataframe (works on any dataframe, not just self.data).

        ``window`` (samples/seconds) and ``units`` ("Seconds"/"Samples")
        override the Export Options spinbox/selector when given, so the live
        preview can drive smoothing from its own independent window controls.
        """
        win = self.spin_rolling_window.value() if window is None else window
        use_seconds = ((self.combo_window_units.currentText() == "Seconds")
                       if units is None else (units == "Seconds"))

        def apply_savgol(group):
            window_length = min(31, len(group))
            if window_length % 2 == 0:
                window_length -= 1
            polyorder = min(2, window_length - 1)
            if len(group) > polyorder:
                return savgol_filter(group, window_length=window_length, polyorder=polyorder)
            return group

        if method == "Savitzky-Golay":
            data['smoothed_x'] = data.groupby('shortid')['location_x'].transform(apply_savgol)
            data['smoothed_y'] = data.groupby('shortid')['location_y'].transform(apply_savgol)
        elif method in ("Rolling Average", "Rolling Median"):
            # The window value is interpreted as seconds (time-based, centred on
            # each tag's Timestamp) or as a fixed sample count, per the units
            # selector next to the spinbox. Time-based is the default and is
            # independent of the irregular, sub-1 Hz reporting rate.
            agg = 'mean' if method == "Rolling Average" else 'median'
            rolling_smooth_xy(data, agg, win, use_seconds)
        elif method == "Forward-Backward Exponentially Weighted Moving Average":
            # Window doubles as the EWMA span. No minimum floor — a span of
            # 1-2 is legitimate (span=1 gives alpha=1, i.e. passthrough).
            span = max(1, win)

            data['smoothed_x'] = data.groupby('shortid')['location_x'].transform(
                lambda x: forward_backward_ewma(x, span))
            data['smoothed_y'] = data.groupby('shortid')['location_y'].transform(
                lambda x: forward_backward_ewma(x, span))

        return data
    
    def reset_filter_stats(self):
        """Clear accumulated filter statistics at the start of an export run."""
        self.filter_stats = {}

    def _filter_and_smooth(self, data, smoothing_method, *, collect_stats=True,
                           velocity=None, jump=None, velocity_thresh=None,
                           jump_thresh=None, window=None, units=None,
                           outlier=None, outlier_window=None,
                           outlier_sigma=None, velocity_min_dt=None,
                           adjacency=None, excursion_s=None):
        """Threshold, then smooth (the original order).

        BOTH the velocity and jump thresholds run on the raw coordinates BEFORE
        smoothing, so the noisy/teleport fixes are removed and the smoother gets
        clean input. (Thresholding the velocity on the smoothed track instead
        let the noise into the smoother and produced a jumpy result.)
        """
        use_vel = self.chk_velocity_filter.isChecked() if velocity is None else velocity
        use_jmp = self.chk_jump_filter.isChecked() if jump is None else jump
        use_out = (self.chk_outlier_filter.isChecked()
                   if outlier is None else outlier)

        if (use_vel or use_jmp or use_out) and len(data):
            data = self.apply_filters_to_data(
                data, collect_stats=collect_stats, velocity=use_vel, jump=use_jmp,
                velocity_thresh=velocity_thresh, jump_thresh=jump_thresh,
                outlier=use_out, outlier_window=outlier_window,
                outlier_sigma=outlier_sigma, velocity_min_dt=velocity_min_dt,
                adjacency=adjacency, excursion_s=excursion_s)
        if smoothing_method != "None" and len(data):
            data = self.apply_smoothing_to_data(data, smoothing_method,
                                                window=window, units=units)
        return data

    def apply_filters_to_data(self, data, collect_stats=True, velocity=None, jump=None,
                              velocity_thresh=None, jump_thresh=None,
                              x_col='location_x', y_col='location_y',
                              outlier=None, outlier_window=None,
                              outlier_sigma=None, velocity_min_dt=None,
                              adjacency=None, excursion_s=None):
        """Apply velocity and jump thresholding with time-window grouping.

        ``velocity``/``jump`` override whether each threshold is applied and
        ``velocity_thresh``/``jump_thresh`` override the cutoff values; when None
        they follow the Export checkboxes/spinboxes. The preview passes its own
        toggles and values so the user can tune thresholding live.

        ``x_col``/``y_col`` choose the coordinates distance/velocity are measured
        on (default the raw location; thresholding runs before smoothing).
        """
        use_velocity = self.chk_velocity_filter.isChecked() if velocity is None else velocity
        use_jump = self.chk_jump_filter.isChecked() if jump is None else jump
        use_outlier = (self.chk_outlier_filter.isChecked()
                       if outlier is None else outlier)
        initial_count = len(data)
        removed_velocity = 0
        removed_jump = 0
        removed_outlier = 0
        
        # Make explicit copy at start to avoid any SettingWithCopyWarning
        data = data.copy()
        
        # Calculate time differences and group by time gaps (prevents filtering across battery restarts)
        time_gap_threshold = self.spin_time_gap.value()
        data['time_diff'] = data.groupby('shortid')['Timestamp'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds()
        data['time_diff_s'] = np.ceil(data['time_diff']).astype(int)
        data['tw_group'] = data.groupby('shortid')['time_diff_s'].apply(
            lambda x: (x > time_gap_threshold).cumsum()
        ).reset_index(level=0, drop=True)
        
        # Calculate distance and velocity within time window groups, on the
        # requested coordinate columns (raw for jump, smoothed for velocity).
        xc = x_col if x_col in data.columns else 'location_x'
        yc = y_col if y_col in data.columns else 'location_y'
        data['distance'] = np.sqrt(
            (data[xc] - data.groupby(['shortid', 'tw_group'])[xc].shift())**2 +
            (data[yc] - data.groupby(['shortid', 'tw_group'])[yc].shift())**2
        )
        # A minimum time base. The raw interval can be a millisecond, and
        # dividing by it turns a few centimetres of positional noise into
        # tens of m/s - so an unfloored velocity threshold mostly measures how
        # closely spaced the reports were, not how fast the animal moved.
        min_dt = (self.spin_velocity_min_dt.value()
                  if velocity_min_dt is None else velocity_min_dt)
        data['velocity'] = data['distance'] / np.maximum(
            data['time_diff'], float(min_dt) if min_dt else 0.0).replace(
                0, np.nan)

        # Spatial-outlier pass FIRST. It is the only one of the three that
        # survives a run of consecutive ghosts, and clearing them here stops
        # the step-based tests below from deleting the GOOD fix that follows a
        # ghost (they measure the distance INTO a fix, so a ghost condemns its
        # innocent successor as well).
        if use_outlier:
            win = (self.spin_outlier_window.value()
                   if outlier_window is None else outlier_window)
            sig = (self.spin_outlier_sigma.value()
                   if outlier_sigma is None else outlier_sigma)
            before_outlier = len(data)
            keep = hampel_keep(
                data[xc].values, data[yc].values, window=win, n_sigma=sig,
                group=data.groupby(['shortid', 'tw_group']).ngroup().values)
            data = data[keep].copy()
            removed_outlier = before_outlier - len(data)
            if removed_outlier > 0 and collect_stats:
                self.log_message(
                    f"  Removed {removed_outlier} spatial outliers "
                    f"(>{sig:g} MAD of a {int(win)}-sample local median)")
            # Steps and speeds describe the track, and the track just changed.
            data['distance'] = np.sqrt(
                (data[xc] - data.groupby(['shortid', 'tw_group'])[xc].shift())**2 +
                (data[yc] - data.groupby(['shortid', 'tw_group'])[yc].shift())**2
            )
            data['velocity'] = data['distance'] / np.maximum(
                data['time_diff'], float(min_dt) if min_dt else 0.0).replace(
                    0, np.nan)

        # Apply velocity filtering if enabled
        if use_velocity:
            velocity_threshold = (self.spin_velocity_threshold.value()
                                  if velocity_thresh is None else velocity_thresh)
            before_velocity = len(data)
            data = data[(data['velocity'] <= velocity_threshold) | (data['velocity'].isna())].copy()
            removed_velocity = before_velocity - len(data)
            if removed_velocity > 0 and collect_stats:
                self.log_message(f"  Removed {removed_velocity} points with "
                                 f"velocity > {velocity_threshold:g} m/s")
        else:
            removed_velocity = 0

        # Apply jump filtering if enabled
        if use_jump:
            jump_threshold = (self.spin_jump_threshold.value()
                              if jump_thresh is None else jump_thresh)
            before_jump = len(data)
            data['is_jump'] = (data['distance'] > jump_threshold)
            data = data[~data['is_jump']].copy()
            removed_jump = before_jump - len(data)
            if removed_jump > 0 and collect_stats:
                self.log_message(f"  Removed {removed_jump} points with "
                                 f"distance jump > {jump_threshold:g} m")
        else:
            removed_jump = 0
        
        # Everything above measured each point against its neighbour in the
        # ORIGINAL sequence and applied one mask. Removal changes who is
        # adjacent to whom, and a run of bad fixes loses only its first and
        # last members that way - the interior survives, joined to the real
        # track by a stride nothing has looked at. Re-test what is left.
        removed_adjacency = 0
        use_adj = (self.chk_adjacency_filter.isChecked()
                   if adjacency is None else adjacency)
        exc_s = (self.spin_excursion_s.value()
                 if excursion_s is None else excursion_s)
        if use_adj and (use_velocity or use_jump) and len(data) > 2:
            _t = (data['Timestamp'].values.astype('datetime64[ns]')
                  .astype('int64') / 1e9)
            _keep = np.ones(len(data), dtype=bool)
            _keep, removed_adjacency, _converged = resolve_adjacency(
                _t, data[xc].values, data[yc].values, _keep,
                group=data.groupby(['shortid', 'tw_group']).ngroup().values,
                v_thresh=velocity_threshold if use_velocity else None,
                min_dt=min_dt, max_excursion_s=exc_s,
                j_thresh=jump_threshold if use_jump else None)
            if removed_adjacency:
                data = data[_keep].copy()
                if collect_stats:
                    self.log_message(
                        f"  Removed {removed_adjacency} points that only became "
                        f"impossible once their neighbours were removed")
                    if not _converged:
                        self.log_message(
                            "    ⚠ discontinuities remain — a displaced run "
                            "that is too long, or one whose removal would not "
                            "make the track continuous, is reported rather "
                            "than deleted. See the threshold read-out.")

        # Clean up temporary columns
        data = data.drop(columns=['time_diff', 'time_diff_s', 'tw_group', 'distance', 'velocity'], errors='ignore')
        if 'is_jump' in data.columns:
            data = data.drop(columns=['is_jump'])
        
        final_count = len(data)
        # Only an export narrates itself. The preview re-filters every tag on
        # every scrub and every spinbox step, and logging each one buried the
        # run in thousands of identical lines; the live figures belong in the
        # threshold read-out, which is where the preview already puts them.
        if initial_count != final_count and collect_stats:
            self.log_message(f"  Total filtered: {initial_count - final_count} points ({100*(initial_count-final_count)/initial_count:.1f}%)")
        
        # Accumulate stats for the run summary. This runs once per tag during
        # export, so the totals must sum across calls — replacing them would
        # report only the last tag while labelling it as the whole run.
        # The preview passes collect_stats=False so scrubbing cannot pollute
        # an export's figures.
        if collect_stats:
            s = getattr(self, 'filter_stats', None) or {}
            self.filter_stats = {
                'initial_count': s.get('initial_count', 0) + initial_count,
                'removed_outlier': s.get('removed_outlier', 0) + removed_outlier,
                'removed_velocity': s.get('removed_velocity', 0) + removed_velocity,
                'removed_jump': s.get('removed_jump', 0) + removed_jump,
                # Counted separately, because these points passed every
                # threshold on their own and only became impossible once
                # their neighbours went. Folding them into the jump count
                # would say the jump threshold caught something it did not.
                'removed_adjacency': (s.get('removed_adjacency', 0)
                                      + removed_adjacency),
                'final_count': s.get('final_count', 0) + final_count,
                'tags_processed': s.get('tags_processed', 0) + 1,
            }
            tot = self.filter_stats['initial_count']
            self.filter_stats['percent_filtered'] = (
                100 * (tot - self.filter_stats['final_count']) / tot if tot else 0)

        return data

    def xml_config_summary(self):
        """Site metadata parsed from the detected XML, or None if there is none.

        Makes the export record self-contained: it captures the anchor/antenna
        positions, zone polygons, scale and map extent read from the site XML
        (all lengths already converted to meters). The embedded map *image*
        itself is not copied — it lives in the XML file referenced by 'path'.

        Every value is coerced to a plain Python type so json.dump never chokes
        on a numpy scalar.
        """
        if not getattr(self, 'xml_config_path', None):
            return None

        summary = {
            'path': self.xml_config_path,
            'filename': os.path.basename(self.xml_config_path),
            'scale_inches_per_px': (float(self.xml_scale)
                                    if self.xml_scale is not None else None),
            'anchor_positions_m': [
                {'shortid': int(a['shortid']),
                 'x': float(a['x']), 'y': float(a['y']), 'z': float(a['z'])}
                for a in (self.anchor_positions or [])
            ],
            'zones': [
                {'name': z.get('name'),
                 'color': z.get('color'),
                 'points_m': (z['points'].tolist() if hasattr(z.get('points'), 'tolist')
                              else z.get('points'))}
                for z in (getattr(self, 'xml_zones', None) or [])
            ],
        }
        if getattr(self, 'xml_map_extent', None) is not None:
            summary['map_extent_m'] = [float(v) for v in self.xml_map_extent]
        if self.bg_width_meters is not None and self.bg_height_meters is not None:
            summary['background_size_m'] = [float(self.bg_width_meters),
                                            float(self.bg_height_meters)]
        return summary

    def get_config_dict(self):
        """Get current configuration as dictionary.

        Syncs the hidden export widgets from the preview first: the preview
        is the single source of truth for filtering/smoothing, and the
        top-level keys below are read from the export mirrors. Without this
        a config written outside the export path (e.g. write_live_config
        after assigning identities) could record stale mirror values that
        disagree with the pane the user is looking at.
        """
        from datetime import datetime
        self._sync_export_from_preview()
        config = {
            'fnt_version': get_fnt_version(),
            'run_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'database_path': self.db_path,
            'database_name': os.path.basename(self.db_path) if self.db_path else None,
            'table_name': self.table_name,
            'selected_tags': [tag for tag, cb in self.tag_checkboxes.items() if cb.isChecked()],
            'timezone': self.combo_timezone.currentText(),
            'velocity_filter': self.chk_velocity_filter.isChecked(),
            'velocity_threshold': self.spin_velocity_threshold.value(),
            'jump_filter': self.chk_jump_filter.isChecked(),
            'jump_threshold': self.spin_jump_threshold.value(),
            'outlier_filter': self.chk_outlier_filter.isChecked(),
            'outlier_window': self.spin_outlier_window.value(),
            'outlier_sigma': self.spin_outlier_sigma.value(),
            'velocity_min_dt': self.spin_velocity_min_dt.value(),
            'adjacency_filter': self.chk_adjacency_filter.isChecked(),
            'excursion_s': self.spin_excursion_s.value(),
            'time_gap': self.spin_time_gap.value(),
            'smoothing_method': self.combo_smoothing.currentText(),
            # rolling_window / rolling_window_units are null unless the method
            # really has a window; EWMA's parameter is written as
            # smoothing_span instead. See smoothing_window_fields.
            **smoothing_window_fields(
                self.combo_smoothing.currentText(),
                self.spin_rolling_window.value(),
                self.combo_window_units.currentText(),
                window_key='rolling_window', units_key='rolling_window_units'),
            'show_anchors': self.chk_show_anchors.isChecked(),
            'show_tracking': self.chk_show_tracking.isChecked(),
            'show_tag_id': self.chk_show_tag_id.isChecked(),
            'tag_id_type': self.combo_tag_id_type.currentText(),
            'preview_tag_size': self.spin_tag_size.value(),
            'label_color': self.combo_label_color.currentData(),
            'label_outline': self.chk_label_outline.isChecked(),
            'show_battery': self.chk_show_battery.isChecked(),
            'export_raw_csv': self.chk_export_raw_csv.isChecked(),
            'export_smoothed_csv': self.chk_export_smoothed_csv.isChecked(),
            'proximity_detection': self.chk_proximity_detection.isChecked(),
            'proximity_threshold': self.spin_proximity_threshold.value(),
            # Analysis settings, so they ARE inherited between trials -
            # unlike the regions themselves, which belong to one arena.
            'export_roi_bouts': self.chk_export_roi_bouts.isChecked(),
            'export_roi_daily': self.chk_export_roi_daily.isChecked(),
            'export_behavior_events': self.chk_export_behavior.isChecked(),
            'save_plots': self.chk_save_plots.isChecked(),
            'save_svg': self.chk_save_svg.isChecked(),
            'plot_types': {k: cb.isChecked() for k, cb in self.plot_type_checkboxes.items()},
            'save_animation': self.chk_save_animation.isChecked(),
            'animation_trail': self.spin_animation_trail.value(),
            'animation_tag_size': self.spin_anim_tag_size.value(),
            'animation_show_battery': self.chk_show_battery_export.isChecked(),
            # Which preview layers are drawn into the video. Saved raw (not
            # resolved against the preview) so a trial reopens with the choices
            # made, and re-resolves them against whatever the preview says now.
            'animation_layers': {
                name: bool(getattr(self, attr).isChecked())
                for attr, name in (
                    ('chk_anim_show_trail', 'trail'),
                    ('chk_anim_show_tag_id', 'tag_id'),
                    ('chk_anim_show_behavior', 'behavior'),
                    ('chk_anim_beh_social', 'social'),
                    ('chk_anim_beh_chase', 'chase'),
                    ('chk_anim_beh_displace', 'displace'))
                if hasattr(self, attr)},
            'animation_speed': self.combo_animation_speed.currentText(),
            'animation_fps': self.combo_animation_fps.currentText(),
            'color_by': self.combo_color_by.currentText(),
            'video_quality': self.combo_video_quality.currentText(),
            'daily_animations': self.chk_daily_animations.isChecked(),
            'full_animation': self.chk_full_animation.isChecked(),
            'tag_identities': self.tag_identities,
            'plot_layers': dict(self.plot_layers),
            # Site-map image copied into the analysis folder (filename + extent),
            # or None. Written by _export_sitemap during export.
            'sitemap': getattr(self, '_exported_sitemap', None),
            'background_image_path': self.background_image_path,
            # Manual background transform (option (b)) so the alignment nudge
            # survives a reload. Scale is in/px; offset is metres.
            'background_scale_inpx': self.bg_scale,
            'background_offset_m': [float(self.bg_offset_x), float(self.bg_offset_y)],
            'arena_zones': self.arena_zones.to_dict('records') if self.arena_zones is not None else None,
            # Regions the user drew over the preview. Theirs, not the XML's,
            # and stored here so they travel with the trial's analysis folder.
            'rois': uwb_roi.rois_to_json(self.rois),
            'show_rois': self.chk_show_rois.isChecked(),
            # Anchor/antenna positions, zones, scale and map extent from the
            # site XML (all in meters); None when no XML was detected.
            'xml_config': self.xml_config_summary(),
            # Outcome of this run's threshold filtering, summed across every
            # tag processed (previously the only non-redundant content of
            # runSummary.csv). None until an export has actually filtered.
            'filter_stats': (dict(self.filter_stats)
                             if getattr(self, 'filter_stats', None) else None),
            # Live-preview display/analysis settings. Nested under one key so
            # they are clearly NOT part of the export's reproducibility record —
            # nothing here affects exported data — while still surviving a
            # reload so the pane comes back the way it was left.
            'preview': self.get_preview_config(),
        }
        return config

    # Preview settings persisted in fnt_config.json: attribute -> (widget, kind).
    # Kind drives how the value is read/written ('text' for combos, 'value' for
    # spinboxes, 'checked' for checkboxes).
    _PREVIEW_CONFIG_WIDGETS = {
        'smoothing': ('combo_preview_smoothing', 'text'),
        'window': ('spin_preview_window', 'value'),
        'window_units': ('combo_preview_window_units', 'text'),
        'velocity_filter': ('chk_preview_velocity', 'checked'),
        'velocity_threshold': ('spin_preview_velocity', 'value'),
        'jump_filter': ('chk_preview_jump', 'checked'),
        'jump_threshold': ('spin_preview_jump', 'value'),
        'outlier_filter': ('chk_preview_outlier', 'checked'),
        'outlier_window': ('spin_preview_outlier_window', 'value'),
        'outlier_sigma': ('spin_preview_outlier_sigma', 'value'),
        'velocity_min_dt': ('spin_preview_velocity_min_dt', 'value'),
        'adjacency_filter': ('chk_preview_adjacency', 'checked'),
        'excursion_s': ('spin_preview_excursion', 'value'),
        'trail_points': ('chk_preview_trail_points', 'checked'),
        'color_by': ('combo_preview_color', 'text'),
        'dark_mode': ('chk_preview_dark', 'checked'),
        'trail_seconds': ('spin_preview_trail', 'value'),
        'chunk_minutes': ('spin_preview_minutes', 'value'),
        'rate': ('combo_preview_hz', 'text'),
        'view_mode': ('combo_view_mode', 'text'),
        'arena_source': ('combo_arena', 'text'),
        'show_background': ('chk_show_background', 'checked'),
        'show_anchors': ('chk_show_anchors', 'checked'),
        'show_zones': ('chk_show_zones', 'checked'),
        'show_tracking': ('chk_show_tracking', 'checked'),
        'show_tag_id': ('chk_show_tag_id', 'checked'),
        'tag_id_type': ('combo_tag_id_type', 'text'),
        'show_battery': ('chk_show_battery', 'checked'),
        'show_speed': ('chk_show_speed', 'checked'),
        'show_step': ('chk_show_step', 'checked'),
        'time_gap': ('spin_time_gap', 'value'),
        # Pinned drawing window. Persisted here because it now governs the
        # exports too, so a trial must reopen framed the way it was exported.
        'xy_range_manual': ('chk_xy_range', 'checked'),
        'xy_xmin': ('spin_xy_xmin', 'value'),
        'xy_xmax': ('spin_xy_xmax', 'value'),
        'xy_ymin': ('spin_xy_ymin', 'value'),
        'xy_ymax': ('spin_xy_ymax', 'value'),
        # Behaviour detection. These drive the preview overlay AND the
        # exported ethogram, so a trial must reopen with the thresholds it
        # was tuned and exported with.
        'show_behavior': ('chk_show_behavior', 'checked'),
        'beh_social': ('chk_beh_social', 'checked'),
        'beh_chase': ('chk_beh_chase', 'checked'),
        'beh_displace': ('chk_beh_displace', 'checked'),
        'social_radius': ('spin_social_radius', 'value'),
        'chase_distance': ('spin_chase_distance', 'value'),
        'chase_speed': ('spin_chase_speed', 'value'),
        'chase_angle_deg': ('spin_chase_angle', 'value'),
        'min_chase_s': ('spin_min_chase', 'value'),
        'heading_lag_s': ('spin_heading_lag', 'value'),
        'displace_distance': ('spin_displace_distance', 'value'),
        'displace_loser_speed': ('spin_displace_loser_speed', 'value'),
        'displace_winner_speed': ('spin_displace_winner_speed', 'value'),
        'displace_leave_distance': ('spin_displace_leave', 'value'),
        'displace_window_s': ('spin_displace_window', 'value'),
        # NOTE: the preview tag marker size persists separately as the
        # top-level 'preview_tag_size' key (kept for config compatibility).
    }

    def get_preview_config(self):
        """Current live-preview settings as a plain dict (missing widgets skipped)."""
        out = {}
        for key, (attr, kind) in self._PREVIEW_CONFIG_WIDGETS.items():
            w = getattr(self, attr, None)
            if w is None:
                continue
            if kind == 'text':
                out[key] = w.currentText()
            elif kind == 'value':
                out[key] = w.value()
            else:
                out[key] = w.isChecked()
        # The preview's window control is the same dual-purpose spinbox the
        # export's is, and the units combo beside it is hidden for EWMA - so
        # the raw values would claim a window and a unit that the method never
        # used. Rewrite them the way the export block does.
        out.update(smoothing_window_fields(
            out.get('smoothing'), out.get('window'), out.get('window_units'),
            window_key='window', units_key='window_units'))
        return out

    # Pre-consolidation config keys -> the preview widget that now owns them.
    # Only consulted when config['preview'] does not already carry the setting.
    _LEGACY_PREVIEW_KEYS = {
        'smoothing_method': ('smoothing', 'combo_preview_smoothing', 'text'),
        'rolling_window': ('window', 'spin_preview_window', 'value'),
        'rolling_window_units': ('window_units', 'combo_preview_window_units', 'text'),
        'velocity_filter': ('velocity_filter', 'chk_preview_velocity', 'checked'),
        'velocity_threshold': ('velocity_threshold', 'spin_preview_velocity', 'value'),
        'jump_filter': ('jump_filter', 'chk_preview_jump', 'checked'),
        'jump_threshold': ('jump_threshold', 'spin_preview_jump', 'value'),
        'color_by': ('color_by', 'combo_preview_color', 'text'),
        'animation_trail': ('trail_seconds', 'spin_preview_trail', 'value'),
    }

    def _backfill_preview_from_legacy(self, config):
        """Seed the preview pane from a config written before consolidation.

        Each entry is applied only when the nested 'preview' block does not
        already define it, so a current config always wins. Combo values are
        matched case-insensitively and normalised the same way the export
        loader does, so old smoothing labels still resolve.
        """
        preview = config.get('preview') or {}
        for legacy_key, (preview_key, attr, kind) in self._LEGACY_PREVIEW_KEYS.items():
            if preview_key in preview or legacy_key not in config:
                continue
            w = getattr(self, attr, None)
            if w is None:
                continue
            value = config[legacy_key]
            try:
                if kind == 'text':
                    want = str(value).replace(" (default)", "")
                    if want == "Forward-Backward EWMA":
                        want = ("Forward-Backward Exponentially Weighted "
                                "Moving Average")
                    idx = next(
                        (i for i in range(w.count())
                         if w.itemText(i).replace(" (default)", "").strip().lower()
                         == want.strip().lower()), -1)
                    if idx >= 0:
                        w.setCurrentIndex(idx)
                elif kind == 'value':
                    w.setValue(type(w.value())(value))
                else:
                    w.setChecked(bool(value))
            except Exception:
                continue

        # The social radius is stored as the export's centre-to-centre
        # proximity threshold; the preview owns half of it.
        if ('social_radius' not in preview and 'proximity_threshold' in config
                and getattr(self, 'spin_social_radius', None) is not None):
            try:
                self.spin_social_radius.setValue(
                    float(config['proximity_threshold']) / 2.0)
            except (TypeError, ValueError):
                pass

    def apply_preview_config(self, cfg):
        """Restore preview settings saved by get_preview_config.

        Every set is guarded: a combo entry that no longer exists (renamed
        smoothing method, removed arena) is skipped rather than clearing the
        control, and a value outside a spinbox's range is clamped by Qt.
        """
        if not isinstance(cfg, dict):
            return
        for key, (attr, kind) in self._PREVIEW_CONFIG_WIDGETS.items():
            if key not in cfg:
                continue
            w = getattr(self, attr, None)
            if w is None:
                continue
            val = cfg[key]
            try:
                if kind == 'text':
                    if w.findText(str(val)) >= 0:
                        w.setCurrentText(str(val))
                elif kind == 'value':
                    w.setValue(type(w.value())(val))
                else:
                    w.setChecked(bool(val))
            except Exception:
                continue
        # The window/span, after the generic pass: a null 'window' is skipped
        # above, so the value has to come from 'smoothing_span' (or, for a
        # config written before the split, from 'window' itself).
        span = smoothing_window_value(cfg, window_key='window')
        if span is not None and getattr(self, 'spin_preview_window', None):
            try:
                self.spin_preview_window.setValue(span)
            except Exception:
                pass
        if not DISPLACEMENT_ENABLED:
            # Trials configured before displacement was parked have
            # beh_displace: true saved. Honouring it would put a hidden
            # detector's output back into the overlay and the events CSV
            # with no visible control to turn it off. The saved thresholds
            # are still restored above, so nothing is lost by re-enabling.
            self.chk_beh_displace.setChecked(False)

    def save_config(self, output_dir):
        """Save current configuration to JSON file"""
        config = self.get_config_dict()
        config_path = os.path.join(output_dir, 'fnt_config.json')

        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            self.log_message(f"Config saved: {os.path.basename(config_path)}")
        except Exception as e:
            self.log_message(f"Warning: Could not save config: {str(e)}")

    def analysis_dir_for_db(self):
        """Path to the '<db>_FNT_analysis' folder for the loaded database.

        This is the same folder the export writes to and that
        load_config_if_exists reads back on open, so a config written here is
        picked up automatically next time the database is loaded. Returns None
        if no database is selected.
        """
        if not self.db_path:
            return None
        db_dir = os.path.dirname(self.db_path)
        db_name = os.path.splitext(os.path.basename(self.db_path))[0]
        return os.path.join(db_dir, f"{db_name}_FNT_analysis")

    def write_live_config(self):
        """Write fnt_config.json to the analysis folder right now.

        Config-only (no CSVs/plots/heavy reads): used to snapshot the current
        settings as soon as the user applies identity assignments, so the
        export-details JSON exists and stays current before a full export runs.
        Creates the analysis folder if it does not exist yet.
        """
        analysis_dir = self.analysis_dir_for_db()
        if not analysis_dir:
            return
        try:
            os.makedirs(analysis_dir, exist_ok=True)
            self.save_config(analysis_dir)
        except Exception as e:
            self.log_message(f"Warning: Could not write live config: {str(e)}")

    def load_config_if_exists(self):
        """Check for existing config file and load it.
        Returns True if a config was loaded, False otherwise."""
        if not self.db_path:
            return False

        db_dir = os.path.dirname(self.db_path)
        db_filename = os.path.basename(self.db_path)
        db_name = os.path.splitext(db_filename)[0]
        analysis_dir = os.path.join(db_dir, f"{db_name}_FNT_analysis")
        config_path = os.path.join(analysis_dir, 'fnt_config.json')

        # The batch queue captures the settings shown in the UI at add-time and
        # replays them here via an in-memory override, instead of reading the
        # DB's saved fnt_config.json. Consume-once so it never leaks to a later
        # interactive load.
        override = getattr(self, '_pending_config_override', None)
        self._pending_config_override = None

        if override is None and not os.path.exists(config_path):
            return False

        try:
            if override is not None:
                config = override
            else:
                with open(config_path, 'r') as f:
                    config = json.load(f)

            # Load configuration into GUI
            if 'table_name' in config and config['table_name']:
                # Store for deferred application after combo_table is populated
                self.pending_table_name = config['table_name']
                # Also try to set directly if combo is already populated
                index = self.combo_table.findText(config['table_name'])
                if index >= 0:
                    self.combo_table.setCurrentIndex(index)
            
            if 'timezone' in config:
                index = self.combo_timezone.findText(config['timezone'])
                if index >= 0:
                    self.combo_timezone.setCurrentIndex(index)
            
            if 'smoothing_method' in config:
                # Match on the base method name so configs saved under the old
                # labels (e.g. "Rolling Average (default)") still resolve after
                # the "(default)" marker moved to None.
                want = config['smoothing_method'].replace(" (default)", "")
                # The EWMA option's label was spelled out in full; map the old
                # abbreviated label so pre-existing configs still restore it.
                if want == "Forward-Backward EWMA":
                    want = "Forward-Backward Exponentially Weighted Moving Average"
                index = next(
                    (i for i in range(self.combo_smoothing.count())
                     if self.combo_smoothing.itemText(i).replace(" (default)", "") == want),
                    -1)
                if index >= 0:
                    self.combo_smoothing.setCurrentIndex(index)
            
            # Null when the method has no window (EWMA / Savitzky-Golay), so
            # neither of these can be fed to a widget unchecked: setValue(None)
            # raises, and this sits inside the loader's single try - one null
            # would abort the load and silently drop every setting after it.
            span = smoothing_window_value(config, window_key='rolling_window')
            if span is not None:
                self.spin_rolling_window.setValue(span)

            if config.get('rolling_window_units'):
                idx = self.combo_window_units.findText(config['rolling_window_units'])
                if idx >= 0:
                    self.combo_window_units.setCurrentIndex(idx)
            
            if 'velocity_filter' in config:
                self.chk_velocity_filter.setChecked(config['velocity_filter'])
            
            if 'velocity_threshold' in config:
                self.spin_velocity_threshold.setValue(config['velocity_threshold'])
            
            if 'jump_filter' in config:
                self.chk_jump_filter.setChecked(config['jump_filter'])
            
            if 'jump_threshold' in config:
                self.spin_jump_threshold.setValue(config['jump_threshold'])

            # A trial configured before the outlier threshold existed was not
            # filtered by it, and one configured before the velocity time base
            # existed divided by the RAW interval. Re-running either has to
            # reproduce what it produced, so an absent key means OFF here
            # rather than the new default - the defaults apply to new work.
            if 'outlier_filter' in config:
                self.chk_outlier_filter.setChecked(bool(config['outlier_filter']))
                self.chk_preview_outlier.setChecked(bool(config['outlier_filter']))
            else:
                self.chk_outlier_filter.setChecked(False)
                self.chk_preview_outlier.setChecked(False)
                self.log_message(
                    "This config predates the outlier threshold, so it is off "
                    "for this trial (re-running it reproduces what it "
                    "produced). Tick it to adopt the new default.")
            for key, wid, pwid in (
                    ('outlier_window', self.spin_outlier_window,
                     self.spin_preview_outlier_window),
                    ('outlier_sigma', self.spin_outlier_sigma,
                     self.spin_preview_outlier_sigma)):
                if config.get(key) is not None:
                    wid.setValue(type(wid.value())(config[key]))
                    pwid.setValue(type(pwid.value())(config[key]))
            # A trial configured before the excursion pass existed was not
            # cleaned by it, so an absent key means OFF - re-running has to
            # reproduce what it produced. New work gets the default.
            _adj = bool(config.get('adjacency_filter', False))
            self.chk_adjacency_filter.setChecked(_adj)
            self.chk_preview_adjacency.setChecked(_adj)
            if config.get('excursion_s') is not None:
                self.spin_excursion_s.setValue(float(config['excursion_s']))
                self.spin_preview_excursion.setValue(float(config['excursion_s']))
            _min_dt = float(config.get('velocity_min_dt') or 0.0)
            self.spin_velocity_min_dt.setValue(_min_dt)
            self.spin_preview_velocity_min_dt.setValue(_min_dt)
            
            if 'time_gap' in config:
                self.spin_time_gap.setValue(config['time_gap'])
            
            if 'show_tracking' in config:
                self.chk_show_tracking.setChecked(config['show_tracking'])

            if 'show_tag_id' in config:
                self.chk_show_tag_id.setChecked(config['show_tag_id'])
            if 'tag_id_type' in config:
                i = self.combo_tag_id_type.findText(config['tag_id_type'])
                if i >= 0:
                    self.combo_tag_id_type.setCurrentIndex(i)

            if 'preview_tag_size' in config:
                self.spin_tag_size.setValue(config['preview_tag_size'])
            if 'label_color' in config:
                # Matched on the stored hex, not the index, so reordering or
                # adding a colour later cannot silently repoint old configs.
                want = config['label_color']
                idx = self.combo_label_color.findData(want)
                self.combo_label_color.setCurrentIndex(max(0, idx))
            if 'label_outline' in config:
                self.chk_label_outline.setChecked(bool(config['label_outline']))

            # Live-preview display/analysis settings (nested; see
            # get_preview_config). Absent in configs written before this key
            # existed, in which case the preview simply keeps its defaults.
            if 'preview' in config:
                self.apply_preview_config(config['preview'])
            # Configs written before these settings moved into the preview
            # pane carry them only as top-level export keys. Without this the
            # visible controls would show defaults while the export mirrors
            # held the real values - and the next _sync_export_from_preview
            # would then overwrite those real values with the defaults,
            # silently discarding the trial's smoothing settings.
            self._backfill_preview_from_legacy(config)

            if 'show_battery' in config:
                self.chk_show_battery.setChecked(config['show_battery'])

            if 'animation_tag_size' in config:
                self.spin_anim_tag_size.setValue(config['animation_tag_size'])

            if 'animation_show_battery' in config:
                self.chk_show_battery_export.setChecked(config['animation_show_battery'])
            saved_layers = config.get('animation_layers')
            if isinstance(saved_layers, dict):
                for attr, name in (('chk_anim_show_trail', 'trail'),
                                   ('chk_anim_show_tag_id', 'tag_id'),
                                   ('chk_anim_show_behavior', 'behavior'),
                                   ('chk_anim_beh_social', 'social'),
                                   ('chk_anim_beh_chase', 'chase'),
                                   ('chk_anim_beh_displace', 'displace')):
                    cb = getattr(self, attr, None)
                    if cb is not None and name in saved_layers:
                        on = bool(saved_layers[name])
                        cb.setChecked(on)
                        # A saved 'off' is a deliberate clear, so re-syncing
                        # from the preview must not turn it back on.
                        cb._user_cleared = not on

            if 'export_raw_csv' in config:
                self.chk_export_raw_csv.setChecked(config['export_raw_csv'])

            if 'export_smoothed_csv' in config:
                self.chk_export_smoothed_csv.setChecked(config['export_smoothed_csv'])

            # 'export_downsampled_csv' / 'downsample_hz' are obsolete (the
            # downsampled export was removed); ignored if present in old configs.

            if isinstance(config.get('plot_layers'), dict):
                self.plot_layers = {
                    k: bool(config['plot_layers'].get(k, DEFAULT_PLOT_LAYERS[k]))
                    for k in DEFAULT_PLOT_LAYERS}

            # 'export_zone_occupancy' is the pre-split key: one checkbox
            # wrote both of the retired occupancy files. It now seeds both new
            # ROI products, so a config saved before the split still exports
            # something rather than silently nothing.
            if 'export_zone_occupancy' in config:
                _occ = bool(config['export_zone_occupancy'])
                self.chk_export_roi_bouts.setChecked(_occ)
                self.chk_export_roi_daily.setChecked(_occ)
            if 'export_roi_bouts' in config:
                self.chk_export_roi_bouts.setChecked(
                    bool(config['export_roi_bouts']))
            if 'export_roi_daily' in config:
                self.chk_export_roi_daily.setChecked(
                    bool(config['export_roi_daily']))
            self.on_zone_occupancy_toggled()
            if 'proximity_detection' in config:
                self.chk_proximity_detection.setChecked(config['proximity_detection'])

            if 'proximity_threshold' in config:
                self.spin_proximity_threshold.setValue(config['proximity_threshold'])

            # 'export_social_network', 'export_gbi', 'export_edgelist' and
            # 'edgelist_window_h' are read from older configs and ignored: the
            # GBI and the edge list were removed, and both were derivable from
            # the bouts file that remains.
            if 'export_behavior_events' in config:
                self.chk_export_behavior.setChecked(config['export_behavior_events'])
            self.on_social_network_toggled()
    
            if 'save_plots' in config:
                self.chk_save_plots.setChecked(config['save_plots'])

            if 'save_svg' in config:
                self.chk_save_svg.setChecked(config['save_svg'])

            if 'plot_types' in config:
                for key, value in config['plot_types'].items():
                    if key in self.plot_type_checkboxes:
                        self.plot_type_checkboxes[key].setChecked(value)
            
            if 'save_animation' in config:
                self.chk_save_animation.setChecked(config['save_animation'])
            
            if 'animation_trail' in config:
                self.spin_animation_trail.setValue(config['animation_trail'])
            
            if 'animation_speed' in config:
                index = self.combo_animation_speed.findText(config['animation_speed'])
                if index >= 0:
                    self.combo_animation_speed.setCurrentIndex(index)
            
            if 'animation_fps' in config:
                index = self.combo_animation_fps.findText(str(config['animation_fps']))
                if index >= 0:
                    self.combo_animation_fps.setCurrentIndex(index)
            
            if 'color_by' in config:
                index = self.combo_color_by.findText(config['color_by'])
                if index >= 0:
                    self.combo_color_by.setCurrentIndex(index)

            if 'video_quality' in config:
                index = self.combo_video_quality.findText(config['video_quality'])
                if index >= 0:
                    self.combo_video_quality.setCurrentIndex(index)

            if 'daily_animations' in config:
                self.chk_daily_animations.setChecked(config['daily_animations'])
            if 'full_animation' in config:
                self.chk_full_animation.setChecked(config['full_animation'])

            if 'show_anchors' in config:
                self.chk_show_anchors.setChecked(config['show_anchors'])

            if 'tag_identities' in config:
                # Convert string keys back to integers if needed
                self.tag_identities = {}
                for key, value in config['tag_identities'].items():
                    tag_key = int(key) if isinstance(key, str) and key.isdigit() else key
                    self.tag_identities[tag_key] = value
            
            # Load background image if path is saved and file exists
            if 'background_image_path' in config and config['background_image_path']:
                bg_path = config['background_image_path']
                # Ensure XML config is parsed first to get scale
                if not self.xml_scale and self.db_path:
                    db_dir = os.path.dirname(self.db_path)
                    xml_files = [f for f in list_visible_files(db_dir) if f.lower().endswith('.xml')]
                    if xml_files:
                        xml_file = next((f for f in xml_files if 'config' in f.lower()), xml_files[0])
                        self.xml_config_path = os.path.join(db_dir, xml_file)
                        try:
                            self.parse_xml_config()
                        except:
                            pass
                
                # Resolve the saved path (absolute, else relative to the DB dir),
                # then load through the shared helper so scale/offset seeding and
                # the alignment log stay consistent with an interactive load.
                resolved = None
                if os.path.exists(bg_path):
                    resolved = bg_path
                elif os.path.exists(os.path.join(db_dir, os.path.basename(bg_path))):
                    resolved = os.path.join(db_dir, os.path.basename(bg_path))

                if resolved:
                    self._apply_background_image(resolved, source="restored")
                else:
                    self.log_message(f"Warning: Saved background image not found: {bg_path}")

            # Honour a saved manual transform (overrides the XML default). Kept
            # outside the image block above because the transform now places the
            # XML site map too, so a nudge made with no floorplan loaded has to
            # survive the reload as well.
            saved_scale = config.get('background_scale_inpx')
            saved_off = config.get('background_offset_m')
            if saved_scale:
                self.bg_scale = float(saved_scale)
            if saved_off and len(saved_off) == 2:
                self.bg_offset_x = float(saved_off[0])
                self.bg_offset_y = float(saved_off[1])
            # Every config carries a [0, 0] offset, so only say something when
            # the trial was actually nudged away from the XML default.
            if saved_scale or self.bg_offset_x or self.bg_offset_y:
                self._recompute_bg_size()
                self._recompute_xml_map_extent()
                self._sync_bg_transform_controls()
                scale_txt = f"{self.bg_scale:.4f} in/px" if self.bg_scale else "XML default"
                self.log_message(
                    f"Restored background transform: scale {scale_txt}, "
                    f"offset ({self.bg_offset_x:.2f}, {self.bg_offset_y:.2f}) m")
            
            # Note: selected_tags will be loaded after tags are populated from table
            if 'selected_tags' in config:
                self.pending_tag_selection = config['selected_tags']
            
            # Load zone data if present
            if 'rois' in config:
                self.rois = uwb_roi.rois_from_json(config.get('rois'))
                if self.rois:
                    self.log_message(
                        f"Loaded {len(self.rois)} region(s) of interest from "
                        f"config: {', '.join(r['name'] for r in self.rois)}")
                self._refresh_roi_list(keep=-1)
            if 'show_rois' in config:
                self.chk_show_rois.blockSignals(True)
                self.chk_show_rois.setChecked(bool(config['show_rois']))
                self.chk_show_rois.blockSignals(False)
            self.on_rois_changed()

            if 'arena_zones' in config and config['arena_zones'] is not None:
                self.arena_zones = pd.DataFrame(config['arena_zones'])
                if not self.arena_zones.empty:
                    num_zones = self.arena_zones['zone'].nunique()
                    num_points = len(self.arena_zones)
                    self.log_message(f"Loaded {num_zones} zones with {num_points} coordinate points from config")
            
            self.log_message(f"Loaded previous configuration from {config_path}")

            # A config can carry zones (and a background path) for a trial whose
            # site XML is missing, so re-offer the layers those restored sources
            # support rather than leaving them greyed out.
            self._sync_layer_toggles()

            # Update tag labels if identities were loaded
            if self.tag_identities and self.tag_checkboxes:
                self.update_tag_labels()

            # --- Migration prompt: move loose files into subfolders ---
            plots_subdir = os.path.join(analysis_dir, 'plots')
            animations_subdir = os.path.join(analysis_dir, 'animation_Tracking')

            loose_plots = []
            loose_animations = []

            # The site-map image is a root-level PNG but it is NOT a plot: the
            # config references it by filename at the analysis-folder root, so
            # sweeping it into plots/ would break that reference. Exclude it
            # (and anything the config names as the site map) from the sweep.
            sitemap_names = {f"{db_name}_sitemap.png".lower()}
            cfg_map = (config or {}).get('sitemap') or {}
            if cfg_map.get('filename'):
                sitemap_names.add(str(cfg_map['filename']).lower())

            def _is_plot(name):
                low = name.lower()
                if low in sitemap_names or low.endswith('_sitemap.png'):
                    return False
                return low.endswith(('.png', '.svg'))

            if not os.path.exists(plots_subdir):
                loose_plots = [f for f in list_visible_files(analysis_dir)
                               if os.path.isfile(os.path.join(analysis_dir, f))
                               and _is_plot(f)]

            if not os.path.exists(animations_subdir):
                loose_animations = [f for f in list_visible_files(analysis_dir)
                                    if os.path.isfile(os.path.join(analysis_dir, f))
                                    and f.lower().endswith('.mp4')]

            if loose_plots or loose_animations:
                parts = []
                if loose_plots:
                    parts.append(f"{len(loose_plots)} plot file(s)")
                if loose_animations:
                    parts.append(f"{len(loose_animations)} animation file(s)")
                file_desc = " and ".join(parts)

                reply = QMessageBox.question(
                    self, "Update Folder Structure",
                    f"Your analysis folder has {file_desc} in the root directory. "
                    f"Would you like to organize them into subfolders (plots/ and animations/) "
                    f"for better organization?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
                )
                if reply == QMessageBox.Yes:
                    moved = 0
                    if loose_plots:
                        ensure_dir(plots_subdir)
                        for plot_file in loose_plots:
                            src = os.path.join(analysis_dir, plot_file)
                            dst = os.path.join(plots_subdir, plot_file)
                            try:
                                shutil.move(src, dst)
                                moved += 1
                            except Exception as move_err:
                                self.log_message(f"Warning: Could not move {plot_file}: {move_err}")
                    if loose_animations:
                        ensure_dir(animations_subdir)
                        for anim_file in loose_animations:
                            src = os.path.join(analysis_dir, anim_file)
                            dst = os.path.join(animations_subdir, anim_file)
                            try:
                                shutil.move(src, dst)
                                moved += 1
                            except Exception as move_err:
                                self.log_message(f"Warning: Could not move {anim_file}: {move_err}")
                    self.log_message(f"Migrated {moved} file(s) into subfolders")

            # The preview pane is now authoritative for filtering and
            # smoothing, so push the just-restored values onto the hidden
            # export mirrors. Without this the mirrors keep whatever the
            # top-level keys set, and the two disagree until the next
            # preview edit.
            self._sync_export_from_preview()
            self.invalidate_preview_cache()
            # Kept so later steps can reuse what the config already knows
            # instead of re-deriving it from the database (see
            # load_tags_from_table).
            self._loaded_config = config

            return True

        except Exception as e:
            print(f"Warning: Could not load config: {str(e)}")
            return False
    
    def apply_pending_tag_selection(self):
        """Apply tag selection from loaded config after tags are populated"""
        if hasattr(self, 'pending_tag_selection') and self.pending_tag_selection:
            for tag, cb in self.tag_checkboxes.items():
                cb.setChecked(tag in self.pending_tag_selection)
            delattr(self, 'pending_tag_selection')
    
    def select_animation_tags(self):
        """Pick which tags the animation includes.

        The list is a sub-selection of the tags currently checked in the
        top-level Tag Selection — you cannot add tags here that weren't chosen
        there. Default is all of them.
        """
        from PyQt5.QtWidgets import QDialog, QDialogButtonBox
        tags = self.selected_preview_tags()
        if not tags:
            QMessageBox.information(
                self, "No Tags Selected",
                "Select one or more tags in the Tag Selection section first — "
                "the animation can only include tags chosen there.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Include Tags in Animation")
        lay = QVBoxLayout()
        lay.addWidget(QLabel("Select the tags to include in the animation:"))
        boxes = {}
        for t in tags:
            info = self.tag_identities.get(t, {}) or {}
            hex_id = hex(int(t)).upper().replace('0X', '')
            label = (f"{info['sex']}-{info['identity']}  (HexID {hex_id})"
                     if info else f"HexID {hex_id}")
            cb = QCheckBox(label)
            cb.setChecked(self._animation_tags is None or t in self._animation_tags)
            boxes[t] = cb
            lay.addWidget(cb)

        sel_row = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_none = QPushButton("Unselect All")
        btn_all.clicked.connect(lambda: [cb.setChecked(True) for cb in boxes.values()])
        btn_none.clicked.connect(lambda: [cb.setChecked(False) for cb in boxes.values()])
        sel_row.addWidget(btn_all)
        sel_row.addWidget(btn_none)
        lay.addLayout(sel_row)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)
        dlg.setLayout(lay)
        if dlg.exec_() == QDialog.Accepted:
            sel = [t for t, cb in boxes.items() if cb.isChecked()]
            if not sel or len(sel) == len(tags):
                self._animation_tags = None
                self.lbl_anim_tags.setText("All configured tags")
            else:
                self._animation_tags = sel
                self.lbl_anim_tags.setText(f"{len(sel)} of {len(tags)} tags")

    def _snapshot_anim_settings(self):
        """Freeze every animation setting at export time.

        The export runs plots on a background thread and renders the animation on
        the main thread with periodic processEvents(), so the window stays live
        the whole time. Reading widgets during the run let a mid-export click
        (e.g. changing the speed) leak into a render already in the queue. This
        captures the values once so the whole export uses a single, consistent
        set of settings regardless of what the user clicks afterwards.
        """
        speed_text = self.combo_animation_speed.currentText()
        all_cbs = self.daily_animation_day_checkboxes
        days = [d for d, cb in all_cbs.items() if cb.isChecked()]
        gen_daily = self.chk_daily_animations.isChecked()
        return {
            'trailing_window': self.spin_animation_trail.value(),
            'fps': int(self.combo_animation_fps.currentText()),
            'speed_text': speed_text,
            'speed_multiplier': int(speed_text.replace('x', '')),
            'color_by': self.combo_color_by.currentText(),
            'generate_daily': gen_daily,
            'generate_full': self.chk_full_animation.isChecked(),
            # Concatenation is only valid when daily is on and EVERY day is chosen.
            'all_days_selected': (gen_daily and len(all_cbs) > 0
                                  and len(days) == len(all_cbs)),
            'selected_days': days,
            'video_quality': self.combo_video_quality.currentText(),
            'tag_size': self.spin_anim_tag_size.value(),
            'show_battery': self.chk_show_battery_export.isChecked(),
            # Inherited straight from the preview's display toggles, like the
            # battery line: the video labels what the preview labels.
            'show_speed': self.chk_show_speed.isChecked(),
            'show_step': self.chk_show_step.isChecked(),
            # Inherited from the preview like the rest of the display choices:
            # the video's labels read the way the ones you tuned do.
            'label_color': self.combo_label_color.currentData(),
            'label_outline': self.chk_label_outline.isChecked(),
            'gap_s': self.spin_time_gap.value(),
            'layers': self.anim_layer_flags(),
            'animation_tags': (list(self._animation_tags)
                               if self._animation_tags else None),
            'use_identities': bool(self.tag_identities),
        }

    def _export_behavior_events(self, csv_path, selected_tags):
        """Directed chase/displacement bouts from the exported smoothed CSV.

        Borrows PlotSaverWorker's classifier rather than duplicating it, so the
        CSV and the behaviour plots can never disagree about what fired.
        """
        # Only what the classifier reads. On a 15 M-row export the full file
        # took ~75 s to parse and most of it - the raw location columns, the
        # epoch timestamp, battery, the identity strings - is never touched
        # here: the labels come from self.tag_identities, not from the frame.
        head = pd.read_csv(csv_path, nrows=0)
        coord = ['smoothed_x', 'smoothed_y']
        if not all(c in head.columns for c in coord):
            coord = ['location_x', 'location_y']
        # 'timestamp' as well as 'Timestamp': the integer column is what
        # smoothed_csv_timestamps reads, and it is far cheaper than parsing
        # the string. The string stays in the list as the fallback.
        want = ['shortid', 'timestamp', 'Timestamp'] + coord
        df = pd.read_csv(csv_path, low_memory=False,
                         usecols=[c for c in want if c in head.columns])
        df['Timestamp'] = smoothed_csv_timestamps(
            df, self.combo_timezone.currentText())
        tags = sorted(t for t in df['shortid'].unique()
                      if not selected_tags or t in selected_tags)
        worker = PlotSaverWorker(
            self.db_path, self.table_name, tags, False, "None",
            plot_types={}, tag_identities=self.tag_identities,
            use_identities=bool(self.tag_identities),
            behavior_params=self._behavior_params())
        worker.progress.connect(self.log_message)
        # One animal, one entity - the same merge the ethogram uses. Without
        # it an animal that wore two tags is classified as two, and the CSV
        # and the raster disagree about what fired.
        df, _remap = worker.merge_tag_data(df)
        tags = sorted(t for t in df['shortid'].unique()
                      if not selected_tags or t in selected_tags)
        return worker.behavior_events_frame(df, tags, tz=df['Timestamp'].dt.tz)

    # Which animation "Show" option each preview toggle governs. The export
    # never offers a layer the preview has switched off: an option nobody can
    # see in the preview is not a choice, it is a way to be surprised by the
    # video. Preview off -> the export row is hidden AND forced off.
    _ANIM_SHOW_LINKS = (
        ('chk_anim_show_trail', 'chk_show_tracking'),
        ('chk_anim_show_tag_id', 'chk_show_tag_id'),
        ('chk_anim_show_behavior', 'chk_show_behavior'),
        ('chk_anim_beh_social', 'chk_beh_social'),
        ('chk_anim_beh_chase', 'chk_beh_chase'),
        ('chk_anim_beh_displace', 'chk_beh_displace'),
    )

    def _anim_show(self, name):
        """Is an animation layer on? False unless the preview offers it too."""
        for export_name, preview_name in self._ANIM_SHOW_LINKS:
            if export_name != name:
                continue
            preview = getattr(self, preview_name, None)
            if preview is not None and not preview.isChecked():
                return False
        cb = getattr(self, name, None)
        return bool(cb is not None and cb.isChecked())

    def _sync_anim_show_options(self):
        """Show/hide the animation layer rows to match the preview toggles."""
        for export_name, preview_name in self._ANIM_SHOW_LINKS:
            cb = getattr(self, export_name, None)
            preview = getattr(self, preview_name, None)
            if cb is None or preview is None:
                continue
            on = preview.isChecked()
            cb.setVisible(on)
            if not on:
                cb.setChecked(False)
            elif not cb.isChecked() and not getattr(cb, '_user_cleared', False):
                cb.setChecked(True)
        # The behaviour sub-rows are meaningless without their master.
        master = self._anim_show('chk_anim_show_behavior')
        for name in ('chk_anim_beh_social',
                     'chk_anim_beh_chase', 'chk_anim_beh_displace'):
            cb = getattr(self, name, None)
            preview = getattr(self, dict(self._ANIM_SHOW_LINKS)[name], None)
            if cb is not None:
                cb.setVisible(master and preview is not None
                              and preview.isChecked())

    def anim_layer_flags(self):
        """Every animation "Show" choice, resolved against the preview."""
        return {
            'trail': self._anim_show('chk_anim_show_trail'),
            'tag_id': self._anim_show('chk_anim_show_tag_id'),
            'behavior': self._anim_show('chk_anim_show_behavior'),
            'social': self._anim_show('chk_anim_beh_social'),
            'chase': self._anim_show('chk_anim_beh_chase'),
            'displace': self._anim_show('chk_anim_beh_displace'),
        }

    def _animation_behavior(self, data, tags, flags=None):
        """Per-frame behaviour overlays for the exported animation, or None.

        Built from the same classifier the preview draws from, at the PREVIEW's
        rate rather than the video's frame rate: the video then samples that
        grid, so a 240x render and a 1x render show the same behaviour at the
        same moments instead of two different approximations.
        """
        from fnt.uwb import behavior_detection as bd
        # Frozen at export time by _snapshot_anim_settings, so a click during a
        # multi-hour render cannot change what a later clip draws.
        flags = flags if flags is not None else self.anim_layer_flags()
        if not flags.get('behavior'):
            return None
        want = {name: bool(flags.get(name))
                for name in ('social', 'chase', 'displace')}
        if not any(want.values()):
            return None

        hz = 1.0
        try:
            hz = float(self.combo_preview_hz.currentText().split()[0])
        except Exception:
            pass
        worker = PlotSaverWorker(
            self.db_path, self.table_name, tags, False, "None",
            plot_types={}, tag_identities=self.tag_identities,
            use_identities=bool(self.tag_identities),
            behavior_params=self._behavior_params())
        worker.progress.connect(self.log_message)
        # Same merge as the plots: a replaced tag is not a second animal to
        # draw a circle around or to link to.
        data, remap = worker.merge_tag_data(data)
        if remap:
            tags = [t for t in tags if t not in remap]
        grid, _have, soc, _events, links, _ft = worker._behavior_classification(
            data, tags, hz=hz, collect_links=True)
        if grid is None:
            return None

        kinds = []
        if want['social']:
            kinds.append(bd.SOC_CONTACT)
        if want['displace']:
            kinds.append(bd.SOC_DISPLACING)
        if want['chase']:
            kinds.append(bd.SOC_CHASING)
        if len(links):
            links = links[np.isin(links[:, 3], kinds)]

        # Social state per tag, rebuilt from the links that SURVIVED the layer
        # choice, so the text can never name a behaviour whose connections are
        # not being drawn. SOC_* codes ascend by precedence, so taking the
        # maximum resolves an animal doing two things at once the same way the
        # preview does.
        soc_shown = np.zeros_like(soc)
        if len(links):
            f, a, b, k = (links[:, 0], links[:, 1], links[:, 2],
                          links[:, 3].astype(np.int8))
            partner = np.where(k == bd.SOC_CHASING, bd.SOC_CHASED,
                               np.where(k == bd.SOC_DISPLACING,
                                        bd.SOC_DISPLACED, bd.SOC_CONTACT))
            np.maximum.at(soc_shown, (f, a), k)
            np.maximum.at(soc_shown, (f, b), partner.astype(np.int8))

        circle_col = self.combo_circle_color.currentData() or "#9fb3c8"
        link_colors = dict(bd.SOCIAL_COLORS)
        # Contact links match the circles they came from, as in the preview.
        link_colors[bd.SOC_CONTACT] = circle_col
        return {
            'radius': (self.spin_social_radius.value() if want['social'] else None),
            'circle_color': circle_col,
            'grid': grid,
            'tags': list(tags),
            'soc': soc_shown,
            'links': links,
            'link_colors': link_colors,
            'state_labels': {'soc': dict(bd.SOCIAL_LABELS)},
            'state_colors': {'soc': dict(bd.SOCIAL_COLORS)},
        }

    # Seconds of VIDEO in a preview clip. Deliberately a video duration and
    # not a data duration: it is what the user is about to sit and watch, and
    # it keeps the render cost fixed (fps x seconds frames) whatever the speed
    # multiplier is set to.
    ANIM_CLIP_SECONDS = 10

    def _animation_speed_multiplier(self):
        try:
            return int(self.combo_animation_speed.currentText().replace("x", ""))
        except (ValueError, AttributeError):
            return 60

    def animation_clip_span_ms(self):
        """(start_ms, end_ms) of tracking time one preview clip needs.

        Starts at the playhead, so the clip shows the moment being looked at
        rather than the top of the trial - which for most trials is the release
        and says nothing about the settings. Falls back to the start of the
        timeline, and pulls the window back inside the end if the playhead is
        parked near it.
        """
        if self.preview_t0 is None or self.preview_t1 is None:
            return None
        span_ms = int(self.ANIM_CLIP_SECONDS
                      * self._animation_speed_multiplier() * 1000)
        span_ms = max(1000, min(span_ms, int(self.preview_t1 - self.preview_t0)))
        start = int(getattr(self, "preview_playhead_ms", 0) or self.preview_t0)
        start = max(int(self.preview_t0), min(start, int(self.preview_t1) - span_ms))
        return start, start + span_ms

    def set_clip_progress(self, text):
        """Show, or with '' hide, the line under the Preview button.

        Pumps the event loop: the clip renders on the GUI thread, so without
        this the label would only repaint once the render it is describing had
        already finished.
        """
        lbl = getattr(self, 'lbl_clip_progress', None)
        if lbl is None:
            return
        lbl.setText(text or "")
        lbl.setVisible(bool(text))
        QApplication.processEvents()

    def preview_animation_clip(self):
        """Render a few seconds of the animation and play it in a window.

        Goes through the SAME adapter the export uses, over the same rows the
        preview is drawing, so what plays here is what the full video will look
        like - trail length, marker size, colouring, context layers and
        behaviour overlays included. The only thing that differs is how much of
        the recording it covers.
        """
        if self.exporting:
            QMessageBox.information(
                self, "Export running",
                "An export is already running. Wait for it to finish before "
                "rendering a preview clip.")
            return
        if not self.db_path or not self.table_name:
            QMessageBox.information(self, "No database", "Load a database first.")
            return
        tags = self.selected_preview_tags()
        span = self.animation_clip_span_ms()
        if not tags or span is None:
            QMessageBox.information(
                self, "Nothing to render",
                "Select at least one tag and let the preview timeline load "
                "first.")
            return

        start_ms, end_ms = span
        fps = int(self.combo_animation_fps.currentText())
        speed_text = self.combo_animation_speed.currentText()
        frame_interval = uwb_animation.frame_interval_seconds(
            self._animation_speed_multiplier(), fps)
        tz = pytz.timezone(self.combo_timezone.currentText())
        stamp = pd.to_datetime(start_ms, unit="ms", utc=True).tz_convert(tz)

        self.log_message(
            f"Rendering a {self.ANIM_CLIP_SECONDS}s preview clip from "
            f"{stamp.strftime('%Y-%m-%d %H:%M:%S')} at {speed_text}...")
        prev_cursor = self.cursor()
        self.setCursor(Qt.WaitCursor)
        # Same inheritance the real export uses, so the clip is drawn over the
        # arena on screen rather than whatever a previous export left behind.
        self.plot_layers_from_preview(announce=False)
        # The adapter reads the frozen snapshot; take one now so the clip uses
        # exactly what is on screen, and put back whatever was there after.
        prev_snapshot = getattr(self, "_anim_settings", None)
        prev_cancelled = self.export_cancelled
        self._anim_settings = self._snapshot_anim_settings()
        self.export_cancelled = False
        tmp_dir = tempfile.mkdtemp(prefix="fnt_clip_")
        try:
            # Read a lead-in before the clip, exactly as the preview loader
            # does: without it the smoothing starts cold and the first frame
            # opens with no trail, neither of which is what the full video
            # will show at that moment. The lead-in rows are dropped from the
            # rendered span by `time_range` below.
            lead_ms = int(self._anim_settings['trailing_window'] * 1000)
            load_from = max(int(self.preview_t0), start_ms - lead_ms)
            conn = connect_ro(self.current_indexed_db() or self.db_path)
            try:
                cols = processing_select_clause(conn, self.table_name)
                placeholders = ",".join(["?"] * len(tags))
                raw = pd.read_sql_query(
                    f"SELECT {cols} FROM {self.table_name} "
                    f"WHERE shortid IN ({placeholders}) "
                    f"AND timestamp BETWEEN ? AND ? ORDER BY shortid, timestamp",
                    conn, params=list(tags) + [load_from, end_ms])
            finally:
                conn.close()

            raw, _ = collapse_multi_solutions(raw, log=self.log_message,
                                              label="clip")
            data = self._processed_slice(raw) if len(raw) else None
            if data is None or data.empty:
                QMessageBox.information(
                    self, "No data in that window",
                    "No fixes survived filtering in the clip's time window. "
                    "Move the playhead somewhere with data, or loosen the "
                    "thresholds.")
                return
            anim_tags = self._anim_settings.get("animation_tags")
            if anim_tags:
                data = data[data["shortid"].isin(anim_tags)]
                if data.empty:
                    QMessageBox.information(
                        self, "No data in that window",
                        "None of the tags chosen for the animation have data "
                        "in this window.")
                    return
            data = uwb_animation.prepare_animation_data(data, self.tag_identities)

            # Same framing rule as the export, so the clip is not drawn in a
            # tighter box than the real video will be.
            _, bg_extent = self._context_bg_source()
            axis_limits = self.xy_range() or uwb_animation.compute_axis_limits(
                data[data['Timestamp'] >= stamp], self.plot_layers, bg_extent)

            self.set_clip_progress("Building preview clip…")
            self._clip_rendering = True
            path = self.create_animation_frames(
                data, tmp_dir, frame_interval,
                self._anim_settings["trailing_window"], fps,
                self._anim_settings["color_by"],
                use_custom_identities=bool(self.tag_identities),
                day_suffix="_clip", speed_text=speed_text,
                axis_limits=axis_limits,
                time_range=(stamp, pd.to_datetime(end_ms, unit="ms", utc=True)
                            .tz_convert(tz)))
            if not path or not os.path.exists(path):
                QMessageBox.warning(
                    self, "Clip failed",
                    "The preview clip could not be rendered. See the message "
                    "log for details.")
                return

            mins = (end_ms - start_ms) / 60000.0
            dlg = AnimationClipDialog(
                path, fps,
                caption=(f"{self.ANIM_CLIP_SECONDS}s at {speed_text} \u2014 "
                         f"{mins:.1f} min of tracking from "
                         f"{stamp.strftime('%Y-%m-%d %H:%M:%S')}, rendered "
                         f"with the current export settings."),
                parent=self)
            self.log_message("Preview clip ready.")
            self.set_clip_progress("")
            dlg.exec_()
        except Exception as e:
            self.log_message(f"\u2717 Preview clip failed: {e}")
            QMessageBox.warning(self, "Clip failed", str(e))
        finally:
            self.setCursor(prev_cursor)
            self.export_cancelled = prev_cancelled
            self._anim_settings = prev_snapshot
            shutil.rmtree(tmp_dir, ignore_errors=True)
            self._clip_rendering = False
            self.set_clip_progress("")
            if hasattr(self, "progress_bar"):
                self.progress_bar.setValue(0)
            if hasattr(self, "lbl_export_progress"):
                self.lbl_export_progress.setText("")

    def _prompt_animation_temp_dir(self, animations_dir):
        """Ask up-front where to write the tracking animation's temp frames.

        Returns the chosen directory, or None if the user cancels (which aborts
        the export). Called at export start so the prompt isn't sprung on the
        user minutes into a long run.

        Built as a plain QDialog rather than a QMessageBox so the buttons appear
        in the order given: QMessageBox re-orders by button role, which varies by
        platform. Mirrors ExportConflictDialog's layout in this module.
        """
        default_temp_dir = os.path.join(animations_dir, 'temp_frames')
        on_network = is_network_path(default_temp_dir)
        cloud = cloud_sync_provider(default_temp_dir)

        CHOOSE, USE_DEFAULT, CANCEL = 1, 2, 0

        dlg = QDialog(self)
        dlg.setWindowTitle("Temporary Frames Location")
        dlg.setMinimumWidth(520)
        lay = QVBoxLayout()

        intro = QLabel("The tracking animation writes temporary frame files "
                       "while it renders.")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        head = QLabel("Default location:")
        head.setStyleSheet("font-weight: bold;")
        lay.addWidget(head)

        loc = QLabel(default_temp_dir)
        loc.setWordWrap(True)
        loc.setStyleSheet("color: #9ecbff;")
        lay.addWidget(loc)

        note = QLabel("These are deleted when the animation finishes.")
        note.setWordWrap(True)
        lay.addWidget(note)

        if on_network or cloud:
            if on_network:
                drive = (os.path.splitdrive(os.path.abspath(default_temp_dir))[0]
                         or "this location")
                body = (drive + " is a network / mapped drive. Rendering "
                        "writes and rewrites frame data continuously, which is "
                        "far slower over the network than on local storage, and "
                        "a long render is exposed to any dropout on the share.")
            else:
                body = (cloud + " syncs this folder to the cloud. The render "
                        "writes several GB of video here, and " + cloud + " will "
                        "keep uploading it while it is still being written "
                        "— competing for disk and network, and risking a lock on "
                        "the file mid-render. (Windows' OneDrive Backup "
                        "redirects your Desktop into OneDrive, so a folder that "
                        "looks local can still be synced.)")
            warn = QLabel(
                "⚠  " + body + "  A plain local folder outside any synced "
                "or network location is recommended. The frames are deleted when "
                "the render finishes, so the space is only needed while it runs.")
            warn.setWordWrap(True)
            warn.setStyleSheet(
                "color: #ffcc66; background-color: #3a2f10; "
                "border: 1px solid #7a5c00; border-radius: 4px; padding: 8px;")
            lay.addWidget(warn)

        row = QHBoxLayout()
        btn_choose = QPushButton("Choose Folder...")
        btn_choose.setToolTip("Pick a different folder for the temporary frames")
        btn_choose.clicked.connect(lambda: dlg.done(CHOOSE))
        row.addWidget(btn_choose)

        btn_default = QPushButton("Use Default")
        btn_default.setToolTip("Write the temporary frames to the default location")
        btn_default.clicked.connect(lambda: dlg.done(USE_DEFAULT))
        row.addWidget(btn_default)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setToolTip("Abort the export")
        btn_cancel.clicked.connect(lambda: dlg.done(CANCEL))
        row.addWidget(btn_cancel)

        lay.addLayout(row)
        dlg.setLayout(lay)
        # On a network default, make picking a local folder the Enter action.
        (btn_choose if (on_network or cloud) else btn_default).setDefault(True)

        result = dlg.exec_()
        if result == CHOOSE:
            custom_dir = QFileDialog.getExistingDirectory(
                self, "Select Folder for Temporary Animation Frames",
                os.path.expanduser("~"))
            if not custom_dir:
                return None
            return os.path.join(custom_dir, 'temp_animation_frames')
        if result == USE_DEFAULT:
            return default_temp_dir
        return None

    def _concat_videos_cv2(self, paths, out_path):
        """Concatenate same-resolution/fps MP4s into one, frame by frame (cv2).

        Used to build the full tracking animation from already-rendered daily
        videos when every day was selected — no re-render needed.
        """
        writer = None
        try:
            for p in paths:
                if not os.path.exists(p):
                    continue
                cap = cv2.VideoCapture(p)
                fps_v = cap.get(cv2.CAP_PROP_FPS) or 20
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if writer is None:
                        h, w = frame.shape[:2]
                        writer = cv2.VideoWriter(
                            out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_v, (w, h))
                    writer.write(frame)
                cap.release()
        finally:
            if writer is not None:
                writer.release()
        return (writer is not None and os.path.exists(out_path)
                and os.path.getsize(out_path) > 0)

    def generate_animation(self, output_dir, total_export_steps=1, current_export_step=1, csv_path=None, animations_dir=None):
        """Generate animation video from tracking data"""
        try:
            if self.export_cancelled:
                return
            
            self.log_message("Preparing animation data...")

            # Resolve animations output directory
            if animations_dir is None:
                animations_dir = os.path.join(output_dir, 'animation_Tracking')
            ensure_dir(animations_dir)

            # Temp-frames folder was chosen up-front when Export was clicked
            # (see _prompt_animation_temp_dir); fall back to the default if the
            # animation is somehow reached without that prompt.
            temp_frames_dir = (getattr(self, '_anim_temp_frames_dir', None)
                               or os.path.join(animations_dir, 'temp_frames'))
            os.makedirs(temp_frames_dir, exist_ok=True)
            self.log_message(f"Temp frames location: {temp_frames_dir}")
            
            # Load data from CSV if provided (for consistency with plots)
            if csv_path and os.path.exists(csv_path):
                self.log_message(f"Loading animation data from CSV...")
                anim_data = pd.read_csv(csv_path, low_memory=False)
                anim_data['Timestamp'] = smoothed_csv_timestamps(
                    anim_data, self.combo_timezone.currentText())
                self.log_message(f"Loaded {len(anim_data)} records from CSV")
            else:
                # Fallback to using self.data
                self.log_message("Using in-memory data for animation...")
                anim_data = self.data.copy()
            
            # Report the coordinate basis, then normalise columns (ID/sex/
            # identities/smoothed-coord fallback) via the shared helper.
            self.log_message(
                "Animation trajectories use "
                f"{'smoothed' if 'smoothed_x' in anim_data.columns else 'raw (unsmoothed)'} coordinates")
            anim_data = uwb_animation.prepare_animation_data(anim_data, self.tag_identities)

            # All animation settings come from the snapshot taken at export time
            # (see _snapshot_anim_settings) so a mid-export click can't change
            # this render.
            s = getattr(self, '_anim_settings', None) or self._snapshot_anim_settings()

            # Restrict to the chosen subset of tags (default: all).
            anim_tags = s['animation_tags']
            if anim_tags:
                anim_data = anim_data[anim_data['shortid'].isin(anim_tags)]
                self.log_message(f"Animation limited to {len(anim_tags)} selected tag(s)")

            # Get animation parameters (from the frozen snapshot)
            trailing_window = s['trailing_window']
            fps = s['fps']
            speed_text = s['speed_text']
            speed_multiplier = s['speed_multiplier']
            color_by = s['color_by']

            frame_interval = uwb_animation.frame_interval_seconds(speed_multiplier, fps)
            self.log_message(f"Animation: {speed_text} speed at {fps} FPS (each frame = {frame_interval:.2f}s of real time)")

            self.log_message("Setting up animation frames...")

            generate_daily = s['generate_daily']
            generate_full = s.get('generate_full', not generate_daily)
            all_days = s.get('all_days_selected', False)
            db_name = os.path.splitext(os.path.basename(self.db_path))[0]
            daily_paths = []   # final daily video paths in day order (for concat)

            # ONE set of axis limits, spanning the whole recording, used by
            # every render below. Letting each daily video compute its own
            # min/max made the arena shift slightly from day to day, so the
            # concatenated result jumped at each day boundary and did not
            # match a from-scratch full render. Computed after the tag subset
            # is applied so excluded animals do not stretch the frame.
            _, bg_extent_for_limits = self._context_bg_source()
            axis_limits = self.xy_range() or uwb_animation.compute_axis_limits(
                anim_data, self.plot_layers, bg_extent_for_limits)
            self.log_message(
                ("Animation frame (fixed XY range): " if self.xy_range()
                 else "Animation frame (shared by every clip): ") +
                f"x {axis_limits[0]:.2f}–{axis_limits[1]:.2f} m, "
                f"y {axis_limits[2]:.2f}–{axis_limits[3]:.2f} m")

            # ---- Daily animations (one per selected day) ----
            if generate_daily:
                selected_days = s['selected_days']
                if not selected_days:
                    self.log_message("⚠ No days selected for daily animations")
                else:
                    self.log_message(f"Generating {len(selected_days)} daily animations...")
                    for day_idx, date_str in enumerate(selected_days):
                        if self.export_cancelled:
                            return
                        self.log_message(f"Processing Day {day_idx + 1}/{len(selected_days)}: {date_str}")
                        data_tz = anim_data['Timestamp'].dt.tz
                        date_obj = pd.to_datetime(date_str).date()
                        day_start = (pd.Timestamp(date_obj, tz=data_tz) if data_tz is not None
                                     else pd.Timestamp(date_obj))
                        day_end = day_start + pd.Timedelta(days=1)
                        day_data = anim_data[(anim_data['Timestamp'] >= day_start)
                                             & (anim_data['Timestamp'] < day_end)].copy()
                        if len(day_data) == 0:
                            self.log_message(f"⚠ No data for {date_str}, skipping")
                            continue
                        video_path = self.create_animation_frames(
                            day_data, temp_frames_dir, frame_interval, trailing_window,
                            fps, color_by, s['use_identities'],
                            total_export_steps, current_export_step,
                            day_suffix=f"_Day{day_idx + 1}_{date_str}", speed_text=speed_text,
                            axis_limits=axis_limits)
                        if video_path and not self.export_cancelled:
                            final_video_path = os.path.join(
                                animations_dir,
                                f"{db_name}_Animation_Day{day_idx + 1}_{date_str}_{fps}fps_{speed_text}.mp4")
                            if os.path.exists(final_video_path):
                                self.log_message(f"⚠ Day {day_idx + 1} animation already exists, skipping: {os.path.basename(final_video_path)}")
                            elif os.path.exists(video_path):
                                shutil.move(video_path, final_video_path)
                                self.log_message(f"✓ Day {day_idx + 1} animation saved: {final_video_path}")
                            if os.path.exists(final_video_path):
                                daily_paths.append(final_video_path)

            # ---- Full animation (whole recording), after the dailies ----
            if generate_full and not self.export_cancelled:
                full_path = os.path.join(animations_dir, f"{db_name}_Animation_{fps}fps_{speed_text}.mp4")
                if os.path.exists(full_path):
                    self.log_message(f"⚠ Full animation already exists, skipping: {os.path.basename(full_path)}")
                elif generate_daily and all_days and len(daily_paths) >= 1:
                    # Every day was rendered — just stitch the daily videos together.
                    self.lbl_export_progress.setText("Concatenating daily animations into the full video...")
                    QApplication.processEvents()
                    self.log_message(f"Building full animation by concatenating {len(daily_paths)} daily animation(s)...")
                    if self._concat_videos_cv2(daily_paths, full_path):
                        self.log_message(f"✓ Full animation saved (concatenated dailies): {os.path.basename(full_path)}")
                    else:
                        self.log_message("⚠ Concatenation failed; full animation not produced")
                else:
                    # No dailies, or only a subset — render the full span from scratch.
                    self.log_message("Generating full animation frames (this may take a while)...")
                    video_path = self.create_animation_frames(
                        anim_data, temp_frames_dir, frame_interval, trailing_window,
                        fps, color_by, s['use_identities'],
                        total_export_steps, current_export_step, speed_text=speed_text,
                        axis_limits=axis_limits)
                    if video_path and not self.export_cancelled and os.path.exists(video_path):
                        shutil.move(video_path, full_path)
                        self.log_message(f"✓ Full animation saved: {os.path.basename(full_path)}")

            if not generate_daily and not generate_full:
                self.log_message("No tracking animation selected (neither daily nor full).")

            # ---- Clean up temp frames (once) ----
            try:
                if os.path.exists(temp_frames_dir):
                    shutil.rmtree(temp_frames_dir)
                    self.log_message("✓ Temp frames cleaned up")
            except Exception as e:
                self.log_message(f"Warning: Could not clean temp frames: {str(e)}")

            self.log_message("✓ Animation generation complete!")
            
            # Reset UI after animation completes
            self.exporting = False
            self.btn_export.setEnabled(True)
            self.btn_stop_export.setVisible(False)
            self.progress_bar.setValue(100)
            self.lbl_export_progress.setText("All exports complete!")
            self._notify_done('info', "Success", "All exports completed successfully!")
            QTimer.singleShot(3000, lambda: self.progress_widget.setVisible(False))
            
        except Exception as e:
            self._last_export_failed = True
            self._notify_done('error', "Animation Error", f"Failed to generate animation: {str(e)}")
            self.log_message(f"✗ Animation generation failed: {str(e)}")
        finally:
            # Animation is the last render stage; drop the temp render CSVs and
            # make sure the export flag is cleared even if rendering raised (the
            # batch queue waits on it to advance to the next trial).
            self._cleanup_plot_working_files()
            self.exporting = False
            self.btn_export.setEnabled(True)
            self.btn_stop_export.setVisible(False)

    def create_animation_frames(self, data, output_dir, frame_interval, trailing_window,
                               fps, color_by, use_custom_identities=False,
                               total_export_steps=1, current_export_step=1, day_suffix="",
                               speed_text="", axis_limits=None, time_range=None):
        """Render one animation video via the shared ``uwb_animation`` core.

        This is a thin adapter: it gathers the GUI-held settings (quality/DPI,
        layer choice, background/zones/anchors, identities) and wires the tool's
        progress bar, cancellation flag and logger into the pure renderer.
        """
        s = getattr(self, '_anim_settings', None) or self._snapshot_anim_settings()
        dpi = uwb_animation.QUALITY_DPI.get(s['video_quality'], 100)
        self.log_message(f"Using {dpi} DPI for video generation")

        # Behaviour is classified from THIS clip's rows, so a daily video
        # only pays for its own day rather than re-classifying the trial.
        layers_on = s.get('layers') or self.anim_layer_flags()
        behavior = None
        if layers_on.get('behavior') and 'shortid' in data.columns:
            beh_tags = sorted(data['shortid'].unique())
            if beh_tags:
                self.log_message(
                    "Classifying behaviour for the animation overlay...")
                if getattr(self, '_clip_rendering', False):
                    self.set_clip_progress("Building preview: classifying behaviour…")
                try:
                    behavior = self._animation_behavior(data, beh_tags, layers_on)
                except Exception as e:
                    # A missing overlay is worth far less than the video, so a
                    # classification failure downgrades the clip rather than
                    # losing the whole render.
                    self.log_message(
                        f"Warning: behaviour overlay skipped ({e})")

        bg_image, bg_extent = self._context_bg_source()
        video_filename = f'animation_temp{day_suffix}.mp4' if day_suffix else 'animation_temp.mp4'
        video_output_path = os.path.join(output_dir, video_filename)

        def _progress(i, total):
            # The clip renders on the GUI thread with no progress bar of its
            # own, so it gets the line under the Preview button. Every frame,
            # not every tenth: a clip is ~300 frames and the point is to look
            # alive.
            if getattr(self, '_clip_rendering', False):
                self.set_clip_progress(
                    f"Building preview: frame {i + 1} / {total}")
            if i % 10 == 0 or i == total - 1:
                pct = int(((current_export_step - 1) + (i + 1) / max(1, total)) / max(1, total_export_steps) * 100)
                self.progress_bar.setValue(pct)
                self.lbl_export_progress.setText(
                    f"Step {current_export_step}/{total_export_steps}: Rendering frame {i+1}/{total}...")
                # One tick a second, plus a guaranteed first and last. The
                # log line overwrites the previous tick, so this cadence only
                # sets how often the counter visibly moves - not how much log
                # it leaves behind. Every-50-frames was both too chatty for the
                # log and, at ~30 frames/s, too coarse to look live.
                now = time.time()
                if (i == 0 or i == total - 1
                        or now - getattr(self, '_frame_tick_at', 0.0) >= 1.0):
                    self._frame_tick_at = now
                    self.log_message(f"Rendering frame {i+1}/{total}...")
                QApplication.processEvents()

        data_view = None
        if self.chk_anim_data_view.isChecked():
            if getattr(self, '_clip_rendering', False):
                self.set_clip_progress("Building preview: Data View counters…")
            try:
                data_view = self.build_data_view_payload()
            except Exception as e:
                # Same downgrade rule as the behaviour overlay: a missing
                # panel is worth far less than the video.
                self.log_message(f"Warning: Data View skipped ({e})")

        return uwb_animation.render_animation(
            data, video_output_path,
            frame_interval=frame_interval, trailing_window=trailing_window, fps=fps,
            dpi=dpi, speed_text=speed_text,
            layers=self.plot_layers, bg_image=bg_image, bg_extent=bg_extent,
            zones_xml=self.xml_zones, arena_zones=self.arena_zones,
            anchors=self.anchor_positions, rois=self.rois,
            tag_identities=self.tag_identities, use_custom_identities=use_custom_identities,
            color_by=color_by, marker_size=s['tag_size'],
            show_battery=s['show_battery'],
            show_speed=s.get('show_speed', False),
            show_step=s.get('show_step', False),
            label_color=s.get('label_color'),
            label_outline=s.get('label_outline', True),
            gap_s=s.get('gap_s', 60.0), axis_limits=axis_limits,
            behavior=behavior,
            show_trail=layers_on.get('trail', True),
            show_labels=layers_on.get('tag_id', True),
            time_range=time_range, data_view=data_view,
            is_cancelled=lambda: self.export_cancelled,
            progress=_progress, log=self.log_message,
        )

    # NOTE: stop_export lives with the other export-control methods further up.

    # ---- Batch queue: preprocess many trials one at a time --------------- #
    # Runs each queued database through the exact interactive load + export
    # path, one at a time, so peak memory stays bounded (helped by the per-tag
    # streamed CSV). Per-trial dialogs and the preview index build are
    # suppressed (see _batch_active). A Python-level failure in one trial marks
    # it Failed and the batch moves on; only a hard native crash would stop it.

    def add_current_to_batch(self):
        """Snapshot the currently loaded database + current settings as a job.

        This captures everything shown in the UI right now — table, tag
        selection, thresholds, smoothing, and every export option — so the
        queued job is self-contained. The user can then load a different
        database up top, give it different settings, and add that as another
        job. Run Batch replays each job with its own captured settings.
        """
        if not self.db_path or not self.table_name:
            QMessageBox.warning(
                self, "No Database Loaded",
                "Load a database and select a table first, adjust the settings "
                "and export options you want, then add it to the queue.")
            return

        save_plots = self.chk_save_plots.isChecked()
        save_animation = self.chk_save_animation.isChecked()

        # Make ALL interactive decisions NOW, at add-to-queue time, so the batch
        # run itself never blocks on a dialog. (1) Plot/animation spatial layers
        # — taken from the preview as it stands NOW and frozen into the job by
        # get_config_dict below, so later preview fiddling cannot change what a
        # queued trial renders.
        if save_plots or save_animation:
            self.plot_layers_from_preview()

        self._sync_export_from_preview()

        # Freeze the animation snapshot so conflict prediction matches what will
        # actually be rendered for this job.
        self._anim_settings = self._snapshot_anim_settings()

        # (2) Overwrite conflict — decided PER JOB against files on disk now.
        conflict_choice = ExportConflictDialog.OVERWRITE   # default when nothing exists
        all_conflicting, all_new = self._predict_export_conflicts()
        if all_conflicting:
            dlg = ExportConflictDialog(all_conflicting, all_new, parent=self)
            result = dlg.exec_()
            if result not in (ExportConflictDialog.SKIP, ExportConflictDialog.OVERWRITE,
                              ExportConflictDialog.NEW_FOLDER,
                              ExportConflictDialog.CLEAR):
                self.log_message("Add to queue cancelled at overwrite prompt.")
                return
            # Confirmed HERE rather than at run time: a batch is started and
            # left alone, and a destructive prompt appearing an hour in would
            # stall the queue behind a dialog nobody is watching.
            if result == ExportConflictDialog.CLEAR:
                if not self.confirm_clear_analysis_folder():
                    self.log_message("Add to queue cancelled at the clear prompt.")
                    return
                self.log_message(
                    "  This job will CLEAR its analysis folder before "
                    "exporting (confirmed now; it will not ask again).")
            conflict_choice = result

        job = {
            'path': self.db_path,
            'table': self.table_name,
            'config': self.get_config_dict(),      # full snapshot (incl. plot_layers)
            'conflict_choice': conflict_choice,     # per-job overwrite decision
            'status': 'Queued',
        }
        self._batch_items.append(job)
        n_tags = len(job['config'].get('selected_tags', []))
        # Say it now rather than after twenty minutes of processing: zone
        # occupancy needs polygons from the site XML, and not every site
        # defines them. The plot is skipped at the very end of the plot run,
        # so without this the queue looks like it is going to produce it.
        zone_cb = self.plot_type_checkboxes.get('zone_occupancy')
        if (zone_cb is not None and zone_cb.isChecked()
                and self.chk_save_plots.isChecked() and not self.region_sets()):
            self.log_message(
                "  Note: Zone Occupancy is ticked but no regions are drawn "
                "for this trial, so that plot will be skipped.")
        if ((self.chk_export_roi_bouts.isChecked()
             or self.chk_export_roi_daily.isChecked())
                and not self.region_sets()):
            self.log_message(
                "  Note: an ROI occupancy CSV is ticked but no regions are "
                "drawn for this trial, so no occupancy file will be written.")
        self.log_message(
            f"Added to queue: {os.path.basename(self.db_path)} "
            f"(table {self.table_name}, {n_tags} tag(s); layers + overwrite choices captured)")
        self._refresh_batch_list()

    def clear_batch(self):
        """Empty the batch queue (ignored while a batch is running)."""
        if self._batch_active:
            return
        self._batch_items = []
        self._refresh_batch_list()

    def _refresh_batch_list(self):
        if not hasattr(self, 'batch_list'):
            return
        self.batch_list.clear()
        for i, it in enumerate(self._batch_items, 1):
            cfg = it.get('config', {})
            outs = []
            if cfg.get('export_smoothed_csv', True): outs.append('smoothed')
            if cfg.get('save_plots'): outs.append('plots')
            if cfg.get('save_animation'): outs.append('anim')
            if cfg.get('proximity_detection'): outs.append('prox')
            if cfg.get('export_behavior_events'): outs.append('behav')
            if cfg.get('export_roi_bouts'): outs.append('ROIbouts')
            if cfg.get('export_roi_daily'): outs.append('ROIdaily')
            ntags = len(cfg.get('selected_tags', []))
            summary = f"{ntags} tags · {', '.join(outs) or 'no outputs'}"
            self.batch_list.addItem(
                f"{i}. [{it['status']}]  {os.path.basename(it['path'])}  ({summary})")
        has_items = len(self._batch_items) > 0
        self.btn_run_batch.setEnabled(has_items and not self._batch_active)
        self.btn_add_batch.setEnabled(not self._batch_active)
        self.btn_clear_batch.setEnabled(has_items and not self._batch_active)
        self.btn_stop_batch.setVisible(self._batch_active)

    def _suppress_dialogs(self):
        """Replace modal QMessageBox pop-ups with logging for the batch run.

        An unattended batch must never block on a modal box. The explicit
        _batch_active guards cover the paths we know about; this catch-all makes
        sure a stray dialog anywhere in the load/export chain can't stall the
        queue. Restored in _restore_dialogs().
        """
        if getattr(self, '_saved_dialogs', None) is not None:
            return
        self._saved_dialogs = (QMessageBox.information, QMessageBox.warning,
                               QMessageBox.critical, QMessageBox.question)

        def _silent(*a, **k):
            title = a[1] if len(a) > 1 else ''
            text = a[2] if len(a) > 2 else ''
            self.log_message(f"[batch] suppressed dialog — {title}: {text}")
            return QMessageBox.Ok

        QMessageBox.information = staticmethod(_silent)
        QMessageBox.warning = staticmethod(_silent)
        QMessageBox.critical = staticmethod(_silent)
        QMessageBox.question = staticmethod(_silent)

    def _restore_dialogs(self):
        saved = getattr(self, '_saved_dialogs', None)
        if saved is None:
            return
        (QMessageBox.information, QMessageBox.warning,
         QMessageBox.critical, QMessageBox.question) = saved
        self._saved_dialogs = None

    def run_batch(self):
        """Start processing the queued databases sequentially."""
        if self._batch_active or not self._batch_items:
            return

        # The ONE decision shared across the whole run: where tracking-animation
        # temp frames are written (reused per trial, emptied between them). Ask
        # it here — before the run starts and before dialogs are suppressed — so
        # the user can then walk away. Everything else was decided per-job at
        # add-to-queue time.
        self._batch_temp_frames_dir = None
        anim_jobs = [it for it in self._batch_items
                     if (it.get('config') or {}).get('save_animation')]
        if anim_jobs:
            first = anim_jobs[0]
            db_dir = os.path.dirname(first['path'])
            db_name = os.path.splitext(os.path.basename(first['path']))[0]
            default_dir = os.path.join(db_dir, f"{db_name}_FNT_analysis", 'animation_Tracking')
            chosen = self._prompt_animation_temp_dir(default_dir)
            if chosen is None:
                self.log_message("Batch cancelled at temp-frames selection.")
                return
            self._batch_temp_frames_dir = chosen

        self._sync_export_from_preview()      # export inherits the preview
        # (job index, phase) -> attempts made this run.
        self._batch_attempts = {}
        for it in self._batch_items:          # fresh run: reset all statuses
            it['status'] = 'Queued'
            # Clear last run's outcome too. This flag suppresses a trial's
            # animation when its data pass failed; left sticky it would keep
            # skipping that render on every later run (silently, since the
            # queue row never changed) while the batch moved on to the next
            # trial's animation.
            it.pop('data_failed', None)

        # THREE-phase plan, ordered by how long a pass takes and how much of
        # it you need first: every trial's CSVs, then every trial's plots, then
        # the video renders. Each pass covers the WHOLE queue before the next
        # begins.
        #
        # The point is that a slow product for trial 1 must not delay a fast
        # product for trial 8. Animations take hours and plots take minutes
        # while the CSVs take less; running them trial-by-trial trickles one
        # complete trial out per day, whereas this has every trial's numbers on
        # disk in the first hour and every trial's figures shortly after.
        #
        # Plots sit in the middle rather than with the data because they are
        # rendered, not computed: they cost real time, they are re-runnable
        # from the smoothed CSV alone, and a failure in one must not hold up
        # another trial's CSVs.
        self._batch_plan = [(i, 'data') for i in range(len(self._batch_items))]
        plot_idx = [i for i, it in enumerate(self._batch_items)
                    if (it.get('config') or {}).get('save_plots')]
        self._batch_plan += [(i, 'plots') for i in plot_idx]
        anim_idx = [i for i, it in enumerate(self._batch_items)
                    if (it.get('config') or {}).get('save_animation')
                    or (it.get('config') or {}).get('social_animation')]
        self._batch_plan += [(i, 'animation') for i in anim_idx]
        self._batch_plan_pos = 0
        self._batch_step_running = False
        # Anchors _smoothed_csv_is_fresh: only a CSV written after this
        # moment can have come from this run.
        self._batch_started_at = time.time()
        for _it in self._batch_items:
            _it.pop('reuse_smoothed', None)

        self._batch_active = True
        self._batch_stop_requested = False
        self._batch_frame_cur = 0
        self._batch_frame_total = 0
        self._batch_stage = ''
        self.batch_progress_bar.setValue(0)
        self.lbl_batch_progress.setText("Starting batch...")
        self.lbl_batch_eta.setText("")
        self.batch_progress_widget.setVisible(True)
        self._suppress_dialogs()
        self.log_message("=" * 50)
        self.log_message(
            f"BATCH START: {len(self._batch_items)} database(s) — "
            f"data products first, then {len(anim_idx)} animation render(s)")
        self._log_batch_settings()
        self._refresh_batch_list()
        QTimer.singleShot(0, self._batch_step)

    # Config keys too bulky to be worth pasting into a bug report, replaced by
    # a one-line summary. Everything else is written out in full.
    _LOG_SUMMARISE_KEYS = ('arena_zones', 'xml_config', 'tag_identities', 'rois')

    def _loggable_config(self, cfg):
        """A copy of a job's config with the bulky keys reduced to a summary.

        Zone polygons and the XML anchor dump run to hundreds of coordinate
        pairs; identities can be dozens of records. Pasting those verbatim
        buries the settings someone is actually trying to read.
        """
        out = {}
        for key, value in (cfg or {}).items():
            if key not in self._LOG_SUMMARISE_KEYS or value in (None, [], {}):
                out[key] = value
                continue
            if key == 'tag_identities':
                # Keep the part that changes behaviour (which tags, and their
                # active windows); drop nothing else of consequence.
                # The MODE, not just the timestamp. A bound on Auto
                # keeps its last-seen value in `stop_time` purely for display -
                # it trims nothing - so printing the bare timestamp made an
                # untrimmed tag look like it was being cut off at a stale date.
                def _bound(v, which):
                    mode = IdentityAssignmentDialog._mode_of(v, which)
                    if mode != 'manual':
                        return 'auto'
                    return v.get(f'{which}_time', '-')

                out[key] = {
                    str(t): (f"{v.get('sex', '?')}{v.get('identity', '?')} "
                             f"{_bound(v, 'start')} → {_bound(v, 'stop')}")
                    for t, v in (value or {}).items()}
            elif key == 'arena_zones':
                names = sorted({r.get('zone') for r in value if isinstance(r, dict)})
                out[key] = f"<{len(names)} zone(s): {', '.join(n for n in names if n)}>"
            elif key == 'rois':
                out[key] = "<{} region(s): {}>".format(
                    len(value), ', '.join(str(r.get('name', '')) for r in value))
            else:
                out[key] = f"<{type(value).__name__} omitted from the log>"
        return out

    def _log_batch_settings(self):
        """Write each queued job's settings into the session log.

        Dumped at batch start so the log a user copies is self-contained: the
        run and the exact settings that produced it are in one paste, without
        having to go and find the fnt_config.json of every trial. These are the
        settings CAPTURED when each job was added to the queue, which is what
        actually runs - not necessarily what the settings pane shows now.
        """
        self.log_message("─" * 20 + " batch settings " + "─" * 20)
        for i, it in enumerate(self._batch_items, 1):
            name = os.path.basename(it.get('path') or '?')
            self.log_message(f"[{i}/{len(self._batch_items)}] {name} "
                             f"(table {it.get('table') or '?'}) — settings as queued:")
            try:
                text = json.dumps(self._loggable_config(it.get('config')),
                                   indent=2, sort_keys=True, default=str)
            except Exception as e:
                self.log_message(f"  (could not serialise settings: {e})")
                continue
            for line in text.splitlines():
                self.log_message("  " + line)
        self.log_message("─" * 56)

    def stop_batch(self):
        """Stop the batch, terminating the trial currently running."""
        if not self._batch_active:
            return
        self.log_message("⚠ Batch stop requested — stopping the current trial...")
        self._batch_stop_requested = True
        # Kill the isolated worker process, if one is running.
        proc = getattr(self, '_batch_proc', None)
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.terminate()
            except Exception:
                pass
        # In-process fallback path.
        self.export_cancelled = True
        if getattr(self, 'worker', None) and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()

    def _batch_step(self):
        """Launch the next queued job in its OWN process, or finish the batch.

        Each trial runs in a separate worker process (see fnt.uwb.uwb_batch_worker).
        A UWB export is a long, native-heavy pipeline, and a memory-safety fault
        anywhere in that native stack kills the whole interpreter — when every
        trial shared one process, one such fault destroyed the rest of the queue.
        Isolating trials bounds the damage to a single job and gives each one a
        fresh heap, so nothing accumulates across an overnight run.
        """
        if self._batch_stop_requested or self._batch_plan_pos >= len(self._batch_plan):
            self._finish_batch()
            return

        # The plan position only advances when a job FINISHES, so a second
        # entry here before then would launch a second worker process for the
        # same trial - two processes writing one trial's CSVs and video at the
        # same time. This is posted from several places (run_batch, each skip
        # branch, the poll loop), and a duplicate post is easy to introduce, so
        # guard the launch rather than trusting every caller.
        if getattr(self, '_batch_step_running', False):
            return
        self._batch_step_running = True

        job_idx, phase = self._batch_plan[self._batch_plan_pos]
        self._batch_index = job_idx
        self._batch_phase = phase
        it = self._batch_items[job_idx]

        # A trial whose data pass failed has no smoothed CSV for either of
        # the later passes to render from.
        if phase in ('plots', 'animation') and it.get('data_failed'):
            self.log_message(
                f"Skipping {phase} for {os.path.basename(it['path'])} "
                f"— its data pass failed in this run.")
            # Must still read as a failure: overwriting it with a neutral
            # 'skipped' string lost the fact that the trial failed at all,
            # and the end-of-run summary then counted it as nothing.
            it['status'] = f'Failed (data) \u2014 {phase} skipped'
            self._refresh_batch_list()
            self._batch_plan_pos += 1
            self._batch_step_running = False
            QTimer.singleShot(0, self._batch_step)
            return

        it['status'] = {'data': 'Processing data',
                        'plots': 'Generating plots',
                        'animation': 'Rendering animation'}[phase]
        self._batch_frame_cur = 0
        self._batch_frame_total = 0
        self._batch_stage = ''
        self._batch_step_started = time.time()
        self._refresh_batch_list()
        self._update_batch_progress()
        self.log_message(
            f"BATCH [{self._batch_plan_pos + 1}/{len(self._batch_plan)}] "
            f"({phase}): "
            f"{os.path.basename(it['path'])} — starting worker process")

        # Split the captured settings across the three passes. Each pass turns
        # OFF what the other two own, so no product is computed twice and a
        # re-run of one pass cannot quietly overwrite another's output.
        cfg = dict(it.get('config') or {})
        DATA_PRODUCTS = ('export_raw_csv', 'proximity_detection',
                         'export_roi_bouts', 'export_roi_daily',
                         'export_behavior_events')
        if phase == 'data':
            # The numbers only: no figures, no video.
            cfg['save_plots'] = False
            cfg['save_animation'] = False
            cfg['social_animation'] = False
        elif phase == 'plots':
            # Figures only, drawn from the smoothed CSV the data pass wrote.
            for key in DATA_PRODUCTS:
                cfg[key] = False
            cfg['save_animation'] = False
            cfg['social_animation'] = False
            cfg['save_plots'] = True
        else:
            # Renders only — every other product already exists on disk.
            for key in DATA_PRODUCTS:
                cfg[key] = False
            cfg['save_plots'] = False

        # Frozen builds can't run `python -m`; fall back to the in-process path
        # so the queue still works (without crash isolation).
        if getattr(sys, 'frozen', False):
            self._batch_run_in_process(it, cfg, phase)
            return

        import json as _json
        import subprocess
        import tempfile

        job = {
            'path': it['path'],
            'table': it.get('table'),
            'config': cfg,
            'conflict_choice': self._phase_conflict_choice(it, phase),
            'temp_frames_dir': getattr(self, '_batch_temp_frames_dir', None),
            # The animation pass renders from the CSV the data pass wrote.
            # The animation pass always reuses what the data pass wrote;
            # a data retry reuses it only when this run already verified
            # it (see _smoothed_csv_is_fresh).
            'reuse_smoothed': (phase in ('plots', 'animation')
                               or bool(it.get('reuse_smoothed'))),
        }
        job_dir = tempfile.mkdtemp(prefix='fnt_batch_')
        job_path = os.path.join(job_dir, 'job.json')
        try:
            with open(job_path, 'w', encoding='utf-8') as f:
                _json.dump(job, f, default=str)
        except Exception as e:
            self.log_message(f"✗ Could not write batch job file: {e}")
            it['status'] = 'Failed'
            if getattr(self, '_batch_phase', 'data') == 'data':
                it['data_failed'] = True
            self._refresh_batch_list()
            self._batch_step_running = False
            self._batch_plan_pos += 1
            QTimer.singleShot(0, self._batch_step)
            return

        env = dict(os.environ)
        env['PYTHONIOENCODING'] = 'utf-8'
        env['QT_QPA_PLATFORM'] = 'offscreen'   # worker renders headless
        kwargs = {}
        if os.name == 'nt':
            kwargs['creationflags'] = getattr(subprocess, 'CREATE_NO_WINDOW', 0)

        log_path = os.path.join(job_dir, 'worker.log')
        try:
            self._batch_log_fh = open(log_path, 'w', encoding='utf-8')
            self._batch_proc = subprocess.Popen(
                [sys.executable, '-m', 'fnt.uwb.uwb_batch_worker', job_path],
                stdout=self._batch_log_fh, stderr=subprocess.STDOUT,
                env=env, **kwargs)
        except Exception as e:
            self.log_message(f"✗ Could not start worker process: {e}")
            it['status'] = 'Failed'
            if getattr(self, '_batch_phase', 'data') == 'data':
                it['data_failed'] = True
            self._refresh_batch_list()
            self._batch_step_running = False
            self._batch_plan_pos += 1
            QTimer.singleShot(0, self._batch_step)
            return

        self._batch_log_path = log_path
        self._batch_log_pos = 0
        QTimer.singleShot(500, self._batch_poll_export)

    def _phase_conflict_choice(self, it, phase):
        """This job's conflict choice, as the given pass should see it.

        CLEAR belongs to the DATA pass alone. Every pass runs in its own
        worker process, so a downgrade applied inside a worker dies with it -
        the plots pass would start fresh, see CLEAR again, and delete the CSVs
        the data pass had just written. The decision therefore has to be made
        here, in the parent, where the plan lives.

        Only CLEAR is rewritten; SKIP and NEW_FOLDER keep whatever behaviour
        they already had across passes.
        """
        choice = it.get('conflict_choice', ExportConflictDialog.OVERWRITE)
        if choice == ExportConflictDialog.CLEAR and phase != 'data':
            return ExportConflictDialog.OVERWRITE
        return choice

    def _batch_run_in_process(self, it, cfg=None, phase='data'):
        """Fallback for frozen builds: export in this process (no isolation)."""
        self._pending_config_override = cfg if cfg is not None else it.get('config')
        self._batch_reuse_smoothed = (phase in ('plots', 'animation')
                                      or bool(it.get('reuse_smoothed')))
        if not self._load_database_path(it['path']):
            it['status'] = 'Failed'
            if getattr(self, '_batch_phase', 'data') == 'data':
                it['data_failed'] = True
            self._refresh_batch_list()
            self._batch_step_running = False
            self._batch_plan_pos += 1
            QTimer.singleShot(0, self._batch_step)
            return
        if not any(cb.isChecked() for cb in self.tag_checkboxes.values()):
            for cb in self.tag_checkboxes.values():
                cb.setChecked(True)
        self._batch_conflict_choice = self._phase_conflict_choice(it, phase)
        self._batch_proc = None
        self.export_data()
        QTimer.singleShot(300, self._batch_poll_export)

    def _drain_worker_log(self):
        """Copy new worker stdout into the session log so progress stays visible."""
        path = getattr(self, '_batch_log_path', None)
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                f.seek(self._batch_log_pos)
                chunk = f.read()
                self._batch_log_pos = f.tell()
        except Exception:
            return
        lines = [ln.strip() for ln in chunk.splitlines()]
        # A chunk drained after a stall can hold hundreds of consecutive ticks.
        # Each would only overwrite the one before it in the log, so mark the
        # ones that are immediately followed by another tick of the same kind
        # and skip them: same final line, without the intermediate rewrites.
        keys = [progress_line_key(ln) if ln else None for ln in lines]
        superseded = [False] * len(lines)
        following = None
        for i in range(len(lines) - 1, -1, -1):
            if not lines[i]:
                continue
            if keys[i] is not None and keys[i] == following:
                superseded[i] = True
            following = keys[i]

        for idx, line in enumerate(lines):
            if not line:
                continue
            # Track render progress for the batch progress bar. Parsed from
            # every line, including ones that are never displayed.
            if line.startswith('Creating ') and 'animation frames' in line:
                try:
                    self._batch_frame_total = int(line.split()[1])
                    self._batch_frame_cur = 0
                    self._update_batch_progress()
                except Exception:
                    pass
            elif line.startswith('Rendering frame'):
                try:
                    part = line.split('Rendering frame', 1)[1].strip()
                    num, rest = part.split('/', 1)
                    self._batch_frame_cur = int(num)
                    self._batch_frame_total = int(rest.split('.')[0].strip())
                    self._update_batch_progress()
                except Exception:
                    pass
            elif line and not line.startswith('['):
                self._batch_stage = line[:70]
                self._update_batch_progress()
            if superseded[idx]:
                continue
            self.log_message(f"  [worker] {line}")

    def _update_batch_progress(self):
        """Refresh the progress bar under the queue.

        Overall position = completed plan steps + how far the current step has
        got (known only while frames are rendering, which is where the hours
        go). Everything is derived from the plan, so it stays correct whether a
        step is a data pass or a render.
        """
        if not hasattr(self, 'batch_progress_bar') or not self._batch_active:
            return
        total_steps = max(1, len(self._batch_plan))
        done = min(self._batch_plan_pos, total_steps)
        cur = getattr(self, '_batch_frame_cur', 0)
        tot = getattr(self, '_batch_frame_total', 0)
        frac = min(1.0, cur / tot) if tot else 0.0
        pct = int(round((done + frac) / total_steps * 100))
        self.batch_progress_bar.setValue(max(0, min(100, pct)))

        it = (self._batch_items[self._batch_index]
              if self._batch_index < len(self._batch_items) else None)
        name = os.path.basename(it['path']) if it else ''
        phase = getattr(self, '_batch_phase', 'data')
        self.lbl_batch_progress.setText(
            f"Step {done + 1} of {total_steps}  ·  {name}  ·  "
            + {'data': 'data products', 'plots': 'plots',
               'animation': 'animation render'}.get(phase, phase))

        detail = getattr(self, '_batch_stage', '') or ''
        if tot:
            detail = f"frame {cur:,} of {tot:,} ({frac * 100:.1f}%)"
            started = getattr(self, '_batch_step_started', None)
            if started and cur > 50:
                elapsed = time.time() - started
                remaining = elapsed / max(frac, 1e-9) - elapsed
                if remaining > 0:
                    h, m = int(remaining // 3600), int((remaining % 3600) // 60)
                    detail += (f"  ·  ~{h}h {m:02d}m left in this render"
                               if h else f"  ·  ~{m}m left in this render")
        self.lbl_batch_eta.setText(detail)

    def _batch_poll_export(self):
        """Wait for the current job to finish, then advance to the next."""
        proc = getattr(self, '_batch_proc', None)

        if proc is None:
            # In-process fallback path.
            if self.exporting:
                QTimer.singleShot(300, self._batch_poll_export)
                return
            failed = bool(getattr(self, '_last_export_failed', False))
            # In-process fallback: an exception was caught and reported, so
            # this is a clean failure, never a dead process.
            rc_for_retry = 1
        else:
            self._drain_worker_log()
            if proc.poll() is None:            # still running
                QTimer.singleShot(1000, self._batch_poll_export)
                return
            self._drain_worker_log()           # final flush
            rc = proc.returncode
            failed = (rc != 0)
            rc_for_retry = rc
            try:
                self._batch_log_fh.close()
            except Exception:
                pass
            self._batch_proc = None
            if failed and not self._batch_stop_requested:
                self.log_message(
                    f"✗ Worker exited with code {rc} — this trial failed, "
                    f"continuing with the rest of the queue.")

        it = self._batch_items[self._batch_index]
        phase = getattr(self, '_batch_phase', 'data')
        cfg_of = it.get('config') or {}
        has_anim = bool(cfg_of.get('save_animation')
                        or cfg_of.get('social_animation'))
        has_plots = bool(cfg_of.get('save_plots'))

        def _queued_after(done_phase):
            """Status for a finished pass, naming whatever is still owed.

            The queue row is the only place a part-finished trial explains
            itself, so it says what is still to come rather than a bare
            'Done' that would read as complete.
            """
            if done_phase == 'data' and has_plots:
                return 'Data ✓ — plots queued'
            if done_phase == 'plots' and has_anim:
                return 'Plots ✓ — animation queued'
            if done_phase == 'data' and has_anim:
                return 'Data ✓ — animation queued'
            return 'Done'

        # A worker that stopped because Stop Batch killed it did not fail -
        # nothing went wrong, the user changed their mind. Calling it a failure
        # buried what HAD worked: a trial whose data products were already on
        # disk was summarised as "Failed (animation)" because an hour into the
        # render someone pressed stop.
        if self._batch_stop_requested:
            # A trial whose earlier passes finished keeps that in its
            # status: the pass in flight was cancelled, the CSVs were not.
            it['status'] = (f'Data ✓ ({phase} cancelled)'
                            if phase in ('plots', 'animation')
                            and it.get('data_ok')
                            else f'Cancelled ({phase})')
            self.log_message(
                f"BATCH {os.path.basename(it['path'])} [{phase}]: "
                f"{it['status']}")
            self._refresh_batch_list()
            self._batch_step_running = False
            self._batch_plan_pos += 1
            QTimer.singleShot(0, self._batch_step)
            return

        # A worker that DIED (native fault) is worth another go: those
        # faults are non-deterministic. A worker that ran and reported an
        # error is not - it would fail identically and only delay the queue.
        if failed and self._batch_should_retry(it, phase, rc_for_retry):
            key = (self._batch_index, phase)
            attempts = self._batch_attempts.get(key, 1) + 1
            self._batch_attempts[key] = attempts
            limit = attempts + self._batch_retries_left(it, phase)
            it['status'] = f'Retrying ({phase}, attempt {attempts})'
            self.log_message(
                ("↺ Output failed verification — retrying "
                 if rc_for_retry == self.EXIT_OUTPUT_CORRUPT else
                 "↺ Worker died on a native fault — retrying ") +
                f"{os.path.basename(it['path'])} [{phase}], "
                f"attempt {attempts} of {limit}.")
            # The smoothed CSV is published atomically and VERIFIED before any
            # of the stages that crash, so if this attempt got that far the
            # file on disk is sound. Re-deriving it costs minutes for a
            # byte-identical result - on a 7-tag trial, ~5 of the ~12 minutes
            # each retry was spending.
            if phase == 'data' and self._smoothed_csv_is_fresh(it):
                it['reuse_smoothed'] = True
                self.log_message(
                    "  its smoothed CSV is already written and verified — "
                    "the retry will reuse it instead of re-processing "
                    "every tag.")
            self._refresh_batch_list()
            # Re-run the SAME plan step: do not advance _batch_plan_pos.
            # The one-launch guard has to be released explicitly here,
            # precisely because this is the one path that relaunches a
            # position on purpose rather than moving past it.
            self._batch_step_running = False
            QTimer.singleShot(0, self._batch_step)
            return

        if failed:
            tries = self._batch_attempts.get((self._batch_index, phase), 1)
            it['status'] = (f'Failed ({phase})' if tries == 1
                            else f'Failed ({phase}) after {tries} attempts')
            if phase == 'data':
                it['data_failed'] = True
        elif phase == 'data':
            # The numbers are on disk; figures and renders come later.
            it['data_ok'] = True
            it['status'] = _queued_after('data')
        elif phase == 'plots':
            it['status'] = _queued_after('plots')
        else:
            it['status'] = 'Done'
        self.log_message(f"BATCH {os.path.basename(it['path'])} [{phase}]: {it['status']}")
        self._refresh_batch_list()
        self._batch_step_running = False
        self._batch_plan_pos += 1
        QTimer.singleShot(0, self._batch_step)

    def _smoothed_csv_is_fresh(self, it):
        """True when THIS batch run already wrote and verified the CSV.

        Two guards, because reusing the wrong file would silently analyse the
        wrong data. The mtime must fall inside this batch run, so a file left
        by an earlier run is never picked up; and the export only reaches the
        stages that crash AFTER it has verified what it wrote, so a file that
        exists and is recent is one that passed. Publication is atomic (a
        .partial renamed with os.replace), so a half-written file cannot be
        mistaken for a finished one.

        A trial exported into a timestamped folder (the New Folder conflict
        choice) simply will not be found here, and falls back to re-processing.
        """
        started = getattr(self, '_batch_started_at', None)
        if started is None:
            return False
        try:
            db_dir = os.path.dirname(it['path'])
            db_name = os.path.splitext(os.path.basename(it['path']))[0]
            csv = os.path.join(db_dir, f"{db_name}_FNT_analysis",
                               f"{db_name}_smoothed.csv")
            return os.path.exists(csv) and os.path.getmtime(csv) >= started
        except Exception:
            return False

    @classmethod
    def _is_native_fault(cls, rc):
        """True when a worker exit is worth retrying.

        The worker reports its own problems by logging and returning 1. Anything
        else - a Windows NTSTATUS such as 0xC0000005 (access violation) or
        0xC0000374 (heap corruption), or a POSIX signal (negative rc) - means
        the process was killed out from under us mid-run.
        """
        if rc is None:
            return False
        if rc < 0:                       # POSIX: killed by signal
            return True
        if rc == cls.EXIT_OUTPUT_CORRUPT:
            # Not a dead process, but the same underlying problem: the
            # file the worker wrote failed its integrity check, and a
            # re-run usually produces a clean one.
            return True
        return (rc & 0xFFFFFFFF) >= 0xC0000000

    # Statuses that mean "still owed something" — a batch that stops has to
    # rewrite every one of them, or a row sits reading 'Rendering animation'
    # forever. Kept as data so adding a pass cannot silently miss one.
    _CANCELLABLE_STATUSES = frozenset({
        'Queued', 'Loading', 'Running',
        'Processing data', 'Generating plots', 'Rendering animation',
        'Data ✓ — plots queued', 'Data ✓ — animation queued',
        'Plots ✓ — animation queued',
    })

    # What each of those becomes when the trial's data pass HAD finished:
    # name the pass that was lost rather than a vague 'remaining passes'.
    _CANCELLED_AS = {
        'Generating plots': 'Data ✓ (plots cancelled)',
        'Rendering animation': 'Data ✓ (animation cancelled)',
        'Data ✓ — plots queued': 'Data ✓ (plots cancelled)',
        'Data ✓ — animation queued': 'Data ✓ (animation cancelled)',
        'Plots ✓ — animation queued': 'Plots ✓ (animation cancelled)',
    }

    def _batch_retries_left(self, it, phase):
        cap = self.spin_batch_retries.value() if hasattr(self, 'spin_batch_retries') else 0
        used = self._batch_attempts.get((self._batch_index, phase), 1) - 1
        return max(0, cap - used)

    def _batch_should_retry(self, it, phase, rc):
        if not self._is_native_fault(rc):
            return False
        return self._batch_retries_left(it, phase) > 0

    def _finish_batch(self):
        self._batch_active = False
        self._batch_step_running = False
        if hasattr(self, 'batch_progress_widget'):
            self.batch_progress_bar.setValue(100)
            self.batch_progress_widget.setVisible(False)
        self._restore_dialogs()   # before the summary box, and re-enable normal UI dialogs
        for it in self._batch_items:
            status = str(it['status'])
            if (status in self._CANCELLABLE_STATUSES
                    or status.startswith('Retrying')):
                # A trial whose data pass finished keeps that in its status:
                # the later passes were cancelled, its CSVs were not. Which
                # pass was outstanding is read off the status it was showing,
                # so the row still says what was lost.
                it['status'] = (
                    self._CANCELLED_AS.get(status,
                                           'Data ✓ (remaining passes cancelled)')
                    if it.get('data_ok') else 'Cancelled')
        self._refresh_batch_list()
        # Every item lands in exactly one bucket, and anything unrecognised
        # is reported rather than dropped - a status that matched none of
        # these is what made a failed run summarise as '0 done, 0 failed,
        # 0 cancelled'.
        done = failed = cancelled = 0
        unknown = []
        for it in self._batch_items:
            status = str(it.get('status', ''))
            if status.startswith('Failed'):
                failed += 1
            elif 'ancelled' in status:
                cancelled += 1
            elif status.startswith('Done') or status.startswith('Data \u2713'):
                done += 1
            else:
                unknown.append(f"{os.path.basename(it['path'])}: {status}")
        total = len(self._batch_items)
        summary = (f"{done} done, {failed} failed, {cancelled} cancelled"
                   + (f", {len(unknown)} unaccounted" if unknown else ""))
        # A stopped run reads as all-cancelled, which undersells it: the CSVs
        # and plots of every trial that got that far are on disk and do not
        # need redoing. Say so, rather than letting '0 done' imply the hours
        # before the stop were wasted.
        data_ok = sum(1 for it in self._batch_items if it.get('data_ok'))
        if data_ok > done:
            summary += (f" ({data_ok} of {total} have their data products "
                        f"complete)")
        self.log_message(f"BATCH FINISHED: {total} trial(s) \u2014 {summary}")
        for u in unknown:
            self.log_message(f"  unaccounted status \u2014 {u}")
        if failed:
            for it in self._batch_items:
                if str(it.get('status', '')).startswith('Failed'):
                    self.log_message(
                        f"  FAILED: {os.path.basename(it['path'])} "
                        f"({it['status']})")
        self._batch_stop_requested = False
        QMessageBox.information(
            self, "Batch complete",
            f"Batch finished: {total} trial(s).\n\n"
            f"{done} succeeded, {failed} failed, {cancelled} cancelled."
            + (f"\n{len(unknown)} finished in an unexpected state \u2014 "
               "see the message log." if unknown else "")
            + ("\n\nFailed trials are listed in the message log; the queue "
               "is kept so you can re-run them." if failed else ""))

    def _cleanup_plot_working_files(self):
        """Delete any temporary render CSVs tracked for this run.

        Plots and animation now render straight from the exported smoothed CSV,
        so no temp render files are written and this list is normally empty. The
        method is kept as a defensive no-op safety net (and to clean anything a
        future render stage might register). The smoothed CSV deliverable is
        never added here.
        """
        for path in getattr(self, '_plot_working_files', []):
            try:
                if path and os.path.exists(path):
                    os.remove(path)
                    self.log_message(f"Removed temporary render file: {os.path.basename(path)}")
            except OSError as e:
                self.log_message(f"Could not remove temp render file {os.path.basename(path)}: {e}")
        self._plot_working_files = []

    # Exit code the batch worker uses for "the file we wrote did not verify".
    # Distinct from 1 (a real error that will recur) so the queue knows a retry
    # is worth attempting.
    EXIT_OUTPUT_CORRUPT = 17

    #: Columns of the smoothed CSV that must parse as numbers.
    _SMOOTHED_NUMERIC_COLS = ('shortid', 'timestamp', 'location_x', 'location_y',
                              'smoothed_x', 'smoothed_y')

    def verify_smoothed_csv(self, path, chunksize=2_000_000):
        """Check every numeric column of the written CSV actually parses.

        This exists because a written value was once observed to come back as
        's\\x00\\xc2\\x0011.606...' - four bytes of memory garbage prepended to an
        otherwise correct float, one cell in sixty million. Nothing in this
        program can produce that; it is the machine corrupting a buffer between
        formatting and disk.

        Without this check the damage surfaced two minutes later as
        "agg function failed [how->mean,dtype->object]" deep inside a pandas
        groupby, because ONE bad cell makes pandas read the whole column as
        strings. Catching it here names the file, the line and the value, and
        lets the batch retry the trial - a re-run usually writes it cleanly.

        Returns (ok, list_of_problem_strings).
        """
        problems = []
        try:
            cols = list(pd.read_csv(path, nrows=0).columns)
        except Exception as e:
            return False, [f"could not read the header: {e}"]
        check = [c for c in self._SMOOTHED_NUMERIC_COLS if c in cols]
        if not check:
            return True, []

        # Fast path first. Read those columns AS NUMBERS: a corrupted cell
        # cannot parse, so pandas hands the whole column back as object dtype
        # - the same signal the string scan below looks for, for a fifth of
        # the cost (6 s against 35 s on a 14 M-row trial). Chunked, so a large
        # file is still bounded in memory.
        try:
            suspect = False
            for chunk in pd.read_csv(path, usecols=check, chunksize=chunksize):
                if any(not pd.api.types.is_numeric_dtype(chunk[c])
                       for c in check):
                    suspect = True
                    break
            if not suspect:
                return True, []
        except Exception as e:
            return False, [f"could not read the file: {e}"]

        # Something did not parse. Re-read as text to name the file, the line
        # and the value - this is the expensive pass, and now it only runs
        # when there is actually something to report.
        line0 = 1                       # 1-based, after the header
        try:
            for chunk in pd.read_csv(path, usecols=check, dtype=str,
                                     chunksize=chunksize):
                for col in check:
                    raw = chunk[col]
                    bad = raw[pd.to_numeric(raw, errors='coerce').isna() & raw.notna()]
                    for pos, value in list(bad.items())[:3]:
                        problems.append(
                            f"line {line0 + pos + 1:,}, column '{col}': {value!r}")
                    if len(bad) > 3:
                        problems.append(
                            f"  ...and {len(bad) - 3} more bad value(s) in '{col}'")
                line0 += len(chunk)
                if len(problems) > 12:
                    problems.append("  (stopping the scan; enough evidence)")
                    break
        except Exception as e:
            return False, [f"scan failed: {e}"]
        return (not problems), problems

    def _predict_export_conflicts(self):
        """Predict output files and split into (conflicting, new) vs. what's on disk.

        Reads the current UI settings + ``self._anim_settings`` + ``db_path`` so
        it is valid both at add-to-queue time (to show the conflict dialog) and
        at export time (to apply a stored choice). Returns
        (all_conflicting, all_new); each is a list of (filename, subfolder_label)
        with "" meaning the analysis-folder root.
        """
        db_dir = os.path.dirname(self.db_path)
        db_name = os.path.splitext(os.path.basename(self.db_path))[0]
        base_output_dir = os.path.join(db_dir, f"{db_name}_FNT_analysis")
        plots_subdir = os.path.join(base_output_dir, 'plots')
        animations_subdir = os.path.join(base_output_dir, 'animation_Tracking')
        sna_subdir = os.path.join(base_output_dir, 'animation_SocialNetworks')

        export_raw_csv = self.chk_export_raw_csv.isChecked()
        # On the animation pass of a two-phase batch the smoothed CSV is an
        # INPUT written by the data pass, not an output of this pass. Predicting
        # it as an output made it "conflicting", and the Overwrite choice then
        # DELETED the freshly written CSV before the render could reuse it.
        export_smoothed_csv = not getattr(self, '_batch_reuse_smoothed', False)
        detect_proximity = self.chk_proximity_detection.isChecked()
        social_animation = False   # social-network animation was removed
        save_animation = self.chk_save_animation.isChecked()
        save_plots = self.chk_save_plots.isChecked()
        save_svg = self.chk_save_svg.isChecked()
        anim = getattr(self, '_anim_settings', None) or self._snapshot_anim_settings()

        def _tag_suffix(tag):
            if self.tag_identities and tag in self.tag_identities:
                info = self.tag_identities[tag]
                return f"{info.get('sex', 'M')}-{info.get('identity', str(tag))}"
            return f"HexID{hex(tag).upper().replace('0X', '')}"

        def _add_with_svg(lst, basename):
            lst.append(basename + '.png')
            if save_svg:
                lst.append(basename + '.svg')

        predicted_files = []
        if export_raw_csv:
            predicted_files.append(f'{db_name}_raw.csv')
        if export_smoothed_csv:
            predicted_files.append(f'{db_name}_smoothed.csv')
        if detect_proximity:
            predicted_files.append(f'{db_name}_SocialOverlapBouts.csv')
        if self.chk_export_behavior.isChecked():
            predicted_files.append(f'{db_name}_behavior_events.csv')
        if not getattr(self, '_batch_reuse_smoothed', False):
            if self.chk_export_roi_bouts.isChecked():
                predicted_files.append(f'{db_name}_ROI_bout_occupancy.csv')
            if self.chk_export_roi_daily.isChecked():
                predicted_files.append(f'{db_name}_ROI_DailySummary.csv')

        predicted_sna_files = []   # social-network animation was removed

        predicted_animation_files = []
        if save_animation:
            fps = anim['fps']
            speed_text = anim['speed_text']
            if anim['generate_daily']:
                for day_idx, date_str in enumerate(anim['selected_days']):
                    predicted_animation_files.append(
                        f'{db_name}_Animation_Day{day_idx + 1}_{date_str}_{fps}fps_{speed_text}.mp4')
            if anim.get('generate_full', not anim['generate_daily']):
                predicted_animation_files.append(f'{db_name}_Animation_{fps}fps_{speed_text}.mp4')

        predicted_plot_files = []
        if save_plots:
            selected_tags = [tag for tag, cb in self.tag_checkboxes.items() if cb.isChecked()]
            plot_types = {k: self.plot_type_checkboxes[k].isChecked() for k in self.plot_type_checkboxes}
            if plot_types.get('daily_paths', False):
                for tag in selected_tags:
                    _add_with_svg(predicted_plot_files, f'{db_name}_DailyPaths_{_tag_suffix(tag)}')
            if plot_types.get('trajectory_overview', False):
                _add_with_svg(predicted_plot_files, f'{db_name}_TrajectoryOverview')
            if plot_types.get('battery_levels', False):
                _add_with_svg(predicted_plot_files, f'{db_name}_BatteryLevels')
            if plot_types.get('activity_timeline', False):
                _add_with_svg(predicted_plot_files, f'{db_name}_ActivityTimeline')
            if plot_types.get('velocity_distribution', False):
                _add_with_svg(predicted_plot_files, f'{db_name}_VelocityDistribution')
            if plot_types.get('cumulative_distance', False):
                _add_with_svg(predicted_plot_files, f'{db_name}_CumulativeDistance')
            if plot_types.get('velocity_timeline', False):
                _add_with_svg(predicted_plot_files, f'{db_name}_VelocityTimeline')
            if plot_types.get('home_range', False):
                for tag in selected_tags:
                    _add_with_svg(predicted_plot_files, f'{db_name}_HomeRange_{_tag_suffix(tag)}')
                _add_with_svg(predicted_plot_files, f'{db_name}_HomeRange_Overview')
            if plot_types.get('mcp_range', False):
                for tag in selected_tags:
                    _add_with_svg(predicted_plot_files, f'{db_name}_MCP_{_tag_suffix(tag)}')
                _add_with_svg(predicted_plot_files, f'{db_name}_MCP_Overview')
            # Only when there is something to measure. The plot skips itself
            # with no drawn regions, and predicting it anyway made Overwrite
            # DELETE the previous run's figure for a pass that then wrote no
            # replacement - the analysis folder losing a good plot to a
            # re-export that never regenerated it.
            if plot_types.get('zone_occupancy', False) and self.region_sets():
                _add_with_svg(predicted_plot_files, f'{db_name}_ZoneOccupancy')
            if plot_types.get('ethogram', False):
                _add_with_svg(predicted_plot_files, f'{db_name}_Ethogram')
            if plot_types.get('social_budget', False):
                _add_with_svg(predicted_plot_files, f'{db_name}_SocialBudget')
            if plot_types.get('behavior_counts', False):
                _add_with_svg(predicted_plot_files, f'{db_name}_BehaviorCounts')
            if plot_types.get('behavior_matrix', False):
                _add_with_svg(predicted_plot_files, f'{db_name}_BehaviorMatrix')
            if plot_types.get('data_quality', False):
                _add_with_svg(predicted_plot_files, f'{db_name}_DataQuality')

        all_conflicting, all_new = [], []
        for f in predicted_files:
            # export_csv_path finds it in csvs/ or, for a trial exported
            # before that subfolder existed, at the top level - so an old
            # layout still reports its conflicts instead of looking untouched
            # and being silently written alongside.
            here = export_csv_path(base_output_dir, f)
            where = (CSV_SUBDIR if os.path.dirname(here).endswith(CSV_SUBDIR)
                     else "")
            (all_conflicting if os.path.exists(here) else all_new).append((f, where))
        for f in predicted_plot_files:
            (all_conflicting if os.path.exists(os.path.join(plots_subdir, f)) else all_new).append((f, "plots"))
        for f in predicted_animation_files:
            (all_conflicting if os.path.exists(os.path.join(animations_subdir, f)) else all_new).append((f, "animation_Tracking"))
        for f in predicted_sna_files:
            (all_conflicting if os.path.exists(os.path.join(sna_subdir, f)) else all_new).append((f, "animation_SocialNetworks"))
        return all_conflicting, all_new

    def export_data(self):
        """Export data and/or plots based on selected options"""
        if not self.db_path:
            QMessageBox.warning(self, "No Database", "Please select a database first")
            return

        # Note: self.data can be None if this is a fresh export
        # This is OK - plots and animations will load data directly from CSV/database

        # Initialize export state
        self.export_cancelled = False
        # Both render on the GUI thread, so a playing preview would fight the
        # export for it and make an already long job longer.
        if self._playing:
            self.stop_preview_playback()
            self.log_message("Preview playback paused for the export.")
        self._last_export_failed = False  # set by error handlers; read by the batch queue
        self._plot_working_files = []  # defensive: normally stays empty now
        self.exporting = True
        self.btn_export.setEnabled(False)
        self.btn_stop_export.setVisible(True)
        self.progress_widget.setVisible(True)
        self.progress_bar.setValue(0)

        # Freeze animation settings NOW, before the (background-threaded) plot
        # export and the main-thread animation render — both leave the window
        # interactive — so clicking the controls mid-export can't change a render
        # already queued. Everything downstream reads this snapshot, not widgets.
        self._anim_settings = self._snapshot_anim_settings()

        # Pull the preview's live values onto the hidden export mirrors BEFORE
        # reading any of them. The preview is the single source of truth, but
        # its widgets are wired only to redraw (e.g. Social radius ->
        # render_preview_frame), so without this the mirrors can still hold the
        # values from whenever they were last synced — and the proximity/social
        # products would silently be computed at a stale contact threshold
        # while fnt_config.json recorded the new one.
        self._sync_export_from_preview()

        # Gather export settings
        db_dir = os.path.dirname(self.db_path)
        db_filename = os.path.basename(self.db_path)
        db_name = os.path.splitext(db_filename)[0]  # Remove extension

        export_raw_csv = self.chk_export_raw_csv.isChecked()
        # Always exported, except on the animation pass of a two-phase batch,
        # where the data pass already wrote it (see _batch_reuse_smoothed).
        export_smoothed_csv = not getattr(self, '_batch_reuse_smoothed', False)
        save_plots = self.chk_save_plots.isChecked()
        save_animation = self.chk_save_animation.isChecked()

        detect_proximity = self.chk_proximity_detection.isChecked()
        proximity_threshold = self.spin_proximity_threshold.value()
        export_behavior = self.chk_export_behavior.isChecked()
        export_roi_bouts = self.chk_export_roi_bouts.isChecked()
        export_roi_daily = self.chk_export_roi_daily.isChecked()
        export_zone_occupancy = export_roi_bouts or export_roi_daily
        social_animation = False   # social-network animation was removed
        # Frozen at export start so mid-export clicks can't change them.

        if not (export_raw_csv or export_smoothed_csv or save_plots
                or save_animation or detect_proximity
                or export_zone_occupancy or export_behavior
                or social_animation):
            QMessageBox.warning(self, "No Export Selected", "Please select at least one export option (CSV, Plots, or Animation)")
            return

        batch = getattr(self, '_batch_active', False)

        # --- Spatial layers for plots/animation ---
        # Interactively: inherited from the preview toggles, so the export draws
        # the arena you were just looking at. In a batch they were frozen at
        # add-to-queue time and are already in self.plot_layers (restored from
        # the job's captured config), so the preview is not consulted here.
        if (save_plots or save_animation) and not batch:
            self.plot_layers_from_preview()

        # --- Animation temp-frames location ---
        # Interactively: ask up-front so the user can start and walk away. In a
        # batch: one shared location was chosen when Export Batch was clicked
        # (self._batch_temp_frames_dir); reuse it for every trial.
        self._anim_temp_frames_dir = None
        if save_animation:
            if batch:
                self._anim_temp_frames_dir = (
                    getattr(self, '_batch_temp_frames_dir', None)
                    or os.path.join(db_dir, f"{db_name}_FNT_analysis", 'animation_Tracking'))
            else:
                _anim_dir = os.path.join(db_dir, f"{db_name}_FNT_analysis", 'animation_Tracking')
                self._anim_temp_frames_dir = self._prompt_animation_temp_dir(_anim_dir)
                if self._anim_temp_frames_dir is None:
                    self.log_message("Export cancelled at temp-frames selection.")
                    self.exporting = False
                    self.btn_export.setEnabled(True)
                    self.btn_stop_export.setVisible(False)
                    self.progress_widget.setVisible(False)
                    return

        # --- Conflict detection ---
        base_output_dir = os.path.join(db_dir, f"{db_name}_FNT_analysis")
        plots_subdir = os.path.join(base_output_dir, 'plots')
        animations_subdir = os.path.join(base_output_dir, 'animation_Tracking')

        # Predict which files this export would produce and which already exist
        # (shared with add-to-queue, which shows the same dialog up-front).
        all_conflicting, all_new = self._predict_export_conflicts()

        skip_existing = False
        output_dir = base_output_dir

        # Clear & Re-export, when that is this job's standing choice. Handled
        # BEFORE and OUTSIDE the conflict branch for two reasons: a folder can
        # hold orphans with nothing conflicting at all (the case the conflict
        # dialog cannot see), and the clear must happen exactly ONCE per trial.
        # With the three-pass plan, running it again before the plots pass
        # would delete the CSVs the data pass had just written.
        if (getattr(self, '_batch_active', False)
                and getattr(self, '_batch_conflict_choice', None)
                == ExportConflictDialog.CLEAR):
            if getattr(self, '_batch_phase', 'data') == 'data':
                self.log_message(
                    f"Clear & Re-export: emptying "
                    f"{os.path.basename(base_output_dir)} before this trial.")
                _rm, _freed, _fail = clear_analysis_folder(
                    base_output_dir, log=self.log_message)
                for _f in _fail:
                    self.log_message(f"  \u26a0 could not remove {_f}")
            else:
                self.log_message(
                    "Clear & Re-export already ran in this trial's data pass; "
                    "this pass writes into the folder it left.")
            # Whatever is in the folder now was written by THIS run, so the
            # ordinary replace-what-I-produce rule is the right one from here.
            self._batch_conflict_choice = ExportConflictDialog.OVERWRITE
            all_conflicting, all_new = self._predict_export_conflicts()

        if all_conflicting:
            if getattr(self, '_batch_active', False):
                # Per-job choice was recorded when the job was queued; replay it
                # silently so the batch never blocks on this dialog.
                result = getattr(self, '_batch_conflict_choice',
                                 ExportConflictDialog.OVERWRITE)
            else:
                dialog = ExportConflictDialog(all_conflicting, all_new, parent=self)
                result = dialog.exec_()

            if result == ExportConflictDialog.SKIP:
                skip_existing = True
                output_dir = base_output_dir
            elif result == ExportConflictDialog.OVERWRITE:
                skip_existing = False
                output_dir = base_output_dir
                # Delete conflicting files so they get cleanly rewritten —
                # except the smoothed CSV, which is published atomically
                # (.partial + os.replace). Pre-deleting it would destroy the
                # previous good copy if the rewrite then crashed.
                _atomic = f'{db_name}_smoothed.csv'
                for fname, subfolder in all_conflicting:
                    if not subfolder and fname == _atomic:
                        continue
                    if subfolder:
                        fpath = os.path.join(base_output_dir, subfolder, fname)
                    else:
                        fpath = os.path.join(base_output_dir, fname)
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
            elif result == ExportConflictDialog.CLEAR:
                # Interactive only - the batch path handled its own clear
                # above, once, before this branch was reached.
                if not self.confirm_clear_analysis_folder():
                    self.exporting = False
                    self.btn_export.setEnabled(True)
                    self.btn_stop_export.setVisible(False)
                    self.progress_widget.setVisible(False)
                    return
                _rm, _freed, _fail = clear_analysis_folder(
                    base_output_dir, log=self.log_message)
                for _f in _fail:
                    self.log_message(f"  \u26a0 could not remove {_f}")
                skip_existing = False
                output_dir = base_output_dir
            elif result == ExportConflictDialog.NEW_FOLDER:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                output_dir = os.path.join(db_dir, f"{db_name}_FNT_analysis_{timestamp}")
                skip_existing = False
            else:
                # User cancelled
                self.exporting = False
                self.btn_export.setEnabled(True)
                self.btn_stop_export.setVisible(False)
                self.progress_widget.setVisible(False)
                return

        # Create output directory and subfolders only as needed
        ensure_dir(output_dir)
        plots_dir = os.path.join(output_dir, 'plots')
        if save_plots:
            ensure_dir(plots_dir)
        animations_dir = os.path.join(output_dir, 'animation_Tracking')
        if save_animation:
            ensure_dir(animations_dir)

        try:
            # Busy cursor for the whole synchronous prep. The heavy pandas /
            # sqlite calls below run on the GUI thread and cannot be interrupted
            # mid-call, so the window can still briefly stop repainting; the
            # cursor plus the finer-grained processEvents() calls keep it from
            # looking dead. Restored in the finally so every exit path clears it.
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.log_message(f"Starting export to {output_dir}")
            # Before the long part, not after: a folder that accepts writes but
            # refuses to be listed exports fine and can never be cleared again,
            # so every later run's output piles onto this one's.
            for _d in (output_dir, csv_output_dir(output_dir)):
                _why = check_dir_usable(_d)
                if _why:
                    self.log_message(f"  ⚠ OUTPUT FOLDER PROBLEM: {_why}")
            self.lbl_export_progress.setText("Initializing export...")

            # Calculate total steps for progress
            total_steps = 0
            if export_raw_csv:
                total_steps += 1
            if export_smoothed_csv:
                total_steps += 1
            if save_plots:
                total_steps += 1
            if save_animation:
                total_steps += 1

            current_step = 0

            # Note: Conflict resolution (skip/overwrite/new folder) was handled above

            # Export raw CSV (unprocessed database dump)
            if export_raw_csv:
                if self.export_cancelled:
                    self.stop_export()
                    return

                current_step += 1
                self.lbl_export_progress.setText(f"Step {current_step}/{total_steps}: Exporting raw CSV...")
                self.progress_bar.setValue(int(current_step / total_steps * 100))
                QApplication.processEvents()

                raw_csv_filename = f'{db_name}_raw.csv'
                raw_csv_path = os.path.join(
                    csv_output_dir(output_dir, create=True), raw_csv_filename)
                if skip_existing and os.path.exists(raw_csv_path):
                    self.log_message(f"Skipped (exists): {raw_csv_filename}")
                else:
                    self.log_message("Exporting raw database to CSV...")
                    QApplication.processEvents()
                    conn = connect_ro(self.db_path)
                    raw_data = pd.read_sql_query(f"SELECT * FROM {self.table_name}", conn)
                    conn.close()
                    raw_data.to_csv(raw_csv_path, index=False)
                    self.log_message(f"✓ Raw CSV exported: {raw_csv_filename}")
                    QApplication.processEvents()

            # Which regions to measure. Read once, before anything is
            # written, so a run measures a fixed set even if the user carries
            # on drawing while the export proceeds.
            region_sets = self.region_sets() if export_zone_occupancy else []
            roi_bout_rows = []

            # Prepare processed data (needed for smoothed CSV, plots, animation, behaviors)
            needs_processed_data = (export_smoothed_csv or save_plots
                                    or save_animation or detect_proximity
                                    or (export_zone_occupancy and region_sets))
            csv_path = None  # Path to the CSV that plots/animation will use
            plot_csv_path = None
            anim_csv_path = None

            # Animation pass of a two-phase batch: the smoothed CSV was already
            # written by the data pass, so render straight from it instead of
            # re-reading and re-smoothing every tag (minutes of pure waste).
            if getattr(self, '_batch_reuse_smoothed', False):
                _existing = export_csv_path(output_dir,
                                            f'{db_name}_smoothed.csv')
                if os.path.exists(_existing):
                    csv_path = _existing
                    plot_csv_path = _existing if save_plots else None
                    anim_csv_path = _existing if save_animation else None
                    needs_processed_data = False
                    self.log_message(
                        f"Reusing the existing {os.path.basename(_existing)} "
                        f"(no re-processing).")
                else:
                    self.log_message(
                        "Animation pass: smoothed CSV missing — reprocessing from the database.")

            if needs_processed_data:
                if self.export_cancelled:
                    self.stop_export()
                    return

                self.log_message("Preparing processed data (per-tag chunked processing)...")

                selected_tags = [tag for tag, cb in self.tag_checkboxes.items() if cb.isChecked()]
                tz = pytz.timezone(self.combo_timezone.currentText())
                smoothing_method = self.get_smoothing_method()
                do_filter = self.chk_velocity_filter.isChecked() or self.chk_jump_filter.isChecked()

                # Filter statistics accumulate across tags; without this reset
                # the run summary would report whatever the last call left
                # behind (a preview chunk, or a previous export).
                self.reset_filter_stats()

                # Stream the smoothed CSV one tag at a time rather than holding
                # the whole processed dataset in RAM and writing it in one shot
                # (the old concat -> sort -> to_csv path peaked at ~3x the full
                # dataset in memory). Tags are processed in ascending shortid
                # order and each chunk is time-sorted, so the streamed file
                # matches the old global (shortid, Timestamp) sort.
                #
                # Written to '<name>.partial' and renamed only after the last
                # tag succeeds. A hard crash (or power loss) mid-run therefore
                # leaves a .partial file, never a truncated '<name>_smoothed.csv'
                # that later runs would mistake for a complete export — the
                # failure mode seen when a crash killed a batch mid-write.
                selected_tags = sorted(selected_tags)
                smoothed_csv_filename = f'{db_name}_smoothed.csv'
                smoothed_csv_path = os.path.join(
                    csv_output_dir(output_dir, create=True), smoothed_csv_filename)
                smoothed_partial_path = smoothed_csv_path + '.partial'
                csv_path = smoothed_csv_path  # plots/animation/proximity read this
                stream_smoothed = export_smoothed_csv and not (
                    skip_existing and os.path.exists(smoothed_csv_path))
                if export_smoothed_csv and not stream_smoothed:
                    self.log_message(f"Skipped (exists): {smoothed_csv_filename}")
                elif stream_smoothed:
                    # Clear any stale partial from an earlier interrupted run.
                    for _p in (smoothed_partial_path, smoothed_csv_path):
                        try:
                            if os.path.exists(_p):
                                os.remove(_p)
                        except OSError:
                            pass

                total_points = 0
                tags_with_data = 0
                smoothed_header_written = False
                conn = connect_ro(self.db_path)

                # Read only the columns the pipeline uses. The unused TEXT
                # columns dominate memory, so this is the difference between
                # ~390 MB and ~37 MB of transient DataFrame per tag.
                proc_cols = processing_select_clause(conn, self.table_name)
                if proc_cols != "*":
                    # The caveat only applies when a raw CSV is actually being
                    # written; with Export Raw CSV off it read as a promise
                    # about a file that never appears.
                    caveat = (" the raw CSV still contains every database column."
                              if self.chk_export_raw_csv.isChecked()
                              else " other columns are not read.")
                    self.log_message(
                        f"Reading columns [{proc_cols}] for processing;{caveat}")

                for i, tag in enumerate(selected_tags):
                    if self.export_cancelled:
                        conn.close()
                        # A cancelled stream leaves a partial CSV — drop it.
                        try:
                            if os.path.exists(smoothed_partial_path):
                                os.remove(smoothed_partial_path)
                        except OSError:
                            pass
                        self.stop_export()
                        return

                    # 'HexID' spelled out: the checkbox list, the identity
                    # dialog and the plot legends all use hex, but the config
                    # JSON and this loop's own tag list are decimal, so a bare
                    # "Processing tag 10" read as shortid 10 when it was 16.
                    hex_id = hex(tag).upper().replace('0X', '')
                    who = ''
                    if self.tag_identities.get(tag):
                        info = self.tag_identities[tag]
                        who = (f" — {info.get('sex', '')}"
                               f"{info.get('identity', '')}")
                    self.log_message(
                        f"  Processing HexID {hex_id}{who} "
                        f"({i+1}/{len(selected_tags)})...")
                    QApplication.processEvents()

                    tag_data = pd.read_sql_query(
                        f"SELECT {proc_cols} FROM {self.table_name} WHERE shortid = ?",
                        conn, params=(tag,))

                    if len(tag_data) == 0:
                        continue

                    # Before anything else touches the track: while two rows
                    # can share a timestamp, every velocity computed from them
                    # is fiction.
                    tag_data, _cs = collapse_multi_solutions(
                        tag_data, log=self.log_message, label=f"HexID {hex_id}")

                    tag_data['Timestamp'] = pd.to_datetime(tag_data['timestamp'], unit='ms', origin='unix', utc=True)
                    tag_data['Timestamp'] = tag_data['Timestamp'].dt.tz_convert(tz)
                    tag_data['location_x'] *= 0.0254
                    tag_data['location_y'] *= 0.0254
                    tag_data = tag_data.sort_values(by='Timestamp')

                    # Per-tag time trimming (same helper the preview uses)
                    data_last = (tag_data['Timestamp'].max()
                                 if len(tag_data) else None)
                    tag_data, trimmed = self._trim_to_tag_window(tag_data, tag, tz)
                    if trimmed > 0:
                        self.log_message(
                            f"    Trimmed {trimmed} points outside the tag's active window")
                        # A window whose stop lands well before the tag's last
                        # fix is usually a STALE default rather than a
                        # decision: the identity dialog pre-fills the stop
                        # with whatever the data extent was when it was last
                        # opened, so a database that has grown since then gets
                        # its new days silently discarded.
                        _, w_stop = self._tag_window(tag, tz)
                        if (w_stop is not None and data_last is not None
                                and w_stop < data_last - pd.Timedelta(hours=12)):
                            lost = (data_last - w_stop).total_seconds() / 86400.0
                            self.log_message(
                                f"    \u26a0 Active window for this tag stops at "
                                f"{w_stop:%Y-%m-%d %H:%M} but it has data to "
                                f"{data_last:%Y-%m-%d %H:%M} — {lost:.1f} days are being "
                                f"discarded. If that is not deliberate, reopen "
                                f"Configure Identities and extend the Stop time.")

                    # Threshold + smooth: jump threshold (raw) → smooth →
                    # velocity threshold (on the smoothed track).
                    if len(tag_data) > 0:
                        tag_data = self._filter_and_smooth(tag_data, smoothing_method)

                    # Let the window repaint / process the Stop button between
                    # tags, right after the heaviest per-tag work.
                    QApplication.processEvents()

                    # Identity mapping
                    if tag in self.tag_identities:
                        tag_data['sex'] = self.tag_identities[tag].get('sex', 'M')
                        tag_data['identity'] = self.tag_identities[tag].get('identity', f'Tag{tag}')
                    else:
                        tag_data['sex'] = 'M'
                        tag_data['identity'] = f'Tag{tag}'

                    # Regions, before the frame is streamed and dropped:
                    # the name columns have to be in the CSV, and the
                    # occupancy summary is accumulated here rather than by
                    # re-reading the finished file. Both come from one
                    # point-in-polygon pass over the tag.
                    if region_sets and len(tag_data):
                        _idx = self.annotate_regions(tag_data, region_sets)
                        roi_bout_rows.extend(self.region_bout_rows(
                            tag_data, tag, region_sets, idx=_idx))

                    # Stream this tag straight to the CSV, then release it so
                    # peak memory stays at ~one tag rather than the whole trial.
                    n = len(tag_data)
                    if stream_smoothed and n:
                        tag_data.to_csv(smoothed_partial_path, index=False, mode='a',
                                        header=not smoothed_header_written)
                        smoothed_header_written = True
                    total_points += n
                    tags_with_data += 1
                    self.log_message(f"    {n} points after processing")
                    del tag_data

                conn.close()

                if total_points:
                    self.log_message(
                        f"Total processed: {total_points} points across {tags_with_data} tags")
                else:
                    self.log_message("Warning: No data after processing")
                QApplication.processEvents()

                # Every tag streamed successfully — publish the finished file by
                # renaming the .partial into place. Until this point a crash
                # leaves only a .partial, never a CSV that looks complete.
                if stream_smoothed and smoothed_header_written:
                    os.replace(smoothed_partial_path, smoothed_csv_path)
                    # A trial exported before the csvs/ subfolder existed has
                    # its smoothed CSV at the top level. The new copy lives in
                    # csvs/, so the old one is now stale but still looks
                    # current - and export_csv_path would fall back to it the
                    # moment the csvs/ copy went missing. Retire it, but only
                    # here: deleting it any earlier would leave the trial with
                    # no smoothed CSV at all if the stream then crashed.
                    _legacy = os.path.join(output_dir, smoothed_csv_filename)
                    if os.path.abspath(_legacy) != os.path.abspath(smoothed_csv_path):
                        try:
                            if os.path.exists(_legacy):
                                os.remove(_legacy)
                                self.log_message(
                                    f"Removed the pre-csvs/ copy of "
                                    f"{smoothed_csv_filename}")
                        except OSError as exc:
                            self.log_message(
                                f"  ⚠ could not remove the old "
                                f"{smoothed_csv_filename}: {exc}")
                    # Verify what actually landed on disk. Everything
                    # downstream - plots, proximity, the GBI, the ethogram -
                    # reads this file, so a corrupted cell here poisons the
                    # whole trial in confusing ways much later.
                    self.log_message("Verifying the smoothed CSV...")
                    QApplication.processEvents()
                    ok, problems = self.verify_smoothed_csv(smoothed_csv_path)
                    if not ok:
                        self.log_message(
                            "\u2717 The smoothed CSV did NOT verify - it contains "
                            "values that are not numbers:")
                        for p in problems:
                            self.log_message(f"    {p}")
                        self.log_message(
                            "  Nothing in FNT writes non-numeric coordinates, so "
                            "this is the file being corrupted between formatting "
                            "and disk. Aborting before the damage spreads into "
                            "the plots and network files; a re-run usually writes "
                            "it cleanly.")
                        self._last_export_failed = True
                        self._export_output_corrupt = True
                        self.stop_export()
                        return
                    self.log_message("\u2713 Smoothed CSV verified")

                # The smoothed CSV was streamed per-tag above; just advance the
                # progress step and report (keeps total_steps accounting intact
                # — increment whenever the smoothed CSV was requested, matching
                # the pre-streaming behaviour even when the write was skipped).
                if export_smoothed_csv:
                    current_step += 1
                    self.lbl_export_progress.setText(
                        f"Step {current_step}/{total_steps}: Smoothed CSV exported")
                    self.progress_bar.setValue(int(current_step / total_steps * 100))
                    if smoothed_header_written:
                        self.log_message(f"✓ Smoothed CSV exported: {smoothed_csv_filename}")
                    QApplication.processEvents()

                # Plots and animation render directly from the full-resolution
                # smoothed CSV — the exact same data product the user inspects.
                # This is deliberate: a downsampled render would look artificially
                # coarser/jerkier than the smoothed track and could mislead the
                # user into over-smoothing. No temporary render files are written.
                plot_csv_path = smoothed_csv_path if save_plots else None
                anim_csv_path = smoothed_csv_path if save_animation else None
            
            # Region occupancy. Written after the smoothed CSV so a failure
            # here cannot cost the expensive product, and before the config so
            # the folder describes what it actually contains.
            if export_zone_occupancy:
                if not region_sets:
                    self.log_message(
                        "ROI occupancy requested but this trial has no drawn "
                        "regions. Draw them in the preview, or import the "
                        "site XML's zones, then re-export. Nothing measured.")
                else:
                    named = ', '.join(f"{len(r)} region(s)"
                                      for _c, r in region_sets)
                    self.log_message(f"ROI occupancy: measuring {named}.")
                    if not roi_bout_rows and csv_path and os.path.exists(csv_path):
                        # The per-tag loop did not run (a batch retry reusing a
                        # verified CSV), so read the regions back off the file.
                        self.log_message(
                            "  Reading regions back from the existing smoothed "
                            "CSV (this pass did not re-process the tags).")
                        roi_bout_rows = self.region_bouts_from_csv(
                            csv_path, region_sets,
                            [t for t, cb in self.tag_checkboxes.items()
                             if cb.isChecked()])
                    # Both files come from the same visits, but each is ticked
                    # on its own: asking for the summary alone must not also
                    # drop a bout file the user did not request.
                    if export_roi_bouts:
                        self.write_roi_bout_csv(
                            roi_bout_rows, output_dir, db_name,
                            skip_existing=skip_existing)
                    if export_roi_daily:
                        self.write_roi_daily_summary_csv(
                            roi_bout_rows, output_dir, db_name,
                            skip_existing=skip_existing)
                QApplication.processEvents()

            # Write the site-map image (if any) so the analysis folder is a
            # self-contained data product, then save the config that references it.
            self._export_sitemap(output_dir, db_name)
            self.save_config(output_dir)

            # Save the message log. There is no separate runSummary.csv: every
            # setting it listed already lives in fnt_config.json, and the one
            # thing it uniquely carried - the filtering statistics - is now a
            # 'filter_stats' block in that same JSON.
            self.save_message_log(output_dir)
            
            # Detect proximity bouts if requested
            # The bouts file is the only social product now: the GBI and the
            # edge list were removed, so nothing else needs this pass.
            if detect_proximity or social_animation:
                if self.export_cancelled:
                    self.stop_export()
                    return

                self.log_message("=" * 50)
                self.log_message("Starting proximity detection...")
                self.lbl_export_progress.setText("Detecting proximity events and bouts...")
                QApplication.processEvents()

                try:
                    from fnt.uwb.proximity_detection import detect_proximity_bouts

                    # Warn if tag identities not configured
                    if not self.tag_identities:
                        self.log_message("Warning: Tag identities not configured. "
                                         "Proximity output will use raw shortid values. "
                                         "Configure identities in Tag Configuration for sex-ID labels.")

                    # Read the smoothed full-resolution data back from the CSV
                    # (streamed per-tag above). Proximity needs every tag together
                    # for pairwise distances, so it is loaded as one frame here —
                    # only when proximity/SNA is actually requested.
                    if csv_path and os.path.exists(csv_path):
                        # Only the four columns detect_proximity uses, and the
                        # time from the INTEGER column: parsing the ISO string
                        # cost ~34 s on a 14 M-row trial for an answer the
                        # integer gives in one, and the six unused columns
                        # doubled the read on top of that.
                        _head = pd.read_csv(csv_path, nrows=0)
                        _coord = (['smoothed_x', 'smoothed_y']
                                  if 'smoothed_x' in _head.columns
                                  else ['location_x', 'location_y'])
                        _want = ['shortid', 'timestamp', 'Timestamp'] + _coord
                        prox_data = pd.read_csv(
                            csv_path, low_memory=False,
                            usecols=[c for c in _want if c in _head.columns])
                        prox_data['Timestamp'] = smoothed_csv_timestamps(
                            prox_data, self.combo_timezone.currentText())
                    else:
                        self.log_message("Error: No smoothed data available for proximity detection")
                        prox_data = None

                    if prox_data is not None:
                        # events = per-second pairwise distances (needed for the
                        # chain-rule GBI); bouts = the threshold-based deliverable.
                        prox_events, proximity_bouts = detect_proximity_bouts(
                            prox_data,
                            threshold=proximity_threshold,
                            gap_s=5,
                            tag_identities=self.tag_identities,
                            log_callback=self.log_message
                        )

                        # Export proximity bouts (only when explicitly requested)
                        if detect_proximity:
                            bouts_path = os.path.join(
                                csv_output_dir(output_dir, create=True),
                                f'{db_name}_SocialOverlapBouts.csv')
                            if skip_existing and os.path.exists(bouts_path):
                                self.log_message(f"Skipped (exists): {os.path.basename(bouts_path)}")
                            else:
                                from fnt.uwb import roi_bouts as _RB
                                self.log_audit(
                                    os.path.basename(bouts_path),
                                    lambda: _RB.audit_social(
                                        proximity_bouts, proximity_threshold))
                                proximity_bouts.to_csv(bouts_path, index=False)
                                self.log_message(f"✓ Exported proximity bouts ({len(proximity_bouts)} bouts): "
                                                 f"{os.path.basename(bouts_path)}")
                            self.log_message("✓ Proximity detection complete!")


                except Exception as e:
                    self.log_message(f"Error during proximity/social-network export: {e}")
                    import traceback
                    traceback.print_exc()

                self.log_message("=" * 50)
            
            # Directed behaviour bouts (chase / displacement). Classified
            # from the smoothed CSV with the preview's thresholds; the only
            # export that records WHO acted on WHOM.
            if export_behavior and not self.export_cancelled:
                self.log_message("=" * 50)
                self.log_message("Extracting behaviour events...")
                self.lbl_export_progress.setText(
                    "Classifying chases and displacements...")
                QApplication.processEvents()
                try:
                    ev_path = os.path.join(
                        csv_output_dir(output_dir, create=True),
                        f'{db_name}_behavior_events.csv')
                    if skip_existing and os.path.exists(ev_path):
                        self.log_message(
                            f"Skipped (exists): {os.path.basename(ev_path)}")
                    elif csv_path and os.path.exists(csv_path):
                        # Read the selection here rather than relying on the
                        # outer binding, which only exists when the smoothed
                        # data was actually (re)generated this run.
                        chosen = [t for t, cb in self.tag_checkboxes.items()
                                  if cb.isChecked()]
                        ev = self._export_behavior_events(csv_path, chosen)
                        if len(ev):
                            # Displacement is parked, so its bouts must never
                            # reach this file; dyad_type must be one of the
                            # four documented values, with MIXED always 'MF'
                            # so a downstream filter on it cannot miss.
                            seen_b = sorted(set(ev['behavior']))
                            seen_d = sorted(set(ev['dyad_type']))
                            self.log_audit(os.path.basename(ev_path), [
                                (ev.notna().all().all().item(),
                                 f"{len(ev):,} bout(s), no missing values"),
                                (set(seen_b) <= {'chase'},
                                 f"behaviours present: {', '.join(seen_b)}"),
                                (set(seen_d) <= {'MM', 'MF', 'FF', 'NA'},
                                 f"dyad types present: {', '.join(seen_d)}"),
                                ((ev['actor'] != ev['target']).all().item(),
                                 "actor is never its own target"),
                                ((ev['duration_s'] > 0).all().item(),
                                 "every duration is positive"),
                                (((ev['bout_stop'] - ev['bout_start'])
                                  .dt.total_seconds() >= 0).all().item(),
                                 "bout_stop is never before bout_start"),
                            ])
                        ev.to_csv(ev_path, index=False)
                        by = ev['behavior'].value_counts().to_dict() if len(ev) else {}
                        self.log_message(
                            f"\u2713 Exported behaviour events ({len(ev)} bouts"
                            + (f": {by}" if by else "")
                            + f"): {os.path.basename(ev_path)}")
                        if not len(ev):
                            self.log_message(
                                "  No chases or displacements fired \u2014 check the "
                                "thresholds against the speed summary in the preview.")
                    else:
                        self.log_message(
                            "Error: No smoothed data available for behaviour events")
                except Exception as e:
                    self.log_message(f"Error during behaviour-event export: {e}")
                    import traceback
                    traceback.print_exc()
                self.log_message("=" * 50)

            # Export plots if requested (BEFORE animation, which takes longer)
            if save_plots:
                if self.export_cancelled:
                    self.stop_export()
                    return
                
                current_step += 1
                self.lbl_export_progress.setText(f"Step {current_step}/{total_steps}: Generating plots...")
                self.progress_bar.setValue(int(current_step / total_steps * 100))
                QApplication.processEvents()
                
                selected_tags = [tag for tag, cb in self.tag_checkboxes.items() if cb.isChecked()]

                # Read every checkbox rather than naming them one by one:
                # the hand-written dict this replaces silently dropped any
                # plot type added later, so a newly ticked plot produced no
                # file and no error. Matches how _predict_export_conflicts
                # enumerates them, so prediction and output cannot diverge.
                plot_types = {k: cb.isChecked()
                              for k, cb in self.plot_type_checkboxes.items()}

                # Get rolling window value
                rolling_window = self.spin_rolling_window.value()

                self.btn_export.setEnabled(False)
                self.log_message("Starting plot generation in background...")

                self.worker = PlotSaverWorker(
                    self.db_path,
                    self.table_name,
                    selected_tags,
                    False,  # downsample: vestigial, feature removed
                    self.get_smoothing_method(),
                    plot_types,
                    skip_existing,
                    rolling_window,
                    self.combo_timezone.currentText(),
                    self.tag_identities,
                    bool(self.tag_identities),  # Use identities if any are configured
                    # Always pass the image; whether it is drawn is decided by
                    # self.plot_layers, which for an interactive export mirrors
                    # the preview toggle and for a batch job replays the layers
                    # the job was queued with.
                    self.background_image,
                    self.bg_width_meters,  # Pass scaled width
                    self.bg_height_meters,  # Pass scaled height
                    plot_csv_path,  # plots render source (full-resolution smoothed CSV)
                    self.chk_save_svg.isChecked(),  # Save SVG copies
                    output_dir,  # Pass output directory
                    plots_dir,  # Pass plots subfolder
                    self.combo_window_units.currentText(),  # seconds vs samples
                    plot_layers=self.plot_layers,
                    xml_zones=self.xml_zones,
                    rois=self.rois,
                    occupancy_sources=[c for c, _ in self.region_sets()],
                    anchor_positions=self.anchor_positions,
                    xml_map_image=self.xml_map_image,
                    xml_map_extent=self.xml_map_extent,
                    bg_offset_x=self.bg_offset_x,
                    bg_offset_y=self.bg_offset_y,
                    behavior_params=self._behavior_params(),
                    axis_limits=self.xy_range(),
                )
                self.worker.progress.connect(self.update_status)
                self.worker.finished.connect(lambda success, msg: self.export_finished(success, msg, save_animation, output_dir, total_steps, current_step, anim_csv_path, animations_dir))
                self.worker.start()
            
            # Animation will be started from export_finished() after plots complete
            elif save_animation:
                # If no plots, start animation directly
                if self.export_cancelled:
                    self.stop_export()
                    return
                
                current_step += 1
                self.lbl_export_progress.setText(f"Step {current_step}/{total_steps}: Generating animation...")
                self.progress_bar.setValue(int((current_step - 1) / total_steps * 100))
                QApplication.processEvents()
                
                self.generate_animation(output_dir, total_steps, current_step, anim_csv_path, animations_dir)

            # If no plots or animation, show success message now
            any_csv = export_raw_csv or export_smoothed_csv
            if (any_csv or detect_proximity) and not save_plots and not save_animation:
                self.log_message("✓ Export completed successfully")
                msg = f"Export completed to:\n{output_dir}"
                # No async plot worker / animation follows on this path, so the
                # export is fully done here — clear the flag the batch waits on.
                self.exporting = False
                self.btn_export.setEnabled(True)
                self.btn_stop_export.setVisible(False)
                self._notify_done('info', "Success", msg)

        except Exception as e:
            self._last_export_failed = True
            if is_corruption_error(e):
                # Name the real cause: the file is damaged, not the export.
                self.report_corrupt_database(e, quiet=getattr(self, '_batch_active', False))
                self._notify_done('error', "Database File Is Damaged",
                                  "Export stopped: the database file is damaged. "
                                  "See the message log for recovery steps.")
            else:
                self._notify_done('error', "Error", f"Export failed: {str(e)}")
            self.log_message(f"✗ Export failed: {str(e)}")
            # If we failed before the async plot worker started, the export is
            # over now — clear the flag the batch waits on. (If a worker IS
            # running, export_finished will clear it instead.)
            if not (getattr(self, 'worker', None) and self.worker.isRunning()):
                self.exporting = False
                self.btn_export.setEnabled(True)
                self.btn_stop_export.setVisible(False)
        finally:
            # Always clear the busy cursor, on success, cancel, or error. The
            # async plot worker (if started) runs in its own thread and does not
            # freeze the UI, so restoring here is correct.
            QApplication.restoreOverrideCursor()

    def update_status(self, message):
        """Update status label and messages window"""
        self.log_message(message)

    def _notify_done(self, kind, title, text):
        """Show an export completion/error dialog, unless a batch is running.

        During a batch the per-trial popups are suppressed (they would each
        block the unattended run on a modal box); the batch shows one summary at
        the end instead. ``kind`` is 'info' or 'error'.
        """
        if getattr(self, '_batch_active', False):
            return
        if kind == 'info':
            QMessageBox.information(self, title, text)
        else:
            QMessageBox.critical(self, title, text)
    
    def export_finished(self, success, message, start_animation=False, output_dir=None, total_steps=1, current_step=1, csv_path=None, animations_dir=None):
        """Handle export completion"""
        if not self.export_cancelled:
            if success:
                if 'FAILED' in str(message):
                    # Partial: the run continues, but the detail must not be
                    # buried under a tick.
                    self.log_message(
                        f"⚠ Plot export finished with failures: {message}")
                    self.log_message(
                        "  The other products are unaffected; the trial is not "
                        "marked failed for this.")
                else:
                    self.log_message("✓ Plot export completed successfully")

                # Start animation if requested (plots are now complete)
                if start_animation and output_dir:
                    self.log_message("Starting animation generation...")
                    self.lbl_export_progress.setText(f"Step {current_step + 1}/{total_steps}: Generating animation...")
                    self.progress_bar.setValue(int(current_step / total_steps * 100))
                    QApplication.processEvents()
                    self.generate_animation(output_dir, total_steps, current_step + 1, csv_path, animations_dir)
                    return  # Don't reset UI yet, animation will do that
                else:
                    self.progress_bar.setValue(100)
                    self.lbl_export_progress.setText("Export complete!")
                    self._notify_done('info', "Success", "Export completed successfully!")
            else:
                self._last_export_failed = True
                self.log_message(f"✗ Plot export failed: {message}")
                self._notify_done('error', "Error", message)

        # Plots are done and no animation follows (that path returned above and
        # cleans up itself), so drop the temp render CSVs now.
        self._cleanup_plot_working_files()

        # Reset UI state
        self.exporting = False
        self.btn_export.setEnabled(True)
        self.btn_stop_export.setVisible(False)

        # Hide progress after a delay
        QTimer.singleShot(3000, lambda: self.progress_widget.setVisible(False))
    
    def closeEvent(self, event):
        """Handle window close event — stop exports and release global hooks."""
        if self.exporting:
            reply = QMessageBox.question(self, 'Export in Progress',
                                        'An export is in progress. Do you want to cancel it and close?',
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:
                self.stop_export()
            else:
                event.ignore()
                return

        self._teardown_preview()
        event.accept()

    def _teardown_preview(self):
        """Release everything the preview owns beyond this window's lifetime.

        The launcher keeps a reference to this window after it is closed, so an
        app-level event filter and running timers/threads would otherwise live
        on: arrow keys anywhere in FNT would still be swallowed to scrub a
        hidden timeline, and a QThread freed while running aborts the process.
        Safe to call more than once.
        """
        self._preview_active = False

        try:
            app = QApplication.instance()
            if app is not None:
                app.removeEventFilter(self)
        except Exception:
            pass

        for name in ("preview_timer", "preview_scrub_timer",
                     "preview_tag_timer", "preview_trail_timer"):
            t = getattr(self, name, None)
            if t is not None:
                try:
                    t.stop()
                except Exception:
                    pass

        # Wait on in-flight readers rather than dropping their last reference.
        for loader in list(getattr(self, "preview_inflight", {}).values()):
            try:
                if loader.isRunning():
                    loader.wait(5000)
            except Exception:
                pass
        try:
            self.preview_inflight.clear()
        except Exception:
            pass

        for name in ("preview_index_builder",):
            th = getattr(self, name, None)
            if th is not None:
                try:
                    if th.isRunning():
                        th.wait(10000)
                except Exception:
                    pass

        for w in list(getattr(self, "_db_workers", [])):
            try:
                if w.isRunning():
                    w.wait(5000)
            except Exception:
                pass


# Kept alive for the whole process so faulthandler's file target isn't GC'd.
_FAULT_LOG = None


def _install_faulthandler():
    """Dump a native-crash traceback to a local log file (and the console).

    A hard crash inside a C extension (numpy/MKL/matplotlib) terminates the
    process with no Python traceback — e.g. the 0xc000001d illegal-instruction
    crash seen on Python 3.14 during plot export. faulthandler installs a fatal-
    signal handler that writes every thread's Python stack on the way down, so a
    recurrence names the exact plot/line (including the export worker thread).

    The trace goes to ~/.fnt/faulthandler_crash.log on the LOCAL disk (never the
    network share, so it survives even if that share is what hiccuped) and the
    path is printed at startup.
    """
    global _FAULT_LOG
    try:
        log_dir = os.path.join(os.path.expanduser("~"), ".fnt")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "faulthandler_crash.log")
        _FAULT_LOG = open(log_path, "a", buffering=1, encoding="utf-8")
        from datetime import datetime
        _FAULT_LOG.write(f"\n===== FNT session {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
        _FAULT_LOG.flush()
        faulthandler.enable(file=_FAULT_LOG, all_threads=True)
        print(f"[FNT] Native-crash log enabled: {log_path}")
    except Exception as e:
        # Never block startup on logging setup — fall back to stderr-only.
        try:
            faulthandler.enable(all_threads=True)
        except Exception:
            pass
        print(f"[FNT] faulthandler on stderr only ({e})")


def main():
    _install_faulthandler()
    app = QApplication(sys.argv)
    window = UWBQuickVisualizationWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
