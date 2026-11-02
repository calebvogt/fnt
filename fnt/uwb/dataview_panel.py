"""The Data View panel drawn beside the tracking animation.

ROI OCCUPANCY - one row per animal, one column per region - accumulating in
lockstep with the video.

Every number comes from a CumulativeSeries read at the frame's own timestamp
(see dataview.py), never from counting frames, so the last frame agrees with
the exported CSV exactly.

SOCIAL OVERLAP IS NOT SHOWN, by choice. It could be: proximity is computable
in time windows without holding the trial in memory. But the panel exists to
be checked against the exported summaries, and computing overlap here rather
than reading the exported bouts would give the trial two implementations of
who was together and for how long - which is precisely the disagreement this
panel is meant to detect. Social overlap lives in _SocialOverlapBouts.csv,
the GBI and the edge list. Two renderings of it survive behind ``dyad_mode``
(``"partner"``, ``"triangle"``) for when the exported bouts are at hand.

The per-frame cost is why the values are laid out as ONE text artist per
COLUMN rather than one per cell: redrawing a few multi-line artists on every
frame of a long render is affordable where redrawing a hundred-odd is not.
Rows still line up because every column shares the monospace font and the
line spacing.
"""

import numpy as np

from fnt.uwb import dataview as DV

# Enough for " 0.0s" through "59.9m" and "19.0d" - see human_duration.
CELL_CHARS = 5
LINESPACING = 1.45
# Room for a row label like "M9657" plus a little air.
LABEL_CHARS = 7


def _fit_fontsize(n_rows, height_in, *, lo=4.0, hi=9.0):
    """Largest font (pt) that fits ``n_rows`` lines into ``height_in`` inches.

    Clamped at the top so a small trial does not render absurdly large text,
    and at the bottom because past a point the panel stops being readable and
    the honest answer is that it does not fit.
    """
    if n_rows <= 0:
        return hi
    return float(np.clip((height_in * 72.0) / (n_rows * LINESPACING), lo, hi))


def panel_width_in(n_animals, n_rois, fontsize, dyad_mode="none"):
    """Inches the panel needs for its widest grid at ``fontsize``.

    Monospace advance is ~0.6 em, so a character is 0.6 * fontsize points.
    With no dyad section the ROI matrix sets the width; "partner" adds two
    narrow columns, and the full triangle needs one per animal, which roughly
    doubles the panel.
    """
    ch_in = 0.6 * fontsize / 72.0
    cols = int(n_rois)
    if dyad_mode == "triangle":
        cols = max(cols, max(int(n_animals) - 1, 0))
    elif dyad_mode == "partner":
        cols = max(cols, 3)                 # partner name + accumulated time
    return (LABEL_CHARS + cols * (CELL_CHARS + 1)) * ch_in + 0.35


class DataViewPanel:
    """Owns the panel's axes, its static furniture and its per-frame numbers.

    ``occupancy`` maps (animal, region_index) -> CumulativeSeries and
    ``overlaps`` maps the canonical pair key -> CumulativeSeries. Both may be
    sparse: a missing entry simply reads zero, which is what an animal that
    never entered a region should show.
    """

    def __init__(self, fig, rect, animals, roi_names, occupancy, overlaps, *,
                 text_color="#e6e6e6", grid_color="#4a4a4a",
                 bg_color="#1b1b1b", accent="#ffd166", fontsize=None,
                 dyad_mode="none"):
        self.animals = list(animals)
        self.roi_names = list(roi_names or [])
        self.occupancy = occupancy or {}
        self.overlaps = overlaps or {}
        self.accent = accent
        self.dyad_mode = dyad_mode
        self._last = {}

        n_a, n_r = len(self.animals), len(self.roi_names)
        self.cells = DV.triangle_cells(self.animals)

        # Rows: ROI header + ROI body, then a dyad section only if one is
        # asked for. "none" is the default: the panel is an ROI counter. Social
        # is NOT recomputed here - the exported bouts are the one authority on
        # who was together and for how long, and a second implementation for
        # the sake of a preview is how the two come to disagree.
        if dyad_mode == "triangle":
            dyad_rows = max(n_a - 1, 0)
        elif dyad_mode == "partner":
            dyad_rows = n_a
        else:
            dyad_rows = 0
        self.n_rows = (2 + n_a) + (2 + (2 + dyad_rows) if dyad_rows else 0)
        height_in = rect[3] * fig.get_figheight()
        self.fontsize = fontsize or _fit_fontsize(self.n_rows, height_in)

        self.ax = fig.add_axes(rect)
        self.ax.set_facecolor(bg_color)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for s in self.ax.spines.values():
            s.set_color(grid_color)

        # One row's height as a fraction of the axes, derived from the font so
        # a multi-line column and the row labels beside it share a grid.
        self.row_h = (self.fontsize * LINESPACING / 72.0) / max(height_in, 1e-6)
        ch = 0.6 * self.fontsize / 72.0 / max(rect[2] * fig.get_figwidth(), 1e-6)
        self.col_w = (CELL_CHARS + 1) * ch
        self.x0 = LABEL_CHARS * ch

        common = dict(family="monospace", fontsize=self.fontsize,
                      color=text_color, linespacing=LINESPACING,
                      transform=self.ax.transAxes)
        self._common = common
        self.value_texts = {}

        y = 1.0 - 0.6 * self.row_h
        y = self._build_section(
            y, "ROI OCCUPANCY", self.roi_names, self.animals,
            [(r, [a for a in self.animals]) for r in range(n_r)],
            kind="roi", grid_color=grid_color, common=common)
        if dyad_mode == "none":
            return
        y -= 1.2 * self.row_h
        if dyad_mode == "triangle":
            # Every pair, as a lower triangle. Kept for when the full dyad
            # matrix is wanted; it costs one column per animal, roughly
            # doubling the panel's width.
            # The triangle's rows are animals[1:]; column j covers rows j+1..n-1.
            self._build_section(
                y, "DYAD OVERLAP", [self._short(a) for a in self.animals[:-1]],
                self.animals[1:],
                [(j, self.animals[j + 1:]) for j in range(max(n_a - 1, 0))],
                kind="dyad", grid_color=grid_color, common=common, stagger=True)
        else:
            # One row per animal: who it has spent the most time beside, and
            # how much. 120 cells of mostly zeros said far less than 16 rows
            # of who-is-with-whom, and it reads at a glance on a moving video.
            self._build_section(
                y, "TOP SOCIAL PARTNER", ["partner", "time"], self.animals,
                [("partner", self.animals), ("time", self.animals)],
                kind="partner", grid_color=grid_color, common=common)

    @staticmethod
    def _short(name):
        return str(name)

    def _build_section(self, y_top, title, col_labels, row_labels, columns, *,
                       kind, grid_color, common, stagger=False):
        """Lay out one grid; returns the y just below it."""
        ax = self.ax
        ax.text(0.005, y_top, title, family="monospace",
                fontsize=self.fontsize, color=self.accent, va="top",
                fontweight="bold", transform=ax.transAxes)
        y_head = y_top - 1.1 * self.row_h

        # Column headers. Rotated for the dyad grid, where a column is an
        # animal ID and there are as many columns as animals.
        for c, lab in enumerate(col_labels):
            x = self.x0 + (c + 1) * self.col_w - 0.3 * self.col_w
            ax.text(x, y_head, str(lab)[:CELL_CHARS + 2],
                    family="monospace", fontsize=self.fontsize * 0.85,
                    color=grid_color if kind == "grid" else "#9fb3c8",
                    va="bottom", ha="right" if not stagger else "left",
                    rotation=0 if not stagger else 60,
                    transform=ax.transAxes)

        y_body = y_head - 0.6 * self.row_h
        # Row labels as ONE multi-line artist, so they share the column texts'
        # line spacing exactly and cannot drift out of alignment.
        ax.text(0.005, y_body, "\n".join(str(r)[:LABEL_CHARS] for r in row_labels),
                va="top", ha="left", **common)

        for c, (key, members) in enumerate(columns):
            x = self.x0 + (c + 1) * self.col_w
            # A dyad column starts partway down: column j has no cell until
            # row j+1, which is what makes the grid a triangle.
            offset = c if stagger else 0
            t = ax.text(x, y_body - offset * self.row_h, "",
                        va="top", ha="right", animated=True, **common)
            self.value_texts[(kind, key)] = (t, list(members))
        return y_body - len(row_labels) * self.row_h

    # ---------------------------------------------------------------- values

    def _occ(self, animal, r, when):
        s = self.occupancy.get((animal, r))
        return float(s.at(when)) if s is not None else 0.0

    def _ovl(self, a, b, when):
        s = self.overlaps.get(DV.pair_key(a, b))
        return float(s.at(when)) if s is not None else 0.0

    def _top_partner(self, animal, when):
        """(partner, seconds) this animal has spent longest beside, so far.

        Ties and an all-zero row both read as no partner yet, rather than
        naming whichever animal happens to sort first.
        """
        best, best_t = None, 0.0
        for other in self.animals:
            if other == animal:
                continue
            v = self._ovl(animal, other, when)
            if v > best_t:
                best, best_t = other, v
        return best, best_t

    def update(self, when_s, fmt):
        """Set every column's text for trial-time ``when_s`` (epoch seconds).

        ``fmt`` formats one duration; it is injected rather than imported so
        the panel and the totals card cannot drift apart.
        """
        for (kind, key), (artist, members) in self.value_texts.items():
            if kind == "roi":
                txt = "\n".join(fmt(self._occ(a, key, when_s)) for a in members)
            elif kind == "partner":
                pairs = [self._top_partner(a, when_s) for a in members]
                if key == "partner":
                    txt = "\n".join(
                        (str(p)[:CELL_CHARS] if p else "-").rjust(CELL_CHARS)
                        for p, _t in pairs)
                else:
                    txt = "\n".join(fmt(t) for _p, t in pairs)
            else:
                anchor = self.animals[key]
                vals = [self._ovl(anchor, b, when_s) for b in members]
                txt = "\n".join(fmt(v) for v in vals)
            # Only touch the artist when the rendered text actually changed:
            # most cells hold still for long stretches, and set_text
            # invalidates matplotlib's cached layout for that artist.
            if self._last.get((kind, key)) != txt:
                artist.set_text(txt)
                self._last[(kind, key)] = txt

    def artists(self):
        return [a for a, _ in self.value_texts.values()]

    def draw(self):
        for a, _ in self.value_texts.values():
            self.ax.draw_artist(a)

    # ----------------------------------------------------------- totals card

    def render_totals_card(self, figsize, dpi, *, title="FINAL TOTALS",
                           bg_color="#1b1b1b", text_color="#e6e6e6",
                           accent="#ffd166"):
        """An end card of every total in SECONDS, as an RGBA frame buffer.

        Drawn on its own figure at the video's size rather than over the
        animation: the arena is not needed once the trial has ended, so the
        whole canvas is free for numbers, and keeping it off the blitted
        figure means the per-frame path is untouched.

        Seconds, because this is the frame the CSV gets compared against and a
        rounded '1.42d' cannot be. Laid out in as many columns as it takes.
        """
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        occ, ovl = self.totals_rows()
        lines = [f"{a:<8s} {r:<10s} {v:12.3f}" for a, r, v in occ]
        lines += [f"{a:<8s} ~{b:<9s} {v:12.3f}" for a, b, v in ovl]

        fig = Figure(figsize=figsize, dpi=dpi)
        FigureCanvasAgg(fig)
        fig.patch.set_facecolor(bg_color)
        fig.text(0.02, 0.965, f"{title} — seconds", color=accent,
                 fontsize=15, fontweight="bold", family="monospace",
                 va="top")
        fig.text(0.02, 0.93,
                 "Compare directly with the exported summary CSV.",
                 color="#9fb3c8", fontsize=10, family="monospace", va="top")

        # Enough columns that the rows fit the height, then size the font to
        # whatever that leaves - a long trial simply gets more columns rather
        # than unreadable text.
        usable_h = 0.88 * figsize[1]
        n_cols = 1
        while n_cols < 8:
            per = int(np.ceil(len(lines) / n_cols))
            fs = _fit_fontsize(per, usable_h, lo=4.0, hi=11.0)
            if fs > 6.0 or n_cols == 7:
                break
            n_cols += 1
        per = int(np.ceil(len(lines) / max(n_cols, 1))) or 1
        fs = _fit_fontsize(per, usable_h, lo=4.0, hi=11.0)
        for c in range(n_cols):
            chunk = lines[c * per:(c + 1) * per]
            if not chunk:
                break
            fig.text(0.02 + c * (0.96 / n_cols), 0.90, "\n".join(chunk),
                     color=text_color, fontsize=fs, family="monospace",
                     va="top", linespacing=LINESPACING)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba()).copy()
        fig.clear()
        return buf

    def totals_rows(self):
        """Final totals in SECONDS, for the end card and for checking.

        Seconds, not the adaptive display units: the card exists so the figure
        on screen can be compared with the CSV exactly, and a rounded '1.42d'
        cannot be.
        """
        occ = [(str(a), self.roi_names[r], float(s.total))
               for (a, r), s in sorted(self.occupancy.items(),
                                       key=lambda kv: (str(kv[0][0]), kv[0][1]))
               if 0 <= r < len(self.roi_names)]
        ovl = [(str(k[0]), str(k[1]), float(s.total))
               for k, s in sorted(self.overlaps.items(),
                                  key=lambda kv: -kv[1].total)]
        return occ, ovl
