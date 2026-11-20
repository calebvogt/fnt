---
hide:
  - navigation
---

# Mask Tracker

Label animals in video frames with SAM, train a YOLO instance-segmentation model
on them, review its predictions, and run it over whole videos. One window does
the labelling, training and review; a command-line tool does the long runs.

Launch from the FNT main GUI: **Video** tab ▸ **FNT Tracking Tools** ▸ **Mask
Tracker**, or from a terminal in the `fnt` env:

```bash
python -m fnt.videoTracking.mask_tracker_gui --project <project folder>
```

The loop:

```
pick frames → label (SAM) → train → predict the unlabelled frames → accept / reject / fix
      ↑                                                                  │
      └──────────── frames where the model is weakest ◄──────────────────┘
```

---

## The project folder

A project lives next to the video it is for:

```
<project>/
    project_config.json        videos, classes, the ROI
    LABELING_GUIDE.md          the project's labelling rules (optional)
    training_frames/*.png      the frames you label
    annotations/annotations.json   labels, predictions, empty/skipped flags (COCO)
    models/<date>_<time>_<arch>_n=<frames>/   one folder per training run
    focus_*.txt                frame lists for --focus sessions
```

Labels save on every action; there is no Save step to forget. Two windows open on
the same project do not see each other's changes: close one before working in
the other, and reopen a window after a training run has written predictions.

## Labelling

The **Training Frames Queue** lists the frames to label. Frames come from the
source videos (arrow keys scrub the video; **E** extracts the frame on screen to
the queue) or from a script (`mask_tracker_cli extract-frames`, below).
`--focus <list.txt>` opens the window on just those frames.

| Key | Does |
|-----|------|
| **M** | SAM: click each animal; a click on or near a pending one adds to it, Shift+click starts a new one, right-click excludes a spot. **Enter** saves each as its own mask. |
| **P** / **X** | Paint / Eraser on a pending or saved mask (wheel = brush size) |
| **U**, Ctrl+Z / Ctrl+Shift+Z | Undo a click or stroke; undo / redo |
| **E** | Mark the frame empty: no animals (it trains as background) |
| **S** | Skip: leave the frame for now and go to the next one to label |
| **N** / **B** | Next / back in the queue |
| **Space** / Shift+Space | Next / previous queue frame along this video's timeline |
| ← → (Shift ±10, Ctrl ±100) | Scrub the source video; landing on a queue frame opens it with its masks |
| **V** | Show / hide the masks (and the ROI shading) |
| **1**–**9**, **0** | Class: save the pending mask as that class, or switch the active class |
| **+** / **−**, scroll | Zoom in / out; or the **−** **+** **Fit** buttons beside the zoom % (bottom right). **Fit** shows the whole frame again (100%) |

The timeline under the frame marks the queue's frames from this video: green
labelled, yellow predicted, white still to do.

**Small animals.** Zoom in (scroll or **+**) before you click: SAM sees what you see.
Label every animal in a frame, or the model learns the ones you left as
background.

### Labelling guide

**Guide** (next to **ROI**) shows the project's rules on hover and opens them to
read or edit. They are kept in `LABELING_GUIDE.md` in the project folder, travel
with portable sessions, and MAD has the same button. Write a rule down the first
time you make a judgement call (a mouse half in an entrance, a mouse seen through
a lid) so every session, and every labeller, makes it the same way.

## Region of interest

A fixed camera sees more than the animals' space: the room around the arena, the
lids of boxes the animals hide under. **ROI** draws, once per project, where the
model may look:

- **Include area**: click the corners of the area the animals can be in. The
  model sees only the box around it, so each animal has more pixels at the same
  input size; everything outside the outline is filled flat grey.
- **Exclude area**: holes in that area the model must ignore (a lid it sees mice
  through).

Close an outline by clicking its first corner again (or double-click,
right-click, Enter); drag a finished outline's corners to adjust it.
**Show what the model sees** previews the prepared frame. The preview then shades
what the model won't see (dark outside, red hatching in holes).

The ROI is saved in `project_config.json` and copied into each model trained
after it is drawn. Training cuts frames and labels to it, and every way of
running that model (post-training prediction, Predict, `infer-videos`, the
behaviour tools, MAD's Fusion QC) prepares frames the same way, maps the masks
back to full-frame coordinates and drops detections centred outside the area.
Models trained before the ROI keep seeing whole frames.

**Margin** grows the area the model sees beyond the drawn outline by that many
pixels (and shrinks excluded holes by as much); the editor shows it as a dashed
line. An animal seen at an angle stands out past the floor it is on, as near the
walls of an overhead camera's view, and an outline drawn on the floor alone cuts
it off.

**One outline per trial.** A camera bumped between trials frames the arena
differently. With a frame from a trial on screen, **ROI** offers **Only for
<trial folder>**: the outline is then kept for that trial's videos alone (the
folder the video is in, or the one above a `cam5`/`Camera5`/`ch1_...` camera
folder). Training frames, videos and predictions each use their own trial's
outline, and the project's outline for trials without one; the preview shows the
one of the frame on screen. Outlines placed by alignment rather than by hand are
marked *proposed* until saved from the editor.

!!! note
    This is not FNT's separate **ROI Tool** (zones for occupancy analysis of
    finished tracks).

## Training

**Train Model** (bottom of the Mask tab) trains on the approved frames: those
with labelled masks or empty. Pending predictions and skipped frames are
left out. The run opens a training window with the loss, validation mAP and
live predictions on a few validation frames.

The same run from a terminal (useful for overnight runs):

```bash
python -m fnt.videoTracking.mask_tracker_gui --project <project> --focus <list.txt> \
    --train --training-window-only --arch yolo11n-seg --imgsz 1600 --batch 2 \
    --epochs 200 --patience 60 --augment --aug-rotation 180 --mosaic 1.0 \
    --copy-paste 0.5 --post-infer --post-infer-conf 0.25
```

With `--training-window-only` only the training window shows, and it closes
itself when the run is over (after the post-training inference, if any). The
model, `training_log.csv` and `training_summary.json` are in the run folder. If
the run fails, the window stays up with the error.

- `--imgsz`: input size. 640 is fine for large animals; small mice in a wide
  view want 1280-1600 (and an ROI). On a 3 GB card, 1600 needs `--batch 2`.
- `--augment --aug-rotation 180`: flips, rotation, scale and brightness; 180 for
  top-down cameras, 15 for side views.
- `--copy-paste`: animals pasted between training frames by their masks; strong
  with few labelled frames.
- `--post-infer`: when training ends, the model predicts the queue's frames that
  have no approved mask, for review.
- `--train-only <list>`: train on a subset of the frames.
- `--optimizer` and `--lr` combine: picking an optimizer resets the learning
  rate to its default (SGD 0.01, AdamW 0.002) only when `--lr` is not given.

The run's model, plots and summary go in the project's `models/` folder. The
YOLO copy of the frames that training reads is built in a temporary folder on
the local disk and removed when the run ends. On Windows, a mapped drive is
opened by its network path (`Z:` becomes `\\server\share`), and without
long-path support a label file 260 or more characters deep cannot be opened,
so YOLO would train that frame as empty; the short local copy avoids that, and
a run whose paths are still too long stops with an error instead.

## Reviewing predictions

Predictions appear yellow. **A** accepts the bold one (click another to pick it),
**R** rejects it, Shift+A accepts all on the frame; a SAM click on a prediction
replaces it with SAM's mask, and Paint / Eraser fix it before you accept.
**Next to Review** jumps to the next frame with predictions. A rejected spot is
not offered again on that frame. A frame whose every prediction you rejected,
with nothing labelled, counts as empty (it trains as background); **E** unmarks
it. The queue's Annotations column counts rejections as ✗.

`--predict <model folder>` (or **File ▸ Predict Unreviewed Frames with a Model…**) runs any
trained model over the queue's unreviewed frames.

The same without the window, e.g. on the CPU while another job has the GPU:

```bash
python -m fnt.videoTracking.mask_tracker_cli predict-frames --project <project> \
    --model <model folder> --focus <list.txt> --device cpu
```

The next frames worth labelling are the ones the model gets wrong: low-scored
masks, animals it missed, counts that jump. Pick them from videos and trials the
model has not seen.

## Whole videos

```bash
python -m fnt.videoTracking.mask_tracker_cli infer-videos --model <model folder> \
    --videos <folders, files or a .txt list> --step 250
```

Segments every `--step`-th frame and writes `<video>_masks.h5` beside each
video (masks as RLE, scores, boxes, centroids). It resumes where it stopped and
skips videos already done with the same model and settings; `--order coarse`
(the default) spreads the work over every folder and camera first, so a stopped
run is still an even sample.

`--gate` (MP4/MOV, fixed cameras) chooses the frames by activity instead of
`--step`. A still scene compresses to tiny frames and a moving animal multiplies
their size, so the frame sizes in each file's index show where something moves
without decoding the video. The gate then segments every frame of the moving
seconds (padded by `--gate-pad`, 1 s), `--gate-low-fps` frames a second (2) where
it is unsure, and only the keyframes elsewhere, where a resting animal can still
be. On the 2021_8x8 resource-zone cameras the moving seconds are about a fifth
of the video. `--gate-hi` and `--gate-lo` (2 and 1.3) are the size ratios over
the camera's still level that mark moving and unsure seconds.

`behavior-window` runs the model on every frame of a stretch of video and links
the masks into tracks, with candidate behaviour bouts for review.

### Tracking every frame

The **Inference** tab tracks every frame of the queued videos and writes
`<video>_MaskTracker/` beside each (`trajectories.csv`, the annotated video, the
masks for proofreading). One video at a time leaves the GPU mostly idle: the work
around the model (reading, tracking, drawing, writing) runs on the CPU one step
after another. **Several videos at once** runs one worker process per video, and
the preview pane shows a small tile per running video with its masks, progress
and speed. Each video's outputs are the same as when it runs alone. **At once:
Auto** takes as many as the machine allows (the smallest of physical CPU cores
/ 3, free GPU memory / 1.5 GB, available RAM / 3 GB, and 8); a number sets it.
On the lab PC (RTX 4070 Ti SUPER, i9-14900K), 1080p open-field videos with
yolo11l-seg masks ran at 23 frames/s one at a time and 88 frames/s seven at once,
with the GPU then fully busy. **Stop** ends every worker after its current frame
and removes the unfinished videos' output folders.

From the command line, the same run in the same window:

```bash
python -m fnt.videoTracking.mask_tracker_cli track --videos <folders, files or a .txt list> \
    --model <model folder> --parallel auto --max-objects 1 --conf 0.5
```

Folders are searched recursively. The project is the one the model folder is in
(`--project` for a model kept elsewhere). Every Inference setting has a flag
(`--masks`/`--no-masks`, `--tracked-video`/`--no-tracked-video`,
`--save-masks`, `--preview`, `--infer-device`, `--resolution`, `--max-disappear`,
`--frame-step`, `--linker` and its settings, `--iou`, `--matching`, `--classifier`
and its four settings); one not given keeps the tab's value.

**Analyse every Nth frame** (`--frame-step N`, default 1) detects and tracks
only every Nth frame: 3 on 30 fps video gives 10 positions per second at about a
third of the model time (every frame is still decoded). Frame numbers and
`time_s` in `trajectories.csv` stay the video's, speeds use the real time between
analysed frames, and the tracked video shows the analysed frames at fps / N, so it
keeps the source's length. **Max disappear** then counts analysed frames.

How detections join tracks (**Linker**, `--linker`):

- **ByteTrack** (default since 2026-11-20). Each analysed frame, a detection's
  cost to a track is its centroid's distance from where the track is predicted
  to be (a constant-velocity Kalman filter, `--motion kalman`) divided by the
  gate, plus `--iou-weight` (1) x (1 - the overlap of its mask with the track's
  last mask moved to that prediction; boxes when there are no masks). Pairs
  beyond the gate never match. Confident detections are matched first and may
  start tracks; detections below `--two-pass-conf` then only continue a track
  that got none, and one that continues none keeps an id of its own (from
  1000001 up), so no detection is lost and a doubtful one never starts a
  track. A track ends after 5 missed analysed frames (**Max disappear: Auto**),
  so an animal lost in a huddle comes back as a new track piece rather than as
  someone else; joining the pieces into animals is the names step's job (see
  Names across cameras). Autos: the gate (`--gate-px`, per video frame, x frame
  step) is half the median animal size (square root of the mask area), at
  least 1.2% of the frame diagonal; the two-pass split is 0.8 x the median
  confidence, at most 0.5. With **Max objects** 1 every detection is the one
  animal (one id for the whole video, as before). On the VC-T005 drop day
  (mesocosm, 6 voles, every 3rd frame) it cut the zone cameras' track pieces in
  the crowded box from 306 to 150 and the frames where one vole is named twice
  at once from 71 to 14, with no more swaps (`--no-takeover`/`--takeover`: at
  Max objects tracks an unmatched detection continues the nearest free track;
  on by default only with Max objects 1).
- **Centroid**: the linker before 2026-11-20, kept to repeat older runs (a run
  recorded without a linker setting repeats with it under `--settings-from`).
  Detections are matched to tracks by centroid distance (Hungarian or greedy,
  **Matching**); a pair further apart than 15% of the frame diagonal never
  matches. Tracks never end by default. A detection left over starts a new
  track; once there are **Max objects** tracks it continues the nearest track
  that got no detection on that frame, wherever it is, and a detection beyond
  the limit with every track taken is left out. Where animals crowd, the few
  ids jump between animals.

`run_settings.json` records the linker settings as given and, under
`linker_resolved`, as used (the autos worked out).

**Linking a finished run again.** The linker only joins detections, so a run
can be relinked from its saved detections without the model:

```bash
python -m fnt.videoTracking.mask_tracker_cli relink <MaskTracker folders, folders holding them or a .txt list>
```

It reads each folder's mask store (`*_tracks.h5`; `trajectories.csv` for
box-only runs), links again with the current linker (linker flags as for
`track`; the run's frame step, object limit and model stay), and writes
`trajectories.csv`, the mask store with the new ids, `run_settings.json` (with
`relinked` and the previous settings) and `inference.log` (the old log, then the
relink's). What the run wrote before moves to
`_TO_DELETE_tracking/<folder>/` beside it, as with `--redo`. The tracked video
showed the old ids, so it moves aside too; `--tracked-video` draws a new one
from the source video. A progress window closes itself when every folder
succeeded (it stays up on a failure); `--no-window` prints instead. The drop
day's four zone cameras (56,000 detections) relinked in under a minute.

Videos that already have a track in their `<video>_MaskTracker` folder are
left out; `--redo` tracks them again, first moving the files tracking wrote to
`_TO_DELETE_tracking/<video>_MaskTracker/` beside them (nothing is deleted). The
folder stays, with the files other tools keep in it, such as the ROI Tool's
drawing (`*_roi*`). `--quit-when-done` closes the window at
the end.

Each `<video>_MaskTracker` folder records its run: `run_settings.json` (the
model, every setting, the FNT version, when, frames per second, and how many
frames had no detection) and `inference.log`. `--settings-from <that folder>`
repeats the run exactly on other videos (flags given as well win). Nothing else is
written beside the videos: each video's record is in its own folder.

### Exits

Tracking that keeps a count of the animals needs to know where an animal can
leave the view: each opening in the wall (in a resource zone, the square
cut-outs). Draw one outline per opening, over the gap and the strip just
outside it. Don't outline the floor or cover inside the view (a water tower,
a nest): an animal that vanishes there is hiding and stays in the count; one
that vanishes within 60 px of an exit has left.

```bash
python -m fnt.videoTracking.mask_tracker_cli exits --videos <folders or files> \
    [--focus <cameras.txt>]
```

In the Mask Tracker window, **Exits...** on the Inference tab's Video Queue
opens the same window for the queued videos' folders (or a folder you pick
when the queue is empty).

The window lists every folder and camera, unchecked ones first (a `--focus`
file of `folder/camera` lines puts those at the top), and shows each camera's
median frame. Click the corners round each exit and close the outline as in the
ROI editor; corners can be dragged and **Delete** removes an exit. **Space**
confirms the camera as it is and moves on; **PgUp** / **PgDn** step through the
list; **Done** (or **Esc**) saves and closes the window. The exits are saved per camera in `exits.json` in each video folder. A
fixed outline cannot follow an exit that moves (a tube that slides in its hole
over a trial), so check those cameras across the trial. Cameras are told apart
by the start of the video name (`CAM13_...`); recorder names that put the camera
second (`VoleCosm_Camera1_{...}_<time>.mp4`, several cameras in one folder) keep
it (`VoleCosm_Camera1`).

### Counting through huddles

In a huddle the model cannot tell animals apart: they lie on and under each
other, and the number of masks it draws there says little about how many
animals there are. The count does not need it. Animals arrive and leave only
through the exits, so a huddle holds the animals that went into it minus those
that came out, even when it was there before the video started (the animals
leaving it show how many it held). An animal lost from view anywhere other than
an exit is hiding (under a feeder, beyond the frame's edge), not gone.

```bash
python -m fnt.videoTracking.mtt_counts <tracked videos or their _MaskTracker folders> \
    [--animals N]
```

It reads each video's masks (`<video>_tracks.h5`; track with **Save masks**,
`--save-masks`), or, without them, the boxes in `trajectories.csv`, and the
camera's exits from `exits.json`. Masks that touch are one group; a group
followed from frame to frame until animals join or leave it is a segment. One
whole number of animals per segment is then solved for the whole video at once:
animals are conserved where segments meet and split; they arrive and leave
cheaply at the exits and at the video's start and end; a segment starting or
ending anywhere else is animals coming out of or going into hiding, and the
number hiding can never go below zero; a lone animal-sized mask is at least one
animal, and a group at least as many as its area holds and as many as the whole
animals it shows apart (masks of at least 0.6 of one animal, not lying over a
bigger mask: animals side by side touch and curl up, so their group has little
more area than one animal, but each keeps its own mask); `--animals` caps the
total. Without exits drawn every start and end counts as an arrival or a
departure and nothing hides. In the `_MaskTracker` folder:

- `counts.csv`: per analysed frame, animals in view, hiding, and both (`in_space`),
  with the groups and masks seen;
- `segments.csv`: each segment's frames, place, masks, area (in animals), whole
  masks, the least it can hold, and solved count;
- `count_events.csv`: arrivals and departures at the exits, and animals going
  into and coming out of hiding.

### Midline keypoints

Keypoints along the body come from the masks themselves, with no keypoint
labelling: each frame's mask is thinned to its spine, the spine's ends are
extended to the mask's edge, and five keypoints are placed along it (nose,
neck, mid, hip, tail base, plus the tail tip when the tail is in the mask).

```bash
python -m fnt.videoTracking.mtt_midline <tracked videos, folders or .txt lists> \
    [--head bright|motion|none] [--objects largest|all] [--scale auto|N] [--workers N] \
    [--frame-workers N] [--skip-existing]
```

`--workers` runs that many videos at once; `--frame-workers N` splits each
video's frames across N processes (a long or crowded video, such as a mesocosm
overhead camera with several enlarged animals). Each frame's midline is worked
out on its own and the head/tail orientation over time is done afterwards on
the joined results, so the CSV is the same as with one process.

`--objects largest` (default) does one animal, the track with the most frames
(an open field). `--objects all` does every tracked animal (a mesocosm or any
group video): each is oriented along its own frames, and the CSV has one row
per animal per analysed frame with its `object_id` (with a frame step, only the
analysed frames).

Small animals are analysed enlarged. The size limits (smallest mask, smallest
head marker, tail length) were set on open-field voles, whose masks are about
7000 px. When a video's median mask is smaller, `--scale auto` (default)
enlarges each crop to that size first: a mesocosm overhead vole of about
560 px, with a hat 6–8 px across, is enlarged about 3.5×. Every value in the
CSV is still in the video's pixels. `--scale N` sets the factor by hand.

The video must have been tracked with **Save masks** (`--save-masks`). Which
end is the head is decided from a head cue and kept consistent from frame to
frame:

- `bright` (default): a bright marker on the head (a tracking tag or headcap),
  the largest patch in the mask brighter than the body by `--bright-delta`
  grey levels (40);
- `motion`: the end leading the direction of travel, for animals without a
  marker. This is a weak cue: on VoleCosm open-field videos it agreed with the
  head marker on only about 70% of moving frames (animals turn, back up and
  move sideways), so check the orientation before relying on it;
- `none`: frame-to-frame continuity only; the first frame's head end is
  arbitrary.

Written to `<video>_MaskTracker/<video>_midline.csv`, one row per frame, in
pixels: the keypoints, body and tail length, head and body direction, the
head-body angle, bend, total turning, the mask and marker area (frames without
a marker hold their orientation from neighbouring frames), and `ok` (the frame
had a usable mask). A 10-minute video takes about 5 minutes per worker.

### Overhead camera to floor metres

```bash
python -m fnt.videoTracking.mtt_floor fit --outline <a run_settings.json or ROI json> \
    --arena X0 X1 Y0 Y1 [--flip-x] [--no-flip-y] --out floor_map.json [--check-frame frame.png]
```

A wide overhead lens bends the arena's straight walls outward, so a plain
perspective mapping is off near the walls. `mtt_floor` uses the floor outline
drawn as the camera's ROI (it follows the walls): it fits the lens's radial
distortion so the outline's points along each wall become straight (points off
a wall's smooth curve, such as notches cut round boxes, are left out),
intersects the four walls for the floor's corners and maps them onto the arena
rectangle in floor-plan metres (`--arena`; by default the image's top is the
plan's high y). `--check-frame` draws the floor grid on a frame to look at.
In Python, `FloorMap.load(path).to_floor(pixels)` gives metres and
`to_pixels(metres)` goes back; `FloorMap.align(pixels, targets)` fits a robust
affine onto another tracking system's coordinates (for example UWB positions
matched to the same animals), so the two can be compared directly. On the
VoleCosm overhead camera the walls' bow went from about 3% to 0.7% of a wall's
length.

### Names across cameras (UWB)

```bash
python -m fnt.videoTracking.mtt_identity identity.json
```

For an arena filmed by several cameras whose animals also wear position tags
(UWB): it names every track and hands names between cameras. Tracks from the
cameras must share frame numbers (clock-synced clips, `fnt-grid --clips`).
UWB is too coarse indoors to name animals moment by moment, but it is
trustworthy when an animal is alone, so names are anchored there and carried
along the tracks:

- **Tracklets**: each camera's tracks, cut where they jump faster than an
  animal moves or are lost for long; an overhead camera's positions go to floor
  metres through its floor map (`mtt_floor`). On the overhead camera a second
  mask on an animal already detected in that frame (two masks overlapping,
  typically the whole body and the head alone) is left out first; the larger
  mask is kept, since the higher-confidence one is often the head. Set
  `"dedupe": true` or `false` on a camera to change this (off for zone cameras).
- **Anchors**: a zone camera that sees one animal while UWB has exactly one
  animal in that box and all others clear of it; an overhead animal whose
  nearest UWB animal is close, the next one far, and no other animal near it.
  A tracklet anchored to two animals holds a swap and is cut between them.
- **Links**: handoffs (a zone tracklet ends in one of its camera's exits, from
  the Exits window, and an overhead tracklet starts by that box within a few
  seconds, or the reverse) and bridges (a tracklet ends and another starts close
  by soon after in the same camera: mask flicker), plus doorway links: in a
  frame where exactly one zone detection touches an exit and exactly one
  overhead detection is at that exit's door on the floor (within 0.15 m), the
  two are one animal seen twice, and their tracklets get a strong link. The
  door positions are taken from the data (the overhead position of the one
  animal by the box while one zone detection touches the exit) and saved as
  `doors.json` in the output folder; a later run reads that file, so a door
  can be moved by hand, and `--derive-doors` places them again.
- **Names**: one integer programme names all tracklets at once. Anchors decide,
  the plain UWB distance breaks ties, and a link whose two ends get different
  names costs its weight, so a name travels along a chain. No animal is on two
  tracklets at once (except the two views of an animal in a doorway: the
  overhead view stays within 0.5 m of the box for 90% of the shared frames) or
  in two places further apart than it could run in the time between. A tracklet
  nothing anchors or links stays unnamed; a chain whose evidence barely favours
  its name is marked unsure.
- **Conflicts**: when one of those rules leaves a tracklet unnamed that its
  anchors or UWB name, because the animal is already on another tracklet for
  part of that time, the tracklet is cut around that stretch (widened by the
  time the animal would need to get between the two places) and the names are
  solved again, up to 3 rounds. Only the stretch in conflict stays unnamed,
  not the whole tracklet (a long zone tracklet once lost all its names to a
  short overhead one).

`identity.json` gives the clock time of frame 0 (`start`), the frame count,
the UWB CSV and its columns, the UWB clock offset (`uwb_offset_s`: UWB second =
video clock second + offset), the zones CSV (box corners in metres) and, per
camera, its Mask Tracker folder plus either its `zone` and `exits` or its
`floor_map`. It writes `identities.csv` (one row per tracklet: camera,
object_id, first and last frame, animal, unsure, anchors), `links.csv`,
`doors.json`, and `duplicates.csv` / `door_links.csv` when there are any.
On a VoleCosm minute it named 83% of tracked detections, 4% of them unsure; a
box holding several animals at once is never anchored, so its tracks take their
names from the doorway handoffs.

An animal counts as in the arena only from its first UWB fix to its last, so
nothing seen before its release (hands at the release point) takes its name.
On the 35-minute VoleCosm VC-T005 drop test each vole was first named within a
minute of its UWB release.

**Corrections**: `corrections.csv` in the `out` folder (camera, object_id,
frame_first, frame_last, animal; `none` = not an animal), written by the
Fusion View window. Tracklets are cut at each correction's ends and the
stretch inside keeps the name given, every other rule still applying, so a
correction renames whatever its chain carries. A correction the rules can't
meet (the same animal twice at once) is reported, not forced. `identities.csv`
marks corrected tracklets (`fixed`).

### Fusion View

```bash
python -m fnt.videoTracking.fusion_view identity.json
```

A window for checking and correcting the names, without rendering a video:
the camera grid drawn live with each animal's mask, keypoints and name.

- **Play and scroll**: slider, play/pause at 0.25-8x, step by an analysed
  frame, a second or ten seconds (arrow keys, Shift, Ctrl; Space plays). A
  timeline has a row per animal, coloured by the camera that names it (faint
  where unsure); click it to go there. Tracks made at a frame step play their
  analysed frames only (option).
- **Zoom**: pinch on a trackpad or touch screen, Ctrl+scroll or the mouse
  wheel zoom about the pointer (up to 12x); drag, or scroll with two fingers,
  to move about; `+` / `-` / `0` (or the -, Fit, + buttons) zoom in, out and
  back to the whole grid. Zoomed in, the grid is drawn at 4K so the cameras and
  labels stay sharp; whole, at 1080 for speed. A click that doesn't drag still
  picks an animal.
- **Show** (toggles): masks, keypoints, names, video tails, UWB positions
  (a filled dot and the animal's name on the overhead camera; who UWB puts in
  or near each box on the zone cameras), UWB tails, a floor map in metres with
  the video positions (squares) and UWB positions (dots and names) side by
  side, and the legend, which gives each animal's distance between its video
  and UWB positions. Tail length is set in seconds. The UWB positions are the
  file named by `uwb` in identity.json as they are: with FNT's smoothed export,
  its smoothing (e.g. a 10-s forward-backward EWMA) shows as UWB cutting
  corners and falling short of fast runs.
- **Go to** (N / P): the next or previous unsure name, name change on a track
  (a possible swap), handoff between cameras, long unnamed track, or
  correction.
- **Correct**: click an animal to see its track, frames, name, margin, anchors
  and its mean UWB distance to every animal over the stretch, then name the
  stretch (or from here on, or up to here) as an animal or as not an animal.
  Corrections are saved at once; **Re-solve names** runs the naming again with
  them held (about half a minute for a 35-minute test) and redraws. Corrected
  names are drawn in a white frame.

Jumping far takes up to a second (the clips have a keyframe every ~8 s);
playing on and stepping are quick.

## Labelling on another machine

A focus list can be labelled anywhere FNT is installed, e.g. on a laptop through
OneDrive:

```bash
python -m fnt.videoTracking.mask_tracker_cli pack-session --project <project> \
    --focus <list.txt> --out "<OneDrive>/sessions" --note "what to do"
```

copies the frames and their labels into a session folder that opens as an
ordinary project (`README.txt` inside says how). When it's done:

```bash
python -m fnt.videoTracking.mask_tracker_cli unpack-session --session <session folder>
```

merges the labels back frame by frame (the project's `annotations.json` is
backed up first). A frame whose labels changed in the project since packing
stops the merge, unless `--force`; `--dry-run` reports without writing.

## Zones in finished tracks (ROI Tool)

FNT's **ROI Tool** (`python -m fnt.sleapProcessing.sleap_roi_tool_pyqt`, or the main GUI)
measures time, entries and distance in zones drawn on each video: for an open field, the
tracking area, a scale bar (e.g. one floor edge, 39.5 cm) and the floor's four corners;
the centre zone is the floor shrunk to half its area. It reads a Mask Tracker track
(`<video>_MaskTracker/<video>_trajectories.csv`, or an older `trajectories.csv`), a SLEAP
analysis CSV or a SimpleTracker CSV. Folders or videos named on the command line open
already loaded.

A video's drawing is saved as `<video>_roiConfig.json` as soon as the video is added to the
queue (in the `_MaskTracker` folder for a Mask Tracker track), so a cancelled run or a closed
window loses nothing. A saved drawing loaded for another kind of track keeps only the
keypoints that track has (a Mask Tracker track names its animals `vole_1`, ...).

Saved drawings run without the window:

```bash
python -m fnt.sleapProcessing.roi_cli <folders or roiConfig.json files> --source mtt --workers 8
```

Every video's CSVs (position coordinates, zone occupancy, `<video>_roiSummary.csv`) are
written first, a few seconds each, then the tracking plots and tracked videos, which take
minutes each, `--workers` at a time. `--data-only` stops after the CSVs; `--visuals-only`
makes only the plots and videos. Outputs go where the window put them; the drawing itself is
not rewritten. Folders named like backups or to-delete are skipped.
