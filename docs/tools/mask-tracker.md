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

The timeline under the frame marks the queue's frames from this video: green
labelled, yellow predicted, white still to do.

**Small animals.** Zoom in (scroll) before you click: SAM sees what you see.
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
`--iou`, `--matching`, `--classifier` and its four settings); one not given keeps
the tab's value. Videos that already have a `<video>_MaskTracker` folder are left
out; `--redo` tracks them again after moving their old folders to
`_TO_DELETE_tracking/` beside them. `--quit-when-done` closes the window at
the end.

Each `<video>_MaskTracker` folder records its run: `run_settings.json` (the
model, every setting, the FNT version, when, frames per second, and how many
frames had no detection) and `inference.log`. `--settings-from <that folder>`
repeats the run exactly on other videos (flags given as well win). At the end of
a run, each video folder gets `MaskTracker_summary.csv`: one row per tracked
video with its frames, frames without a detection, tracks, speed and settings.
`mask_tracker_cli track-summary --videos <folders>` rebuilds it for any folder;
output from before these records is read from its trajectories and mask store.

### Exits

Tracking that keeps a count of the animals needs to know where an animal can
leave the view (the mouth of an entrance tube):

```bash
python -m fnt.videoTracking.mask_tracker_cli exits --videos <folders or files> \
    [--focus <cameras.txt>]
```

The window lists every folder and camera, unchecked ones first (a `--focus`
file of `folder/camera` lines puts those at the top), and shows each camera's
median frame. Click the corners round each exit and close the outline as in the
ROI editor; corners can be dragged and **Delete** removes an exit. **Space**
confirms the camera as it is and moves on; **PgUp** / **PgDn** step through the
list. The exits are saved per camera in `exits.json` in each video folder. A
fixed outline cannot follow an exit that moves (a tube that slides in its hole
over a trial), so check those cameras across the trial.

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
