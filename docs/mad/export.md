# Exporting tables

**File ▸ Export…** writes folder-wide tables from the `.mad` files beside your
recordings: one row per recording, one row per detection and one row per
labelled call, all in one output folder. They are meant for analysis in R or
pandas. `mad export` writes the same tables from the command line, with the
same options.

This is different from **File ▸ Export Annotations to CSV…**, which writes one
`<recording>_FNT_MAD_annotations.csv` beside each wav. Use that one to look at a
single recording; use Export… to analyse a run.

## What it writes

| File | One row per | Contents |
|---|---|---|
| `mad_recordings.csv` | recording whose last inference was the chosen model | where it is, its start time and channel when the file name gives them, its duration, the model and when it ran, its detection and call counts |
| `mad_detections.csv` | detection of those recordings | onset, score, review status, harmonic number, contour measures of its mask, brightness ridge |
| `mad_labelled_calls.csv` | labelled (accepted) call, from any model or painted by hand | onset and offset, frequency box, MAD's call metrics recomputed from the audio, contour measures, brightness ridge, note and tags |
| `mad_call_images/<id>.png` | labelled call | the call and 10 ms either side, every frequency bin (highest at the top); grey = level as MAD stores it, alpha = the call's own mask |
| `mad_export.json` | — | how and when each table was last written (model, settings, inputs, FNT version) |

The columns are generic. Anything particular to your experiment — trial,
treatment, which animals were present, which zone a microphone covers — is not
in the tables; join it in your analysis code, by `folder`, `recording`,
`channel` or `start`.

Detections are the model's output: each row is the detection's mask as the
model drew it, whatever its review status. Labelled calls are the reviewed
masks, including edits and hand-painted calls.

## Options

| Option | Meaning |
|---|---|
| Recordings | The open project's recordings (the Audio list's with no project open), or folders, each searched to the bottom. Only recordings with a `.mad` file are exported; a recording whose wav has moved is still exported from its `.mad`. |
| Output folder | Where the tables go. Created if needed. |
| Recordings and detections | `mad_recordings.csv` and `mad_detections.csv`. No audio is read. |
| Model | Only recordings whose last inference was this model. The default is the model that analysed the most recordings; the list shows how many each model analysed. Labelled calls are not filtered by model. |
| Labelled calls | `mad_labelled_calls.csv`. Each labelled recording's audio is read to compute the call metrics, so this is the slow table. If the audio is missing the rows are written with those columns blank. |
| Spectrogram images | `mad_call_images/`, one PNG per labelled call. Implies Labelled calls. |
| Include harmonics | Off (the default): fundamentals only, i.e. calls whose `harmonic_n` is blank or 1. On: every detection and labelled call. |
| Skip recordings unchanged since the last export | On (the default): see below. |
| Leave out .mad files written in the last N min | Default 10. A `.mad` that changed this recently may still be open in MAD (an inference run in progress), so it is left out until the next export. A `.mad` that cannot be opened is left out the same way. Set 0 to include everything. |

The export runs in the background and shows its progress. **Stop** ends it
after the current recording; the table being written is discarded and the
previous one is kept. Closing the window stops it too.

## Exporting again

Export into the same folder again and only new and changed recordings are read.
The rows of every recording whose `.mad` has the same modification time as last
time are copied from the existing tables, so an export of a run in progress
mostly costs the recordings analysed since. A recording whose rows in the old
table do not add up to the count in `mad_recordings.csv` is read again.

Nothing is copied when the settings differ from the last export (another
harmonics setting, images turned on or off) or when the table format has
changed. Turn **Skip recordings unchanged** off (`--full` on the command line)
after updating FNT, so every value is computed with the current code.

Each table is written to a temporary file and swapped in only when complete,
so a crash leaves the previous table as it was.

## From the command line

```bash
# Recordings and detections of the most common model, every .mad under a folder
mad export --input /data/UltrasonicVocalizations_proc --out /data/usv_tables

# Which models analysed which recordings
mad export --input /data/UltrasonicVocalizations_proc --list-models

# A named model (or the unique start of its name), harmonics included
mad export --input /data/UltrasonicVocalizations_proc --out /data/usv_tables \
           --model 20261014_181834 --harmonics

# A project's labelled calls, with their images
mad export --project my_project/ --out /data/usv_tables --labels --images
```

| Flag | Meaning |
|---|---|
| `--input` | Folders (searched to the bottom) and/or wav files. |
| `--project` | The recordings registered in a MAD project (with or instead of `--input`). |
| `--out` | The output folder. |
| `--detections` | `mad_recordings.csv` and `mad_detections.csv`. The default when no table is named. |
| `--labels` | `mad_labelled_calls.csv`. |
| `--images` | `mad_call_images/` (implies `--labels`). |
| `--model` | The model to export; default the most common. |
| `--list-models` | Print each model and its number of recordings, then stop. |
| `--harmonics` | Include harmonics. |
| `--full` | Read every `.mad` again instead of copying unchanged recordings. |
| `--settle-minutes` | Leave out `.mad` files written in the last N minutes (default 10). |

It prints a line for each step and a progress line every 500 recordings.

## Columns

### mad_recordings.csv

| Column | Meaning |
|---|---|
| `path` | Full path of the recording. |
| `folder` | Name of the folder the recording is in. |
| `recording` | File name of the recording. |
| `channel` | Microphone channel from the file name (`ch<N>…`), when it has one. |
| `start` | Start time from the file name, `yyyy-mm-dd HH:MM:SS`, when it matches Avisoft's `ch<N><yyyy-mm-dd>_<HH-MM-SS>_<n>.wav`; otherwise blank. This is the recorder's clock. |
| `duration_s` | Length of the analysed spectrogram, in seconds. |
| `model` | Model of the last inference (its run folder name). |
| `inferred_at` | When that inference ran. |
| `threshold` | Its probability threshold. |
| `detections` | Number of detections, harmonics included. |
| `calls` | Number of detections that are fundamentals (`harmonic_n` blank or 1). |
| `mad_mtime` | Modification time of the `.mad`, in seconds since 1970 (UTC). Used by the next export to tell what changed. |

### mad_detections.csv

`folder` and `recording` join each row to its recording.

| Column | Meaning |
|---|---|
| `blob_id` | The detection's id in the `.mad`. |
| `start_s` | Onset, seconds from the start of the recording (the same time axis as the per-recording CSV). |
| `score` | The model's mean probability over the detection. |
| `status` | `pending`, `accepted` or `rejected`, from the review. |
| `harmonic_n`, `harmonic_call_id` | Harmonic grouping: 1 is the fundamental, 2 and up its harmonics; the call id groups them. Blank if grouping has not run. |
| contour measures | See below. |
| `ridge_turns`, `ridge_rate_hz`, `ridge_depth_khz`, `ridge_trill` | Brightness ridge (`mad add-ridge`): turns of the brightest continuous path through the mask, half the turns per second of call (Hz), median kHz between successive turns, and 1 for a trill (at least 4 turns at 20 Hz or faster). Blank where the `.mad` holds no ridge. |

### mad_labelled_calls.csv

| Column | Meaning |
|---|---|
| `path`, `folder`, `recording`, `channel` | As in `mad_recordings.csv`. |
| `rec_start` | The recording's start time from its file name (as `start` above). |
| `id` | The label's id in the `.mad`; also the image's file name. |
| `blob_id` | The detection it was accepted from; blank for a call painted by hand. |
| `painted` | `TRUE` for a call painted by hand. |
| `class` | Call class. |
| `start_s`, `stop_s` | Onset and offset, seconds from the start of the recording. |
| `min_freq_hz`, `max_freq_hz` | The call's frequency box. |
| `harmonic_n`, `harmonic_call_id`, `f0_hz` | Harmonic grouping. |
| `score`, `model_name` | Score and model of the detection it was accepted from. |
| `note`, `tags` | What the reviewer wrote on the call. |
| `peak_freq_hz` … `aspect_ratio` | MAD's call metrics, recomputed from the audio, as in the per-recording CSV (see the technical reference). |
| contour measures | See below. |
| `ridge_*` | The label's own brightness ridge. |
| `image`, `img_col0`, `img_col1`, `f_bin0`, `f_bin1` | The image's file name, and where the call is in it, counted from 1: its first and last columns, and its lowest and highest frequency bins (bin 1 is 0 Hz). Blank without images. |
| `mad_mtime` | As in `mad_recordings.csv`. |

### Contour measures

Measured from the call's own mask, without the audio. The contour is the
mask's mean frequency in each time frame.

| Column | Meaning |
|---|---|
| `duration_ms` | Width of the mask. |
| `f_mean_khz` | Mean of the contour. |
| `f_min_khz`, `f_max_khz` | Lowest and highest frequency of the mask. |
| `f_start_khz`, `f_end_khz` | First and last value of the contour. |
| `slope_khz_ms` | (end − start) / duration. |
| `contour_sd_khz` | Standard deviation of the contour. |
| `turns` | Reversals of the contour by more than one frequency bin. |
| `max_jump_khz` | Largest frame-to-frame change of the contour. |
| `thickness_khz` | Mean height of the mask per time frame. |
| `area_px` | Pixels in the mask. |

`turns` follows the mask's centre and counts small reversals; `ridge_turns`
follows the brightest line inside the mask and counts excursions of at least
3 kHz. A blob-shaped mask over a fast trill has few `turns` and many
`ridge_turns`.
