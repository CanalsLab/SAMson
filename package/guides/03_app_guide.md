# 3. The SAMson app

Start (after guide 01): `conda activate samson && samson app --folder /path/to/my/images`, then open
http://localhost:8501 in a web browser (`--port N` changes the port). The app listens on the local computer only by
default (`--address` changes this; it can read and write files on the machine, so do not expose it to untrusted
networks). On a remote server use an SSH tunnel: `ssh -L 8501:localhost:8501 user@server`. Stop with `Ctrl+C`.

The app only sets up and runs the same procedure as the command line: the settings chosen in steps 1-3 form the dataset
specification (`samson_spec.json`, downloadable from the left bar and saved next to the results), and step 4 prints the
equivalent `samson segment` command. A narrated screen-recorded tour of all five steps is in
`../app_tutorial/SAMson_app_tutorial.mp4` (8 min 12 s, captions in `.srt`).
Screenshots of every step are in `../source/docs/images/`.

| Step | Purpose | What you do |
|---|---|---|
| 1 Images | choose a dataset | type the folder (or try the bundled example mouse); the app lists images with shape and voxel size; choose a **representative image**; for 4D files choose the volume (0 = first echo / first volume) |
| 2 Orientation | make SAMson see the data correctly | check, set, verify |
| 3 Anteroposterior range | match slices to the template | see below |
| 4 Segment | run SAMson | model `vit_h`, device `auto`, mode Full-Auto or Semi-Auto, images (representative only, all, or a selection), output folder |
| 5 Review | inspect, correct, export | see below |

## Step 2: Orientation page

SAMson needs coronal slices, dorsal side up, slice 0 at the caudal end (guide 04). The page shows the representative image
as SAMson will see it next to the template: the **coronal view must show the dorsal side up** and the **side view must
show the cerebellum on the left**. If not, set (i) which image axis runs from back to front of the head, (ii) which runs
from bottom to top, and tick *Reverse anteroposterior axis* / *Reverse dorsoventral axis* (left-right reversal is
optional: it does not change the masks). *Guess from the image header* reads the direction letters (R/L, A/P, S/I) of
the NIfTI header; for rodent data these are often wrong, so always verify the two views. The files are never modified; the
masks are written on the grid of the original images.

## Step 3: choosing the anteroposterior range

SAMson places one box prompt per slice from the template slice that matches it, so it must know which **template slices
(0 = caudal end, 39 = rostral end, 0.4 mm apart)** match the first and last processed slice.

1. **Top: side views.** Left, the sagittal view of the representative image; right, the sagittal view of the template.
   Horizontal lines mark the processed slices of the subject and the corresponding positions in the template (same colour =
   matching slices). Use *Side view position (left-right)* to move the sagittal plane.
2. **Bottom: coronal pairs**, one pair per sampled slice: the subject slice above its matched template slice, with SAMson's
   box prompt in green (*Show the box prompts*; the median_otsu setting used to place the box is selectable).
3. Move *Template slices matching the first and last processed slice* until every pair shows the same anatomy and the green
   boxes enclose the brain. *Slices to process* excludes slices without brain at either end (they stay empty in the mask).
   *Link the two range lengths* keeps the template range as long, in mm, as the subject range (ratio 1.0-1.25 in the
   article data), so only the start has to be moved. Set *Header scale* to 0.1 when the header stores voxel sizes x10
   (it affects only the lengths shown, not the masks).
4. The page lists the two range lengths in mm and yellow warnings for implausible settings (length ratio outside
   1.0-1.25; template range not including the template centre, slice 19.5, which affects the anchor slice).
5. Optional: *Suggest the range from manual masks* ranks ranges by the agreement between SAMson's template boxes and the
   bounding boxes of manual masks of a few animals, with one-click buttons to apply a suggestion; always check visually.

Partial coverage (e.g. six 1 mm coronal slices): choose the template slices that show the same anatomy as the first and
last slice (example in `../source/examples/specs/partial_coverage_6_slices.json`: 12-22).

## Step 4: Full-Auto or Semi-Auto

Full-Auto gives one mask per slice. Semi-Auto additionally stores SAM's three candidate masks per slice, to be chosen from
in step 5. The article's Semi-Auto numbers use an oracle selection (the candidate with the highest Jaccard index against
the manual mask); in the app the user chooses by eye. Existing masks are skipped unless *Redo* is ticked. The only
dataset-specific setting that SAMson requires from the user is the anteroposterior range (plus the orientation, which
is a property of how the files were stored).

## Step 5: Review and export

Choose an image (from this session, or *A results folder* from earlier runs or the command line). The page shows every
processed slice with the mask (blue) and box (green); *Zoom on the brain* crops around the brain. In Semi-Auto mode move the
*Slice* slider, click the best of the four options (Full-Auto mask, candidates 0-2) or *Set every slice to ...*;
**Save the mask** rewrites `<name>_mask.nii.gz` (and `_brain.nii.gz` if requested) and updates `<name>_samson.json`.
With *Compare with a reference mask* the Jaccard index and Dice coefficient are computed and the reference is drawn in yellow;
the button *Select the best candidate of every slice* reproduces the article's oracle Semi-Auto. Exports: masks
(`_mask.nii.gz`), candidates, JSON sidecar, QC picture, `samson_spec.json`, `samson_summary.csv` (guide 04).

## Verification status

The app's five steps are exercised by the test suite (Streamlit's headless application tester, `tests/test_app.py`,
3 tests, pass; guide 05). The tutorial video is a real recorded session (see `../app_tutorial/README.md` for the example
image used and a data-provenance note).
