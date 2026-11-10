# Swalker: project summary

How the Swalker website was built: what it does, the method behind each part, how the CASIA-B
dataset was used, what was measured, and what is still open. Written 2026-11-10.

Swalker is the website for **GLTNet**, a gait recognition model, a Birzeit University ECE graduation
project by Nabeel Sawafta, Ahmad Karmi and Kareem Hamza, supervised by Dr. Rawan Soboh. The model
identifies a person from the way they walk, without their face. The website shows the model at
work: a visitor uploads a walking clip and the site either names the person from its Archive or
says the walker is unknown.

Other documents go deeper: [`DATASET.md`](DATASET.md) (the dataset),
[`PROJECT_STATUS.md`](PROJECT_STATUS.md) (what is built), [`CHANGELOG.md`](CHANGELOG.md) (every
change with its date and PR), [`HANDOFF.md`](HANDOFF.md) (the model and the original plan),
[`backend/README.md`](backend/README.md) and [`frontend/README.md`](frontend/README.md).

## 1. The idea in one picture

```
walking clip (phone or CASIA-B video)
  -> resampled to 25 frames per second
  -> YOLOv8-seg finds the walker in each frame and cuts out their shape (a mask)
  -> each mask is framed like a CASIA-B silhouette: 64 x 44 pixels, white person on black
  -> up to 120 consecutive silhouettes go into GLTNet
  -> GLTNet gives one 4096-number vector: the walk's "gait signature"
  -> compared with every clip in the Archive by Euclidean distance (smaller = more alike)
  -> closest person under the match line = identified; otherwise = unknown walker
```

The site has a detective theme: the upload page is **The Crime Scene**, the stored people are
**The Archive**, the result is **The Verdict**, and a detective duck appears on every page.

## 2. The model (trained outside this repo)

GLTNet, also called Combined GLFE + MSTFE, has about 6.5M parameters.

- **What it learns from:** a spatial part (GLFE) that looks at body parts, two temporal branches
  that read motion between neighbouring frames and over 3-frame steps (MSTFE), and pooling per
  horizontal strip of the body: 16 strips × 256 numbers = 4096.
- **Training:** on CASIA-B with triplet loss and cross-entropy, on Kaggle. The website uses the
  checkpoint `stage2_iter70000.pth` and never changes the architecture or the training.
- **Accuracy:** on the standard CASIA-B test (people 075-124, all 11 camera angles), Rank-1 is 99.27%
  for normal walking, 93.26% carrying a bag and 72.36% in a coat. These are the corrected numbers
  from 2026-09-22 and are shown on the About page.

## 3. How the work was done

The rules that shaped every step:

1. **Measure before trusting.** Each change that could affect the answers was measured first,
   against a baseline that reproduces the training numbers exactly. No accuracy claim appears
   without the measurement behind it.
2. **The website must feed the model what it was trained on.** Most of the work in the pipeline
   went into making a phone clip look like CASIA-B to the model.
3. **Small, reviewed changes.** Every change went through a branch and a pull request (25 so far),
   with automatic checks (CI): lint and build for the frontend, lint and 88 tests for the backend.
   Each change has a dated line in `CHANGELOG.md`.
4. **Keep test data apart.** People the model trained on (001-074) are never used for accuracy
   claims, and test clips are held out of the Archive they are tested against.

### Timeline

| Date | What was done |
|---|---|
| 2026-09-17 to 09-19 | Frontend with demo data: navigation, Crime Scene, Verdict, Archive, About (#1-#5) |
| 2026-10-19 | Handoff plan for the backend; review of the original Python scripts found 6 problems (`HANDOFF.md` §6) |
| 2026-10-28 | Backend built and wired to the frontend, all demo data removed (#7, #8); evaluation on Kaggle reproduces the training numbers exactly; identify runs as a job with real progress (#9-#11) |
| 2026-10-29 | Framing of silhouettes chosen by measurement (#12, #16, #17); one-click launcher `start.bat` (#13-#15); match line 8.2, then 8.5 |
| 2026-10-30 | First phone recordings of 10 people measured; several clips per person (#20); smaller fixes (#18, #19, #21); changelog (#22) |
| 2026-11-05 | Raw CASIA-B videos imported into the Archive and measured (#23); match line set to 8 and `DATASET.md` (#24); phone layout (#25) |

## 4. The pipeline, step by step, and why

The original inference scripts (`silhouette_extractor.py`, `identify.py`) ran, but a review found
that several of their choices would make the website's answers worse than the model's real accuracy
without showing any error. Each step below says what the backend does now and why.

| Step | What the backend does | Why |
|---|---|---|
| Frame rate | Resample the clip to 25 fps | CASIA-B is 25 fps and the temporal branches learned walking speed at that rate; a 60 fps phone clip would look like slow motion |
| Finding the walker | YOLOv8n-seg, people only, masks at the frame's own resolution | The old masks came from a letterboxed image and didn't line up with the frame |
| Following one person | Start with the biggest confident person, then follow the box that overlaps most (overlap of at least 0.2) | The old script took "the most confident person" in each frame, so bystanders got mixed into the walk |
| Framing the silhouette | Person height 50.1% of the frame, width squashed to 0.52, top of the box at 23.3%, centre at 51.4%; drawn as a binary 320×240 frame and shrunk to 44×64 with PIL bilinear | This is how a CASIA-B silhouette looks. The old tight crop made the person fill the whole image, which the model never saw |
| Scale and position | One scale and one position for the whole clip (`casia_clip`) | Keeps the bobbing and the movement across the frame that the camera saw |
| Which frames | Up to 120 consecutive frames, at least 15, reading stops after 450 | The old script spread 30 frames across the clip, so the walk looked several times faster than in training |
| Gait signature | GLTNet loaded once at startup, `eval()` and `no_grad()`, only the `feat` output | Running in training mode once made a working model look broken (normal walking 84.91% read as 22.55%). The classifier output is meaningless for new people |
| Matching | Euclidean distance to every clip; each person counts once, by their closest clip; top 3 | One person can have a normal clip, a bag clip and a coat clip |
| Decision | Closest person at or under the match line is a match, otherwise unknown | The old script always named someone, even a stranger |
| "Strength" | Shares of the top 3 that add up to 100%, labelled as relative | It is not a probability, so the decision uses the raw distance |

### How the framing was chosen (measured on Kaggle, 2026-10-28/29)

`backend/scripts/casia_eval.py` runs the model on the CASIA-B test people (075-124) using CASIA-B's
own silhouette masks, framed in different ways. It first checks that the training framing gives
exactly the training numbers. It does (99.27 / 93.26 / 72.36), so any gap below is caused by the framing.

Rank-1 for normal / bag / coat:

| Framing | Normal | Bag | Coat | Note |
|---|---|---|---|---|
| Training framing (whole CASIA-B frame) | 99.27 | 93.26 | 72.36 | The baseline; needs CASIA-B's own camera, so it can't be used on phone video |
| Old tight crop (`crop`) | 69.45 | 41.17 | 12.09 | What the original script did |
| CASIA-B framing, per frame (`casia`) | 96.09 | 78.05 | 35.73 | |
| One scale and position per clip, old drawing (`casia_clip` v1) | 97.82 | 90.06 | 56.64 | |
| **One scale and position per clip, training's drawing (`casia_clip` v2)** | **97.82** | **90.07** | **63.91** | **What the website uses** |

How the silhouette is drawn mattered as much as where it is placed: using the training code's PIL
shrink instead of area averaging took coat from 56.64 to 63.91. The remaining gap to the baseline comes
from fitting each clip's scale and position. Phone video needs that step, because the person's size
and place in the frame depend on the camera. These numbers use CASIA-B's clean masks; YOLO's masks
are tested in section 7.

## 5. The backend

FastAPI and PyTorch in `backend/`.

- **Models load once** at startup. If loading fails, the server still starts, and `/api/health`
  says why.
- **Identify runs as a job.** `POST /api/identify` returns at once, and the page polls for the stage
  (finding the walker, silhouettes, signature, searching) and the progress, which drives the loading
  screen.
- **The Archive stores signatures, not videos.** For each clip it keeps the 4096-number vector, about
  10 small silhouettes for display, and the clip's name and date. Uploaded videos are deleted
  after processing.
- **Consistency check.** The Archive records the framing, its version, the checkpoint and the frame
  rate it was built with. If any of these change, the server refuses to compare instead of giving
  wrong answers.
- **Editing.** Add a person from a video or from a walk just identified, add more clips to someone,
  remove a clip, rename, delete.
- **Settings** in `backend/.env`: match line, device, frame limits, upload size.

## 6. The frontend

React 19, TypeScript, Vite and MUI in `frontend/`.

- **The Crime Scene:** pick a video, which is checked for type and size before upload.
- **The Verdict:**
  - a loading screen that follows the real job
  - the decision on a distance scale with the match line
  - three key numbers
  - the clue's silhouettes playing next to the matched person's
  - the closest 3 walks
  - how the result was reached

  An unknown walker can be saved to the Archive, as a new person or as another clip of someone.
- **The Archive:** every person as a card. A case file shows each clip's silhouettes and signature,
  with Add clip, Rename and Delete.
- **About:** how it works, what gait recognition is, the model and its accuracy.
- **Layout:**
  - On PC, pages don't scroll as a whole; each page scrolls inside a fixed area.
  - On phones, the navigation moves to a bar along the bottom (#25).
  - The PC layout was checked unchanged: 70 captures at 7 window sizes, compared before and after.
- **Launcher:** `start.bat` starts both servers and opens the site.

## 7. How the CASIA-B dataset was used

CASIA-B (Institute of Automation, Chinese Academy of Sciences) has 124 people. Each is filmed
walking normally 6 times, with a bag 2 times and in a coat 2 times, each from 11 camera angles: 110
walking videos per person, 13,640 in all. The details are in [`DATASET.md`](DATASET.md).

It was used in three ways:

| Use | Data | Where |
|---|---|---|
| Training the model | Silhouettes of people 001-074 | Kaggle, outside this repo |
| Choosing the framing (section 4) | Silhouettes of people 075-124 | Kaggle, `casia_eval.py` |
| Filling and testing the Archive | Raw videos of all 124 people, through the website's own pipeline | This laptop, `casia_videos.py` |

### The Archive built from CASIA-B (2026-11-05)

- **Archive:** each person has a made-up first name (001 = Ahmad, 002 = Omar) and holds their 6
  normal walks plus the first bag walk and the first coat walk (`bg-01`, `cl-01`). That is 88 clips
  per person, 10,909 in all. 3 videos could not be read, all of person 109.
- **Held out for testing:** the second bag walk and the second coat walk (`bg-02`, `cl-02`), 2,728
  videos in `datasets\tests\<name>\`. They are not in the Archive.
- **Normal walking test:** the 6th normal walk (`nm-06`) is left out of the comparison for its own test.
- **Strangers:** to measure how often a stranger gets a name, each test clip is compared with
  everyone except its own person.
- **Speed:** all 13,637 readable videos went through the real pipeline in about 5.5 hours on the
  laptop's GPU, about 1.5 s per video.

Results for the people the model never trained on (075-124), with the Archive as built:

| Test clips | Closest is right (Rank-1) | Right person accepted at line 8 | Stranger gets a name at line 8 |
|---|---|---|---|
| Normal, side view (50 clips) | 100% | 98% | 38% |
| Bag, side view (50 clips) | 98% | 88% | 8% |
| Coat, side view (50 clips) | 100% | 96% | 8% |
| Normal, all angles (550 clips) | 99.82% | 96% | 16.55% |
| Bag, all angles (550 clips) | 96.55% | 80% | 7.45% |
| Coat, all angles (550 clips) | 96.91% | 86% | 4.18% |

**Several clips per person help.** With only normal clips in the Archive, Rank-1 over all angles
was 99.64 / 92.91 / 62.18 (normal / bag / coat). Adding one bag clip and one coat clip per person
raised it to 99.82 / 96.55 / 96.91. The added and the tested bag and coat walks come from the same
CASIA-B recordings, so a different bag or coat may gain less.

**These are not the model's benchmark numbers.** The standard test doesn't allow bag and coat clips
in the Archive. The comparable row is the normal-only Archive (coat 62.18 against 72.36 on the
benchmark), and the difference is the website's framing and YOLO's masks.

**Checked through the real website backend:** 5 held-out side-view clips named the right person at
distances 5.55-6.74, with the next person 8.79 or more away.

## 8. The match line

The match line is the distance under which the closest person counts as identified. A lower line
names fewer strangers but also turns away more of the right people. It changed as more was
measured:

| When | Line | Based on |
|---|---|---|
| 2026-10-29 | 8.2, then 8.5 | CASIA-B silhouettes, normal against normal, same angle: 1% of different-person pairs fall under 8.52 at the side view, while 96.50% of same-person pairs do |
| 2026-11-05 | 7.2 | The full Archive: with about 88 clips for each of 123 other people, a stranger's nearest distance is smaller. At 8.5, 21.82% of strangers got a name; 1% of strangers is at 7.17 |
| 2026-11-05 | **8** (now) | The team's choice, to accept more of the right people (table in section 7). The cost: more strangers named, most of all strangers walking normally from the side (38% of 50 clips) |

The site still calls the line **provisional**, because it was measured on CASIA-B only.

## 9. Real phone recordings

**First recordings (2026-10-30).** 10 people were recorded with a phone, one normal side-view walk
each, plus bag and coat test clips of three of them:

- Distances between different people started at 9.80 (median 12.66).
- The right person was 10.97-12.66 away for the bag and coat clips, overlapping with different
  people, so no line separated them.
- The closest person was right for 2 of the 3 clips.

Judging each distance against that person's usual distances (score normalisation) was tried and
rejected: the wrong clip stayed wrong.

**Against the CASIA-B Archive (2026-11-05).** All 13 phone clips came out unknown, which is
correct, because none of these people are in it. The nearest CASIA-B person was 9.43-13.09 away.

**What this means.** Distances between phone clips are much larger than between CASIA-B clips
(in CASIA-B the median distance to the right person is 5.7-6.7; for the phone bag and coat clips it
was 11-12.7). So a person
added from a phone clip would probably come out unknown at line 8, even when it is the right
person. The missing measurement is a second normal clip of the same people, to see how far apart
two phone walks of one person are.

## 10. Limits and what is left

- **Record phone clips** of 4-5 people, two normal walks each plus a bag or coat walk, and run
  `scripts/calibrate_threshold` on them to set the match line for phone video.
- **Decide who is in the Archive for the demo:** the CASIA-B people, the team, or both.
- **Different bag or coat:** the gain from extra clips was measured with the same bag and coat as in
  the Archive.
- **YOLO on real video:** the framing measurements use CASIA-B's clean masks. Phone video adds
  YOLO's mask errors, measured only through the phone clips above.
- **Phone layout:** checked in a headless browser at phone sizes, not on a real phone.
- **One backend at a time:** the Archive is a folder of files used by one server process.
