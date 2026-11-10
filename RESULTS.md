# Swalker: results

Every measurement made for the website, in one place, each with what it was measured on. How the
measurements were made and why: [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md). The dataset:
[`DATASET.md`](DATASET.md).

**Words used here:**
- **Rank-1:** how often the closest person in the Archive is the right one.
- **Distance:** how far apart two gait signatures are; smaller is more alike.
- **Match line:** the distance under which the closest person counts as identified.
- **Accepted:** the closest person is right *and* under the match line.
- **Wrong person accepted:** the closest person is someone else and under the line.
- **Stranger gets a name:** a walker who is not in the Archive is wrongly identified. Measured by
  comparing each test clip with everyone except its own person.
- **Normal / bag / coat:** walking normally, carrying a bag, wearing a coat.
- **New people:** CASIA-B people 075-124, whom the model never saw in training. Only they count for
  accuracy claims.

## Results in brief

- **The model** (CASIA-B benchmark): Rank-1 99.27% normal, 93.26% bag, 72.36% coat.
- **The website's pipeline** on CASIA-B silhouettes keeps most of that: 97.82 / 90.07 / 63.91. The
  original script's framing gave 69.45 / 41.17 / 12.09.
- **On raw CASIA-B video through the website** (new people, all angles), with a bag and a coat clip
  of each person in the Archive: Rank-1 99.82 / 96.55 / 96.91. With normal clips only: 99.64 / 92.91 / 62.18.
- **At the match line of 8:** the right person is accepted 96% / 80% / 86% of the time over all angles,
  and 98% / 88% / 96% at the side view. A stranger gets a name 16.55% / 7.45% / 4.18% of the time
  over all angles, and 38% / 8% / 8% at the side view.
- **Phone video:** all 13 phone clips came out unknown against the CASIA-B Archive, which is right.
  Between the team's own phone clips the right person was closest for 2 of 3 bag and coat clips,
  but too far away to pass any match line. The match line for phone video is not set yet.

## 1. The model on the CASIA-B benchmark

Standard test: people 075-124, 11 camera angles. Measured on Kaggle during training.

| | Normal | Bag | Coat |
|---|---|---|---|
| First published | 99.18% | 92.44% | 72.36% |
| Re-measured with the evaluation bug fixed (2026-09-22), shown on the About page | **99.27%** | **93.26%** | **72.36%** |

The first numbers encoded the Archive and the test clips in different BatchNorm modes. The checkpoint
is the same one (`stage2_iter70000.pth`).

## 2. Choosing how silhouettes are framed (2026-10-28/29)

`backend/scripts/casia_eval.py` on Kaggle: CASIA-B's own silhouette masks of people 075-124, framed
in different ways, then the standard test. The `train` row reproduces the training code exactly,
which shows the test itself is right.

Rank-1, all angles together:

| Framing | Normal | Bag | Coat | What it is |
|---|---|---|---|---|
| `train` | 99.27 | 93.26 | 72.36 | The training code's framing: the whole CASIA-B frame. Needs CASIA-B's camera, so it can't be used on other video |
| `native_bin` | 99.27 | 93.26 | 72.36 | The same from a black-and-white mask: identical, so binary masks lose nothing |
| `native` | 97.27 | 91.79 | 64.27 | CASIA-B's real geometry, the website's first way of drawing |
| `native_pos` | 98.00 | 89.97 | 63.73 | First drawing, scale fitted per clip |
| `native_scale` | 97.73 | 91.25 | 61.55 | First drawing, position fitted per clip |
| **`clip_pil`** | **97.82** | **90.07** | **63.91** | **The website (`casia_clip` version 2): scale and position fitted per clip, drawn like the training code** |
| `casia_seq_x` | 97.82 | 90.06 | 56.64 | `casia_clip` version 1: the same fitting, first drawing |
| `casia` | 96.09 | 78.05 | 35.73 | Fitted per frame |
| `casia_seq` | 92.55 | 76.40 | 40.18 | Scale fitted per clip, each frame centred |
| `crop` | 69.45 | 41.17 | 12.09 | The original script: a tight crop around the person |

Judged one test clip at a time against all angles, the result closest to how the website works
(normal / bag / coat): `train` 99.82 / 98.18 / 82.91, `clip_pil` 99.64 / 95.99 / 74.45.

What this shows:
- **Drawing:** drawing the silhouette like the training code (PIL bilinear shrink of a 320×240 frame)
  took coat from 56.64 to 63.91.
- **The remaining gap:** what is left to `train` comes from fitting each clip's scale and position,
  which other video needs.

## 3. First match line, from silhouettes (2026-10-29)

`clip_pil`, normal against normal, same camera angle. Each cell is the line that lets the given share
of different-person pairs through, and the share of same-person pairs under it:

| Angles | 0.1% different-person | 1% | 5% | 10% |
|---|---|---|---|---|
| All 11 | 8.27 (86.63%) | 9.41 (95.00%) | 10.55 (97.71%) | 11.18 (98.40%) |
| Side view (090) | 7.47 (93.00%) | 8.52 (96.50%) | 9.65 (98.75%) | 10.29 (99.75%) |

This gave the match line of 8.5. It compares pairs of clips, though, and the Archive compares a walk
with every clip of every person. That made 8.5 too loose for the full Archive (section 4).

## 4. Raw CASIA-B videos through the website (2026-11-05)

`backend/scripts/casia_videos.py`: every walking video of all 124 people through the website's own
pipeline (YOLO, framing, GLTNet). 13,637 of 13,640 videos could be read.

- **The Archive** holds each person's 6 normal walks plus `bg-01` and `cl-01`: 88 clips per person.
- **Tests:** `bg-02` and `cl-02` are held out. `nm-06` is left out of the comparison for its own test.
- **Strangers:** each test clip compared with everyone except its own person.

### Rank-1

| Who | Archive | Test | All angles | Side view |
|---|---|---|---|---|
| New people | Normal clips only | Normal (`nm-06`) | 99.64 (550 clips) | 100 (50) |
| New people | Normal clips only | Bag (`bg-02`) | 92.91 | 92.00 |
| New people | Normal clips only | Coat (`cl-02`) | 62.18 | 74.00 |
| New people | **+ `bg-01` + `cl-01` (as built)** | Normal | **99.82** | **100** |
| New people | **+ `bg-01` + `cl-01` (as built)** | Bag | **96.55** | **98.00** |
| New people | **+ `bg-01` + `cl-01` (as built)** | Coat | **96.91** | **100** |
| People from training | Normal clips only | Normal / bag / coat | 99.02 / 95.45 / 84.40 (814 each) | 100 / 97.30 / 90.54 (74 each) |
| People from training | + `bg-01` + `cl-01` | Normal / bag / coat | 99.14 / 98.28 / 98.16 | 100 / 97.30 / 100 |

The people from training are shown only for comparison; the model has seen them.

### At three match lines (new people, Archive as built)

| Line | Test | Accepted, all angles | Wrong person accepted | Stranger gets a name | Accepted, side view | Stranger, side view |
|---|---|---|---|---|---|---|
| 8.5 | Normal | 98.36 | 0 | 31.64 | 98 | 62 |
| 8.5 | Bag | 86.73 | 0.73 | 16.91 | 88 | 26 |
| 8.5 | Coat | 89.45 | 1.27 | 16.91 | 98 | 22 |
| **8 (now)** | **Normal** | **96.00** | **0** | **16.55** | **98** | **38** |
| **8 (now)** | **Bag** | **80.00** | **0.18** | **7.45** | **88** | **8** |
| **8 (now)** | **Coat** | **86.00** | **0.36** | **4.18** | **96** | **8** |
| 7.2 | Normal | 90.55 | 0 | 2.73 | 94 | 6 |
| 7.2 | Bag | 63.27 | 0 | 0.91 | 84 | 0 |
| 7.2 | Coat | 70.18 | 0.18 | 0.18 | 90 | 0 |

All angles is 550 clips per test, side view 50. At the side view one clip is 2 points and no wrong
person was accepted at any of the three lines.

With **normal clips only** in the Archive, line 8 accepts 96.00 / 52.55 / 14.36% over all angles: the
bag and coat walks are found (Rank-1 above) but are too far away to pass the line.

### Every line from 7 to 8.5 (new people, Archive as built)

"Under the line" counts the right person under the line even when someone else is closer, so it can
be slightly higher than "accepted" above.

All angles (550 clips per test):

| Line | Right person under the line: normal / bag / coat / all | Stranger gets a name: normal / bag / coat / all |
|---|---|---|
| 7.00 | 88.91 / 59.64 / 63.82 / 70.79 | 0.91 / 0.73 / 0 / 0.55 |
| 7.25 | 90.55 / 64.18 / 70.91 / 75.21 | 2.91 / 1.09 / 0.18 / 1.39 |
| 7.50 | 92.91 / 70.18 / 76.36 / 79.82 | 4.73 / 2.18 / 0.55 / 2.48 |
| 7.75 | 94.91 / 76.00 / 80.55 / 83.82 | 9.64 / 4.36 / 2.00 / 5.33 |
| **8.00** | **96.00 / 80.00 / 86.18 / 87.39** | **16.55 / 7.45 / 4.18 / 9.39** |
| 8.25 | 97.27 / 83.82 / 88.00 / 89.70 | 23.09 / 11.45 / 8.18 / 14.24 |
| 8.50 | 98.36 / 86.91 / 90.36 / 91.88 | 31.64 / 16.91 / 16.91 / 21.82 |

Side view (50 clips per test):

| Line | Right person under the line: normal / bag / coat / all | Stranger gets a name: normal / bag / coat / all |
|---|---|---|
| 7.00 | 94 / 84 / 88 / 88.67 | 0 / 0 / 0 / 0 |
| 7.25 | 94 / 84 / 90 / 89.33 | 6 / 0 / 0 / 2.00 |
| 7.50 | 96 / 86 / 94 / 92.00 | 12 / 2 / 0 / 4.67 |
| 7.75 | 98 / 88 / 96 / 94.00 | 20 / 4 / 2 / 8.67 |
| **8.00** | **98 / 88 / 96 / 94.00** | **38 / 8 / 8 / 18.00** |
| 8.25 | 98 / 88 / 96 / 94.00 | 52 / 12 / 14 / 26.00 |
| 8.50 | 98 / 88 / 98 / 94.67 | 62 / 26 / 22 / 36.67 |

1% of strangers get a name at 7.17 over all angles and at 7.16 at the side view, which is where the
earlier line of 7.2 came from.

### Typical distances (new people, Archive as built)

| | Normal | Bag | Coat |
|---|---|---|---|
| To the right person, median (90% under), all angles | 5.72 (7.14) | 6.72 (8.77) | 6.57 (8.44) |
| To the right person, median, side view | 4.81 | 6.01 | 5.63 |
| A stranger's nearest person, median (1% under), all angles | 8.92 (7.08) | 9.39 (7.29) | 9.33 (7.60) |

### Limits of these numbers

- **Same bag and coat:** `bg-02` and `cl-02` were recorded in the same CASIA-B setup as the `bg-01`
  and `cl-01` in the Archive. A different bag or coat may do worse.
- **Not the benchmark:** the standard test doesn't allow bag and coat clips in the Archive, so these
  are not the model's benchmark numbers. The normal-only rows are the closer comparison: coat 62.18
  here against 72.36 on the benchmark. That gap is the website's framing plus YOLO's masks.

## 5. Through the real website backend (2026-11-05)

- **Held-out CASIA-B clips:** 5 side-view clips from `datasets\tests` (bag and coat, people 075, 076
  and 109). All 5 named the right person, at distances 5.55-6.74; the next person was 8.79 or more
  away. Each took 1.5-2.1 s on the laptop's GPU.
- **Phone clips:** the team's 13 phone clips, none of them in this Archive, were all unknown. The
  nearest CASIA-B person was 9.43-13.09 away.

## 6. Phone recordings of the team (2026-10-30)

10 people, one normal side-view walk each, in a small Archive of their own. Three test clips, two
with a bag and one in a coat:

| Test clip | Closest | Right person | Next person |
|---|---|---|---|
| Bag, person A | Right, 12.66 | 12.66 | 14.52 |
| Coat, person B | Right, 11.24 | 11.24 | 12.30 |
| Bag, person C | **Wrong**, 10.53 | 10.97 (second) | |

- **Different people:** in the same Archive they were 9.80-18.60 apart (median 12.66).
- **No line works:** the right person (10.97-12.66) is as far away as different people, so no match
  line separates them.
- **Score normalisation:** judging each distance against that person's usual distances was tried
  and rejected, because the wrong clip stayed wrong.
- **Missing:** two normal phone walks of the same person, to see how far apart one person's walks
  are on phone video.

## 7. Checks on the website itself

- **Same answer every time:** the same clip gives the identical 4096-number vector twice (CUDA,
  about 1.3 s once warm).
- **Tests:** 88 backend tests and ruff lint; frontend lint and build. All run on every pull request.
- **Phone layout (2026-11-05):**
  - Phones: headless Edge at 320, 360, 390 and 430 px wide, a phone on its side and a small tablet
    (78 captures). Nothing wider than the screen and no console errors.
  - PC, before and after: 70 captures at 7 window sizes. Only the random case number's tag moved,
    40 of 35,812 element measurements.
