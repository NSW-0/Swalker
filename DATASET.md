# The dataset: CASIA-B

CASIA-B is a public gait dataset from the Institute of Automation, Chinese Academy of Sciences
(Yu, Tan and Tan, ICPR 2006). It is the benchmark GLTNet was trained on and the data the website's
Archive holds now. The people are not known to us; the first names in the Archive are made up.

## What is in it

124 people, filmed indoors at 320×240 and 25 fps. Each recording is filmed from 11 camera angles
(000, 018, ... 180; 090 is the side view), so each person has 121 videos.

| Kind | In the file name | Recordings | Videos per person | Total videos |
|---|---|---|---|---|
| Normal walking | `nm` | 01-06 | 66 | 8,184 |
| Carrying a bag | `bg` | 01-02 | 22 | 2,728 |
| Wearing a coat | `cl` | 01-02 | 22 | 2,728 |
| Empty background, nobody walking (not used) | `bkgrd` | - | 11 | 1,364 |

`075-cl-02-090.avi` means person 075, in a coat, recording 02, side view.

## How we use it

- **Training (on Kaggle, not in this repo):** silhouette images of people 001-074. People 075-124
  are the test set (the LT protocol), so **only 075-124 count for accuracy claims**.
- **Framing checks (`backend/scripts/casia_eval.py`, on Kaggle):** the same silhouette images.
- **The website's Archive (`backend/scripts/casia_videos.py`, on this laptop):** the raw videos, run
  through the same YOLO pipeline as an uploaded clip.

| Goes to | What | Videos |
|---|---|---|
| Archive | `nm` 01-06, `bg-01`, `cl-01`: 88 clips per person | 10,909 |
| `datasets\tests\<name>\` | `bg-02`, `cl-02`: 22 clips per person, for testing | 2,728 |
| Could not be read | `109-nm-01` at 126, 144 and 162 degrees (Mohsen has 85 clips) | 3 |

The Archive keeps vectors and small silhouettes, not videos. The videos, the Archive and the vector
cache are not in this repo. `datasets\dataset_report.md` lists every person, and the commands to
rebuild everything are at the top of `casia_videos.py`.

## Results on it

Measured 2026-11-05 at the match line of 8, on new people (075-124), with the Archive as built. A
stranger is a test clip with its own person taken out of the comparison.

| Test clips | Closest is right (Rank-1) | Right person accepted | Stranger gets a name |
|---|---|---|---|
| Normal (`nm-06`), side view, 50 clips | 100% | 98% | 38% |
| Bag (`bg-02`), side view, 50 clips | 98% | 88% | 8% |
| Coat (`cl-02`), side view, 50 clips | 100% | 96% | 8% |
| Normal (`nm-06`), all angles, 550 clips | 99.82% | 96% | 16.55% |
| Bag (`bg-02`), all angles, 550 clips | 96.55% | 80% | 7.45% |
| Coat (`cl-02`), all angles, 550 clips | 96.91% | 86% | 4.18% |

Reading these numbers:

- `nm-06` was left out of the Archive for its test, so it is not compared with itself.
- **This is not the model's benchmark accuracy.** The Archive already holds a bag and a coat clip of
  each person, which the standard protocol does not allow. Compare 99.27 / 93.26 / 72.36 (normal / bag
  / coat, standard protocol) with the normal-clips-only Archive here: 99.64 / 92.91 / 62.18.
- `bg-02` and `cl-02` come from the same recording setup as `bg-01` and `cl-01`, so a different bag
  or coat may do worse.
- Phone video has not been measured against phone video at this Archive size. See `SESSION_NOTES.md`.
