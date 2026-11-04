import json
from pathlib import Path

import numpy as np
import pytest

from app.errors import ApiError
from app.gallery.store import GalleryMeta, GalleryStore, validate_name
from app.ml.embed import FEATURE_DIM
from scripts.casia_videos import (
    NAMES,
    count_people,
    evaluate_clips,
    find_videos,
    make_clips,
    nearest_per_person,
    parse_video,
    person_name,
)


def test_parse_video_and_held_out_recordings():
    video = parse_video(Path("075-BG-02-090.avi"))
    assert (video.subject, video.cond, video.seq, video.view) == (75, "bg", 2, 90)
    assert video.held_out and parse_video(Path("075-cl-02-000.avi")).held_out
    assert not parse_video(Path("075-bg-01-090.avi")).held_out
    assert not parse_video(Path("075-nm-06-090.avi")).held_out
    assert parse_video(Path("075-bkgrd-090.avi")) is None


def test_every_person_gets_their_own_valid_name():
    assert person_name(1) == "Ahmad" and person_name(2) == "Omar"
    names = [person_name(subject) for subject in range(1, 125)]
    assert len({n.casefold() for n in names}) == 124
    assert all(validate_name(n) == n for n in names)
    # Not the names of the team, who were in the Archive before and would be confusing.
    team = {"abdr", "hamed", "hodayfah", "ibrahim", "eyad", "mohamad", "mostafa", "nabeel", "raid"}
    assert not team & {n.casefold() for n in NAMES}


def test_count_people_says_where_each_kind_of_video_goes(tmp_path):
    folder = tmp_path / "DatasetB-1" / "video"
    folder.mkdir(parents=True)
    for name in ["nm-01-090", "nm-06-090", "bg-01-090", "bg-02-090", "cl-01-090", "cl-02-090", "bkgrd-090"]:
        (folder / f"002-{name}.avi").write_bytes(b"")
    [row] = count_people(tmp_path, tmp_path / "no_cache_yet")
    assert (row["number"], row["name"]) == ("002", "Omar")
    assert (row["normal"], row["bag"], row["coat"], row["empty background"]) == (2, 2, 2, 1)
    assert (row["normal in Archive"], row["bag in Archive"], row["coat in Archive"]) == (2, 1, 1)
    assert (row["bag in tests"], row["coat in tests"], row["could not be read"]) == (1, 1, 0)


def test_find_videos_only_reads_video_folders(tmp_path):
    for rel in ["B-1/video/001-nm-01-090.avi", "B-1/video/001-bkgrd-090.avi", "B-2/video/063-cl-02-090.avi"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"")
    (tmp_path / "test_clips/Person 063").mkdir(parents=True)
    (tmp_path / "test_clips/Person 063/063-bg-02-090.avi").write_bytes(b"")  # a copy, not a source
    people = find_videos(tmp_path)
    assert sorted(people) == [1, 63]
    assert [v.path.name for v in people[63]] == ["063-cl-02-090.avi"]


def test_nearest_per_person_takes_each_persons_closest_clip():
    gallery = np.array([[9.0, 0], [1.0, 0], [5.0, 0], [3.0, 0]], np.float32)
    d, people = nearest_per_person(np.zeros((1, 2), np.float32), gallery, np.array([1, 1, 2, 2]))
    assert people.tolist() == [1, 2]
    assert d[0] == pytest.approx([1.0, 3.0])


def fake_clips():
    """Each person has a point of their own. Normal clips are right on it, bag clips 2 away from it
    in one direction, coat clips 2 away in another; people are about 141 apart."""
    names, vectors = [], []
    for subject in (1, 74, 75, 76):
        centre = np.zeros(64, np.float32)
        centre[subject % 32] = 100.0
        for cond, seqs, offset in (("nm", range(1, 7), None), ("bg", (1, 2), 40), ("cl", (1, 2), 50)):
            for seq in seqs:
                for view in (0, 90):
                    v = centre.copy()
                    if offset:
                        v[offset] += 2.0
                    names.append(f"{subject:03d}-{cond}-{seq:02d}-{view:03d}.avi")
                    vectors.append(v)
    return make_clips(names, np.stack(vectors))


def test_evaluate_shows_what_extra_clips_do_and_separates_new_people():
    report = evaluate_clips(fake_clips(), threshold=1.0)
    new = "new people (075-124, not in training)"
    only_normal = report["results"][f"{new} | Archive with normal clips only | bag (bg-02) | all angles"]
    with_bags = report["results"][f"{new} | Archive with normal + bg-01 + cl-01 | bag (bg-02) | all angles"]
    # 2 people x 2 angles. Bag vs normal is 2 away (past the line); bag vs bag is 0.
    assert only_normal["clips"] == 4 and only_normal["rank1"] == 100 and only_normal["accepted"] == 0
    assert with_bags["accepted"] == 100 and with_bags["wrong_person_accepted"] == 0
    assert with_bags["stranger_matched"] == 0
    seen = report["results"][
        "people from training (001-074) | Archive with normal clips only | bag (bg-02) | all angles"
    ]
    assert seen["clips"] == 4
    # nm-06 must not be compared with itself: it is left out of the gallery, so it matches nm-01..05.
    normal = report["results"][f"{new} | Archive with normal clips only | normal (nm-06) | side view"]
    assert normal["clips"] == 2 and normal["rank1"] == 100
    json.dumps(report)  # no arrays left: this gets saved
    table = report["threshold table, all angles"]
    assert [row["strangers"] for row in table] == pytest.approx([0.1, 1, 5, 10])


def test_add_many_writes_once_and_matches_rows(tmp_path):
    meta = GalleryMeta(preprocess_mode="casia", checkpoint="a.pth")
    store = GalleryStore(tmp_path, meta)
    sils = [np.full((64, 44), 0.5, np.float32)]
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(3, FEATURE_DIM)).astype(np.float32)
    store.add_many("Person 001", [("a.avi", vectors[0], sils), ("b.avi", vectors[1], sils)], persist=False)
    store.add_many("Person 002", [("c.avi", vectors[2], sils)], persist=False)
    assert not (tmp_path / "index.json").exists()  # nothing written until save()
    store.save()

    reloaded = GalleryStore(tmp_path, meta)
    person, found = reloaded.find("Person 001")
    assert [c.videoPath for c in person.clips] == ["a.avi", "b.avi"]
    np.testing.assert_array_equal(found, vectors[:2])
    np.testing.assert_array_equal(reloaded.find("Person 002")[1], vectors[2:])
    with pytest.raises(ApiError):
        store.add_many("person 001", [("d.avi", vectors[0], sils)])
    with pytest.raises(ValueError):
        store.add_many("Person 003", [])
