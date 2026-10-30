import json

import numpy as np
import pytest

from app.errors import ApiError
from app.gallery.store import GalleryMeta, GalleryStore, row_owners, validate_name
from app.ml.embed import FEATURE_DIM

META = GalleryMeta(preprocess_mode="casia", checkpoint="a.pth")


def vec(seed):
    return np.random.default_rng(seed).normal(size=FEATURE_DIM).astype(np.float32)


def sils(n=3):
    return [np.full((64, 44), 0.5, np.float32) for _ in range(n)]


def test_add_persists_vectors_pngs_and_metadata(tmp_path):
    store = GalleryStore(tmp_path, META)
    person = store.add("  Nabeel   Sawafta ", "C:/videos/walk1.mp4", vec(1), sils())
    assert person.personId == "Nabeel Sawafta"
    [clip] = person.clips
    assert clip.videoPath == "walk1.mp4"  # file name only, no local paths leak out
    assert len(clip.silhouetteFiles) == 3
    assert all((tmp_path / "silhouettes" / f).exists() for f in clip.silhouetteFiles)

    reloaded = GalleryStore(tmp_path, META)
    assert len(reloaded) == 1
    found, vectors = reloaded.find("Nabeel Sawafta")
    assert found == person
    np.testing.assert_array_equal(vectors, vec(1)[None, :])
    # Only numbers, images and JSON are stored, never the video.
    suffixes = {p.suffix for p in tmp_path.rglob("*") if p.is_file()}
    assert suffixes <= {".json", ".npy", ".png"}


def test_several_clips_per_person_keep_rows_matched(tmp_path):
    store = GalleryStore(tmp_path, META)
    store.add("A", "a1.mp4", vec(1), sils(1))
    store.add("B", "b1.mp4", vec(2), sils(1))
    store.add_clip("A", "a_bag.mp4", vec(3), sils(2))
    store.add_clip("B", "b_coat.mp4", vec(4), sils(1))
    store.add_clip("A", "a_coat.mp4", vec(5), sils(1))

    reloaded = GalleryStore(tmp_path, META)
    assert len(reloaded) == 2 and reloaded.features.shape == (5, FEATURE_DIM)
    a, a_vectors = reloaded.find("A")
    assert [c.videoPath for c in a.clips] == ["a1.mp4", "a_bag.mp4", "a_coat.mp4"]
    np.testing.assert_array_equal(a_vectors, np.stack([vec(1), vec(3), vec(5)]))
    np.testing.assert_array_equal(reloaded.find("B")[1], np.stack([vec(2), vec(4)]))
    assert row_owners(reloaded.people) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    # Each clip has its own silhouettes.
    assert len({f for c in a.clips for f in c.silhouetteFiles}) == 4

    with pytest.raises(ApiError) as err:
        store.add_clip("Nobody", "x.mp4", vec(6), [])
    assert err.value.status == 404


def test_remove_clip(tmp_path):
    store = GalleryStore(tmp_path, META)
    store.add("A", "a1.mp4", vec(1), sils(1))
    a = store.add_clip("A", "a2.mp4", vec(2), sils(1))
    store.add("B", "b1.mp4", vec(3), sils(1))
    first, second = a.clips

    after = store.remove_clip("A", first.clipId)
    assert [c.clipId for c in after.clips] == [second.clipId]
    assert not any((tmp_path / "silhouettes" / f).exists() for f in first.silhouetteFiles)
    reloaded = GalleryStore(tmp_path, META)
    np.testing.assert_array_equal(reloaded.find("A")[1], vec(2)[None, :])
    np.testing.assert_array_equal(reloaded.find("B")[1], vec(3)[None, :])

    with pytest.raises(ApiError) as err:
        store.remove_clip("A", second.clipId)  # their only clip: delete the person instead
    assert err.value.status == 409 and err.value.code == "last_clip"
    with pytest.raises(ApiError) as err:
        store.remove_clip("A", "nope")
    assert err.value.status == 404


def test_version_1_archives_are_read_as_one_clip_each(tmp_path):
    (tmp_path / "silhouettes" / "a").mkdir(parents=True)
    (tmp_path / "silhouettes" / "a" / "00.png").write_bytes(b"png")
    np.save(tmp_path / "features.npy", np.stack([vec(1), vec(2)]))
    old = {
        "meta": {"preprocess_mode": "casia", "checkpoint": "a.pth"},
        "entries": [
            {"personId": "A", "slug": "a", "videoPath": "a.mp4", "addedAt": "2026-10-30T20:00:00+00:00",
             "silhouetteFiles": ["a/00.png"]},
            {"personId": "B", "slug": "b", "videoPath": "b.mp4", "addedAt": "2026-10-30T20:01:00+00:00",
             "silhouetteFiles": []},
        ],
    }  # fmt: skip
    (tmp_path / "index.json").write_text(json.dumps(old), encoding="utf-8")

    store = GalleryStore(tmp_path, META)
    a, a_vectors = store.find("A")
    assert a.addedAt == "2026-10-30T20:00:00+00:00"
    assert [(c.videoPath, c.silhouetteFiles) for c in a.clips] == [("a.mp4", ["a/00.png"])]
    np.testing.assert_array_equal(a_vectors, vec(1)[None, :])

    # The next change writes version 2; old PNG paths keep working.
    store.add_clip("A", "a_bag.mp4", vec(3), sils(1))
    doc = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert doc["version"] == 2 and "entries" not in doc
    reloaded = GalleryStore(tmp_path, META)
    assert reloaded.find("A")[0].clips[0].silhouetteFiles == ["a/00.png"]
    np.testing.assert_array_equal(reloaded.find("B")[1], vec(2)[None, :])


def test_duplicate_names_are_rejected_case_insensitively(tmp_path):
    store = GalleryStore(tmp_path, META)
    store.add("Ahmad", "a.mp4", vec(1), [])
    with pytest.raises(ApiError) as err:
        store.add("ahmad", "b.mp4", vec(2), [])
    assert err.value.status == 409 and err.value.code == "name_taken"


def test_unicode_names_get_distinct_ascii_folders(tmp_path):
    store = GalleryStore(tmp_path, META)
    a = store.add("\u0643\u0631\u064a\u0645", "a.mp4", vec(1), sils(1))  # Arabic "Kareem"
    b = store.add("\u0646\u0628\u064a\u0644", "b.mp4", vec(2), sils(1))  # Arabic "Nabeel"
    assert a.slug != b.slug and a.slug.isascii() and b.slug.isascii()


@pytest.mark.parametrize("bad", ["", "   ", "../etc", "a/b", "x" * 41, "<script>"])
def test_invalid_names(bad):
    with pytest.raises(ApiError):
        validate_name(bad)


def test_mismatched_preprocessing_or_checkpoint_is_incompatible(tmp_path):
    GalleryStore(tmp_path, META).add("Kareem", "k.mp4", vec(1), [])
    other = GalleryStore(tmp_path, GalleryMeta(preprocess_mode="crop", checkpoint="a.pth"))
    assert not other.compatible
    with pytest.raises(ApiError) as err:
        other.add("New", "n.mp4", vec(2), [])
    assert err.value.code == "gallery_incompatible"
    with pytest.raises(ApiError) as err:
        other.add_clip("Kareem", "k2.mp4", vec(3), [])
    assert err.value.code == "gallery_incompatible"
    assert not GalleryStore(tmp_path, GalleryMeta(preprocess_mode="casia", checkpoint="b.pth")).compatible


def test_empty_gallery_adopts_current_settings(tmp_path):
    GalleryStore(tmp_path, META)
    assert GalleryStore(tmp_path, GalleryMeta(preprocess_mode="crop", checkpoint="z.pth")).compatible


def test_corrupt_row_count_is_detected(tmp_path):
    store = GalleryStore(tmp_path, META)
    store.add("A", "a.mp4", vec(1), [])
    doc = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    doc["people"][0]["clips"].append({**doc["people"][0]["clips"][0], "clipId": "extra"})
    (tmp_path / "index.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(RuntimeError):
        GalleryStore(tmp_path, META)


def test_rename_keeps_vectors_and_silhouettes(tmp_path):
    store = GalleryStore(tmp_path, META)
    person = store.add("Ahmad", "a.mp4", vec(1), sils(2))
    store.add_clip("Ahmad", "a2.mp4", vec(2), sils(1))
    renamed = store.rename("Ahmad", "  Ahmad   Karmi ")
    assert renamed.personId == "Ahmad Karmi"
    assert renamed.clips[0] == person.clips[0] and len(renamed.clips) == 2
    reloaded = GalleryStore(tmp_path, META)
    assert reloaded.find("Ahmad") is None
    np.testing.assert_array_equal(reloaded.find("Ahmad Karmi")[1], np.stack([vec(1), vec(2)]))


def test_rename_rules(tmp_path):
    store = GalleryStore(tmp_path, META)
    store.add("Ahmad", "a.mp4", vec(1), [])
    store.add("Kareem", "k.mp4", vec(2), [])
    with pytest.raises(ApiError) as err:
        store.rename("Ahmad", "kareem")
    assert err.value.status == 409
    with pytest.raises(ApiError) as err:
        store.rename("Nobody", "X")
    assert err.value.status == 404
    with pytest.raises(ApiError):
        store.rename("Ahmad", "../x")
    assert store.rename("Ahmad", "AHMAD").personId == "AHMAD"  # changing only the case is fine


def test_remove_drops_all_their_rows_and_silhouettes(tmp_path):
    store = GalleryStore(tmp_path, META)
    store.add("A", "a.mp4", vec(1), sils(2))
    b = store.add("B", "b.mp4", vec(2), sils(2))
    store.add_clip("B", "b2.mp4", vec(4), sils(1))
    store.add("C", "c.mp4", vec(3), sils(2))
    store.remove("B")
    assert not (tmp_path / "silhouettes" / b.slug).exists()
    reloaded = GalleryStore(tmp_path, META)
    assert [p.personId for p in reloaded.people] == ["A", "C"]
    np.testing.assert_array_equal(reloaded.find("C")[1], vec(3)[None, :])  # rows stay matched to names
    with pytest.raises(ApiError) as err:
        store.remove("B")
    assert err.value.status == 404


def test_old_archives_without_a_frame_rate_are_incompatible(tmp_path):
    old = GalleryMeta(preprocess_mode="casia", checkpoint="a.pth")  # target_fps defaults to 0
    GalleryStore(tmp_path, old).add("A", "a.mp4", vec(1), [])
    assert not GalleryStore(
        tmp_path, GalleryMeta(preprocess_mode="casia", checkpoint="a.pth", target_fps=25)
    ).compatible


def test_name_from_filename():
    from app.gallery.store import name_from_filename

    assert name_from_filename("hamed_bag.mp4") == "hamed_bag"
    assert name_from_filename("IMG (2)+final.MOV") == "IMG 2 final"
    assert name_from_filename("نبيل.mp4") == "نبيل"
    assert name_from_filename("((().mp4") == ""
    assert len(name_from_filename("x" * 80 + ".mp4")) == 40
