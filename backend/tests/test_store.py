import json

import numpy as np
import pytest

from app.errors import ApiError
from app.gallery.store import GalleryMeta, GalleryStore, validate_name
from app.ml.embed import FEATURE_DIM

META = GalleryMeta(preprocess_mode="casia", checkpoint="a.pth")


def vec(seed):
    return np.random.default_rng(seed).normal(size=FEATURE_DIM).astype(np.float32)


def sils(n=3):
    return [np.full((64, 44), 0.5, np.float32) for _ in range(n)]


def test_add_persists_vectors_pngs_and_metadata(tmp_path):
    store = GalleryStore(tmp_path, META)
    entry = store.add("  Nabeel   Sawafta ", "C:/videos/walk1.mp4", vec(1), sils())
    assert entry.personId == "Nabeel Sawafta"
    assert entry.videoPath == "walk1.mp4"  # file name only, no local paths leak out
    assert len(entry.silhouetteFiles) == 3
    assert all((tmp_path / "silhouettes" / f).exists() for f in entry.silhouetteFiles)

    reloaded = GalleryStore(tmp_path, META)
    assert len(reloaded) == 1
    found_entry, found_vec = reloaded.find("Nabeel Sawafta")
    assert found_entry == entry
    np.testing.assert_array_equal(found_vec, vec(1))
    # Only numbers, images and JSON are stored, never the video.
    suffixes = {p.suffix for p in tmp_path.rglob("*") if p.is_file()}
    assert suffixes <= {".json", ".npy", ".png"}


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
    assert not GalleryStore(tmp_path, GalleryMeta(preprocess_mode="casia", checkpoint="b.pth")).compatible


def test_empty_gallery_adopts_current_settings(tmp_path):
    GalleryStore(tmp_path, META)
    assert GalleryStore(tmp_path, GalleryMeta(preprocess_mode="crop", checkpoint="z.pth")).compatible


def test_corrupt_row_count_is_detected(tmp_path):
    store = GalleryStore(tmp_path, META)
    store.add("A", "a.mp4", vec(1), [])
    doc = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    doc["entries"].append({**doc["entries"][0], "personId": "B", "slug": "b"})
    (tmp_path / "index.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(RuntimeError):
        GalleryStore(tmp_path, META)
