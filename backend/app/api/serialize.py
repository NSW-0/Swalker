import base64

import numpy as np

from app.gallery.store import GalleryEntry, encode_png

MEDIA_PREFIX = "/api/media"
PREVIEW_LEN = 8


def silhouette_urls(entry: GalleryEntry) -> list[str]:
    return [f"{MEDIA_PREFIX}/{f}" for f in entry.silhouetteFiles]


def thumbnail_url(entry: GalleryEntry) -> str | None:
    urls = silhouette_urls(entry)
    return urls[len(urls) // 2] if urls else None


def png_data_url(silhouette: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(encode_png(silhouette)).decode()


def entry_summary(entry: GalleryEntry, vector: np.ndarray) -> dict:
    return {
        "personId": entry.personId,
        "videoPath": entry.videoPath,
        "addedAt": entry.addedAt,
        "vectorPreview": [round(float(v), 6) for v in vector[:PREVIEW_LEN]],
        "l2Norm": float(np.linalg.norm(vector)),
        "featureDim": int(vector.shape[0]),
        "thumbnailUrl": thumbnail_url(entry),
    }


def entry_detail(entry: GalleryEntry, vector: np.ndarray) -> dict:
    return {**entry_summary(entry, vector), "silhouetteUrls": silhouette_urls(entry)}
