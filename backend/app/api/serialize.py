import base64

import numpy as np

from app.gallery.store import Clip, Person, encode_png

MEDIA_PREFIX = "/api/media"
PREVIEW_LEN = 8


def silhouette_urls(clip: Clip) -> list[str]:
    return [f"{MEDIA_PREFIX}/{f}" for f in clip.silhouetteFiles]


def thumbnail_url(clip: Clip) -> str | None:
    urls = silhouette_urls(clip)
    return urls[len(urls) // 2] if urls else None


def png_data_url(silhouette: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(encode_png(silhouette)).decode()


def _vector_info(vector: np.ndarray) -> dict:
    return {
        "vectorPreview": [round(float(v), 6) for v in vector[:PREVIEW_LEN]],
        "l2Norm": float(np.linalg.norm(vector)),
        "featureDim": int(vector.shape[0]),
    }


def person_summary(person: Person, vectors: np.ndarray) -> dict:
    """The Archive card. videoPath, the vector fields and the thumbnail are the first clip's."""
    first = person.clips[0]
    return {
        "personId": person.personId,
        "videoPath": first.videoPath,
        "addedAt": person.addedAt,
        "clipCount": len(person.clips),
        **_vector_info(vectors[0]),
        "thumbnailUrl": thumbnail_url(first),
    }


def person_detail(person: Person, vectors: np.ndarray) -> dict:
    return {
        **person_summary(person, vectors),
        "silhouetteUrls": silhouette_urls(person.clips[0]),
        "clips": [
            {
                "clipId": clip.clipId,
                "videoPath": clip.videoPath,
                "addedAt": clip.addedAt,
                **_vector_info(vector),
                "silhouetteUrls": silhouette_urls(clip),
            }
            for clip, vector in zip(person.clips, vectors, strict=True)
        ],
    }
