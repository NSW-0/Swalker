"""Uploaded videos go to a temp file for the length of one request, then are deleted."""

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from fastapi import UploadFile

from app.errors import ApiError

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
CHUNK = 1024 * 1024


@contextmanager
def saved_upload(video: UploadFile, max_mb: int) -> Iterator[Path]:
    name = video.filename or ""
    suffix = Path(name).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise ApiError(
            "unsupported_type",
            f"'{name or 'that file'}' isn't a supported video. Use {', '.join(VIDEO_EXTENSIONS)}.",
            415,
        )

    limit = max_mb * 1024 * 1024
    # delete=False: on Windows OpenCV can't open a file another handle still holds.
    fd, tmp_name = tempfile.mkstemp(suffix=suffix, prefix="swalker_")
    tmp = Path(tmp_name)
    try:
        written = 0
        with os.fdopen(fd, "wb") as out:
            while chunk := video.file.read(CHUNK):
                written += len(chunk)
                if written > limit:
                    raise ApiError("file_too_large", f"The video is larger than {max_mb} MB.", 413)
                out.write(chunk)
        if written == 0:
            raise ApiError("unreadable_video", "The uploaded file is empty.", 422)
        yield tmp
    finally:
        tmp.unlink(missing_ok=True)
