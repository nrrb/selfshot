"""Face detection + ArcFace embeddings via InsightFace, on CPU (fine on an 8 GB MacBook Air).

The buffalo_l model pack (~280 MB) downloads to ~/.insightface on first use.
InsightFace's pretrained models are licensed for non-commercial research — fine for personal use.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


@lru_cache(maxsize=1)
def _app():
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def detect(path: Path):
    """Return the list of detected faces (InsightFace Face objects)."""
    img = np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1]  # RGB -> BGR
    return _app().get(img)


def _area(face) -> float:
    x1, y1, x2, y2 = face.bbox
    return float((x2 - x1) * (y2 - y1))


def embed(path: Path) -> np.ndarray | None:
    """Normalized embedding of the largest face, or None if no face is found."""
    faces = detect(path)
    if not faces:
        return None
    return max(faces, key=_area).normed_embedding


def similarity(emb: np.ndarray, refs: np.ndarray) -> tuple[float, float]:
    """(mean, max) cosine similarity of one embedding against the reference set."""
    sims = refs @ emb
    return float(sims.mean()), float(sims.max())
