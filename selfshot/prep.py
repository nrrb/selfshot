"""Turn a folder of your photos into a clean training zip + a reference face set for scoring."""

import zipfile
from pathlib import Path

import numpy as np
import typer
from PIL import Image, ImageOps

from . import faces
from .config import run_dir

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # HEIC support is optional
    pass

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
MAX_SIDE = 2048
MIN_SIDE = 1024


def prepare(photos: Path, run: str, trigger: str, check_faces: bool = True) -> Path:
    out = run_dir(run)
    clean = out / "dataset"
    clean.mkdir(exist_ok=True)
    for old in clean.iterdir():
        old.unlink()

    sources = sorted(p for p in photos.iterdir() if p.suffix.lower() in EXTS)
    if len(sources) < 10:
        typer.secho(f"Only {len(sources)} photos found; 15-25 varied photos train noticeably better.", fg="yellow")

    refs = []
    for i, src in enumerate(sources):
        img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")  # fixes sideways phone photos
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        if min(img.size) < MIN_SIDE:
            typer.secho(f"  {src.name}: {img.size[0]}x{img.size[1]} is small; may teach the model blur", fg="yellow")

        dest = clean / f"{i:03d}.jpg"
        img.save(dest, quality=95)

        caption = src.with_suffix(".txt")
        if caption.exists():
            text = caption.read_text().strip().replace("[trigger]", trigger)
            (clean / f"{i:03d}.txt").write_text(text)

        if check_faces:
            found = faces.detect(dest)
            if not found:
                typer.secho(f"  {src.name}: no face found — fine for a tattoo close-up, otherwise check it",
                            fg="yellow")
            elif len(found) > 1:
                typer.secho(f"  {src.name}: {len(found)} faces detected — crop to just you, or drop it", fg="red")
            else:
                refs.append(found[0].normed_embedding)

    if refs:
        np.save(out / "refs.npy", np.stack(refs))
        typer.echo(f"Saved {len(refs)} reference face embeddings for scoring.")

    zip_path = out / "dataset.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(clean.iterdir()):
            z.write(f, f.name)
    typer.echo(f"Wrote {zip_path} ({len(sources)} images).")
    return zip_path
