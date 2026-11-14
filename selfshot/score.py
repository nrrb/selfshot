"""Score every generation against your real photos with ArcFace, and mark keepers.

The threshold is calibrated on YOU: we first measure how similar your real photos are to
each other (they're never 1.0 — lighting, angle and expression vary), then keep generations
that land within a margin of that baseline. Anything far below it is the "AI cousin" look.
"""

import json

import numpy as np
import typer

from . import faces
from .config import run_dir


def baseline(refs: np.ndarray) -> float:
    """Mean leave-one-out similarity among your real photos."""
    sims = refs @ refs.T
    n = len(refs)
    return float((sims.sum() - np.trace(sims)) / (n * (n - 1)))


def score(run: str, threshold: float | None, margin: float, rescore: bool) -> None:
    d = run_dir(run)
    refs_path, manifest_path = d / "refs.npy", d / "manifest.json"
    if not refs_path.exists():
        raise typer.BadParameter("No refs.npy — run `selfshot prep` (with face checks on) first.")
    if not manifest_path.exists():
        raise typer.BadParameter("No generations yet — run `selfshot generate` first.")

    refs = np.load(refs_path)
    base = baseline(refs) if len(refs) > 1 else 0.6
    cut = threshold if threshold is not None else base * margin
    typer.echo(f"Your real photos agree with each other at {base:.3f}. Keeping generations >= {cut:.3f}.")

    manifest = json.loads(manifest_path.read_text())
    for i, m in enumerate(manifest, 1):
        if "score" in m and not rescore:
            continue
        found = faces.detect(m["file"])
        m["faces"] = len(found)
        if found:
            emb = max(found, key=faces._area).normed_embedding
            m["score"], m["score_max"] = faces.similarity(emb, refs)
        else:
            m["score"], m["score_max"] = 0.0, 0.0
        typer.echo(f"  [{i}/{len(manifest)}] {m['style']} seed {m['seed']}: {m['score']:.3f}")

    for m in manifest:
        m["keep"] = m["faces"] == 1 and m["score"] >= cut

    manifest_path.write_text(json.dumps(manifest, indent=2))
    kept = sum(m["keep"] for m in manifest)
    typer.secho(f"Kept {kept}/{len(manifest)}.", fg="green")

    by_style: dict[str, list[float]] = {}
    for m in manifest:
        by_style.setdefault(m["style"], []).append(m["score"])
    for style, vals in sorted(by_style.items(), key=lambda kv: -np.mean(kv[1])):
        typer.echo(f"  {style:<36} mean {np.mean(vals):.3f}  best {max(vals):.3f}  (n={len(vals)})")
