"""Plain Real-ESRGAN upscale of your picks.

Deliberately NOT a "creative"/diffusion upscaler and NOT face restoration (GFPGAN/CodeFormer):
those re-draw skin and eyes, which is exactly where the uncanny look comes from.
"""

import json
from pathlib import Path

import fal_client
import httpx
import typer

from .config import UPSCALE_PRICE_ESTIMATE, UPSCALER, run_dir


def upscale(run: str, picks: list[str] | None, top: int | None, scale: int, yes: bool) -> None:
    d = run_dir(run)
    manifest = json.loads((d / "manifest.json").read_text())

    if picks:
        # picks look like "studio-charcoal/1_s1.00.png" (as the contact sheet writes them)
        chosen = [m for m in manifest if any(Path(m["file"]).as_posix().endswith(p) for p in picks)]
    else:
        pool = [m for m in manifest if m.get("keep")]
        pool.sort(key=lambda m: -m.get("score", 0))
        chosen = pool[:top] if top else pool
    if not chosen:
        raise typer.BadParameter("Nothing to upscale — pass file names, or run `selfshot score` first.")

    if not yes:
        typer.confirm(f"Upscale {len(chosen)} images {scale}x, ~${len(chosen) * UPSCALE_PRICE_ESTIMATE:.2f}?",
                      abort=True)

    out_dir = d / "final"
    out_dir.mkdir(exist_ok=True)
    for m in chosen:
        src = Path(m["file"])
        url = fal_client.upload_file(str(src))
        result = fal_client.subscribe(
            UPSCALER,
            arguments={"image_url": url, "scale": scale, "model": "RealESRGAN_x4plus", "output_format": "png"},
        )
        dest = out_dir / f"{m['style']}_{src.stem}_x{scale}.png"
        dest.write_bytes(httpx.get(result["image"]["url"], timeout=120, follow_redirects=True).content)
        m["final"] = str(dest)
        typer.echo(f"  {dest}")

    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    typer.secho(f"Finals in {out_dir}", fg="green")
