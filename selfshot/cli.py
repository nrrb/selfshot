import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

from .config import BACKENDS, DEFAULT_TRIGGER

load_dotenv()
app = typer.Typer(no_args_is_help=True, help="Personal AI headshots: prep → train → generate → score → sheet → upscale.")

RunOpt = typer.Option("me", "--run", "-r", help="Run name; everything lives in runs/<name>/")
YesOpt = typer.Option(False, "--yes", "-y", help="Skip the cost confirmation prompt")


def _need_key() -> None:
    if not os.environ.get("FAL_KEY"):
        raise typer.BadParameter("FAL_KEY not set. Copy .env.example to .env and add your key.")


@app.command()
def prep(
    photos: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder of your photos"),
    run: str = RunOpt,
    trigger: str = typer.Option(DEFAULT_TRIGGER, help="Trigger phrase; replaces [trigger] in captions"),
    no_face_check: bool = typer.Option(False, help="Skip face detection (also skips reference embeddings)"),
):
    """Clean photos (EXIF rotation, HEIC, resize), check faces, build dataset.zip + reference embeddings."""
    from .prep import prepare
    prepare(photos, run, trigger, check_faces=not no_face_check)


@app.command()
def train(
    run: str = RunOpt,
    backend: str = typer.Option("portrait", help=f"One of: {', '.join(BACKENDS)}"),
    trigger: str = typer.Option(DEFAULT_TRIGGER),
    steps: Optional[int] = typer.Option(None, help="Training steps (default depends on backend)"),
    yes: bool = YesOpt,
):
    """Train a LoRA on fal. Resumable — rerun to pick up a job in progress."""
    if backend not in BACKENDS:
        raise typer.BadParameter(f"backend must be one of {list(BACKENDS)}")
    _need_key()
    from .train import train as _train
    _train(run, backend, trigger, steps, yes)


PresetsOpt = typer.Option(Path("presets.yaml"), "--presets", exists=True, help="Outfits, scenes and looks")


@app.command()
def generate(
    run: str = RunOpt,
    presets: Path = PresetsOpt,
    per_style: int = typer.Option(8, "--n", help="Images per look (different seeds)"),
    look: Optional[list[str]] = typer.Option(None, "--look", help="Named look(s) from presets (repeatable)"),
    outfit: Optional[list[str]] = typer.Option(None, "--outfit", help="Outfit(s); combined with every --scene"),
    scene: Optional[list[str]] = typer.Option(None, "--scene", help="Scene(s); combined with every --outfit"),
    extra: Optional[list[str]] = typer.Option(None, "--extra", help="Extras like glasses (repeatable)"),
    scale: float = typer.Option(1.0, help="LoRA strength. Lower = more natural, less 'you'; try 0.85–1.1"),
    seed_start: int = typer.Option(1),
    workers: int = typer.Option(2, help="Parallel requests (new fal accounts allow 2 at once)"),
    yes: bool = YesOpt,
):
    """Generate looks × seeds. No flags = every named look. --outfit/--scene = their combinations."""
    _need_key()
    from .generate import generate as _gen
    _gen(run, presets, per_style, look, outfit, scene, extra, scale, seed_start, workers, yes)


@app.command("presets")
def presets_cmd(presets: Path = PresetsOpt):
    """List available looks, outfits, scenes and extras."""
    from .generate import list_presets
    list_presets(presets)


@app.command()
def score(
    run: str = RunOpt,
    threshold: Optional[float] = typer.Option(None, help="Absolute cutoff; default is calibrated on your photos"),
    margin: float = typer.Option(0.85, help="Keep if score >= baseline * margin"),
    rescore: bool = typer.Option(False),
):
    """ArcFace-score every generation against your real photos and mark keepers."""
    from .score import score as _score
    _score(run, threshold, margin, rescore)


@app.command()
def sheet(run: str = RunOpt, no_open: bool = typer.Option(False)):
    """Build and open the HTML contact sheet."""
    from .sheet import build
    typer.echo(build(run, open_browser=not no_open))


@app.command()
def upscale(
    picks: Optional[list[str]] = typer.Argument(None, help="Picks from the contact sheet, e.g. studio-charcoal/3_s1.00.png"),
    run: str = RunOpt,
    top: Optional[int] = typer.Option(None, help="With no picks: upscale the N best keepers"),
    scale: int = typer.Option(2),
    yes: bool = YesOpt,
):
    """Plain 2x Real-ESRGAN upscale (no face restoration)."""
    _need_key()
    from .upscale import upscale as _up
    _up(run, picks, top, scale, yes)


if __name__ == "__main__":
    app()
