"""Over-generate: every chosen look x N seeds. Filtering happens later, like the commercial apps do."""

import itertools
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import fal_client
import httpx
import typer
import yaml

from . import state
from .config import BACKENDS, DEFAULT_SIZE, run_dir


@dataclass
class Look:
    id: str
    scene: str
    outfit: str
    extras: list[str] = field(default_factory=list)

    def prompt(self, presets: dict, trigger: str) -> str:
        sc = presets["scenes"][self.scene]
        wear = presets["outfits"][self.outfit]
        if self.extras:
            wear += ", " + ", ".join(presets["extras"][e] for e in self.extras)
        parts = [trigger, sc["framing"], f"wearing {wear}", sc["setting"].strip(),
                 "natural skin texture", sc.get("expression", "")]
        return ", ".join(" ".join(p.split()) for p in parts if p)


def _check(kind: str, names: list[str], available: dict) -> None:
    unknown = [n for n in names if n not in available]
    if unknown:
        raise typer.BadParameter(f"Unknown {kind}: {unknown}. Available: {', '.join(available)}")


def choose_looks(presets: dict, looks: list[str] | None, outfits: list[str] | None,
                 scenes: list[str] | None, extras: list[str] | None) -> list[Look]:
    extras = extras or []
    _check("extra", extras, presets.get("extras", {}))

    if outfits or scenes:  # build combinations; an unspecified side means "all"
        outfits = outfits or list(presets["outfits"])
        scenes = scenes or list(presets["scenes"])
        _check("outfit", outfits, presets["outfits"])
        _check("scene", scenes, presets["scenes"])
        suffix = "+" + "+".join(extras) if extras else ""
        return [Look(f"{s}__{o}{suffix}", s, o, extras) for s, o in itertools.product(scenes, outfits)]

    named = presets.get("looks", {})
    names = looks or list(named)
    _check("look", names, named)
    chosen = []
    for n in names:
        spec = named[n]
        look_extras = list(dict.fromkeys(spec.get("extras", []) + extras))
        _check("extra", look_extras, presets.get("extras", {}))
        chosen.append(Look(n + ("+" + "+".join(extras) if extras else ""), spec["scene"], spec["outfit"], look_extras))
    return chosen


def ensure_lora(run: str) -> str:
    """fal keeps generated files for at least 7 days. If the LoRA link has lapsed, re-upload our local copy."""
    s = state.load(run)
    url = s["lora_url"]
    try:
        if httpx.head(url, timeout=20, follow_redirects=True).status_code < 400:
            return url
    except httpx.HTTPError:
        pass
    local = s.get("lora_local")
    if not local or not Path(local).exists():
        raise typer.BadParameter("LoRA link expired and no local copy found — retrain with `selfshot train`.")
    typer.echo("LoRA link expired; re-uploading local copy...")
    url = fal_client.upload_file(local)
    state.update(run, lora_url=url)
    return url


def _args(backend: str, prompt: str, lora_url: str, scale: float, seed: int, size: dict) -> dict:
    args = {"prompt": prompt, "loras": [{"path": lora_url, "scale": scale}], "image_size": size, "seed": seed}
    if backend == "portrait":
        args |= {"guidance_scale": 3.5, "num_inference_steps": 28, "output_format": "png"}
    else:  # klein: keep our exact prompt so the trigger isn't rewritten away
        args |= {"enable_prompt_expansion": False}
    return args


def _one(backend: str, endpoint: str, look: Look, prompt: str, lora_url: str,
         scale: float, seed: int, out: Path) -> dict:
    result = fal_client.subscribe(endpoint, arguments=_args(backend, prompt, lora_url, scale, seed, DEFAULT_SIZE))
    url = result["images"][0]["url"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(httpx.get(url, timeout=120, follow_redirects=True).content)
    return {"file": str(out), "url": url, "style": look.id, "scene": look.scene, "outfit": look.outfit,
            "extras": look.extras, "seed": seed, "prompt": prompt, "scale": scale}


def generate(run: str, presets_path: Path, per_style: int, looks: list[str] | None,
             outfits: list[str] | None, scenes: list[str] | None, extras: list[str] | None,
             scale: float, seed_start: int, workers: int, yes: bool) -> None:
    s = state.load(run)
    if not s.get("lora_url"):
        raise typer.BadParameter("No trained LoRA for this run — run `selfshot train` first.")
    backend, trigger = s["backend"], s["trigger"]
    b = BACKENDS[backend]
    presets = yaml.safe_load(presets_path.read_text())

    gen_dir = run_dir(run) / "gen"
    manifest_path = run_dir(run) / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    done = {m["file"] for m in manifest}

    jobs = []
    for look in choose_looks(presets, looks, outfits, scenes, extras):
        prompt = look.prompt(presets, trigger)
        for seed in range(seed_start, seed_start + per_style):
            out = gen_dir / look.id / f"{seed}_s{scale:.2f}.png"
            if str(out) not in done:
                jobs.append((look, prompt, seed, out))

    if not jobs:
        typer.echo("Nothing new to generate.")
        return
    n_looks = len({j[0].id for j in jobs})
    cost = len(jobs) * b.price_per_image
    if not yes:
        typer.confirm(f"Generate {len(jobs)} images ({n_looks} looks × {per_style} seeds) on {b.inference}, "
                      f"~${cost:.2f}?", abort=True)

    lora_url = ensure_lora(run)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, backend, b.inference, lk, pr, lora_url, scale, seed, out)
                   for lk, pr, seed, out in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                entry = fut.result()
                manifest.append(entry)
                manifest_path.write_text(json.dumps(manifest, indent=2))  # save as we go
                typer.echo(f"  [{i}/{len(jobs)}] {entry['style']} seed {entry['seed']}")
            except Exception as e:  # one failed image shouldn't sink the batch
                typer.secho(f"  [{i}/{len(jobs)}] failed: {e}", fg="red")

    typer.secho(f"Done. {len(manifest)} images in {gen_dir}", fg="green")


def list_presets(presets_path: Path) -> None:
    p = yaml.safe_load(presets_path.read_text())
    for kind in ("looks", "outfits", "scenes", "extras"):
        typer.secho(kind, bold=True)
        for name, val in p.get(kind, {}).items():
            if kind == "looks":
                desc = f"{val['scene']} + {val['outfit']}" + (f" + {', '.join(val['extras'])}" if val.get("extras") else "")
            elif kind == "scenes":
                desc = val["framing"]
            else:
                desc = val
            typer.echo(f"  {name:<22} {desc}")
