"""Submit a LoRA training job to fal and wait for it. Resumable: the request id is saved first."""

import time

import fal_client
import httpx
import typer

from . import state
from .config import BACKENDS, run_dir


def estimate(backend: str, steps: int) -> float:
    b = BACKENDS[backend]
    return max(b.min_train_price, steps * b.price_per_step)


def train(run: str, backend: str, trigger: str, steps: int | None, yes: bool) -> str:
    b = BACKENDS[backend]
    s = state.load(run)

    if s.get("lora_url"):
        typer.echo(f"Already trained: {s['lora_url']}")
        return s["lora_url"]

    request_id = s.get("train_request_id")
    if request_id:
        typer.echo(f"Resuming training job {request_id}")
    else:
        steps = steps or b.default_steps
        zip_path = run_dir(run) / "dataset.zip"
        if not zip_path.exists():
            raise typer.BadParameter("No dataset.zip — run `selfshot prep` first.")
        cost = estimate(backend, steps)
        if not yes:
            typer.confirm(f"Train {backend} LoRA, {steps} steps, ~${cost:.2f}?", abort=True)

        dataset_url = fal_client.upload_file(str(zip_path))
        handle = fal_client.submit(
            b.trainer,
            arguments={"images_data_url": dataset_url, b.trigger_key: trigger, "steps": steps},
        )
        request_id = handle.request_id
        state.update(run, train_request_id=request_id, backend=backend, trigger=trigger, steps=steps)
        typer.echo(f"Submitted {request_id}. Safe to Ctrl-C; rerun `selfshot train` to resume.")

    last = None
    while True:
        status = fal_client.status(b.trainer, request_id, with_logs=True)
        if isinstance(status, fal_client.InProgress) and status.logs:
            msg = status.logs[-1]["message"]
            if msg != last:
                typer.echo(f"  {msg}")
                last = msg
        if isinstance(status, fal_client.Completed):
            break
        time.sleep(15)

    result = fal_client.result(b.trainer, request_id)
    lora = result.get("diffusers_lora_file") or result.get("lora_file")
    if not lora:
        raise RuntimeError(f"Unexpected trainer output keys: {list(result)}")
    # fal only guarantees generated files for 7 days, so keep our own copy of the weights.
    local = run_dir(run) / "lora.safetensors"
    with httpx.stream("GET", lora["url"], timeout=300, follow_redirects=True) as r:
        r.raise_for_status()
        with open(local, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    state.update(run, lora_url=lora["url"], lora_local=str(local))
    typer.secho(f"LoRA ready: {lora['url']} (local copy: {local})", fg="green")
    return lora["url"]
