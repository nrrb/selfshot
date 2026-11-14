"""Backends, prices and defaults. Prices are fal's published rates as of Nov 2026 — verify before big runs."""

from dataclasses import dataclass
from pathlib import Path

RUNS_DIR = Path("runs")


@dataclass(frozen=True)
class Backend:
    trainer: str            # fal training endpoint
    trigger_key: str        # name of the trigger field the trainer expects
    default_steps: int
    price_per_step: float
    min_train_price: float
    inference: str          # fal endpoint that serves the LoRA
    price_per_image: float  # at <= 1 megapixel


BACKENDS: dict[str, Backend] = {
    # FLUX.1 [dev] trainer tuned for faces — closest to what consumer headshot apps do. Default.
    "portrait": Backend(
        trainer="fal-ai/flux-lora-portrait-trainer",
        trigger_key="trigger_phrase",
        default_steps=2500,
        price_per_step=0.0024,
        min_train_price=2.40,
        inference="fal-ai/flux-lora",
        price_per_image=0.035,
    ),
    # Newer, cheaper per image (distilled endpoint). Worth an A/B against "portrait".
    "klein": Backend(
        trainer="blackforestlabs/flux-3/klein/base/trainer",
        trigger_key="trigger_phrase",
        default_steps=1000,
        price_per_step=0.0043,
        min_train_price=0.43,
        inference="blackforestlabs/flux-3/klein/base/text-to-image/lora",
        price_per_image=0.035,
    ),
}

UPSCALER = "fal-ai/esrgan"
UPSCALE_PRICE_ESTIMATE = 0.01  # billed per compute second; a 2x upscale is a few seconds

# 832x1152 = 0.96 MP. fal-ai/flux-lora rounds UP to the next megapixel, so staying
# just under 1 MP halves the cost versus e.g. 896x1152.
DEFAULT_SIZE = {"width": 832, "height": 1152}

DEFAULT_TOKEN = "NCHLSB"           # a nonsense word the model has no prior meaning for
DEFAULT_TRIGGER = "a photo of NCHLSB man"


def run_dir(name: str) -> Path:
    d = RUNS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d
