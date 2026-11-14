# selfshot

Personal AI headshots, built the way consumer headshot apps likely work:

```
your photos ──► prep ──► train LoRA ──► generate (looks × seeds) ──► score vs. real you ──► review ──► upscale
               (local)     (fal)            (fal)                      (local, ArcFace)      (HTML)     (fal)
```

Everything heavy runs on fal's hosted GPUs, so there's nothing to spin up or tear down and you pay only per call.
Your Mac does the light work: photo cleanup, face scoring on CPU, and the review page. Works on an 8 GB MacBook Air.

## What it costs

| Step | Price (fal, Nov 2026) |
|---|---|
| Train (portrait trainer, 2,500 steps) | ~$6.00 once |
| Generate 7 looks × 8 seeds = 56 images | ~$2.00 |
| Upscale 10 picks | ~$0.10 |

Each command prints an estimate and asks before spending. Pass `-y` to skip the prompt.

## Setting up fal

fal is pay-as-you-go: no subscription, you buy credits and each call draws them down.

1. **Sign up** at [fal.ai](https://fal.ai) (GitHub or Google login works).
2. **Add credits** in the billing dashboard (fal.ai/dashboard/billing). **$10–20 is plenty** for a training run plus
   several generation batches. Purchased credits expire after 365 days. Turn **auto top-up off** for now, so the most
   you can ever spend is what you loaded. When your balance runs out, requests are simply rejected.
3. **Create an API key** at fal.ai/dashboard/keys → "Create Key", and choose the **API** scope (the narrowest preset
   that can call models). Copy it immediately; fal won't show it again.
4. Put the key in `.env` (see Setup below). Never commit it; `.gitignore` already excludes `.env`.

Good to know:
- **New accounts can run 2 requests at once.** Extra requests wait in a queue, so `generate` defaults to 2 workers.
  The limit rises automatically as you buy more credits.
- **fal keeps generated files for at least 7 days.** `train` saves a local copy of your LoRA
  (`runs/<run>/lora.safetensors`), and `generate` re-uploads it automatically if the original link has expired.
- You can watch spending and each request in the fal dashboard.

## Setup

```bash
brew install uv            # if you don't have it
cd selfshot
uv sync
cp .env.example .env       # then paste your fal key
```

The first `prep` or `score` downloads InsightFace's face model (~280 MB) to `~/.insightface`.

## 1. Photos (this matters more than anything else)

Put 15–25 photos of yourself in `photos/`. Phone photos are fine, HEIC included.

- **Vary angles:** straight on, three-quarter left and right, a couple of profiles, slightly above and below.
- **Vary light and setting:** window light, outdoors in shade, indoors at night. If every photo shares one background,
  the model learns the background too.
- **Vary clothes and expression:** serious, smiling, laughing. Include photos with and without glasses.
- **Only you in frame**, sharp focus, no filters, no heavy beauty mode. Crop other people out.
- **Skip** sunglasses, hats (unless you want them learned), and photos older than a year or two.

### Getting the tattoo right

The model learns the tattoo only from what it sees, and a tattoo has fine, exact geometry, so give it more evidence
than you'd think:

- Make **at least half your photos show the whole tattoo**: open collars, crew necks, or no shirt.
- Add **3–5 sharp close-ups** of your neck straight on, in even, shadow-free light (outdoor shade is ideal), with the
  chin slightly raised. These can include the lower face or none of it. `prep` will note "0 faces" for close-ups
  without a face; that's expected and they still train.
- Avoid photos where a collar, beard shadow or hard light cuts across the ink, since the model learns those as part
  of the design.
- Expect it to come out *close* but not exact. The planned tattoo-composite step (see "Ideas for later") is what
  gets it exact.

### Optional captions

Next to `photos/IMG_0042.jpg`, add `photos/IMG_0042.txt` describing **what varies** and naming you only as `[trigger]`:

```
[trigger], wearing clear glasses and a gray hoodie, outdoors in shade, three-quarter view, smiling
```

Don't caption your face, tattoo or piercings. Leaving constant features out of captions binds them to the trigger
word, which is how the model learns "this is what NCHLSB looks like." Mention glasses whenever you're wearing them
so the model treats them as optional.

## 2. Run it

```bash
uv run selfshot prep photos/           # clean, face-check, zip, save reference embeddings
uv run selfshot train                  # ~$6; Ctrl-C safe, rerun to resume
uv run selfshot presets                # list looks, outfits, scenes, extras
uv run selfshot generate --n 8         # every named look × 8 seeds
uv run selfshot score                  # ArcFace similarity vs. your real photos
uv run selfshot sheet                  # opens the contact sheet in your browser
uv run selfshot upscale ms-studio-black/3_s1.00.png ms-library-suit/5_s1.00.png
```

`prep` flags photos with zero or multiple faces. `score` first measures how similar your *real* photos are to each
other (never 1.0), then keeps generations within 85% of that baseline. It also prints which looks hold your likeness
best.

In the contact sheet, click any image to compare it full size with a real photo. Check ears, piercings, hairline,
stubble pattern, and **tattoo linework**, which is the detail a face score can't check. Tick "pick" on the ones you
like, and copy the generated `upscale` command.

## 3. Outfits, scenes and looks

`presets.yaml` separates **outfits** (what you wear), **scenes** (framing, background, light, expression) and
**extras** (glasses, stubble). A **look** is a named combination. The four `ms-*` looks recreate the Magic Studio
images.

```bash
uv run selfshot generate --look ms-library-suit --n 12            # one named look
uv run selfshot generate --outfit navy-shirt --outfit suit-tie \
                         --scene studio-charcoal --scene library  # 2 outfits × 2 scenes = 4 looks
uv run selfshot generate --outfit pink-shirt --extra glasses      # one outfit in every scene
uv run selfshot generate --scene window-light                     # every outfit in one scene
```

Each command shows the image count and cost before running, so a large grid never surprises you. Only images that
don't exist yet are generated, so re-running a command is safe.

To add a look, write one line in plain English under `outfits:` (for example
`grey-turtleneck: a charcoal merino turtleneck sweater`) and use it. Turtlenecks and buttoned collars with ties will
partly cover the tattoo; that's fine, just expected.

## Tuning

- **Likeness weak or face looks "averaged":** add more varied photos, or retrain with `--steps 3000`.
- **Looks too much like your training photos (same shirt, same pose):** use fewer steps, or try `--scale 0.85`.
- **Plastic or uncanny skin:** keep "natural skin texture" in prompts, and never add face restoration.
- **An outfit keeps coming out wrong:** describe it more concretely (color, fabric, collar style) in `presets.yaml`.
- **Compare base models:** `selfshot prep photos/ --run klein`, then `selfshot train --run klein --backend klein`
  trains on the newer FLUX 3 Image Klein. Generate the same looks in both runs and compare scores.

Everything for a run lives in `runs/<name>/`: the dataset, `state.json` (training job and LoRA URL),
`lora.safetensors`, `manifest.json` (every image, prompt, seed and score), `sheet.html`, and `final/`.

## Privacy

Your photos and LoRA are uploaded to fal's storage to run the jobs. The LoRA URL in `runs/*/state.json` is
effectively a key to your face: don't publish it. `.gitignore` already excludes `runs/`, `photos/` and `.env`.

## Ideas for later

- **Tattoo composite:** segment the tattoo in a real photo, warp it onto the generated neck, and blend it in.
- **Outfit switch on a keeper:** mask the clothing on an image you already love and repaint only that region.
- **Web interface:** the commands map directly onto endpoints; the contact sheet is already the start of a UI.
- **Modal backend:** run ComfyUI on Modal's per-second GPUs for workflows fal doesn't host.
