"""Static HTML contact sheet: every generation ranked by face score, next to your real photos.

Click an image to compare it side by side with a real photo at full size (check ears,
piercings, tattoo linework). Tick "pick" on the ones you like and copy the upscale command.
"""

import html
import json
import os
import webbrowser
from pathlib import Path

from .config import run_dir

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>selfshot · __RUN__</title>
<style>
:root{--bg:#141414;--fg:#eee;--dim:#999;--ok:#5ad08a;--bad:#e06c6c;--card:#1f1f1f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);padding:12px 16px;border-bottom:1px solid #333;
display:flex;gap:16px;align-items:center;flex-wrap:wrap}
h1{font-size:16px;margin:0}.dim{color:var(--dim)}
.refs{display:flex;gap:6px;overflow-x:auto;padding:12px 16px}.refs img{height:120px;border-radius:4px;cursor:pointer}
.refs img.sel{outline:2px solid var(--ok)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;padding:16px}
.card{background:var(--card);border-radius:6px;overflow:hidden;border:2px solid transparent}
.card.keep{border-color:var(--ok)}.card img{width:100%;display:block;cursor:zoom-in}
.meta{padding:8px;display:flex;justify-content:space-between;align-items:center;gap:6px}
.score{font-variant-numeric:tabular-nums;font-weight:600}.keep .score{color:var(--ok)}.card:not(.keep) .score{color:var(--bad)}
.hide-rej .card:not(.keep){display:none}
#cmp{position:fixed;inset:0;background:#000e;display:none;z-index:5;padding:16px;gap:16px;justify-content:center;align-items:center}
#cmp.on{display:flex}#cmp img{max-height:92vh;max-width:48vw;object-fit:contain}
#cmd{flex:1;min-width:240px;font:12px ui-monospace,monospace;background:#000;color:var(--fg);border:1px solid #333;padding:6px}
button{background:#333;color:var(--fg);border:0;padding:6px 10px;border-radius:4px;cursor:pointer}
</style></head><body>
<header><h1>__RUN__</h1><span class="dim">__SUMMARY__</span>
<label><input type="checkbox" id="rej"> keepers only</label>
<input id="cmd" readonly placeholder="tick “pick” to build an upscale command"><button id="copy">copy</button></header>
<div class="refs" id="refs">__REFS__</div>
<p class="dim" style="padding:0 16px;margin:0">Real photos above. Click one to set it as the comparison reference.</p>
<div class="grid" id="grid">__CARDS__</div>
<div id="cmp"><img id="a"><img id="b"></div>
<script>
const run="__RUN__";let ref=document.querySelector('.refs img');ref&&ref.classList.add('sel');
document.getElementById('refs').onclick=e=>{if(e.target.tagName!=='IMG')return;
 ref&&ref.classList.remove('sel');ref=e.target;ref.classList.add('sel')};
const cmp=document.getElementById('cmp');
document.getElementById('grid').onclick=e=>{if(e.target.tagName==='IMG'){
 document.getElementById('a').src=e.target.src;document.getElementById('b').src=ref?ref.src:'';cmp.classList.add('on')}};
cmp.onclick=()=>cmp.classList.remove('on');
document.addEventListener('keydown',e=>{if(e.key==='Escape')cmp.classList.remove('on')});
document.getElementById('rej').onchange=e=>document.body.classList.toggle('hide-rej',e.target.checked);
const cmd=document.getElementById('cmd');
function build(){const p=[...document.querySelectorAll('.pick:checked')].map(c=>c.dataset.f);
 cmd.value=p.length?`selfshot upscale --run ${run} `+p.join(' '):''}
document.querySelectorAll('.pick').forEach(c=>c.onchange=build);
document.getElementById('copy').onclick=()=>{cmd.select();navigator.clipboard&&navigator.clipboard.writeText(cmd.value)};
</script></body></html>"""


def build(run: str, open_browser: bool = True) -> Path:
    d = run_dir(run)
    manifest = json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").exists() else []
    manifest.sort(key=lambda m: -m.get("score", -1))

    def rel(p: str) -> str:
        return html.escape(os.path.relpath(p, d))

    refs = "".join(f'<img src="{rel(str(p))}" alt="reference">'
                   for p in sorted((d / "dataset").glob("*.jpg")))
    cards = []
    for m in manifest:
        keep = m.get("keep")
        sc = f'{m["score"]:.3f}' if "score" in m else "—"
        name = f'{m["style"]}/{Path(m["file"]).name}'  # seeds repeat across styles, so keep the folder
        cards.append(
            f'<div class="card{" keep" if keep else ""}"><img loading="lazy" src="{rel(m["file"])}" '
            f'title="{html.escape(m["prompt"])}"><div class="meta"><span class="score">{sc}</span>'
            f'<span class="dim">{html.escape(m["style"])} · {m["seed"]}</span>'
            f'<label><input type="checkbox" class="pick" data-f="{html.escape(name)}"> pick</label></div></div>'
        )
    kept = sum(1 for m in manifest if m.get("keep"))
    summary = f"{len(manifest)} generated · {kept} keepers"

    page = (PAGE.replace("__RUN__", html.escape(run)).replace("__SUMMARY__", summary)
            .replace("__REFS__", refs).replace("__CARDS__", "".join(cards)))
    out = d / "sheet.html"
    out.write_text(page)
    if open_browser:
        webbrowser.open(out.resolve().as_uri())
    return out
