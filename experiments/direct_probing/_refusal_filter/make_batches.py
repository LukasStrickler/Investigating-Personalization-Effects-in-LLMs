"""Split the dumped response JSONs into fixed-size batches for classifier agents.

Each batch file holds a list of {"prompt_id", "response"} (plus model name) — small
enough for one agent to read and classify. Writes batches/<model>__<i>.json and prints
a manifest the workflow iterates over.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BATCHES = HERE / "batches"
BATCHES.mkdir(exist_ok=True)
BATCH_SIZE = 50

manifest = []
for jf in sorted(HERE.glob("*.responses.json")):
    model = jf.name.replace(".responses.json", "")
    items = json.loads(jf.read_text())
    for i in range(0, len(items), BATCH_SIZE):
        chunk = items[i:i + BATCH_SIZE]
        batch = [{"prompt_id": it["prompt_id"], "response": it["response"]} for it in chunk]
        bf = BATCHES / f"{model}__{i // BATCH_SIZE:02d}.json"
        bf.write_text(json.dumps({"model": model, "items": batch}, ensure_ascii=False))
        manifest.append(bf.name)

(HERE / "batch_manifest.json").write_text(json.dumps(manifest, indent=1))
print(f"wrote {len(manifest)} batches to {BATCHES}")
for m in manifest:
    print("  ", m)
