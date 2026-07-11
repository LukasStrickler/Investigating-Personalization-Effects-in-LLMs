"""
After hand-marking "correct" (1/0) in the analysis CSV, this script
keeps only correct==1 rows, joins their full conversation turns, and writes
personas in the personas.jsonl schema. Gender comes from the CSV's own
`predicted_gender` column — no classifier needed here.

Dataset reference:
    Zhao, W., Ren, X., Hessel, J., Cardie, C., Choi, Y., & Deng, Y. (2024).
    WildChat: 1M ChatGPT Interaction Logs in the Wild. ICLR 2024.
    https://huggingface.co/datasets/allenai/WildChat-1M

Requirements:
    `datasets`
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
 
 
OUTPUT_DIR = Path(__file__).resolve().parent
 
 
_STRICT = {
    "male_self_identified": "Male",
    "female_self_identified": "Female",
}
_LOOSE = {
    "male_self_identified": "Male",
    "female_self_identified": "Female",
    "male_contextual_evidence": "Male",
    "female_contextual_evidence": "Female",
}
 
def to_persona_gender(label, strict):
    return (_STRICT if strict else _LOOSE).get((label or "").strip())
 
 
def combination_id(gender_value):
    """Short, readable id, one fixed value per gender (e.g. Female -> 'female_001')."""
    if gender_value is None:
        return None
    return f"{gender_value.lower()}_001"
 
 
def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
 
 
def is_kept(cell, keep_value):
    """Treat 1 / 1.0 / true / yes (case-insensitive) as keep; everything else drop."""
    if cell is None:
        return False
    s = str(cell).strip().lower()
    if keep_value == "1":
        return s in {"1", "1.0", "true", "yes", "y"}
    return s == keep_value.strip().lower()
 
 
def load_keep_map(checked_csv, correct_col, id_col, gender_col, keep_value, strict):
    """Return {conversation_id: persona_gender} for verified-correct, gendered rows."""
    keep = {}
    stats = {"checked_1": 0, "no_gender_skipped": 0, "missing_id": 0, "total_rows": 0}
    with open(checked_csv, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for col in (correct_col, id_col, gender_col):
            if col not in reader.fieldnames:
                sys.exit(f"ERROR: column '{col}' not found in {checked_csv}.\n"
                         f"       Columns present: {reader.fieldnames}")
        for r in reader:
            stats["total_rows"] += 1
            if not is_kept(r.get(correct_col), keep_value):
                continue
            stats["checked_1"] += 1
            cid = (r.get(id_col) or "").strip()
            if not cid:
                stats["missing_id"] += 1
                continue
            gender = to_persona_gender(r.get(gender_col), strict)
            if gender is None:
                stats["no_gender_skipped"] += 1
                continue
            keep[cid] = gender
    return keep, stats
 
 
def wildchat_conversation_id(example):
    """Same unique ID scheme as conversation_histories_extraction.py:
    the first turn's globally unique `turn_identifier` (conversation_hash
    is a content hash and is NOT unique across distinct conversations)."""
    for m in example.get("conversation", []) or []:
        if isinstance(m, dict) and m.get("turn_identifier") is not None:
            return f"wc_{m['turn_identifier']}"
    h = example.get("conversation_hash")
    return f"hash_{h}" if h else None
 
 
def clean_messages(conversation):
    out = []
    for turn in conversation or []:
        if not isinstance(turn, dict):
            continue
        role, content = turn.get("role"), turn.get("content")
        if role is None or content is None:
            continue
        out.append({"role": role, "content": content})
    return out
 
 
def flatten_transcript(messages, max_chars):
    text = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + " …[truncated]"
    return text
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checked-csv", default=OUTPUT_DIR / "wildchat_gender_evidence_results_checked.csv",
                    help="The hand-checked CSV (with the 'correct' column).")
    ap.add_argument("--out", default=OUTPUT_DIR / "wildchat_personas.jsonl",
                    help="Output JSONL path (personas schema, full conversations).")
    ap.add_argument("--csv", default=OUTPUT_DIR / "wildchat_personas.csv",
                    help="Output CSV path (one row per kept conversation).")
    ap.add_argument("--csv-max-chars", type=int, default=8000,
                    help="Truncate the transcript cell to this many chars (0 = no limit).")
    ap.add_argument("--conversations", default=OUTPUT_DIR / "wildchat_conversations.jsonl",
                    help="Local JSONL of full conversations from analysis10 "
                         "(keyed by conversation_id). If found, the builder joins "
                         "locally and is instant. If missing, it falls back to "
                         "streaming WildChat.")
    ap.add_argument("--split", default="train")
    ap.add_argument("--correct-column", default="correct",
                    help="Header of the 1/0 review column (default: 'correct').")
    ap.add_argument("--keep-value", default="1",
                    help="Cell value that means keep (default: '1').")
    ap.add_argument("--id-column", default="conversation_id")
    ap.add_argument("--gender-column", default="predicted_gender")
    ap.add_argument("--strict", action="store_true",
                    help="Keep only first-person self-ID labels as Male/Female.")
    ap.add_argument("--max-scan", type=int, default=None,
                    help="Stop re-streaming after this many WildChat examples "
                         "(safety cap; default: scan until all ids are found).")
    args = ap.parse_args()
 
    keep, stats = load_keep_map(args.checked_csv, args.correct_column,
                                args.id_column, args.gender_column,
                                args.keep_value, args.strict)
    print(f"Checked CSV: {stats['total_rows']:,} rows | "
          f"correct==keep: {stats['checked_1']:,} | "
          f"usable (gendered): {len(keep):,} | "
          f"dropped no-gender: {stats['no_gender_skipped']:,} | "
          f"missing id: {stats['missing_id']:,}")
    if not keep:
        sys.exit("Nothing to build — no usable gendered rows marked as kept.")
 
    pending = dict(keep)  
    written = [0]
    csv_fields = ["history_id", "Gender", "num_turns", "generated_at", "transcript"]
    out = open(args.out, "w", encoding="utf-8")
    csv_fh = open(args.csv, "w", encoding="utf-8", newline="")
    csv_writer = csv.DictWriter(csv_fh, fieldnames=csv_fields)
    csv_writer.writeheader()
 
    def emit(history_id, gender, messages):
        """history_id is the ORIGINAL WildChat conversation_hash — the final
        personas file keeps the old ID scheme so records stay comparable with
        experiments already run. The unique wc_<turn_identifier> key is used
        only internally to join the CSV against the sidecar without collapsing
        distinct conversations that share a hash."""
        if not messages:
            return
        generated_at = _now_iso()
        record = {
            "history_id": history_id,
            "persona": {"Race": None, "Gender": gender},
            "combination_ids": {"Gender": combination_id(gender)},
            "messages": messages,
            "generated_at": generated_at,
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        csv_writer.writerow({
            "history_id": history_id,
            "Gender": gender,
            "num_turns": len(messages),
            "generated_at": generated_at,
            "transcript": flatten_transcript(messages, args.csv_max_chars),
        })
        written[0] += 1
 
    used_local = os.path.exists(args.conversations)
    try:
        if used_local:
            print(f"Reading conversations locally from {args.conversations} ...")
            with open(args.conversations, encoding="utf-8") as cf:
                for line in cf:
                    if not pending:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    cid = rec.get("conversation_id")
                    if cid in pending:
                        gender = pending.pop(cid)
                        history_id = rec.get("conversation_hash") or cid
                        emit(history_id, gender, clean_messages(rec.get("messages", [])))
        else:
            print(f"Local conversations file '{args.conversations}' not found — "
                  f"streaming WildChat instead (slower).")
            from datasets import load_dataset
            ds = load_dataset("allenai/WildChat-1M", split=args.split, streaming=True)
            scanned = 0
            for example in ds:
                if not pending:
                    break
                if args.max_scan is not None and scanned >= args.max_scan:
                    break
                scanned += 1
                cid = wildchat_conversation_id(example)
                if cid in pending:
                    gender = pending.pop(cid)
                    history_id = example.get("conversation_hash") or cid
                    emit(history_id, gender, clean_messages(example.get("conversation", []) or []))
                    if written[0] % 1000 == 0:
                        print(f"  matched={written[0]}/{len(keep)} scanned={scanned:,} ...")
    finally:
        out.close()
        csv_fh.close()
 
    print(f"\nDone. Wrote {written[0]:,} records "
          f"({'local join' if used_local else 'WildChat stream'}) to:")
    print(f"  JSON : {args.out}")
    print(f"  CSV  : {args.csv}")
    if pending:
        print(f"WARNING: {len(pending):,} kept ids were NOT found"
              f"{' in the conversations file' if used_local else ' in the stream'}.")
        if used_local:
            print("  The sidecar may be from a different/older analysis run than the CSV.")
        else:
            print("  Try leaving --max-scan unset, or check --split.")
        for cid in list(pending)[:5]:
            print(f"    unmatched: {cid}")
 
 
if __name__ == "__main__":
    main()