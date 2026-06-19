"""
Convert personas.jsonl to the dataset JSON format expected by the pipeline.

Produces a dataset with:
  - history_ids: list of history_id strings
  - conversations: flat string format ("User: ...\nAssistant: ...")
  - labels: {"gender": ["Male", "Female", ...]}

Ablation modes (--ablation):
  - none:             Full conversation (default)
  - no_race:          Remove Race template (first exchange with Name + Artist)
  - no_movie:         Mask movie genre in Gender template
  - no_hobby:         Mask hobby in Gender template
  - no_name:          Mask the persona name (in Race template)
  - assistant_only:   Keep only assistant responses
  - gender_template:  Keep only the Gender template exchange (movie+hobby)

Usage:
    python prepare_personas.py                                    # Full dataset
    python prepare_personas.py --subsample 200                    # 200 balanced
    python prepare_personas.py --ablation no_race --subsample 200
    python prepare_personas.py --ablation no_movie --subsample 200
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


def load_personas(path: str):
    """Load personas.jsonl and return entries that have a Gender label."""
    entries = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            gender = d.get("persona", {}).get("Gender")
            if gender:
                entries.append(d)
    return entries


def to_flat_string(messages):
    """Convert list of {role, content} messages to 'User: ...\nAssistant: ...' format."""
    parts = []
    for msg in messages:
        role = msg["role"].capitalize()
        if role == "User":
            parts.append(f"User: {msg['content']}")
        else:
            parts.append(f"Assistant: {msg['content']}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Ablation helpers
# ---------------------------------------------------------------------------

_RACE_TEMPLATE_RE = re.compile(
    r"I am ([^,]+), I really like the music from ([^,]+),"
)
_MOVIE_RE = re.compile(r"I really enjoy watching (.+?) movies")
_HOBBY_RE = re.compile(r"I spend a lot of time on (.+?)\.")


def _is_race_exchange(user_msg: str) -> bool:
    """Check if a user message follows the Race template."""
    return bool(_RACE_TEMPLATE_RE.search(user_msg))


def _is_gender_exchange(user_msg: str) -> bool:
    """Check if a user message follows the Gender template."""
    return bool(_MOVIE_RE.search(user_msg))


def apply_ablation(messages: list, ablation: str) -> list:
    """
    Apply an ablation transformation to a message list.
    Returns the modified message list (or a subset).
    """
    if ablation == "none":
        return messages

    if ablation == "no_race":
        # Remove the Race template exchange (first user+assistant pair if it matches)
        result = []
        skip_next_assistant = False
        for msg in messages:
            if msg["role"] == "user" and _is_race_exchange(msg["content"]):
                skip_next_assistant = True
                continue
            if msg["role"] == "assistant" and skip_next_assistant:
                skip_next_assistant = False
                continue
            skip_next_assistant = False
            result.append(msg)
        return result

    if ablation == "gender_template":
        # Keep only the Gender template exchange(s)
        result = []
        include_next_assistant = False
        for msg in messages:
            if msg["role"] == "user" and _is_gender_exchange(msg["content"]):
                result.append(msg)
                include_next_assistant = True
            elif msg["role"] == "assistant" and include_next_assistant:
                result.append(msg)
                include_next_assistant = False
            else:
                include_next_assistant = False
        return result

    if ablation == "no_movie":
        # Replace movie genre with a placeholder
        result = []
        for msg in messages:
            content = msg["content"]
            if msg["role"] == "user":
                content = _MOVIE_RE.sub("I really enjoy watching [MOVIE] movies", content)
            result.append({"role": msg["role"], "content": content})
        return result

    if ablation == "no_hobby":
        # Replace hobby with a placeholder
        result = []
        for msg in messages:
            content = msg["content"]
            if msg["role"] == "user":
                content = _HOBBY_RE.sub("I spend a lot of time on [HOBBY].", content)
            result.append({"role": msg["role"], "content": content})
        return result

    if ablation == "no_name":
        # Replace the persona name with a placeholder
        result = []
        for msg in messages:
            content = msg["content"]
            content = _RACE_TEMPLATE_RE.sub(
                "I am [NAME], I really like the music from \\2,", content
            )
            result.append({"role": msg["role"], "content": content})
        return result

    if ablation == "assistant_only":
        # Keep only assistant responses
        return [msg for msg in messages if msg["role"] == "assistant"]

    raise ValueError(f"Unknown ablation: {ablation}")


def balanced_subsample(entries, n, seed=42):
    """Take a balanced subsample of n entries across gender classes."""
    rng = random.Random(seed)
    by_gender = {}
    for e in entries:
        g = e["persona"]["Gender"]
        by_gender.setdefault(g, []).append(e)

    per_class = n // len(by_gender)
    sampled = []
    for g, items in by_gender.items():
        rng.shuffle(items)
        sampled.extend(items[:per_class])
    rng.shuffle(sampled)
    return sampled


def main():
    parser = argparse.ArgumentParser(description="Prepare personas dataset for gender probing")
    parser.add_argument("--input", type=str, default="data/personas.jsonl",
                        help="Path to personas.jsonl")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: data/dataset_gender_<ablation>.json)")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Total number of samples (balanced across classes)")
    parser.add_argument("--ablation", type=str, default="none",
                        choices=["none", "no_race", "no_movie", "no_hobby",
                                 "no_name", "assistant_only", "gender_template"],
                        help="Ablation mode to apply")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output is None:
        suffix = args.ablation if args.ablation != "none" else "full"
        args.output = f"data/dataset_gender_{suffix}.json"

    entries = load_personas(args.input)
    print(f"Loaded {len(entries)} entries with Gender labels")
    print(f"Distribution: {dict(Counter(e['persona']['Gender'] for e in entries))}")

    if args.subsample:
        entries = balanced_subsample(entries, args.subsample, seed=args.seed)
        print(f"Subsampled to {len(entries)} entries")
        print(f"Distribution: {dict(Counter(e['persona']['Gender'] for e in entries))}")

    print(f"Ablation mode: {args.ablation}")

    history_ids = []
    conversations = []
    gender_labels = []
    skipped = 0

    for e in entries:
        msgs = apply_ablation(e["messages"], args.ablation)
        if not msgs:
            skipped += 1
            continue
        history_ids.append(e["history_id"])
        conversations.append(to_flat_string(msgs))
        gender_labels.append(e["persona"]["Gender"])

    if skipped:
        print(f"  Skipped {skipped} entries (empty after ablation)")

    dataset = {
        "history_ids": history_ids,
        "conversations": conversations,
        "labels": {
            "gender": gender_labels,
        },
        "metadata": {
            "ablation": args.ablation,
            "subsample": args.subsample,
            "seed": args.seed,
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved dataset to {args.output} ({len(conversations)} conversations)")


if __name__ == "__main__":
    main()
