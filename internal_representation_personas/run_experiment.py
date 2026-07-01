"""
run_experiment.py — Gender probing: single-indicator + generalization test
==========================================================================
Two persona dimensions present in each conversation history:

  Gender: Movie  (e.g. "action") | Hobby   (e.g. "yoga")
  Race:   Name   (e.g. "Michael Johnson") | Artist (e.g. "Linkin Park")

SIX conditions:
  FULL         All four indicators present (baseline)
  ONLY_MOVIE   Only Movie present; Hobby/Name/Artist -> placeholders everywhere
  ONLY_HOBBY   Only Hobby present; Movie/Name/Artist -> placeholders everywhere
  ONLY_NAME    Only Name  present; Movie/Hobby/Artist -> placeholders everywhere
  ONLY_ARTIST  Only Artist present; Movie/Hobby/Name -> placeholders everywhere
  CTRL         Shuffled gender labels

"Everywhere" means every message in the conversation (user turn AND assistant
response), so the removed indicator never leaks through the LLM's own answer.

GENERALIZATION TEST — for each ONLY_* condition:
  One Female-coded and one Male-coded indicator VALUE are held out of probe
  training entirely.  The probe is trained on all other histories, then tested
  on the held-out histories.  High accuracy = the model learned a real gender
  representation, not just a word->gender lookup table.

Probe target : Gender (Male / Female)
Model        : SmolLM2-360M-Instruct (frozen, 32 layers, hidden dim 960)
N histories  : ~392 (196 Female + 196 Male, full-conversation context)
Output       : run_results/
"""

import csv
import json
import re
import sys
import traceback
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

HERE = Path(__file__).resolve().parent
OUT  = HERE / "run_results"
(OUT / "plots").mkdir(parents=True, exist_ok=True)
(OUT / "data").mkdir(parents=True, exist_ok=True)

LOG = open(OUT / "run.log", "w", encoding="utf-8", buffering=1)


def log(msg: str) -> None:
    LOG.write(msg + "\n")
    LOG.flush()
    print(msg, flush=True)


log("=== Gender probe: single-indicator + generalization test ===")

try:
    sys.path.insert(0, str(HERE))
    from config import DataConfig, ModelConfig, ProbeConfig
    from dataset import _format_messages, prepare_dataset
    from extraction import extract_hidden_states, load_model_and_tokenizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    log("imports OK")

    # ── paths ──────────────────────────────────────────────────────────────────
    ROOT          = HERE.parent.parent
    GENDER_CSV    = ROOT / "src" / "generate_backgrounds" / "dimension_value_mapping" / "gender.csv"
    RACE_CSV      = ROOT / "src" / "generate_backgrounds" / "dimension_value_mapping" / "race.csv"
    PERSONAS_FILE = ROOT / "src" / "generate_backgrounds" / "data" / "personas" / "personas.jsonl"
    N_PER_GENDER  = 200

    # ── Gender indicators ─────────────────────────────────────────────────────
    gender_phrase_to_slot:   dict[str, str] = {}
    gender_phrase_to_gender: dict[str, str] = {}
    with open(GENDER_CSV, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            p = row["Indicator_value"].strip()
            gender_phrase_to_slot[p]   = row["Indicator_name"].strip()
            gender_phrase_to_gender[p] = row["Dimension_value"].strip()

    movie_phrases: set[str] = {p for p, s in gender_phrase_to_slot.items() if s == "Movie"}
    hobby_phrases: set[str] = {p for p, s in gender_phrase_to_slot.items() if s == "Hobby"}
    log(f"Gender Movie ({len(movie_phrases)}): {sorted(movie_phrases)}")
    log(f"Gender Hobby ({len(hobby_phrases)}): {sorted(hobby_phrases)}")

    # ── Race indicators ───────────────────────────────────────────────────────
    race_phrase_to_slot: dict[str, str] = {}
    race_phrase_to_race: dict[str, str] = {}
    with open(RACE_CSV, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            p       = row["Indicator_value"].strip()
            dim_col = "Dimension value" if "Dimension value" in row else "Dimension_value"
            race_phrase_to_slot[p] = row["Indicator_name"].strip()
            race_phrase_to_race[p] = row[dim_col].strip()

    name_phrases:   set[str] = {p for p, s in race_phrase_to_slot.items() if s == "Name"}
    artist_phrases: set[str] = {p for p, s in race_phrase_to_slot.items() if s == "Artist"}
    log(f"Race Name   ({len(name_phrases)}): {sorted(name_phrases)[:4]}...")
    log(f"Race Artist ({len(artist_phrases)}): {sorted(artist_phrases)[:4]}...")

    # ── Placeholder map ───────────────────────────────────────────────────────
    phrase_to_placeholder: dict[str, str] = {}
    for p, slot in gender_phrase_to_slot.items():
        phrase_to_placeholder[p] = f"<{slot}>"
    for p, slot in race_phrase_to_slot.items():
        phrase_to_placeholder[p] = f"<{slot}>"

    all_phrases: set[str] = set(phrase_to_placeholder)

    PATS: dict[str, re.Pattern] = {
        p: re.compile(rf"(?<!\w){re.escape(p)}(?!\w)", re.IGNORECASE)
        for p in all_phrases
    }

    def _ablate(messages: list[dict], phrases_to_remove: set[str]) -> list[dict]:
        """Replace indicator phrases in ALL messages (user AND assistant).

        Scanning every role ensures the removed indicator never leaks through
        the LLM's own previous response either.  Longest-first ordering prevents
        partial matches ('romantic comedy' before 'comedy').
        """
        ordered = sorted(phrases_to_remove, key=len, reverse=True)
        result  = deepcopy(messages)
        for msg in result:
            txt = msg["content"]
            for p in ordered:
                txt = PATS[p].sub(phrase_to_placeholder[p], txt)
            msg["content"] = txt
        return result

    # ── dataset ────────────────────────────────────────────────────────────────
    dataset_path = OUT / "data" / "dataset_personas.json"
    if dataset_path.exists():
        with open(dataset_path, encoding="utf-8") as fh:
            _old = json.load(fh)
        if _old.get("context_mode") != "full":
            log("Cached dataset uses wrong context mode — regenerating.")
            dataset_path.unlink()

    if dataset_path.exists():
        log("Loading existing dataset ...")
        with open(dataset_path, encoding="utf-8") as fh:
            dataset = json.load(fh)
    else:
        log("Generating dataset (context_mode='full') ...")
        dc = DataConfig(
            personas_file=str(PERSONAS_FILE),
            data_dir=str(OUT / "data"),
            attributes=["Gender"],
            samples_per_group=N_PER_GENDER,
            context_mode="full",
            seed=42,
        )
        dataset = prepare_dataset(dc)
        log("Dataset saved")

    histories: list = dataset["conversations_chat"]
    labels:    list = dataset["labels"]["Gender"]

    # Keep only histories with both a Gender turn and a Race turn.
    GENDER_MARKER = "I really enjoy watching "
    RACE_MARKER   = "I really like the music from "
    valid_idx = [
        i for i, h in enumerate(histories)
        if any(GENDER_MARKER in m.get("content", "") for m in h)
        and any(RACE_MARKER   in m.get("content", "") for m in h)
    ]
    if len(valid_idx) < len(histories):
        log(f"Dropped {len(histories) - len(valid_idx)} histories missing a turn.")
    histories = [histories[i] for i in valid_idx]
    labels    = [labels[i]    for i in valid_idx]

    # Balance genders.
    female_idx = [i for i, l in enumerate(labels) if l == "Female"]
    male_idx   = [i for i, l in enumerate(labels) if l == "Male"]
    n_each     = min(len(female_idx), len(male_idx), N_PER_GENDER)
    rng_bal    = np.random.default_rng(42)
    keep = sorted(
        rng_bal.choice(female_idx, n_each, replace=False).tolist() +
        rng_bal.choice(male_idx,   n_each, replace=False).tolist()
    )
    histories = [histories[i] for i in keep]
    labels    = [labels[i]    for i in keep]
    N: int = len(histories)
    log(f"Dataset: {N} histories ({labels.count('Female')}F / {labels.count('Male')}M)")

    labels_arr  = np.array(labels)
    female_mask = labels_arr == "Female"
    male_mask   = labels_arr == "Male"

    # ── per-phrase frequency (user turns) ─────────────────────────────────────
    phrase_in_history: dict[str, list[int]] = {p: [] for p in all_phrases}
    for i, hist in enumerate(histories):
        user_text = " ".join(m["content"] for m in hist if m.get("role") == "user")
        for p, pat in PATS.items():
            if pat.search(user_text):
                phrase_in_history[p].append(i)

    log("Gender indicator frequencies:")
    for p in sorted(gender_phrase_to_slot):
        idxs = phrase_in_history[p]
        if idxs:
            log(f"  {p!r:28s} ({gender_phrase_to_slot[p]}/{gender_phrase_to_gender[p]}) -> {len(idxs)}")

    # ── slot-specific canonical indicator detection ───────────────────────────
    MOVIE_SLOT_RE  = re.compile(r"enjoy watching\s+(.+?)\s+movies",       re.IGNORECASE)
    HOBBY_SLOT_RE  = re.compile(r"time on\s+(.+?)[\.,]",                  re.IGNORECASE)
    NAME_SLOT_RE   = re.compile(r"I am\s+(.+?),\s*I really like",         re.IGNORECASE)
    ARTIST_SLOT_RE = re.compile(r"the music from\s+(.+?),\s*can you",     re.IGNORECASE)

    history_to_movie:  dict[int, str] = {}
    history_to_hobby:  dict[int, str] = {}
    history_to_name:   dict[int, str] = {}
    history_to_artist: dict[int, str] = {}

    _s_movie   = sorted(movie_phrases,  key=len, reverse=True)
    _s_hobby   = sorted(hobby_phrases,  key=len, reverse=True)
    _s_names   = sorted(name_phrases,   key=len, reverse=True)
    _s_artists = sorted(artist_phrases, key=len, reverse=True)

    for i, hist in enumerate(histories):
        for msg in hist:
            if msg.get("role") != "user":
                continue
            c = msg["content"]
            if GENDER_MARKER in c:
                m = MOVIE_SLOT_RE.search(c)
                if m:
                    for p in _s_movie:
                        if PATS[p].search(m.group(1).strip()):
                            history_to_movie[i] = p; break
                h = HOBBY_SLOT_RE.search(c)
                if h:
                    for p in _s_hobby:
                        if PATS[p].search(h.group(1).strip()):
                            history_to_hobby[i] = p; break
            if RACE_MARKER in c:
                n = NAME_SLOT_RE.search(c)
                if n:
                    for p in _s_names:
                        if PATS[p].search(n.group(1).strip()):
                            history_to_name[i] = p; break
                a = ARTIST_SLOT_RE.search(c)
                if a:
                    for p in _s_artists:
                        if PATS[p].search(a.group(1).strip()):
                            history_to_artist[i] = p; break

    # Slot-type assertions
    for i, p in history_to_movie.items():
        assert gender_phrase_to_slot[p] == "Movie"
    for i, p in history_to_hobby.items():
        assert gender_phrase_to_slot[p] == "Hobby"
    for i, p in history_to_name.items():
        assert race_phrase_to_slot[p] == "Name"
    for i, p in history_to_artist.items():
        assert race_phrase_to_slot[p] == "Artist"
    log(f"Slot detection: {len(history_to_movie)} movie | {len(history_to_hobby)} hobby | "
        f"{len(history_to_name)} name | {len(history_to_artist)} artist — all valid")

    log("Sample assignments (first 5):")
    for i in range(min(5, N)):
        log(f"  [{i:3d} {labels[i]:6s}] "
            f"movie='{history_to_movie.get(i,'?')}' | "
            f"hobby='{history_to_hobby.get(i,'?')}' | "
            f"name='{history_to_name.get(i,'?')}' | "
            f"artist='{history_to_artist.get(i,'?')}'")

    # ── five ablated text variants ─────────────────────────────────────────────
    # In each ONLY_* condition every message (user + assistant) is scanned and
    # the three removed indicator types are replaced with their placeholders.
    texts_full        = [_format_messages(h) for h in histories]
    texts_only_movie  = [_format_messages(_ablate(h, hobby_phrases  | name_phrases | artist_phrases)) for h in histories]
    texts_only_hobby  = [_format_messages(_ablate(h, movie_phrases  | name_phrases | artist_phrases)) for h in histories]
    texts_only_name   = [_format_messages(_ablate(h, movie_phrases  | hobby_phrases | artist_phrases)) for h in histories]
    texts_only_artist = [_format_messages(_ablate(h, movie_phrases  | hobby_phrases | name_phrases)) for h in histories]

    log(f"ONLY_MOVIE  unique texts: {len(set(texts_only_movie))}")
    log(f"ONLY_HOBBY  unique texts: {len(set(texts_only_hobby))}")
    log(f"ONLY_NAME   unique texts: {len(set(texts_only_name))}")
    log(f"ONLY_ARTIST unique texts: {len(set(texts_only_artist))}")
    log(f"FULL example[0] (first 400 chars):\n{texts_full[0][:400]}")
    log(f"ONLY_NAME example[0] (first 400 chars):\n{texts_only_name[0][:400]}")

    # ── load or extract hidden states ─────────────────────────────────────────
    hs_path    = OUT / "data" / "hs_single_indicator.npz"
    COND_NAMES = ["full", "only_movie", "only_hobby", "only_name", "only_artist"]

    if not hs_path.exists():
        log("Loading model ...")
        mc = ModelConfig(
            model_name=str(HERE / "models" / "SmolLM2-360M-Instruct"),
            device_map="cpu",
            torch_dtype="float32",
        )
        model, tokenizer = load_model_and_tokenizer(mc)
        n_layers   = len(model.model.layers)
        all_layers = list(range(n_layers + 1))
        pc = ProbeConfig(layers=all_layers, token_position="last")
        log(f"Model: {n_layers} layers, {len(all_layers)} probe positions")
    else:
        log("Cache found — skipping model load")
        _tmp = np.load(hs_path)
        n_layers   = max(int(k.rsplit("_", 1)[1]) for k in _tmp.files)
        all_layers = list(range(n_layers + 1))
        del _tmp
        mc = model = tokenizer = pc = None
        log(f"n_layers={n_layers}, layers 0..{n_layers}")

    if hs_path.exists():
        log("Loading cached hidden states ...")
        _d = np.load(hs_path)
        hs: dict[str, dict[int, np.ndarray]] = {
            c: {l: _d[f"{c}_{l}"] for l in all_layers}
            for c in COND_NAMES
        }
        del _d
        log("Hidden states loaded")
    else:
        all_texts = (texts_full + texts_only_movie + texts_only_hobby +
                     texts_only_name + texts_only_artist)
        log(f"Extracting hidden states ({len(all_texts)} texts) ...")
        _raw = extract_hidden_states(model, tokenizer, all_texts, mc, pc, batch_size=4)
        hs = {
            c: {l: _raw[l][idx * N: (idx + 1) * N] for l in all_layers}
            for idx, c in enumerate(COND_NAMES)
        }
        save_d = {f"{c}_{l}": hs[c][l] for c in COND_NAMES for l in all_layers}
        np.savez_compressed(hs_path, **save_d)
        log(f"Hidden states saved -> {hs_path}")

    # ── labels ────────────────────────────────────────────────────────────────
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    male_class = int(np.where(le.classes_ == "Male")[0][0])
    y_shuffled = np.random.default_rng(42).permutation(y)

    # ── standard cross-validated probe ───────────────────────────────────────
    def probe_cv(X: np.ndarray, y_arr: np.ndarray,
                 n_splits: int = 5, seed: int = 42) -> float:
        """5-fold stratified CV; scaler fit per fold to prevent leakage."""
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores = []
        for tr, te in kf.split(X, y_arr):
            sc  = StandardScaler()
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
            clf.fit(sc.fit_transform(X[tr]), y_arr[tr])
            scores.append(accuracy_score(y_arr[te], clf.predict(sc.transform(X[te]))))
        return float(np.mean(scores))

    # ── generalization probe (train without held-out, test on held-out) ───────
    def probe_generalization(X: np.ndarray, y_arr: np.ndarray,
                             held_out_mask: np.ndarray, seed: int = 42):
        """Train on ~held_out_mask, test on held_out_mask.

        Returns (accuracy, n_test).  Returns (nan, 0) if either split is too
        small for a meaningful result.
        """
        train_mask = ~held_out_mask
        n_train, n_test = int(train_mask.sum()), int(held_out_mask.sum())
        if n_train < 10 or n_test < 4:
            return float("nan"), n_test
        # Need at least both classes in both splits
        if len(np.unique(y_arr[train_mask])) < 2 or len(np.unique(y_arr[held_out_mask])) < 2:
            return float("nan"), n_test
        sc  = StandardScaler()
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        clf.fit(sc.fit_transform(X[train_mask]), y_arr[train_mask])
        acc = accuracy_score(y_arr[held_out_mask], clf.predict(sc.transform(X[held_out_mask])))
        return float(acc), n_test

    # ── probe training at every layer ─────────────────────────────────────────
    log("Training standard probes per layer for all 6 conditions ...")
    acc: dict[str, dict[int, float]] = {c: {} for c in COND_NAMES + ["ctrl"]}
    for layer in all_layers:
        for c in COND_NAMES:
            acc[c][layer] = probe_cv(hs[c][layer], y)
        acc["ctrl"][layer] = probe_cv(hs["full"][layer], y_shuffled)
        if layer % 8 == 0:
            vals = "  ".join(f"{c}={acc[c][layer]:.3f}" for c in COND_NAMES)
            log(f"  L{layer:2d}: {vals}  ctrl={acc['ctrl'][layer]:.3f}")
    log(f"Layer 0 (embedding only): " + " ".join(f"{c}={acc[c][0]:.3f}" for c in COND_NAMES))
    log("Probe training done")

    # Best layer per condition (used for generalization probes)
    best_layer_by_cond: dict[str, int] = {
        c: max(all_layers, key=lambda l: acc[c][l]) for c in COND_NAMES
    }
    for c, bl in best_layer_by_cond.items():
        log(f"  Best layer [{c:12s}]: {bl}  acc={acc[c][bl]:.3f}")

    # ── frozen FULL probe ─────────────────────────────────────────────────────
    best_layer = best_layer_by_cond["full"]
    scaler_best = StandardScaler()
    X_best      = scaler_best.fit_transform(hs["full"][best_layer])
    clf_best    = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    clf_best.fit(X_best, y)

    def apply_probe(hs_dict: dict[int, np.ndarray]) -> np.ndarray:
        return clf_best.predict_proba(
            scaler_best.transform(hs_dict[best_layer])
        )[:, male_class]

    pm: dict[str, np.ndarray] = {c: apply_probe(hs[c]) for c in COND_NAMES}
    for c, pmv in pm.items():
        log(f"P(Male) [{c:12s}]: overall={pmv.mean():.3f}  "
            f"Female={pmv[female_mask].mean():.3f}  Male={pmv[male_mask].mean():.3f}")

    # ── out-of-fold P(Male) per condition at each condition's best layer ──────
    # Used in strip plots so the title "layer X" actually matches the data shown.
    def compute_pm_oof(X: np.ndarray, y_arr: np.ndarray,
                       n_splits: int = 5, seed: int = 42) -> np.ndarray:
        """Return out-of-fold P(Male) using the same 5-fold CV as probe_cv."""
        pm_oof = np.zeros(len(y_arr))
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in kf.split(X, y_arr):
            sc  = StandardScaler()
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
            clf.fit(sc.fit_transform(X[tr]), y_arr[tr])
            pm_oof[te] = clf.predict_proba(sc.transform(X[te]))[:, male_class]
        return pm_oof

    pm_oof: dict[str, np.ndarray] = {
        c: compute_pm_oof(hs[c][best_layer_by_cond[c]], y)
        for c in COND_NAMES
    }
    log("OOF P(Male) computed (used in strip plots)")

    # ════════════════════════════════════════════════════════════════════════════
    # GENERALIZATION TEST
    # ════════════════════════════════════════════════════════════════════════════
    # For each ONLY_* condition, hold out 1 Female-coded and 1 Male-coded
    # indicator value.  Train the probe without those histories, test on them.

    # ── compute per-name and per-artist gender distribution in the dataset ────
    name_gender_dist:   dict[str, Counter] = {}
    artist_gender_dist: dict[str, Counter] = {}
    for i in range(N):
        nm = history_to_name.get(i)
        ar = history_to_artist.get(i)
        g  = labels[i]
        if nm:
            name_gender_dist.setdefault(nm, Counter())[g] += 1
        if ar:
            artist_gender_dist.setdefault(ar, Counter())[g] += 1

    log("\nName gender distribution (from dataset):")
    for nm, gc in sorted(name_gender_dist.items()):
        log(f"  {nm!r:32s}: {gc['Female']}F / {gc['Male']}M")

    log("Artist gender distribution (from dataset):")
    for ar, gc in sorted(artist_gender_dist.items()):
        log(f"  {ar!r:28s}: {gc['Female']}F / {gc['Male']}M")

    def _pick_holdout(gender_dist: dict[str, Counter], label_map: dict[str, str],
                      is_gender_csv: bool, min_n: int = 5):
        """Pick 1 Female-coded and 1 Male-coded indicator value to hold out.

        For Gender CSV indicators the gender label comes directly from the CSV.
        For Race indicators we compute the predominant gender from the dataset
        (pure if possible; otherwise the most skewed available).
        Returns [female_val, male_val].
        """
        female_candidates = []
        male_candidates   = []
        if is_gender_csv:
            # gender_dist maps phrase -> list[int]; label_map is phrase_to_gender
            for p, idxs in gender_dist.items():
                n = len(idxs)
                if n < min_n:
                    continue
                if label_map[p] == "Female":
                    female_candidates.append((p, n))
                else:
                    male_candidates.append((p, n))
        else:
            # gender_dist maps phrase -> Counter({Female: n, Male: m})
            for p, gc in gender_dist.items():
                n_f, n_m = gc["Female"], gc["Male"]
                total = n_f + n_m
                if total < min_n:
                    continue
                if n_f > 0 and n_m == 0:
                    female_candidates.append((p, n_f))
                elif n_m > 0 and n_f == 0:
                    male_candidates.append((p, n_m))
                elif n_f >= n_m * 3:      # at least 3:1 Female-skewed
                    female_candidates.append((p, n_f))
                elif n_m >= n_f * 3:      # at least 3:1 Male-skewed
                    male_candidates.append((p, n_m))

        female_candidates.sort(key=lambda x: -x[1])
        male_candidates.sort(key=lambda x:   -x[1])
        f_val = female_candidates[0][0] if female_candidates else None
        m_val = male_candidates[0][0]   if male_candidates   else None
        return f_val, m_val

    # Movie holdout: Female-coded movie with most histories + Male-coded with most
    movie_f, movie_m = _pick_holdout(
        {p: v for p, v in phrase_in_history.items() if p in movie_phrases},
        gender_phrase_to_gender,
        is_gender_csv=True,
    )
    # Hobby holdout
    hobby_f, hobby_m = _pick_holdout(
        {p: v for p, v in phrase_in_history.items() if p in hobby_phrases},
        gender_phrase_to_gender,
        is_gender_csv=True,
    )
    # Name holdout: purely Female and purely Male name (by dataset label)
    name_f, name_m = _pick_holdout(name_gender_dist, {}, is_gender_csv=False)
    # Artist holdout
    artist_f, artist_m = _pick_holdout(artist_gender_dist, {}, is_gender_csv=False)

    HOLDOUT: dict[str, list] = {
        "only_movie":  [v for v in [movie_f,  movie_m]  if v],
        "only_hobby":  [v for v in [hobby_f,  hobby_m]  if v],
        "only_name":   [v for v in [name_f,   name_m]   if v],
        "only_artist": [v for v in [artist_f, artist_m] if v],
    }

    log("\nHeld-out indicator values (excluded from probe training):")
    for cond, vals in HOLDOUT.items():
        log(f"  {cond}: {vals}")

    # Per-condition slot lookup: which dict maps history index -> indicator value?
    SLOT_LOOKUP = {
        "only_movie":  history_to_movie,
        "only_hobby":  history_to_hobby,
        "only_name":   history_to_name,
        "only_artist": history_to_artist,
    }

    # ── run generalization probes ─────────────────────────────────────────────
    gen_results: dict[str, dict] = {}
    log("\n=== GENERALIZATION TEST ===")
    log("(Probe trained on all non-held-out histories, tested on held-out histories)")
    for cond, holdout_vals in HOLDOUT.items():
        if not holdout_vals:
            log(f"  {cond}: no valid holdout found — skipping")
            continue
        slot_map   = SLOT_LOOKUP[cond]
        bl         = best_layer_by_cond[cond]
        X_cond     = hs[cond][bl]

        # Mask of held-out histories
        held_out_mask = np.array([
            slot_map.get(i, "") in holdout_vals
            for i in range(N)
        ], dtype=bool)

        n_train = int((~held_out_mask).sum())
        n_test  = int(held_out_mask.sum())
        log(f"\n  [{cond}]  held-out={holdout_vals}")
        log(f"    train={n_train}  test={n_test}  layer={bl}")

        # Gender breakdown of test set
        test_f = int((held_out_mask & female_mask).sum())
        test_m = int((held_out_mask & male_mask).sum())
        log(f"    test gender: {test_f}F / {test_m}M")

        std_acc   = acc[cond][bl]          # standard CV accuracy at best layer
        gen_acc, _ = probe_generalization(X_cond, y, held_out_mask)
        log(f"    standard CV acc  = {std_acc:.3f}")
        log(f"    generalization   = {gen_acc:.3f}" +
            ("  (nan: too few examples or single class)" if np.isnan(gen_acc) else ""))

        gen_results[cond] = {
            "holdout_vals": holdout_vals,
            "n_train":      n_train,
            "n_test":       n_test,
            "test_f":       test_f,
            "test_m":       test_m,
            "best_layer":   bl,
            "std_acc":      std_acc,
            "gen_acc":      gen_acc,
        }

    # ── ensure only_artist appears in gen_results (no gender-coded holdout) ────
    if "only_artist" not in gen_results:
        bl_ar = best_layer_by_cond["only_artist"]
        gen_results["only_artist"] = {
            "holdout_vals": [],
            "n_train": N,
            "n_test":  0,
            "test_f":  0,
            "test_m":  0,
            "best_layer": bl_ar,
            "std_acc":  acc["only_artist"][bl_ar],
            "gen_acc":  float("nan"),
        }
        log("  [only_artist]: artists are ~50/50 gender split in dataset — no valid holdout")

    # ─── Save generalization results ─────────────────────────────────────────
    with open(OUT / "data" / "generalization_results.csv", "w",
              newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "condition", "holdout_vals", "n_train", "n_test",
            "test_f", "test_m", "best_layer", "std_acc", "gen_acc"
        ])
        w.writeheader()
        for cond, r in gen_results.items():
            w.writerow({
                "condition":   cond,
                "holdout_vals": "|".join(r["holdout_vals"]),
                **{k: v for k, v in r.items() if k != "holdout_vals"},
            })
    log("\nGeneralization results saved")

    # ═══════════════════════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════════════════════
    FCOL   = "#2878b5"
    MCOL   = "#d97706"
    GREY   = "#888888"
    CCOLS  = {
        "full":        "#333333",
        "only_movie":  "#e63946",
        "only_hobby":  "#457b9d",
        "only_name":   "#2a9d8f",
        "only_artist": "#e9a800",
        "ctrl":        GREY,
    }
    CLABELS = {
        "full":        "FULL (all 4 indicators)",
        "only_movie":  "ONLY_MOVIE",
        "only_hobby":  "ONLY_HOBBY",
        "only_name":   "ONLY_NAME",
        "only_artist": "ONLY_ARTIST",
        "ctrl":        "Shuffled-label control",
    }

    layers = all_layers
    rng_j  = np.random.default_rng(0)
    jitter = rng_j.uniform(-0.15, 0.15, N)

    def strip_panel(ax, pm_vals: np.ndarray, title: str, show_legend: bool = False) -> None:
        ax.scatter(jitter[female_mask], pm_vals[female_mask],
                   c=FCOL, alpha=0.45, s=16, linewidths=0, label="Female history")
        ax.scatter(1 + jitter[male_mask], pm_vals[male_mask],
                   c=MCOL, alpha=0.45, s=16, linewidths=0, label="Male history")
        ax.axhline(0.5, color="grey", lw=0.9, ls="--")
        ax.set_xticks([0, 1], ["Female\nhistories", "Male\nhistories"])
        ax.set_ylabel("P(Male)", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(-0.05, 1.08)
        if show_legend:
            ax.legend(fontsize=7, loc="upper center")

    # ── Plot 1: probe accuracy per layer ──────────────────────────────────────
    markers_map = {"full":"o-","only_movie":"s--","only_hobby":"^--",
                   "only_name":"D--","only_artist":"P--","ctrl":":"}
    fig, ax = plt.subplots(figsize=(13, 5))
    for c in COND_NAMES + ["ctrl"]:
        ax.plot(layers, [acc[c][l] for l in layers],
                markers_map[c], color=CCOLS[c],
                lw=1.5 if c == "ctrl" else 2, ms=5, label=CLABELS[c])
    ax.axhline(0.5, color="#cccccc", lw=0.9, ls=":")
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("5-fold CV accuracy", fontsize=11)
    ax.set_title(
        "Linear probe accuracy — single-indicator conditions\n"
        "(fresh probe retrained per condition; probe target = Gender)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0.4, 1.05)
    ax.set_xlim(-0.5, max(layers) + 0.5)
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "probe_accuracy.png", dpi=160)
    plt.close(fig)
    log("Saved probe_accuracy.png")

    # ── Plot 2: P(Male) strip plots (2×3 grid) ───────────────────────────────
    # Use OOF predictions at each condition's best layer so title layer matches data.
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    strip_panel(axes[0, 0], pm_oof["full"],
                f"FULL (all 4 indicators)\nOOF probe · layer {best_layer_by_cond['full']}",
                show_legend=True)
    strip_panel(axes[0, 1], pm_oof["only_movie"],
                f"ONLY_MOVIE\nOOF probe · layer {best_layer_by_cond['only_movie']}")
    strip_panel(axes[0, 2], pm_oof["only_hobby"],
                f"ONLY_HOBBY\nOOF probe · layer {best_layer_by_cond['only_hobby']}")
    strip_panel(axes[1, 0], pm_oof["only_name"],
                f"ONLY_NAME\nOOF probe · layer {best_layer_by_cond['only_name']}")
    strip_panel(axes[1, 1], pm_oof["only_artist"],
                f"ONLY_ARTIST\nOOF probe · layer {best_layer_by_cond['only_artist']}")
    axes[1, 2].axis("off")
    fig.suptitle(
        "P(Male) per history — single-indicator conditions\n"
        "Blue = Female history  |  Orange = Male history\n"
        "Each panel uses a fresh out-of-fold probe at its condition's best layer",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "p_male_distribution.png", dpi=160)
    plt.close(fig)
    log("Saved p_male_distribution.png")

    # ── Plot 3: generalization test results (bar chart) ───────────────────────
    gen_conds = [c for c in ["only_movie", "only_hobby", "only_name", "only_artist"]
                 if c in gen_results]
    std_accs  = [gen_results[c]["std_acc"] for c in gen_conds]
    gen_accs  = [gen_results[c]["gen_acc"] for c in gen_conds]
    x_pos     = np.arange(len(gen_conds))
    bar_w     = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_std = ax.bar(x_pos - bar_w / 2, std_accs, bar_w,
                      color=[CCOLS[c] for c in gen_conds], alpha=0.9,
                      label="Standard CV accuracy (all indicator values)")
    # Replace NaN gen_accs with 0 for rendering; annotate separately.
    _gen_render = [g if not np.isnan(g) else 0.0 for g in gen_accs]
    bars_gen = ax.bar(x_pos + bar_w / 2, _gen_render, bar_w,
                      color=[CCOLS[c] for c in gen_conds], alpha=0.45,
                      hatch="///", label="Generalization accuracy (held-out values only)")

    ax.axhline(0.5, color="black", lw=1, ls="--", label="Chance (0.5)")
    ax.set_xticks(x_pos)

    def _xtick(c):
        vals = gen_results[c]["holdout_vals"]
        if vals:
            return f"{CLABELS[c]}\nheld-out: {', '.join(vals)}"
        return f"{CLABELS[c]}\n(no gender-coded holdout\nin dataset)"

    ax.set_xticklabels([_xtick(c) for c in gen_conds], fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0.0, 1.12)   # full range so bars below 0.4 (e.g. 0.378) are visible
    ax.set_title(
        "Standard vs Generalization probe accuracy — single-indicator conditions\n"
        "Solid bar = trained and tested on ALL values (5-fold CV)\n"
        "Hatched bar = trained WITHOUT held-out values, tested ONLY on held-out values",
        fontsize=10,
    )
    ax.legend(fontsize=9)

    # Annotate every bar; handle sub-0.5 generalization values explicitly
    for i, (bar_s, bar_g, ga) in enumerate(zip(bars_std, bars_gen, gen_accs)):
        # std bar annotation
        hs_h = bar_s.get_height()
        ax.text(bar_s.get_x() + bar_s.get_width() / 2, hs_h + 0.01,
                f"{hs_h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        # gen bar annotation
        if not np.isnan(ga):
            ax.text(bar_g.get_x() + bar_g.get_width() / 2, ga + 0.01,
                    f"{ga:.3f}", ha="center", va="bottom", fontsize=9)
        else:
            # no valid holdout — explain why
            ax.text(x_pos[i] + bar_w / 2, 0.03,
                    "artists ~50/50\nacross genders\n→ no valid holdout",
                    ha="center", va="bottom", fontsize=7,
                    style="italic", color="#777777")

    fig.tight_layout()
    fig.savefig(OUT / "plots" / "generalization_test.png", dpi=160)
    plt.close(fig)
    log("Saved generalization_test.png")

    # ── Plot 4: per-indicator P(Male) under its own ONLY condition ───────────
    # For each indicator value, show the mean P(Male) from the frozen FULL probe
    # when that indicator is the ONLY visible signal.
    gender_stats: list[dict] = []
    race_stats:   list[dict] = []
    for p, idxs in phrase_in_history.items():
        if not idxs:
            continue
        ia = np.array(idxs)
        if p in gender_phrase_to_slot:
            slot = gender_phrase_to_slot[p]
            cond = "only_movie" if slot == "Movie" else "only_hobby"
            gender_stats.append({
                "phrase":     p,
                "slot":       slot,
                "gender_csv": gender_phrase_to_gender[p],
                "n":          len(idxs),
                "pm_full":    float(pm["full"][ia].mean()),
                "pm_cond":    float(pm[cond][ia].mean()),
            })
        else:
            slot   = race_phrase_to_slot[p]
            cond   = "only_name" if slot == "Name" else "only_artist"
            gc     = name_gender_dist.get(p) or artist_gender_dist.get(p) or Counter()
            gender_stats_label = ("Female" if gc["Female"] > gc["Male"] else
                                  "Male"   if gc["Male"]   > gc["Female"] else "Mixed")
            race_stats.append({
                "phrase":      p,
                "slot":        slot,
                "race_group":  race_phrase_to_race[p],
                "gender_data": gender_stats_label,
                "n":           len(idxs),
                "pm_full":     float(pm["full"][ia].mean()),
                "pm_cond":     float(pm[cond][ia].mean()),
            })

    gender_stats.sort(key=lambda r: r["pm_cond"])
    race_stats.sort(  key=lambda r: r["pm_cond"])

    n_race_plot = len(race_stats)
    fig_h = max(12, max(len(gender_stats), n_race_plot) * 0.35 + 2)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(22, fig_h))

    def _dot_panel(ax, stats, title, cond_key, x_label="pm_cond"):
        y_pos = np.arange(len(stats))
        lbl   = [f"{r['phrase']}  ({r['slot']})" for r in stats]
        for i, row in enumerate(stats):
            vf = row["pm_full"] - 0.5
            vc = row[x_label]   - 0.5
            base = FCOL if row.get("gender_csv") == "Female" or row.get("gender_data") == "Female" else \
                   MCOL if row.get("gender_csv") == "Male"   or row.get("gender_data") == "Male"   else \
                   GREY
            ax.plot([vf, vc], [i, i], color="grey", alpha=0.2, lw=1, zorder=1)
            ax.scatter(vf, i, color=base, s=70, marker="o", zorder=4)
            ax.scatter(vc, i, color=base, s=70, marker="D", zorder=4,
                       facecolors="none" if vc == vf else base,
                       edgecolors=base, linewidths=1.5)
        ax.set_yticks(y_pos, lbl, fontsize=8 if len(stats) > 25 else 9)
        ax.set_xlim(-0.6, 0.6)
        ax.axvline(0, color="black", lw=1.1)
        ax.set_xticks([-0.5,-0.25,0,0.25,0.5],
                      ["100%\nFemale","75%\nFemale","Neutral","75%\nMale","100%\nMale"],
                      fontsize=8.5)
        ax.set_xlabel("P(Male) - 0.5  (FULL-trained frozen probe)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(
            handles=[
                Line2D([0],[0], marker="o", color="grey", linewidth=0, ms=8, label="● FULL"),
                Line2D([0],[0], marker="D", color="grey", linewidth=0, ms=8, label="◆ ONLY this type"),
                Patch(facecolor=FCOL, alpha=0.9, label="Female-coded"),
                Patch(facecolor=MCOL, alpha=0.9, label="Male-coded"),
            ],
            fontsize=8, loc="lower right",
        )

    _dot_panel(ax_l, gender_stats,
               "Gender indicators: P(Male) in FULL vs ONLY_MOVIE / ONLY_HOBBY\n"
               "(● FULL with all 4  |  ◆ only this indicator type present)",
               "only_movie")
    _dot_panel(ax_r, race_stats,
               "Race indicators: P(Male) in FULL vs ONLY_NAME / ONLY_ARTIST\n"
               "(● FULL with all 4  |  ◆ only this indicator type present)",
               "only_name")

    fig.suptitle(
        "Per-indicator P(Male) — how probe output changes when only one indicator type is present\n"
        "Blue = Female-coded  |  Orange = Male-coded  |  Frozen FULL-trained probe",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "per_indicator_single.png", dpi=160)
    plt.close(fig)
    log("Saved per_indicator_single.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n=== SUMMARY ===")
    for c in COND_NAMES:
        bl = best_layer_by_cond[c]
        log(f"  [{c:12s}] best_layer={bl}  acc={acc[c][bl]:.3f}  "
            f"Female P(Male)={pm[c][female_mask].mean():.3f}  "
            f"Male P(Male)={pm[c][male_mask].mean():.3f}")
    log(f"  [ctrl        ] best_layer={best_layer}  acc={acc['ctrl'][best_layer]:.3f}")
    log("")
    log("Generalization test:")
    for cond, r in gen_results.items():
        log(f"  [{cond:12s}] std_acc={r['std_acc']:.3f}  "
            f"gen_acc={r['gen_acc']:.3f}  "
            f"held-out={r['holdout_vals']}  "
            f"(train={r['n_train']}, test={r['n_test']} [{r['test_f']}F/{r['test_m']}M])")
    log("=== ALL DONE ===")

except Exception as exc:
    log(f"EXCEPTION: {exc}\n{traceback.format_exc()}")
    raise

finally:
    LOG.close()
