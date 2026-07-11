"""
WildChat Dataset Explicit Gender Evidence Analysis 
=======================================================

Dataset reference:
    Zhao, W., Ren, X., Hessel, J., Cardie, C., Choi, Y., & Deng, Y. (2024).
    WildChat: 1M ChatGPT Interaction Logs in the Wild. ICLR 2024.
    https://huggingface.co/datasets/allenai/WildChat-1M

Requirements:
    pip install datasets pandas --break-system-packages
"""

import re
from pathlib import Path
import pandas as pd
from datasets import load_dataset

# All outputs are saved next to this script, regardless of the working directory.
OUTPUT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 0. Filter settings
# ---------------------------------------------------------------------------

FILTER_LANGUAGE = "English"

# Drop conversations WildChat flagged as toxic (see is_toxic_conversation)
FILTER_TOXIC = True

MIN_FIRST_PROMPT_WORDS = 3
MAX_FIRST_PROMPT_WORDS = 300

DEFAULT_MAX_SAMPLES = 1000000
DEFAULT_MAX_SCAN = 20000000


# ---------------------------------------------------------------------------
# 1. Age shorthand setting
# ---------------------------------------------------------------------------

# Ages 18-99 only, to avoid false positives like "8m" or "mg/m3".
AGE_PATTERN = r"(?:1[8-9]|[2-9][0-9])"


# ---------------------------------------------------------------------------
# 1b. Fast English heuristic
# ---------------------------------------------------------------------------
# Backup check for rows where the `language` field is wrong: rejects text
# with non-Latin scripts or less than 85% ASCII characters.

_NON_LATIN_RE = re.compile(
    r'['
    r'\u0600-\u06FF'   # Arabic
    r'\u0400-\u04FF'   # Cyrillic
    r'\u4E00-\u9FFF'   # CJK Unified Ideographs
    r'\uF900-\uFAFF'   # CJK Compatibility Ideographs
    r'\u3040-\u30FF'   # Hiragana / Katakana
    r'\uAC00-\uD7AF'   # Korean Hangul Syllables
    r'\u1100-\u11FF'   # Korean Jamo
    r'\u3130-\u318F'   # Korean Compatibility Jamo
    r'\u0900-\u097F'   # Devanagari (Hindi)
    r'\u0E00-\u0E7F'   # Thai
    r'\u0600-\u06FF'   # Arabic (duplicate guard)
    r']'
)

def is_likely_english(text: str, min_ascii_ratio: float = 0.85) -> bool:
    """Returns False if the text is clearly not English. Short texts always pass."""
    if len(text) < 20:
        return True
    if _NON_LATIN_RE.search(text):
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return (ascii_chars / len(text)) >= min_ascii_ratio


# ---------------------------------------------------------------------------
# 2. Strong explicit self-identification patterns
# ---------------------------------------------------------------------------

MALE_SELF_PATTERNS = [
    # Direct first-person identity
    r"\b(?:i\s+am|i'm|im)\s+(?:a\s+)?(?:man|male|guy|boy)\b",
    r"\bi\s+identify\s+as\s+(?:a\s+)?(?:man|male)\b",
    r"\bi,\s*(?:a\s+)?(?:man|male|guy|boy)\b",

    # "As a man, I..." with nearby first-person word
    r"(?:^|[.!?]\s*)as\s+(?:a\s+)?(?:man|male|guy|boy)\b(?=[^.!?]{0,200}\b(?:i|me|my|mine|myself)\b)",

    # First-person perspective/body/identity
    r"\bfrom\s+my\s+(?:male|man's|man|masculine)\s+perspective\b",
    r"\bmy\s+(?:male|masculine)\s+(?:body|identity|experience|perspective|health)\b",

    # Age + gender written out, e.g. "I'm a 25-year-old man"
    r"\b(?:i\s+am|i'm|im)\s+(?:an?\s+)?\d{1,2}[-\s]?(?:years?[-\s]?old)?\s+(?:man|male|guy|boy)\b",

    # "I am a [descriptor] man/male/guy" — blocklist excludes third-party or
    # fictional subjects (e.g. "I am looking for a guy", "I'm creating a male character")
    r"\b(?:i\s+am|i'm|im)\s+(?:an?\s+)?(?:(?!(?:not|never|told|said|think|want|need|writing|asking|looking|seeking|searching|creating|drawing|painting|sketching|making|designing|developing|building|playing|roleplaying|portraying|voicing|controlling|generating|rendering|animating|modeling|modelling|describing|imagining|picturing|for|about|with|into)\b)\w+\s+){1,3}(?:man|male|guy)\b",

    # Pronouns
    r"\bmy\s+pronouns\s+are\s+(?:he/him|he\s*/\s*him)\b",
    r"\bi\s+use\s+(?:he/him|he\s*/\s*him)\s+pronouns\b",
]


FEMALE_SELF_PATTERNS = [
    # Direct first-person identity
    r"\b(?:i\s+am|i'm|im)\s+(?:a\s+)?(?:woman|female|girl|lady)\b",
    r"\bi\s+identify\s+as\s+(?:a\s+)?(?:woman|female)\b",
    r"\bi,\s*(?:a\s+)?(?:woman|female|girl|lady)\b",

    # "As a woman, I..." with nearby first-person word
    r"(?:^|[.!?]\s*)as\s+(?:a\s+)?(?:woman|female|girl|lady)\b(?=[^.!?]{0,200}\b(?:i|me|my|mine|myself)\b)",

    # First-person perspective/body/identity
    r"\bfrom\s+my\s+(?:female|woman's|woman|feminine)\s+perspective\b",
    r"\bmy\s+(?:female|feminine)\s+(?:body|identity|experience|perspective|health)\b",

    # Age + gender written out, e.g. "I'm a 30-year-old woman"
    r"\b(?:i\s+am|i'm|im)\s+(?:an?\s+)?\d{1,2}[-\s]?(?:years?[-\s]?old)?\s+(?:woman|female|girl|lady)\b",

    # "I am a [descriptor] woman/girl/female" — blocklist excludes third-party or
    # fictional subjects (e.g. "I am looking for a girl", "I'm drawing a woman")
    r"\b(?:i\s+am|i'm|im)\s+(?:an?\s+)?(?:(?!(?:not|never|told|said|think|want|need|writing|asking|looking|seeking|searching|creating|drawing|painting|sketching|making|designing|developing|building|playing|roleplaying|portraying|voicing|controlling|generating|rendering|animating|modeling|modelling|describing|imagining|picturing|for|about|with|into)\b)\w+\s+){1,3}(?:woman|female|girl|lady)\b",

    # Pronouns
    r"\bmy\s+pronouns\s+are\s+(?:she/her|she\s*/\s*her)\b",
    r"\bi\s+use\s+(?:she/her|she\s*/\s*her)\s+pronouns\b",
]


# ---------------------------------------------------------------------------
# 3. Shorthand patterns: 25M, 30F, M25, F30
# ---------------------------------------------------------------------------

MALE_SHORTHAND_PATTERNS = [
    # "I am 25M" — age and letter glued, so "25 minutes" / "30 million" don't match
    rf"\b(?:i\s+am|i'm|im)\s+(?:a\s+)?\(?{AGE_PATTERN}m\)?(?![a-z])",

    # "I (25M) need advice"
    rf"\bi\s*\(\s*{AGE_PATTERN}\s*m\s*\)",

    # "25M here" / "M25 looking"
    rf"(?<![a-z0-9])(?:{AGE_PATTERN}m|m{AGE_PATTERN})\s+(?:here|looking|seeking|needing|wanting)\b",

    # Spelled-out "28 male" near a first-person word (bare "m" excluded — unit collision)
    r"(?:"
      r"(?:i\b|i'm|im\b|i\s+am|i\s+need|me\b).{0,60}?"
      r"\b(?:" + AGE_PATTERN + r"[\s,]?\s*male\b|male[\s,]+" + AGE_PATTERN + r"\b)"
    r"|"
      r"(?:" + AGE_PATTERN + r"[\s,]?\s*male\b|male[\s,]+" + AGE_PATTERN + r"\b)"
      r".{0,60}?(?:\b(?:here|i\b|i'm|im\b|me\b|my\b))"
    r")",
]


FEMALE_SHORTHAND_PATTERNS = [
    # "I am 25F" — age and letter glued, so "bake at 80f" / "32 f" don't match
    rf"\b(?:i\s+am|i'm|im)\s+(?:a\s+)?\(?{AGE_PATTERN}f\)?(?![a-z])",

    # "I (25F) need advice"
    rf"\bi\s*\(\s*{AGE_PATTERN}\s*f\s*\)",

    # "25F here" / "F25 looking"
    rf"(?<![a-z0-9])(?:{AGE_PATTERN}f|f{AGE_PATTERN})\s+(?:here|looking|seeking|needing|wanting)\b",

    # "28/F" slash notation ("F/28" excluded — collides with aperture values like f/22)
    rf"\b{AGE_PATTERN}\s*/\s*f\b",

    # Spelled-out "28 female" near a first-person word (bare "f" excluded — Fahrenheit collision)
    r"(?:"
      r"(?:i\b|i'm|im\b|i\s+am|i\s+need|me\b).{0,60}?"
      r"\b(?:" + AGE_PATTERN + r"[\s,]?\s*female\b|female[\s,]+" + AGE_PATTERN + r"\b)"
    r"|"
      r"(?:" + AGE_PATTERN + r"[\s,]?\s*female\b|female[\s,]+" + AGE_PATTERN + r"\b)"
      r".{0,60}?(?:\b(?:here|i\b|i'm|im\b|me\b|my\b))"
    r")",
]


# ---------------------------------------------------------------------------
# 4. Gender questioning / transition context
# ---------------------------------------------------------------------------

GENDER_TRANSITION_CONTEXT_PATTERNS = [
    r"\b(?:i\s+am|i'm|im)\s+(?:a\s+)?boy\b.{0,120}\b(?:want|wanna|wish|would\s+like)\s+(?:to\s+)?(?:be|become|look|present)\s+(?:like\s+)?(?:a\s+)?girl\b",
    r"\b(?:i\s+am|i'm|im)\s+(?:a\s+)?girl\b.{0,120}\b(?:want|wanna|wish|would\s+like)\s+(?:to\s+)?(?:be|become|look|present)\s+(?:like\s+)?(?:a\s+)?boy\b",

    r"\b(?:i\s+am|i'm|im)\s+transgender\b",
    r"\b(?:i\s+am|i'm|im)\s+(?:a\s+)?trans\s+(?:man|woman|girl|boy|male|female)\b",
    r"\b(?:i\s+identify\s+as)\s+(?:a\s+)?trans\s+(?:man|woman|girl|boy|male|female)\b",
    r"\bi\s+(?:am|was)\s+assigned\s+(?:male|female)\s+at\s+birth\b",
    r"\b(?:amab|afab)\b",
]


# ---------------------------------------------------------------------------
# 5. Medium confidence contextual evidence
# ---------------------------------------------------------------------------
# Two checks per prompt:
#   A: first-person gendered role/phrase
#   B: personal topic keyword AND "for a [gender]" / "as a [gender]" phrase

MALE_ROLE_PATTERNS = [
    # First-person gendered family/relationship roles (strict adjacency)
    r"\b(?:i\s+am|i'm|im)\s+(?:a\s+)?(?:father|dad|daddy|husband|boyfriend|son|uncle|brother|grandfather|grandpa|stepfather|stepdad)\b",

    # Descriptor/age-tolerant roles, e.g. "I'm a 30yo dad" — blocklist prevents
    # matches like "I'm writing a dad joke"
    r"\b(?:i\s+am|i'm|im)\s+(?:a\s+)?(?:(?!(?:not|never|of|told|said|think|want|need|writing|asking|looking|seeking|searching|creating|drawing|painting|sketching|making|designing|developing|building|playing|roleplaying|portraying|voicing|controlling|generating|rendering|animating|modeling|modelling|describing|imagining|picturing|for|about|with|into)\b)\w+\s+){0,3}"
    r"(?:father|dad|daddy|husband|grandfather|grandpa|stepfather|stepdad)\b",

    # "as a father/dad, I..."
    r"as\s+(?:a\s+)?(?:father|dad|daddy|husband|boyfriend|son|uncle|brother|grandfather|grandpa)\b(?=[^.!?]{0,200}\b(?:i|me|my)\b)",
]

FEMALE_ROLE_PATTERNS = [
    # First-person gendered roles (strict adjacency); relational roles appear
    # here only, to avoid false hits like "I'm a fan of my sister"
    r"\b(?:i\s+am|i'm|im)\s+(?:a\s+)?(?:woman|female|mother|mom|mommy|mum|mummy|wife|girlfriend|daughter|aunt|auntie|sister|grandmother|grandma|nana|niece|stepmom|stepmother|nursing\s+mother|breastfeeding\s+mother)\b",

    # Descriptor/age-tolerant roles, e.g. "I'm a 23yo mom" — blocklist prevents
    # matches like "I'm writing a mom blog"
    r"\b(?:i\s+am|i'm|im)\s+(?:a\s+)?(?:(?!(?:not|never|of|told|said|think|want|need|writing|asking|looking|seeking|searching|creating|drawing|painting|sketching|making|designing|developing|building|playing|roleplaying|portraying|voicing|controlling|generating|rendering|animating|modeling|modelling|describing|imagining|picturing|for|about|with|into)\b)\w+\s+){0,3}"
    r"(?:mother|mom|mommy|mum|mummy|wife|grandmother|grandma|nana|stepmom|stepmother)\b",

    # "as a mother/wife, I..."
    r"as\s+(?:a\s+)?(?:mother|mom|mum|wife|girlfriend|daughter|aunt|sister|grandmother|grandma)\b(?=[^.!?]{0,200}\b(?:i|me|my)\b)",

    # Compound role phrases
    r"\bsingle\s+mom\b",
    r"\bstay[\s-]?at[\s-]?home\s+mom\b",
    r"\bmom\s+of\s+\d\b",
    r"\bbeing\s+a\s+mom\b",
    r"\bi(?:'m|\s+am|\s+just\s+became)\s+(?:a\s+)?grandma\b",
    r"\bi'?m?\s+(?:the\s+)?maid\s+of\s+honor\b",
    r"\bi'?m?\s+(?:a\s+)?bridesmaid\b",

]

# Request-style context ("give me a workout for a woman") — applied as an
# AND check in score_initial_prompt().

PERSONAL_TOPIC_PATTERN = re.compile(
    r"\b(?:diet|meal\s*plan|workout|exercise\s*plan|fitness\s*plan|training\s*plan|"
    r"routine|skincare|skin\s*care|haircut|hairstyle|clothing|outfit|style|wardrobe|"
    r"health|fitness|nutrition|calorie|weight\s*loss|weight\s*gain|hormone|supplement)\b",
    re.IGNORECASE,
)

MALE_REQUEST_GENDER_PATTERN = re.compile(
    r"\bfor\s+(?:a\s+)?(?:man|male|guy|boy)\b"
    r"|\bas\s+(?:a\s+)?(?:man|male|guy|boy)\b",
    re.IGNORECASE,
)

FEMALE_REQUEST_GENDER_PATTERN = re.compile(
    r"\bfor\s+(?:a\s+)?(?:woman|female|girl|lady)\b"
    r"|\bas\s+(?:a\s+)?(?:woman|female|girl|lady)\b",
    re.IGNORECASE,
)

FIRST_PERSON_PATTERN = re.compile(
    r"\b(?:i|me|my|mine|myself|i'm|im|i\s+am|i\s+need|i\s+want|give\s+me|help\s+me|make\s+me)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 6. Compile regex
# ---------------------------------------------------------------------------

MALE_SELF_REGEX = [
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in MALE_SELF_PATTERNS
]

FEMALE_SELF_REGEX = [
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in FEMALE_SELF_PATTERNS
]

MALE_SHORTHAND_REGEX = [
    re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    for pattern in MALE_SHORTHAND_PATTERNS
]

FEMALE_SHORTHAND_REGEX = [
    re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    for pattern in FEMALE_SHORTHAND_PATTERNS
]

GENDER_TRANSITION_CONTEXT_REGEX = [
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in GENDER_TRANSITION_CONTEXT_PATTERNS
]

MALE_ROLE_REGEX = [
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in MALE_ROLE_PATTERNS
]

FEMALE_ROLE_REGEX = [
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in FEMALE_ROLE_PATTERNS
]


# ---------------------------------------------------------------------------
# 7. Helper functions
# ---------------------------------------------------------------------------

def normalize_text(text):
    """Normalize whitespace and apostrophes."""
    text = text or ""
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("`", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_first_user_prompt(messages):
    """Returns only the first user message in the conversation."""
    return next(
        (
            normalize_text(m.get("content", "") or "")
            for m in messages
            if m.get("role") == "user"
        ),
        ""
    )


def find_pattern_hits(regex_list, text):
    """Returns matched text plus the regex pattern that fired."""
    hits = []
    for regex in regex_list:
        match = regex.search(text)
        if match:
            matched_text = match.group(0)
            hits.append(f"{matched_text!r} <= {regex.pattern[:80]}")
    return hits


# ---------------------------------------------------------------------------
# 8. Filter functions
# ---------------------------------------------------------------------------

def is_toxic_conversation(example, first_prompt: str) -> bool:
    """
    True if the conversation should be dropped on safety grounds.
    Primary signal: WildChat's conversation-level `toxic` flag.
    Backup signal: OpenAI moderation result on the first user turn.
    """
    # Conversation-level flag shipped with the dataset
    if example.get("toxic", False):
        return True

    # Optional: also drop redacted rows — uncomment if wanted
    # if example.get("redacted", False):
    #     return True

    # Backup: OpenAI moderation on the first user turn
    mod = example.get("openai_moderation")
    if isinstance(mod, list) and mod:
        first_turn_mod = mod[0]
        if isinstance(first_turn_mod, dict) and first_turn_mod.get("flagged"):
            return True

    return False


def passes_language_gate(example) -> tuple[bool, str]:
    """Gate 1: language field check, applied before the max_scan budget."""
    if example.get("language", "") != FILTER_LANGUAGE:
        return False, "non_english_field"
    return True, "ok"


def passes_filters(example, first_prompt: str) -> tuple[bool, str]:
    """Gate 2 (English heuristic) + safety and content filters."""
    # Safety gate: drop conversations WildChat flags as toxic
    if FILTER_TOXIC and is_toxic_conversation(example, first_prompt):
        return False, "toxic"

    # Backup language check
    if not is_likely_english(first_prompt):
        return False, "non_english_text"

    if not first_prompt:
        return False, "empty_first_prompt"

    word_count = len(first_prompt.split())

    if word_count < MIN_FIRST_PROMPT_WORDS:
        return False, "too_short"

    if word_count > MAX_FIRST_PROMPT_WORDS:
        return False, "too_long"

    return True, "ok"


# ---------------------------------------------------------------------------
# 9. Scoring function
# ---------------------------------------------------------------------------

def score_initial_prompt(first_prompt):
    """
    Classifies whether the initial prompt contains explicit gender evidence.
    High: direct self-identification or shorthand. Medium: gendered role or
    personal-domain request. Unknown: no first-person evidence.
    """

    text = normalize_text(first_prompt)

    gender_transition_hits = find_pattern_hits(
        GENDER_TRANSITION_CONTEXT_REGEX, text
    )

    male_self_hits = (
        find_pattern_hits(MALE_SELF_REGEX, text)
        + find_pattern_hits(MALE_SHORTHAND_REGEX, text)
    )

    female_self_hits = (
        find_pattern_hits(FEMALE_SELF_REGEX, text)
        + find_pattern_hits(FEMALE_SHORTHAND_REGEX, text)
    )

    # First-person roles count as self-identification (high confidence)
    male_self_hits += find_pattern_hits(MALE_ROLE_REGEX, text)
    female_self_hits += find_pattern_hits(FEMALE_ROLE_REGEX, text)

    # Context hits hold only request-style evidence
    male_context_hits = []
    female_context_hits = []

    # Request-style context: personal topic AND gendered request AND first-person signal
    has_personal_topic = bool(PERSONAL_TOPIC_PATTERN.search(text))
    has_first_person = bool(FIRST_PERSON_PATTERN.search(text))

    if has_personal_topic and has_first_person:
        male_request = MALE_REQUEST_GENDER_PATTERN.search(text)
        female_request = FEMALE_REQUEST_GENDER_PATTERN.search(text)

        if male_request:
            male_context_hits.append(
                f"{male_request.group(0)!r} [+personal topic]"
            )
        if female_request:
            female_context_hits.append(
                f"{female_request.group(0)!r} [+personal topic]"
            )

    male_score = len(male_self_hits) * 3 + len(male_context_hits)
    female_score = len(female_self_hits) * 3 + len(female_context_hits)

    # Transition/questioning context takes priority
    if gender_transition_hits:
        predicted_gender = "gender_questioning_or_transition_context"
        confidence = "medium"

    elif male_score > 0 and female_score > 0:
        predicted_gender = "mixed_or_conflicting"
        confidence = "low"

    elif len(male_self_hits) > 0:
        predicted_gender = "male_self_identified"
        confidence = "high"

    elif len(female_self_hits) > 0:
        predicted_gender = "female_self_identified"
        confidence = "high"

    elif len(male_context_hits) > 0:
        predicted_gender = "male_contextual_evidence"
        confidence = "medium"

    elif len(female_context_hits) > 0:
        predicted_gender = "female_contextual_evidence"
        confidence = "medium"

    else:
        predicted_gender = "unknown"
        confidence = "low"

    return {
        "female_score": female_score,
        "male_score": male_score,
        "predicted_gender": predicted_gender,
        "confidence": confidence,
        "male_self_hits": " || ".join(male_self_hits),
        "female_self_hits": " || ".join(female_self_hits),
        "male_context_hits": " || ".join(male_context_hits),
        "female_context_hits": " || ".join(female_context_hits),
        "gender_transition_hits": " || ".join(gender_transition_hits),
    }


# ---------------------------------------------------------------------------
# 10. Main pipeline
# ---------------------------------------------------------------------------

def analyze_wildchat(
    split="train",
    max_samples=DEFAULT_MAX_SAMPLES,
    max_scan=DEFAULT_MAX_SCAN,
    conversations_path=OUTPUT_DIR / "wildchat_conversations.jsonl",
):
    ds = load_dataset(
        "allenai/WildChat-1M",
        split=split,
        streaming=True
    )

    rows = []
    conversations = []  # full conversations (role+content) keyed by id
    scanned = 0

    for example in ds:
        # Gate 1: non-English rows don't count toward max_scan
        lang_ok, _ = passes_language_gate(example)
        if not lang_ok:
            continue

        if len(rows) >= max_samples or scanned >= max_scan:
            break

        scanned += 1

        messages = example.get("conversation", [])
        if not messages:
            continue

        first_prompt = get_first_user_prompt(messages)

        # Gate 2 + safety and content filters
        keep, _ = passes_filters(example, first_prompt)
        if not keep:
            continue
        prompt_words = len(first_prompt.split())

        result = score_initial_prompt(first_prompt)

        # WildChat's unique identifier is conversation_hash
        conversation_id = example.get("conversation_hash", str(scanned))
        country = example.get("country", "")
        language = example.get("language", "")
        model = example.get("model", "")

        row = {
            "conversation_id": conversation_id,
            "country": country,
            "language": language,
            "model": model,
            "num_turns": len(messages),
            "first_prompt_words": prompt_words,

            "predicted_gender": result["predicted_gender"],
            "confidence": result["confidence"],
            "female_score": result["female_score"],
            "male_score": result["male_score"],

            "male_self_hits": result["male_self_hits"],
            "female_self_hits": result["female_self_hits"],
            "male_context_hits": result["male_context_hits"],
            "female_context_hits": result["female_context_hits"],
            "gender_transition_hits": result["gender_transition_hits"],

            "initial_prompt": first_prompt,
        }

        rows.append(row)

        # Save the full conversation keyed by id for local joining later
        conversations.append({
            "conversation_id": conversation_id,
            "messages": [
                {"role": m.get("role"), "content": m.get("content")}
                for m in messages
                if m.get("role") is not None and m.get("content") is not None
            ],
        })

    df = pd.DataFrame(rows)

    # Sidecar file with full conversations, needed for the downstream analysis
    import json as _json
    with open(conversations_path, "w", encoding="utf-8") as _cf:
        for _c in conversations:
            _cf.write(_json.dumps(_c, ensure_ascii=False) + "\n")

    return df


# ---------------------------------------------------------------------------
# 11. Save outputs
# ---------------------------------------------------------------------------

def save_gender_evidence(
    df,
    output_csv=OUTPUT_DIR / "wildchat_gender_evidence_results.csv",
    checked_csv=OUTPUT_DIR / "wildchat_gender_evidence_results_checked.csv",
):
    evidence = df[
        df["predicted_gender"].isin([
            "male_self_identified",
            "female_self_identified",
            "male_contextual_evidence",
            "female_contextual_evidence",
            "gender_questioning_or_transition_context",
        ])
    ].copy()

    # Empty column for manual review: put 1 (keep) or 0 (drop) on each row.
    evidence["correct"] = ""

    # Lay the review columns out so column P = 'correct' (empty), then the prompt.
    # All the analysis columns (A–O) keep their order in front; 'correct' becomes the
    # 16th column (P) and 'initial_prompt' follows it (Q).
    front = [c for c in evidence.columns if c not in ("correct", "initial_prompt")]
    evidence = evidence[front + ["correct", "initial_prompt"]]

    evidence.to_csv(output_csv, index=False)
    evidence.to_csv(checked_csv, index=False)

    return evidence


# ---------------------------------------------------------------------------
# 12. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    df = analyze_wildchat(
        split="train",
        max_samples=DEFAULT_MAX_SAMPLES,
        max_scan=DEFAULT_MAX_SCAN,
    )

    save_gender_evidence(df)