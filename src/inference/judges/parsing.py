"""Strict, deterministic parser for the <final_answer> sentinel.

No fuzzy / case-insensitive / substring matching anywhere. Case-sensitive equality only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from inference.judges.types import NONE_SENTINEL, ParseStatus

# DOTALL so a sentinel that crosses newlines still matches. The tag itself is matched
# case-insensitively (we don't care if a judge writes <Final_Answer>), but the *contents*
# are preserved verbatim for case-sensitive comparison against classes.
_SENTINEL_RE = re.compile(
    r"<\s*final_answer\s*>(.*?)<\s*/\s*final_answer\s*>",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class ParseOutcome:
    final_class: str | None
    none_declared: bool
    sentinel_found: bool
    sentinel_raw: str | None
    parse_status: ParseStatus


def parse_final_answer(raw: str, classes: list[str] | None) -> ParseOutcome:
    """Extract and validate the <final_answer> sentinel.

    If ``classes`` is None we return ``parse_status=FREE_FORM`` — the caller is in
    free-form mode and final_class stays None.
    """
    if classes is None:
        return ParseOutcome(
            final_class=None,
            none_declared=False,
            sentinel_found=False,
            sentinel_raw=None,
            parse_status=ParseStatus.FREE_FORM,
        )

    if not isinstance(raw, str) or not raw:
        return ParseOutcome(
            final_class=None,
            none_declared=False,
            sentinel_found=False,
            sentinel_raw=None,
            parse_status=ParseStatus.MISSING_SENTINEL,
        )

    matches = list(_SENTINEL_RE.finditer(raw))
    if not matches:
        return ParseOutcome(
            final_class=None,
            none_declared=False,
            sentinel_found=False,
            sentinel_raw=None,
            parse_status=ParseStatus.MISSING_SENTINEL,
        )

    last = matches[-1]
    if raw[last.end() :]:
        return ParseOutcome(
            final_class=None,
            none_declared=False,
            sentinel_found=True,
            sentinel_raw=None,
            parse_status=ParseStatus.UNMATCHED,
        )

    inner = last.group(1).strip()
    if inner == NONE_SENTINEL:
        return ParseOutcome(
            final_class=None,
            none_declared=True,
            sentinel_found=True,
            sentinel_raw=inner,
            parse_status=ParseStatus.NONE_DECLARED,
        )

    # Strict case-sensitive equality against the configured class list.
    for c in classes:
        if inner == c:
            return ParseOutcome(
                final_class=c,
                none_declared=False,
                sentinel_found=True,
                sentinel_raw=inner,
                parse_status=ParseStatus.MATCHED,
            )

    return ParseOutcome(
        final_class=None,
        none_declared=False,
        sentinel_found=True,
        sentinel_raw=inner,
        # Sentinel found but contents weren't a valid class — the single judge call
        # has already happened, so this becomes CLASSIFICATION_FAILED at the runner.
        parse_status=ParseStatus.UNMATCHED,
    )


__all__ = ["ParseOutcome", "parse_final_answer"]
