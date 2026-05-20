"""Prompt builder for the single-call judge.

The chat being judged is presented as a labelled <conversation> block inside one user
message (MT-Bench / FastChat / LangSmith convention) so the judge stays an outside
observer. The model is permitted to reason inline, but the system prompt enforces a
hard contract: the final line MUST be ``<final_answer>X</final_answer>`` where X is
verbatim one of the configured classes (or the NONE sentinel as a last resort).

There is intentionally only one call per (subject, judge) row — no extraction pass.
"""

from __future__ import annotations

from typing import Any

from inference.judges.types import (
    NONE_SENTINEL,
    JudgeConfig,
    JudgeSubject,
)


def _escape(text: str) -> str:
    # Light, reversible escaping of the closing tag only; transcripts may contain XML-ish
    # content from the subject which we shouldn't let collapse the surrounding tag.
    return text.replace("</turn>", "</ turn>")


def render_transcript(messages: list[dict[str, Any]]) -> str:
    """Render a chat as a labelled transcript inside one block. Skips system turns."""
    lines = ["<conversation>"]
    for m in messages:
        role = str(m.get("role", "")).strip().lower()
        if role == "system":
            continue
        content = m.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        lines.append(f'<turn role="{role}">{_escape(content)}</turn>')
    lines.append("</conversation>")
    return "\n".join(lines)


def _class_block(classes: list[str]) -> str:
    bullet = "\n".join(f"- {c}" for c in classes)
    return (
        "## Classification contract\n"
        "You will choose exactly ONE label from the list below. You may reason briefly\n"
        "before answering, but your reply MUST end with a single line of exactly this form:\n\n"
        "<final_answer>LABEL</final_answer>\n\n"
        "where LABEL is *verbatim* one of:\n\n"
        f"{bullet}\n\n"
        "Hard rules — failure to follow these makes your reply unusable:\n"
        "1. Copy the label EXACTLY (case, spacing, punctuation). Do not paraphrase or "
        "add quotes.\n"
        "2. The <final_answer>...</final_answer> line must appear EXACTLY ONCE in your "
        "reply.\n"
        "3. It MUST be the LAST line. No text, whitespace, punctuation, or commentary "
        "may follow the closing tag.\n"
        "4. Pick the single best-fitting label even when the choice is hard. Only as a "
        "last resort, if no label fits at all, emit:\n"
        f"   <final_answer>{NONE_SENTINEL}</final_answer>"
    )


def build_judge_messages(config: JudgeConfig, subject: JudgeSubject) -> list[dict[str, str]]:
    """Build the OpenAI-style messages list for the single judge call."""
    system_parts = [config.judge_prompt.strip()]
    if config.classes:
        system_parts.append("")
        system_parts.append(_class_block(config.classes))
    system = "\n".join(system_parts)

    user_parts: list[str] = []
    if subject.messages:
        user_parts.append("You are evaluating the following conversation:")
        user_parts.append(render_transcript(subject.messages))
    elif subject.subject_content is not None:
        user_parts.append("You are evaluating the following content:")
        user_parts.append("<content>")
        user_parts.append(subject.subject_content)
        user_parts.append("</content>")
    else:
        # JudgeSubject validation prevents this, but guard anyway.
        raise ValueError("JudgeSubject must carry subject_content or messages")

    if config.include_metadata_in_prompt and subject.metadata:
        user_parts.append("")
        user_parts.append("<metadata>")
        for k, v in subject.metadata.items():
            user_parts.append(f"{k}: {v}")
        user_parts.append("</metadata>")

    if config.classes:
        user_parts.append("")
        user_parts.append(
            "Reason briefly if needed, then end with the single "
            "<final_answer>...</final_answer> line and nothing after it."
        )
    else:
        user_parts.append("")
        user_parts.append("Provide your evaluation.")

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


__all__ = [
    "build_judge_messages",
    "render_transcript",
]
