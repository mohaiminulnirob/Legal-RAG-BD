"""Build traceable legal-evidence context for baseline RAG."""

from __future__ import annotations

from typing import Any, Sequence


def citation_label(evidence: dict[str, Any]) -> str:
    """Return the citation label the model is allowed to use."""
    title = str(evidence.get("act_title") or "Unknown Act")
    section = evidence.get("section_id")
    return f"{title}, section {section}" if section not in (None, "") else title


def build_context(evidence: Sequence[dict[str, Any]], *, max_chars_per_section: int = 4_000) -> str:
    """Render retrieved statutes as bounded, explicitly identified evidence blocks."""
    if max_chars_per_section < 1:
        raise ValueError("max_chars_per_section must be at least 1.")
    blocks: list[str] = []
    for position, item in enumerate(evidence, start=1):
        content = str(item.get("section_content", "")).strip()
        if len(content) > max_chars_per_section:
            content = f"{content[:max_chars_per_section].rstrip()} […]"
        blocks.append(
            "\n".join(
                (
                    f"[Evidence {position}]",
                    f"Citation: {citation_label(item)}",
                    f"Chunk ID: {item.get('chunk_id', 'unknown')}",
                    f"Source: {item.get('source_url') or 'not available'}",
                    "Content:",
                    content or "(No section text available.)",
                )
            )
        )
    return "\n\n".join(blocks)
