"""Normalize Bangladesh legal-act records into section-level retrieval documents.

Run this only against an unmodified source JSON file. The source is read-only; the
normalized output is written to ``data/processed/legal_sections.json`` by default.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


ACT_LIST_KEYS = ("acts", "data", "records", "items")
SECTION_LIST_KEYS = ("sections", "section", "provisions")


def load_acts(input_path: Path) -> list[dict[str, Any]]:
    """Load a top-level list of acts or a mapping containing a known list key."""
    with input_path.open("r", encoding="utf-8") as source:
        payload = json.load(source)

    if isinstance(payload, list):
        acts = payload
    elif isinstance(payload, dict):
        acts = next((payload[key] for key in ACT_LIST_KEYS if isinstance(payload.get(key), list)), None)
    else:
        acts = None

    if not isinstance(acts, list) or not all(isinstance(item, dict) for item in acts):
        raise ValueError(
            "Expected a JSON list of act records, or an object containing one under "
            f"one of: {', '.join(ACT_LIST_KEYS)}. Inspect the dataset schema and adjust "
            "the parser before processing."
        )
    return acts


def first_value(record: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present non-empty field among compatible source names."""
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return default


def section_records(act: dict[str, Any]) -> Iterable[dict[str, Any]]:
    sections = first_value(act, *SECTION_LIST_KEYS, default=[])
    if not isinstance(sections, list):
        return []
    return (section for section in sections if isinstance(section, dict))


def explicit_section_identifier(section: dict[str, Any]) -> str | None:
    """Return an explicit section number when the source record provides one."""
    explicit_id = first_value(section, "section_id", "id", "section_no", "number")
    if explicit_id is not None:
        return str(explicit_id)

    content = str(first_value(section, "section_content", "content", "text", default=""))
    match = re.match(r"^\s*\.?\s*(?:\d+\[)?(?P<number>\d+[A-Za-z]?)\s*[.:-]", content)
    return match.group("number") if match else None


def section_identifier(
    section: dict[str, Any],
    fallback: int,
    *,
    parent_section: str | None = None,
) -> str:
    """Use an explicit label, associate exception continuations with their parent,
    or retain the old ordinal fallback for unlabeled non-continuation records.
    """
    explicit_id = explicit_section_identifier(section)
    if explicit_id is not None:
        return explicit_id

    content = str(first_value(section, "section_content", "content", "text", default=""))
    exception = re.match(r"^\s*Exception\s+(?P<number>\d+)\b", content, re.I)
    if exception and parent_section:
        return f"{parent_section} (Exception {exception.group('number')})"
    return str(fallback)


def normalize(acts: list[dict[str, Any]], source_name: str) -> list[dict[str, Any]]:
    """Produce one retrieval document per legal section while retaining provenance."""
    documents: list[dict[str, Any]] = []
    for act_index, act in enumerate(acts, start=1):
        title = first_value(act, "act_title", "title", "name", default="Untitled Act")
        # Act numbers are not guaranteed to be unique. The source file/index is stable.
        act_id = str(first_value(act, "act_id", "id", "source_file", default=act_index))
        act_id = Path(act_id).stem
        csv_metadata = act.get("csv_metadata") if isinstance(act.get("csv_metadata"), dict) else {}
        parent_section: str | None = None
        for section_index, section in enumerate(section_records(act), start=1):
            explicit_section = explicit_section_identifier(section)
            section_id = section_identifier(section, section_index, parent_section=parent_section)
            if explicit_section is not None:
                parent_section = explicit_section
            content = first_value(section, "section_content", "content", "text", "description", default="")
            documents.append(
                {
                    # Some source acts repeat section labels; source position is unique.
                    "chunk_id": f"act_{act_id}_section_{section_index:04d}",
                    "act_title": title,
                    "act_no": first_value(act, "act_no", "act_number"),
                    "act_year": first_value(act, "act_year", "year"),
                    "language": first_value(act, "language"),
                    "is_repealed": first_value(
                        act, "is_repealed", "repealed", default=csv_metadata.get("is_repealed")
                    ),
                    "section_id": section_id,
                    "section_position": section_index,
                    "section_title": first_value(section, "section_title", "title", "heading"),
                    "section_content": content,
                    "footnotes": first_value(section, "footnotes", "footnote", default=act.get("footnotes", [])),
                    "source_file": first_value(act, "source_file", default=source_name),
                    "source_url": first_value(act, "source_url", "url", "source"),
                    "government_context": first_value(act, "government_context", default={}),
                    "legal_system_context": first_value(act, "legal_system_context", default={}),
                }
            )
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Create section-level legal retrieval documents.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the raw legal acts JSON file.")
    parser.add_argument("--output", type=Path, default=Path("data/processed/legal_sections.json"))
    args = parser.parse_args()

    acts = load_acts(args.input)
    documents = normalize(acts, args.input.name)
    if not documents:
        raise ValueError("No section records were found; inspect the dataset schema before writing output.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        json.dump(documents, destination, ensure_ascii=False, indent=2)

    print(f"Processed {len(acts):,} acts into {len(documents):,} section-level records.")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
