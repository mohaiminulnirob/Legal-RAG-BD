"""Build and query a persistent BM25 index over legal section records.

Examples:
    python -m src.retrieval.bm25 build
    python -m src.retrieval.bm25 search --query "What punishment applies for murder?"
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


DEFAULT_INPUT = Path("data/processed/legal_sections.json")
DEFAULT_INDEX = Path("data/processed/bm25_index.pkl")
INDEX_VERSION = 1
TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", flags=re.UNICODE)
RESULT_FIELDS = (
    "chunk_id",
    "act_title",
    "act_no",
    "act_year",
    "section_id",
    "section_position",
    "section_title",
    "section_content",
    "source_url",
    "source_file",
    "is_repealed",
)


def tokenize(text: str) -> list[str]:
    """Tokenize English legal text while preserving section numbers and contractions."""
    return TOKEN_PATTERN.findall(text.lower())


def retrieval_text(record: dict[str, Any]) -> str:
    """Select the fields that should influence lexical legal retrieval."""
    fields = (
        record.get("act_title", ""),
        record.get("act_title", ""),  # Modest title emphasis for named-act queries.
        record.get("act_year", ""),
        record.get("section_title", ""),
        record.get("section_content", ""),
    )
    return " ".join(str(value) for value in fields if value)


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    """Keep only metadata needed to cite and display a retrieved legal section."""
    return {field: record.get(field) for field in RESULT_FIELDS}


def build_index(input_path: Path = DEFAULT_INPUT, index_path: Path = DEFAULT_INDEX) -> int:
    """Create a serializable BM25 index and return the indexed document count."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Processed legal sections not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as source:
        records = json.load(source)
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("The processed file must contain a JSON list of section records.")

    documents = [compact_record(record) for record in records if str(record.get("section_content", "")).strip()]
    if not documents:
        raise ValueError("No non-empty section records were found in the processed input.")

    corpus = [tokenize(retrieval_text(record)) for record in documents]
    if not all(corpus):
        raise ValueError("At least one indexed legal section has no searchable text.")

    payload = {
        "version": INDEX_VERSION,
        "document_count": len(documents),
        "documents": documents,
        "bm25": BM25Okapi(corpus),
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("wb") as destination:
        pickle.dump(payload, destination, protocol=pickle.HIGHEST_PROTOCOL)
    return len(documents)


def load_index(index_path: Path = DEFAULT_INDEX) -> dict[str, Any]:
    """Load a BM25 index created by :func:`build_index`."""
    if not index_path.is_file():
        raise FileNotFoundError(f"BM25 index not found: {index_path}. Run the build command first.")
    with index_path.open("rb") as source:
        payload = pickle.load(source)
    if not isinstance(payload, dict) or payload.get("version") != INDEX_VERSION:
        raise ValueError("Unsupported BM25 index format. Rebuild the index.")
    return payload


def search(query: str, index_path: Path = DEFAULT_INDEX, top_k: int = 10) -> list[dict[str, Any]]:
    """Return the top scored legal sections for a non-empty query."""
    query_tokens = tokenize(query)
    if not query_tokens:
        raise ValueError("The query must contain at least one searchable term.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    payload = load_index(index_path)
    scores = payload["bm25"].get_scores(query_tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:top_k]
    return [{**payload["documents"][index], "score": round(float(scores[index]), 6)} for index in ranked_indices]


def print_results(query: str, results: list[dict[str, Any]]) -> None:
    """Render search results as readable, citation-oriented command-line output."""
    print(f'Query: "{query}"')
    for rank, result in enumerate(results, start=1):
        citation = " — ".join(
            value for value in (result.get("act_title"), f"section {result.get('section_id')}") if value
        )
        print(f"\n{rank}. {citation} (score: {result['score']:.4f})")
        print(f"   ID: {result['chunk_id']}")
        print(f"   {str(result.get('section_content', '')).strip()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or query the legal BM25 index.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    build_parser = subcommands.add_parser("build", help="Build a BM25 index from processed sections.")
    build_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    build_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)

    search_parser = subcommands.add_parser("search", help="Search a built BM25 index.")
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    search_parser.add_argument("--top-k", type=int, default=10)

    args = parser.parse_args()
    if args.command == "build":
        count = build_index(args.input, args.index)
        print(f"Built BM25 index for {count:,} legal sections: {args.index}")
    else:
        print_results(args.query, search(args.query, args.index, args.top_k))


if __name__ == "__main__":
    main()
