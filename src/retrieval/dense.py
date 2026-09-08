"""Build and query persistent dense retrieval for Bangladesh legal sections.

Examples:
    python -m src.retrieval.dense build --reset
    python -m src.retrieval.dense search --query "A person intentionally killed another"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Protocol, Sequence

import chromadb
from chromadb.errors import NotFoundError

from src.retrieval.bm25 import retrieval_text


DEFAULT_INPUT = Path("data/processed/legal_sections.json")
DEFAULT_DATABASE = Path("chroma_db")
MODEL_CACHE = Path("data/model_cache")
COLLECTION_NAME = "bangladesh_legal_sections"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_BATCH_SIZE = 32
METADATA_FIELDS = (
    "act_title",
    "act_no",
    "act_year",
    "section_id",
    "section_position",
    "source_url",
    "source_file",
    "is_repealed",
)


class TextEmbedder(Protocol):
    """Minimal interface for an embedding model."""

    def encode(self, texts: Sequence[str], **kwargs: Any) -> Any: ...


def create_embedding_model(model_name: str = MODEL_NAME) -> TextEmbedder:
    """Load the local embedding model, downloading it on first use."""
    from sentence_transformers import SentenceTransformer

    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    return SentenceTransformer(model_name, cache_folder=str(MODEL_CACHE))


def document_text(record: dict[str, Any]) -> str:
    """Produce compact semantic content for an individual legal section."""
    return retrieval_text(record)


def chroma_metadata(record: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Extract scalar citation metadata supported by Chroma."""
    metadata: dict[str, str | int | float | bool] = {}
    for field in METADATA_FIELDS:
        value = record.get(field)
        if value is not None and value != "":
            metadata[field] = value
    return metadata


def encode(model: TextEmbedder, texts: Sequence[str], *, query: bool = False) -> list[list[float]]:
    """Encode text using normalized vectors for cosine similarity."""
    prepared = list(texts)
    if query:
        prepared = [f"Represent this sentence for searching relevant passages: {text}" for text in prepared]
    vectors = model.encode(prepared, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist() if hasattr(vectors, "tolist") else [list(vector) for vector in vectors]


def get_collection(database_path: Path = DEFAULT_DATABASE, *, reset: bool = False):
    """Open the persistent legal-sections collection, optionally replacing it."""
    client = chromadb.PersistentClient(path=str(database_path))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except NotFoundError:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        metadata={"embedding_model": MODEL_NAME, "hnsw:space": "cosine"},
    )


def index_records(
    records: Sequence[dict[str, Any]],
    model: TextEmbedder,
    *,
    database_path: Path = DEFAULT_DATABASE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    reset: bool = False,
) -> int:
    """Embed and persist non-empty records in Chroma, returning the collection count."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    usable_records = [record for record in records if str(record.get("section_content", "")).strip()]
    if not usable_records:
        raise ValueError("No non-empty section records were provided.")

    collection = get_collection(database_path, reset=reset)
    if collection.count() and not reset:
        raise ValueError("The dense collection already contains records. Use --reset to rebuild it.")

    for start in range(0, len(usable_records), batch_size):
        batch = usable_records[start : start + batch_size]
        texts = [document_text(record) for record in batch]
        collection.add(
            ids=[str(record["chunk_id"]) for record in batch],
            documents=texts,
            metadatas=[chroma_metadata(record) for record in batch],
            embeddings=encode(model, texts),
        )
    return collection.count()


def build_index(
    input_path: Path = DEFAULT_INPUT,
    database_path: Path = DEFAULT_DATABASE,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    reset: bool = False,
    model_name: str = MODEL_NAME,
) -> int:
    """Load processed sections, embed them, and persist a Chroma collection."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Processed legal sections not found: {input_path}")
    with input_path.open("r", encoding="utf-8") as source:
        records = json.load(source)
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("The processed file must contain a JSON list of section records.")
    return index_records(
        records,
        create_embedding_model(model_name),
        database_path=database_path,
        batch_size=batch_size,
        reset=reset,
    )


def search(
    query: str,
    model: TextEmbedder,
    *,
    database_path: Path = DEFAULT_DATABASE,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Return semantically similar legal sections for a non-empty natural-language query."""
    if not query.strip():
        raise ValueError("The query must not be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    collection = get_collection(database_path)
    if not collection.count():
        raise ValueError("Dense collection is empty. Run the build command first.")
    result = collection.query(
        query_embeddings=encode(model, [query], query=True),
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "chunk_id": chunk_id,
            "section_content": document,
            **metadata,
            "similarity": round(1 - float(distance), 6),
        }
        for chunk_id, document, metadata, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


def print_results(query: str, results: Sequence[dict[str, Any]]) -> None:
    """Print citation-oriented semantic retrieval results."""
    print(f'Query: "{query}"')
    for rank, result in enumerate(results, start=1):
        print(
            f"\n{rank}. {result.get('act_title', 'Unknown Act')} — section "
            f"{result.get('section_id', '?')} (similarity: {result['similarity']:.4f})"
        )
        print(f"   ID: {result['chunk_id']}")
        print(f"   {result['section_content']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or query dense legal retrieval in Chroma.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    build_parser = subcommands.add_parser("build", help="Embed processed sections into Chroma.")
    build_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    build_parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    build_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    build_parser.add_argument("--model", default=MODEL_NAME)
    build_parser.add_argument("--reset", action="store_true", help="Replace any existing dense collection.")

    search_parser = subcommands.add_parser("search", help="Search the dense legal collection.")
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    search_parser.add_argument("--top-k", type=int, default=10)
    search_parser.add_argument("--model", default=MODEL_NAME)

    args = parser.parse_args()
    if args.command == "build":
        count = build_index(args.input, args.database, batch_size=args.batch_size, reset=args.reset, model_name=args.model)
        print(f"Built dense index for {count:,} legal sections in {args.database}")
    else:
        model = create_embedding_model(args.model)
        print_results(args.query, search(args.query, model, database_path=args.database, top_k=args.top_k))


if __name__ == "__main__":
    main()
