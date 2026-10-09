from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymupdf

from app.config import Settings
from app.models import Embedder, OpenAICompatibleClient


SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


@dataclass
class Chunk:
    source: str
    page: int | None
    section: str | None
    text: str

    @property
    def embedding_text(self) -> str:
        context = f"Source: {self.source}"
        if self.section:
            context += f"\nSection: {self.section}"
        if self.page:
            context += f"\nPage: {self.page}"
        return f"{context}\n\n{self.text}"


def document_paths(documents_dir: Path) -> list[Path]:
    return sorted(
        path for path in documents_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def corpus_digest(documents_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in document_paths(documents_dir):
        digest.update(path.relative_to(documents_dir).as_posix().encode())
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _word_chunks(text: str, size: int = 300, overlap: int = 40) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def _text_sections(text: str, markdown: bool) -> list[tuple[str | None, str]]:
    if not markdown:
        return [(None, text)]
    sections: list[tuple[str | None, str]] = []
    heading: str | None = None
    lines: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.+)$", line)
        if match:
            if lines:
                sections.append((heading, "\n".join(lines)))
            heading, lines = match.group(1).strip(), []
        else:
            lines.append(line)
    if lines:
        sections.append((heading, "\n".join(lines)))
    return sections


def extract_chunks(documents_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in document_paths(documents_dir):
        source = path.relative_to(documents_dir).as_posix()
        if path.suffix.lower() == ".pdf":
            with pymupdf.open(path) as document:
                for page_number, page in enumerate(document, start=1):
                    text = page.get_text("text")
                    for chunk in _word_chunks(text):
                        chunks.append(Chunk(source, page_number, None, chunk))
        else:
            text = path.read_text(encoding="utf-8")
            for section, content in _text_sections(text, path.suffix.lower() == ".md"):
                for chunk in _word_chunks(content):
                    chunks.append(Chunk(source, None, section, f"{section}\n{chunk}" if section else chunk))
    return chunks


def index_is_current(settings: Settings) -> bool:
    if not settings.chunks_db.is_file() or not settings.index_manifest.is_file():
        return False
    try:
        manifest = json.loads(settings.index_manifest.read_text())
        connection = sqlite3.connect(settings.chunks_db)
        try:
            chunk_count = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
        finally:
            connection.close()
        vector_ok = not settings.embedding_model or (
            settings.vectors_file.is_file()
            and np.load(settings.vectors_file, mmap_mode="r").shape[0]
            == manifest.get("chunk_count")
        )
    except (OSError, ValueError, json.JSONDecodeError, sqlite3.DatabaseError):
        return False
    return (
        manifest.get("format_version") == 3
        and manifest.get("corpus_digest") == corpus_digest(settings.documents_dir)
        and manifest.get("embedding_model") == settings.embedding_model
        and manifest.get("chunk_count") == chunk_count
        and vector_ok
    )


def build_index(
    settings: Settings,
    embedder: Embedder | None = None,
    force: bool = False,
    batch_size: int = 48,
) -> dict:
    settings.validate_assets()
    if not force and index_is_current(settings):
        return json.loads(settings.index_manifest.read_text())
    chunks = extract_chunks(settings.documents_dir)
    if not chunks:
        raise RuntimeError(f"No text extracted from {settings.documents_dir}")

    matrix = None
    if settings.embedding_model or embedder is not None:
        embedder = embedder or OpenAICompatibleClient(settings)
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), batch_size):
            vectors.extend(
                embedder.embed([chunk.embedding_text for chunk in chunks[start:start + batch_size]])
            )
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
            raise RuntimeError("Embedding response shape does not match extracted chunks")
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    if settings.chunks_db.exists():
        settings.chunks_db.unlink()
    connection = sqlite3.connect(settings.chunks_db)
    try:
        connection.execute(
            "CREATE VIRTUAL TABLE chunks USING fts5("
            "text, source UNINDEXED, page UNINDEXED, section UNINDEXED, "
            "tokenize='porter unicode61')"
        )
        connection.executemany(
            "INSERT INTO chunks(rowid, text, source, page, section) VALUES (?, ?, ?, ?, ?)",
            [
                (index, chunk.text, chunk.source, chunk.page, chunk.section)
                for index, chunk in enumerate(chunks, start=1)
            ],
        )
        connection.commit()
    finally:
        connection.close()
    if matrix is not None:
        np.save(settings.vectors_file, matrix)
    elif settings.vectors_file.exists():
        settings.vectors_file.unlink()
    manifest = {
        "format_version": 3,
        "corpus_digest": corpus_digest(settings.documents_dir),
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": int(matrix.shape[1]) if matrix is not None else None,
        "chunk_count": len(chunks),
        "files": [p.relative_to(settings.documents_dir).as_posix() for p in document_paths(settings.documents_dir)],
    }
    settings.index_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the document search index")
    parser.add_argument("--force", action="store_true", help="Rebuild an existing index")
    args = parser.parse_args()
    manifest = build_index(Settings(), force=args.force)
    print(f"Index ready: {manifest['chunk_count']} chunks from {len(manifest['files'])} files")


if __name__ == "__main__":
    main()
