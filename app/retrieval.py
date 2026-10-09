from __future__ import annotations

import re
import sqlite3
import numpy as np

from app.config import Settings
from app.ingest import build_index, index_is_current
from app.models import Embedder, OpenAICompatibleClient


STOP_WORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "can",
    "could", "do", "does", "for", "from", "get", "how", "if", "in",
    "is", "it", "me", "my", "of", "on", "or", "please", "should",
    "that", "the", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "why", "with", "would", "you", "your",
}


class DocumentRetriever:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder | None = None,
        build_if_missing: bool = True,
    ):
        self.settings = settings
        self.embedder = embedder
        if not index_is_current(settings):
            if not build_if_missing:
                raise RuntimeError("Document index is missing or stale")
            build_index(settings, embedder=embedder)
        connection = sqlite3.connect(settings.chunks_db)
        connection.row_factory = sqlite3.Row
        try:
            self.rows = [
                dict(row) for row in connection.execute(
                    "SELECT rowid AS id, text, source, page, section FROM chunks ORDER BY rowid"
                )
            ]
        finally:
            connection.close()
        self.vectors = np.load(settings.vectors_file) if settings.embedding_model else None
        if self.vectors is not None and len(self.rows) != self.vectors.shape[0]:
            raise RuntimeError("Document metadata and vectors are out of sync")

    @staticmethod
    def _fts_query(question: str) -> str:
        terms = [
            term.lower()
            for term in re.findall(r"[A-Za-z][A-Za-z0-9'-]{1,}", question)
            if term.lower() not in STOP_WORDS
        ]
        return " OR ".join(f'"{term}"' for term in dict.fromkeys(terms[:20]))

    def _lexical_rank(self, query: str, source: str | None, limit: int) -> list[int]:
        fts_query = self._fts_query(query)
        if not fts_query:
            return []
        sql = "SELECT rowid FROM chunks WHERE chunks MATCH ?"
        params: list[object] = [fts_query]
        if source:
            sql += " AND source = ?"
            params.append(source)
        sql += " ORDER BY bm25(chunks) LIMIT ?"
        params.append(limit)
        connection = sqlite3.connect(self.settings.chunks_db)
        try:
            return [row[0] - 1 for row in connection.execute(sql, params)]
        finally:
            connection.close()

    def _vector_rank(self, query: str, source: str | None, limit: int) -> list[int]:
        if self.vectors is None:
            return []
        embedder = self.embedder or OpenAICompatibleClient(self.settings)
        vector = np.asarray(embedder.embed([query])[0], dtype=np.float32)
        if vector.shape != (self.vectors.shape[1],):
            raise ValueError("Query embedding dimensions differ from the index")
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        scores = self.vectors @ vector
        allowed = [i for i, row in enumerate(self.rows) if source is None or row["source"] == source]
        return sorted(allowed, key=lambda i: scores[i], reverse=True)[:limit]

    def search(self, query: str, source: str | None = None, top_k: int | None = None) -> list[dict]:
        top_k = top_k or self.settings.retrieval_top_k
        limit = max(15, top_k * 4)
        vector_hits = self._vector_rank(query, source, limit)
        lexical_hits = self._lexical_rank(query, source, limit)
        fused: dict[int, float] = {}
        for hits in (vector_hits, lexical_hits):
            for rank, index in enumerate(hits, start=1):
                fused[index] = fused.get(index, 0.0) + 1.0 / (60 + rank)
        candidates = sorted(fused, key=fused.get, reverse=True)
        evidence = []
        seen: set[tuple[str, int | None, str | None]] = set()
        for index in candidates:
            row = self.rows[index]
            key = (row["source"], row["page"], row["section"])
            if key in seen:
                continue
            seen.add(key)
            evidence.append({
                "source_type": "document",
                "source": row["source"],
                "page": row["page"],
                "section": row["section"],
                "excerpt": row["text"],
                "score": round(fused[index], 6),
            })
            if len(evidence) >= top_k:
                break
        return evidence
