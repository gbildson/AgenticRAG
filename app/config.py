from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def _path_env(name: str, default: Path | None) -> Path | None:
    value = os.getenv(name)
    if value and value.lower() == "none" and default is not None and name == "SQL_DATABASE_PATH":
        return None
    return Path(value).expanduser() if value else default


def _sql_allowed_schema() -> dict[str, list[str]] | None:
    value = os.getenv("SQL_ALLOWED_SCHEMA")
    if not value:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict) or not parsed or not all(
        isinstance(table, str)
        and isinstance(columns, list)
        and columns
        and all(isinstance(column, str) and column for column in columns)
        for table, columns in parsed.items()
    ):
        raise ValueError("SQL_ALLOWED_SCHEMA must map table names to nonempty column lists")
    return parsed


@dataclass(frozen=True)
class Settings:
    documents_dir: Path = field(
        default_factory=lambda: _path_env(
            "DOCUMENTS_DIR", ROOT_DIR / "examples" / "helpdesk" / "docs"
        )
    )
    sql_database: Path | None = field(
        default_factory=lambda: _path_env(
            "SQL_DATABASE_PATH", ROOT_DIR / "examples" / "helpdesk" / "helpdesk.db"
        )
    )
    sql_allowed_schema: dict[str, list[str]] | None = field(default_factory=_sql_allowed_schema)
    index_dir: Path = field(
        default_factory=lambda: _path_env("INDEX_DIR", ROOT_DIR / "data" / "rag_index")
    )
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "local"))
    chat_model: str | None = field(default_factory=lambda: os.getenv("CHAT_MODEL"))
    embedding_model: str | None = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL"))
    retrieval_top_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "6")))
    max_agent_steps: int = field(default_factory=lambda: int(os.getenv("MAX_AGENT_STEPS", "6")))

    @property
    def chunks_db(self) -> Path:
        return self.index_dir / "chunks.db"

    @property
    def vectors_file(self) -> Path:
        return self.index_dir / "vectors.npy"

    @property
    def index_manifest(self) -> Path:
        return self.index_dir / "manifest.json"

    def validate_assets(self, require_model: bool = False) -> None:
        if self.max_agent_steps < 1 or self.retrieval_top_k < 1:
            raise ValueError("MAX_AGENT_STEPS and RETRIEVAL_TOP_K must be positive")
        if not self.documents_dir.is_dir() or not any(
            path.suffix.lower() in {".md", ".txt", ".pdf"}
            for path in self.documents_dir.rglob("*")
        ):
            raise RuntimeError(f"No supported documents in {self.documents_dir}")
        if self.sql_database is not None and not self.sql_database.is_file():
            raise RuntimeError(f"SQLite database not found: {self.sql_database}")
        if require_model and not self.chat_model:
            raise RuntimeError("CHAT_MODEL is not set")
