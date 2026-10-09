from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Mapping, Sequence
from pathlib import Path


_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}


class SQLValidationError(ValueError):
    pass


class ReadOnlySQLTool:
    """Query a SQLite database through a discovered, enforced schema allowlist."""

    def __init__(
        self,
        database_path: Path,
        max_rows: int = 100,
        allowed_schema: Mapping[str, Sequence[str]] | None = None,
    ):
        self.database_path = database_path
        self.max_rows = max_rows
        self.schema = self._discover_schema(allowed_schema)
        self._allowed = {
            table.lower(): {column.lower() for column in columns}
            for table, columns in self.schema.items()
        }

    @staticmethod
    def _quote_identifier(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(
            self.database_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5
        )

    def _discover_schema(
        self, allowed_schema: Mapping[str, Sequence[str]] | None
    ) -> dict[str, tuple[str, ...]]:
        connection = self._connect()
        try:
            # table_list distinguishes ordinary/virtual tables from FTS shadow tables.
            names = {
                row[1].lower(): row[1]
                for row in connection.execute("PRAGMA table_list")
                if row[0] == "main" and row[2] in {"table", "virtual"}
                and not row[1].lower().startswith("sqlite_")
            }
            if allowed_schema is not None:
                requested_tables = {name.lower(): name for name in allowed_schema}
                unknown = set(requested_tables) - set(names)
                if unknown:
                    raise ValueError(f"Unknown allowed SQL tables: {', '.join(sorted(unknown))}")
                names = {name: names[name] for name in requested_tables}

            schema: dict[str, tuple[str, ...]] = {}
            for key, name in sorted(names.items()):
                available = {
                    row[1].lower(): row[1]
                    for row in connection.execute(
                        f"PRAGMA table_xinfo({self._quote_identifier(name)})"
                    )
                    if row[6] != 1  # Exclude hidden virtual-table columns.
                }
                requested = (
                    None if allowed_schema is None else allowed_schema[requested_tables[key]]
                )
                if requested is not None:
                    if isinstance(requested, str) or not requested:
                        raise ValueError(f"Allowed columns for {name} must be a nonempty list")
                    missing = {column.lower() for column in requested} - set(available)
                    if missing:
                        raise ValueError(
                            f"Unknown allowed columns in {name}: {', '.join(sorted(missing))}"
                        )
                    columns = tuple(available[column.lower()] for column in requested)
                else:
                    columns = tuple(available.values())
                if columns:
                    schema[name] = columns
            if not schema:
                raise ValueError("No SQL tables and columns are available")
            return schema
        finally:
            connection.close()

    @property
    def schema_description(self) -> str:
        return "\n".join(
            f"{table}({', '.join(columns)})" for table, columns in self.schema.items()
        )

    @staticmethod
    def validate(sql: str) -> str:
        normalized = sql.strip()
        if not normalized:
            raise SQLValidationError("SQL query is empty")
        if ";" in normalized.rstrip(";"):
            raise SQLValidationError("Only one SQL statement is allowed")
        normalized = normalized.rstrip(";").strip()
        if not re.match(r"^(SELECT|WITH)\b", normalized, re.IGNORECASE):
            raise SQLValidationError("Only SELECT or WITH queries are allowed")
        blocked = re.compile(
            r"\b(ATTACH|DETACH|PRAGMA|INSERT|UPDATE|DELETE|REPLACE|"
            r"CREATE|ALTER|DROP|VACUUM|REINDEX)\b",
            re.IGNORECASE,
        )
        if blocked.search(normalized):
            raise SQLValidationError("The query contains a disallowed operation")
        return normalized

    def _authorizer(self, action: int, table, column, _db_name, _trigger) -> int:
        if action == sqlite3.SQLITE_READ:
            columns = self._allowed.get(str(table).lower())
            # SQLite passes an empty column name for count(*) on a table.
            if columns and (not column or str(column).lower() in columns):
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY

    def execute(self, sql: str) -> dict:
        query = self.validate(sql)
        connection = self._connect()
        connection.row_factory = sqlite3.Row
        deadline = time.monotonic() + 2.0
        connection.set_authorizer(self._authorizer)
        connection.set_progress_handler(
            lambda: 1 if time.monotonic() > deadline else 0, 1_000
        )
        try:
            cursor = connection.execute(query)
            fetched = cursor.fetchmany(self.max_rows + 1)
            truncated = len(fetched) > self.max_rows
            rows = [dict(row) for row in fetched[: self.max_rows]]
            columns = [description[0] for description in cursor.description or []]
            return {
                "source_type": "sql",
                "sql": query,
                "columns": columns,
                "rows": rows,
                "truncated": truncated,
            }
        except sqlite3.Error as error:
            raise SQLValidationError(f"SQLite rejected the query: {error}") from error
        finally:
            connection.close()
