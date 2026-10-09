from __future__ import annotations

import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "examples" / "helpdesk" / "helpdesk.db"


def prepare(database: Path = DATABASE) -> Path:
    if database.exists():
        return database
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            """
            CREATE TABLE assets (
                id INTEGER PRIMARY KEY,
                device_type TEXT NOT NULL,
                model TEXT NOT NULL,
                status TEXT NOT NULL,
                location TEXT NOT NULL
            );
            CREATE TABLE tickets (
                id INTEGER PRIMARY KEY,
                asset_id INTEGER,
                issue TEXT NOT NULL,
                status TEXT NOT NULL,
                opened_at TEXT NOT NULL,
                FOREIGN KEY (asset_id) REFERENCES assets(id)
            );
            INSERT INTO assets VALUES
                (1, 'laptop', 'CedarBook 14', 'available', 'North office'),
                (2, 'laptop', 'CedarBook 16', 'available', 'South office'),
                (3, 'laptop', 'Maple Pro 13', 'assigned', 'North office'),
                (4, 'monitor', 'Harbor 27', 'available', 'South office');
            INSERT INTO tickets VALUES
                (101, 3, 'Screen cracked; technician assessing repair', 'open', '2026-09-28'),
                (102, NULL, 'VPN sign-in fails after password reset', 'open', '2026-09-29'),
                (103, 2, 'Battery replacement completed', 'closed', '2026-09-20');
            """
        )
        connection.commit()
    finally:
        connection.close()
    return database


if __name__ == "__main__":
    print(f"Example database ready: {prepare()}")
