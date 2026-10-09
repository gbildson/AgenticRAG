"""Read simple shell-style environment files without executing them."""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, raw_value = line.partition("=")
        key = key.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid environment assignment in {path} at line {number}")
        try:
            parts = shlex.split(raw_value, comments=True)
        except ValueError as error:
            raise ValueError(f"Invalid value in {path} at line {number}") from error
        if len(parts) > 1:
            raise ValueError(f"Quote values containing spaces in {path} at line {number}")
        os.environ.setdefault(key, parts[0] if parts else "")
