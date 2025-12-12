from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write text to the target path.

    Writes to a temporary file in the same directory, flushes the contents to
    disk, then replaces the destination. This approach avoids partially written
    files if the process is interrupted mid-write.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=path.parent, delete=False, prefix=path.name, suffix=".tmp") as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_name = Path(tmp.name)
    os.replace(temp_name, path)


def atomic_write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    serialized = json.dumps(data, indent=indent)
    atomic_write_text(path, serialized)

