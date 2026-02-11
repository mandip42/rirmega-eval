from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(obj, option=orjson.OPT_INDENT_2))


def read_json(path: Path) -> Any:
    return orjson.loads(path.read_bytes())

