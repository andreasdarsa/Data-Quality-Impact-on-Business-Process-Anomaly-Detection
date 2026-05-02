import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_logs(path: str | Path) -> list[dict]:
    logs = load_json(path)

    if not isinstance(logs, list):
        raise ValueError("Expected logs to be a list of cases.")

    return logs


def save_logs(logs: list[dict], path: str | Path) -> None:
    save_json(logs, path)
