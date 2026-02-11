from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson
from rirmega_eval.impact.privacy import get_local_salt, get_or_create_install_id, privacy_hash


def _utc_now() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


@dataclass
class ImpactLedger:
    data: dict[str, Any]

    @staticmethod
    def load(path: str) -> ImpactLedger:
        p = Path(path)
        if not p.exists():
            return ImpactLedger(
                data={
                    "schema_version": 1,
                    "created_utc": _utc_now(),
                    "salt_hint": "local-file-salt",
                    "totals": {"events": 0, "eval_runs": 0, "build_runs": 0, "publish_runs": 0},
                    "by_task": {},
                    "unique_users_approx": 0,
                    "unique_runs_approx": 0,
                    "recent": [],
                    "_seen_users": [],
                    "_seen_runs": [],
                }
            )
        return ImpactLedger(data=orjson.loads(p.read_bytes()))

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(orjson.dumps(self.data, option=orjson.OPT_INDENT_2))

    def log_event(self, event: str, task: str | None = None) -> None:
        salt = get_local_salt()
        install_id = get_or_create_install_id()
        user_hash = privacy_hash(install_id, salt=salt)

        run_id = privacy_hash(_utc_now(), salt=salt)

        self.data["totals"]["events"] += 1
        if event == "eval_run":
            self.data["totals"]["eval_runs"] += 1
        elif event == "build_run":
            self.data["totals"]["build_runs"] += 1
        elif event == "publish_run":
            self.data["totals"]["publish_runs"] += 1

        if task:
            by = self.data.get("by_task", {})
            by[task] = by.get(task, 0) + 1
            self.data["by_task"] = by

        seen_users = set(self.data.get("_seen_users", []))
        seen_runs = set(self.data.get("_seen_runs", []))

        if user_hash not in seen_users:
            seen_users.add(user_hash)
            self.data["unique_users_approx"] = len(seen_users)
        if run_id not in seen_runs:
            seen_runs.add(run_id)
            self.data["unique_runs_approx"] = len(seen_runs)

        self.data["_seen_users"] = list(seen_users)
        self.data["_seen_runs"] = list(seen_runs)

        rec = self.data.get("recent", [])
        rec.append({"utc": _utc_now(), "event": event, "task": task, "user": user_hash[:12], "run": run_id[:12]})
        self.data["recent"] = rec[-200:]

