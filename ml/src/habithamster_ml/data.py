"""
Data validation and CSV boundary; no dependency on Django or a live database.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SCORES = {"done": 1.0, "partial": 0.6, "skipped": 0.1, "missed": 0.0}
REQUIRED = {
    "user_id",
    "habit_id",
    "status",
    "log_date",
    "habit_title",
    "target_type",
    "target_value",
    "user_level",
    "current_streak",
}


def validate(
    logs: pd.DataFrame, tags: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = REQUIRED - set(logs.columns)
    if missing or not {"habit_id", "tag_slug"} <= set(tags.columns):
        raise ValueError(
            f"Invalid columns; missing log columns: {sorted(missing)}"
        )
    logs, tags = logs.copy(), tags.copy()
    if logs.empty or logs[list(REQUIRED)].isna().any().any():
        raise ValueError(
            "Logs must be non-empty and contain no missing required values"
        )
    if not logs["status"].isin(SCORES).all():
        raise ValueError("Unknown interaction status")
    if not logs["target_type"].isin(["check", "minutes", "count"]).all():
        raise ValueError("Unknown target type")
    for column in [
        "user_id",
        "habit_id",
        "target_value",
        "user_level",
        "current_streak",
    ]:
        logs[column] = pd.to_numeric(logs[column], errors="raise")
        if not np.isfinite(logs[column]).all() or (logs[column] < 0).any():
            raise ValueError(f"Invalid numeric values in {column}")
        if not (logs[column] % 1 == 0).all():
            raise ValueError(f"Expected integers in {column}")
    logs["log_date"] = pd.to_datetime(
        logs["log_date"], errors="raise", utc=True
    ).dt.tz_localize(None)
    if logs.duplicated(["habit_id", "log_date"]).any():
        raise ValueError("Duplicate habit/date log")
    owners = logs.groupby("habit_id")["user_id"].nunique()
    if (owners > 1).any():
        raise ValueError("A habit must belong to one user")
    logs["score"] = logs["status"].map(SCORES)
    return logs, tags


def load_csv(
    logs_path: Path, tags_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return validate(pd.read_csv(logs_path), pd.read_csv(tags_path))


def demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic synthetic data, for integration checks only."""
    rows, tags = [], []
    for uid in range(1, 11):
        for slot in range(4):
            hid = uid * 10 + slot
            tags.append({"habit_id": hid, "tag_slug": f"category-{slot}"})
            for day in range(10):
                rows.append(
                    {
                        "user_id": uid,
                        "habit_id": hid,
                        "status": "done" if slot % 2 == 0 else "missed",
                        "log_date": pd.Timestamp("2026-01-01")
                        + pd.Timedelta(days=day),
                        "habit_title": f"Habit {hid}",
                        "target_type": "check",
                        "target_value": slot + 1,
                        "user_level": uid,
                        "current_streak": slot,
                    }
                )
    return validate(pd.DataFrame(rows), pd.DataFrame(tags))
