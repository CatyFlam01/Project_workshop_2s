"""
Shared training/inference features adapted from the original hybrid
recommender.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

FEATURES = [
    "user_level",
    "user_streak",
    "user_avg_score",
    "habit_difficulty",
    "tag_match",
    "cf_score",
]


def interaction_matrix(
    logs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = logs["log_date"].max()
    weights = np.exp(-(reference - logs["log_date"]).dt.days / 30) + 0.01
    frame = logs.assign(weight=weights, weighted=logs["score"] * weights)
    grouped = frame.groupby(["user_id", "habit_id"])[
        ["weight", "weighted"]
    ].sum()
    grouped["weighted_score"] = grouped["weighted"] / grouped["weight"]
    interactions = grouped.reset_index()
    matrix = interactions.pivot(
        index="user_id", columns="habit_id", values="weighted_score"
    ).fillna(0)
    return interactions, matrix


def collaborative_score(uid: int, hid: int, matrix: pd.DataFrame) -> float:
    if uid not in matrix.index or hid not in matrix.columns:
        return 0.5
    similarities = cosine_similarity(matrix.loc[[uid]], matrix)[0]
    # Compare actual user IDs, not row positions. Exclude the current
    # user first.
    neighbors = [
        (float(similarities[i]), other)
        for i, other in enumerate(matrix.index)
        if other != uid
    ]
    neighbors.sort(reverse=True)
    scores = [
        matrix.loc[other, hid]
        for _, other in neighbors[:5]
        if matrix.loc[other, hid] > 0
    ]
    return float(np.mean(scores)) if scores else 0.5


def build_features(
    uid: int,
    hid: int,
    logs: pd.DataFrame,
    tags: pd.DataFrame,
    matrix: pd.DataFrame,
) -> dict[str, float]:
    user_logs = logs[logs["user_id"] == uid]
    habit = logs[logs["habit_id"] == hid].iloc[0]
    user_tags = set(
        tags.loc[
            tags["habit_id"].isin(user_logs["habit_id"]), "tag_slug"
        ].dropna()
    )
    habit_tags = set(tags.loc[tags["habit_id"] == hid, "tag_slug"].dropna())
    union = user_tags | habit_tags
    difficulty = (
        habit["target_value"]
        * {"check": 1, "minutes": 0.5, "count": 1.5}[habit["target_type"]]
    )
    return {
        "user_level": float(user_logs["user_level"].mean()),
        "user_streak": float(user_logs["current_streak"].max()),
        "user_avg_score": float(user_logs["score"].mean()),
        "habit_difficulty": float(min(difficulty / 50, 1)),
        "tag_match": len(user_tags & habit_tags) / len(union)
        if union
        else 0.5,
        "cf_score": collaborative_score(uid, hid, matrix),
    }
