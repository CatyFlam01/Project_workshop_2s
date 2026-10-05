"""
Model lifecycle: explicit training, atomic persistence, deterministic
inference.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import pandas as pd
from lightgbm import LGBMClassifier

from habithamster_ml.data import validate
from habithamster_ml.features import (
    FEATURES,
    build_features,
    interaction_matrix,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class HybridRecommender:
    classifier: LGBMClassifier
    matrix: pd.DataFrame

    @classmethod
    def train(
        cls, logs: pd.DataFrame, tags: pd.DataFrame
    ) -> "HybridRecommender":
        logs, tags = validate(logs, tags)
        interactions, matrix = interaction_matrix(logs)
        if len(logs) < 50 or len(interactions) < 30:
            raise ValueError("Need at least 50 logs and 30 user/habit pairs")
        target = (interactions["weighted_score"] >= 0.7).astype(int)
        if target.nunique() != 2:
            raise ValueError(
                "Training requires both success and failure examples"
            )
        features = pd.DataFrame(
            [
                build_features(
                    int(row.user_id), int(row.habit_id), logs, tags, matrix
                )
                for row in interactions.itertuples()
            ]
        )[FEATURES]
        classifier = LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
            min_child_samples=5,
            verbosity=-1,
        )
        classifier.fit(features, target)
        LOGGER.info(
            "Training complete: logs=%d pairs=%d features=%d",
            len(logs),
            len(target),
            len(FEATURES),
        )
        return cls(classifier, matrix)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
        try:
            joblib.dump(
                {
                    "version": 1,
                    "model": self.classifier,
                    "matrix": self.matrix,
                    "features": FEATURES,
                },
                temporary,
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path) -> "HybridRecommender":
        """Load only trusted local artifacts: joblib uses pickle internally."""
        payload = joblib.load(path)
        if payload.get("version") != 1 or payload.get("features") != FEATURES:
            raise ValueError("Unsupported model artifact schema")
        return cls(payload["model"], payload["matrix"])

    def recommend(
        self,
        uid: int,
        logs: pd.DataFrame,
        tags: pd.DataFrame,
        top_k: int = 5,
        min_score: float = 0.5,
    ) -> list[dict]:
        if top_k < 1 or not 0 <= min_score <= 1:
            raise ValueError(
                "top_k must be positive; min_score must be in [0, 1]"
            )
        logs, tags = validate(logs, tags)
        own = set(logs.loc[logs["user_id"] == uid, "habit_id"])
        if not own:
            popular = (
                logs.groupby("habit_id")["score"]
                .mean()
                .sort_values(ascending=False)
            )
            return [
                {
                    "habit_id": int(hid),
                    "habit_title": logs.loc[
                        logs["habit_id"] == hid, "habit_title"
                    ].iloc[0],
                    "score": float(score),
                    "explanation": "popular habit",
                }
                for hid, score in popular.items()
                if score >= min_score
            ][:top_k]
        result = []
        for hid in sorted(set(logs["habit_id"]) - own):
            features = build_features(uid, int(hid), logs, tags, self.matrix)
            score = float(
                self.classifier.predict_proba(
                    pd.DataFrame([features])[FEATURES]
                )[0, 1]
            )
            if score >= min_score:
                result.append(
                    {
                        "habit_id": int(hid),
                        "habit_title": logs.loc[
                            logs["habit_id"] == hid, "habit_title"
                        ].iloc[0],
                        "score": score,
                        "explanation": "shared interests"
                        if features["tag_match"] > 0.4
                        else "personalized",
                    }
                )
        return sorted(
            result, key=lambda rec: (-rec["score"], rec["habit_id"])
        )[:top_k]
