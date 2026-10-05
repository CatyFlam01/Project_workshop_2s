import json
import subprocess
import sys

import pandas as pd
import pytest

from habithamster_ml.data import demo_data, validate
from habithamster_ml.features import collaborative_score
from habithamster_ml.model import HybridRecommender
from habithamster_ml.onboarding import build_user_vector, recommend_habits


@pytest.fixture(scope="module")
def trained():
    logs, tags = demo_data()
    return HybridRecommender.train(logs, tags), logs, tags


def test_round_trip_and_deterministic_predictions(trained, tmp_path):
    model, logs, tags = trained
    path = tmp_path / "model.joblib"
    model.save(path)
    restored = HybridRecommender.load(path)
    before = model.recommend(1, logs, tags, min_score=0)
    assert before
    all_results = model.recommend(1, logs, tags, top_k=100, min_score=0)
    assert len({round(rec["score"], 3) for rec in all_results}) > 1
    assert restored.recommend(1, logs, tags, min_score=0) == before
    assert model.recommend(1, logs, tags, min_score=0) == before
    assert all(rec["habit_id"] not in {10, 11, 12, 13} for rec in before)
    assert all(0 <= rec["score"] <= 1 for rec in before)


def test_cf_excludes_actual_user_id():
    matrix = pd.DataFrame({10: [1.0, 0.2]}, index=[101, 202])
    assert collaborative_score(101, 10, matrix) == pytest.approx(0.2)


def test_cold_start(trained):
    model, logs, tags = trained
    result = model.recommend(999, logs, tags)
    assert len(result) == 5
    assert all(rec["explanation"] == "popular habit" for rec in result)


@pytest.mark.parametrize(
    "case", ["status", "missing", "duplicate", "negative"]
)
def test_invalid_data(case):
    logs, tags = demo_data()
    if case == "status":
        logs.loc[0, "status"] = "invalid"
    elif case == "missing":
        logs = logs.drop(columns="user_id")
    elif case == "duplicate":
        logs = pd.concat([logs, logs.iloc[[0]]])
    else:
        logs.loc[0, "target_value"] = -1
    with pytest.raises(ValueError):
        validate(logs, tags)


def test_requires_two_classes():
    logs, tags = demo_data()
    logs["status"] = "done"
    with pytest.raises(ValueError, match="both success and failure"):
        HybridRecommender.train(logs, tags)


def test_requires_sufficient_data():
    logs, tags = demo_data()
    with pytest.raises(ValueError, match="50 logs"):
        HybridRecommender.train(logs.iloc[:20], tags)


def test_onboarding_accepts_string_and_form_list():
    base = {
        "goals": ["sport"],
        "difficulties": [],
        "preferred_time": ["morning"],
    }
    scalar = dict(base, sport_frequency="every day")
    form = dict(base, sport_frequency=["every day"])
    assert (build_user_vector(scalar) == build_user_vector(form)).all()
    assert recommend_habits(json.dumps(scalar)) == recommend_habits(form)


def test_cli_demo(tmp_path):
    path = tmp_path / "demo.joblib"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from habithamster_ml.cli import main; main()",
            "demo",
            "--model",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert path.exists()
    assert isinstance(json.loads(result.stdout), list)


def test_invalid_prediction_arguments(trained):
    model, logs, tags = trained
    with pytest.raises(ValueError):
        model.recommend(1, logs, tags, top_k=0)
