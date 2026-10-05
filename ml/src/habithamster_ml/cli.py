"""Command-line entry point for standalone training and prediction."""

import argparse
import json
import logging
from pathlib import Path

from habithamster_ml.data import demo_data, load_csv
from habithamster_ml.model import HybridRecommender


def main() -> None:
    parser = argparse.ArgumentParser(description="HabitHamster ML")
    parser.add_argument("command", choices=["demo", "train", "recommend"])
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--tags", type=Path)
    parser.add_argument(
        "--model", type=Path, default=Path("artifacts/model.joblib")
    )
    parser.add_argument("--user", type=int, default=1)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.command == "demo":
        logs, tags = demo_data()
    else:
        if args.logs is None or args.tags is None:
            parser.error("--logs and --tags are required")
        logs, tags = load_csv(args.logs, args.tags)
    try:
        if args.command in {"demo", "train"}:
            model = HybridRecommender.train(logs, tags)
            model.save(args.model)
        else:
            model = HybridRecommender.load(args.model)
        if args.command != "train":
            print(
                json.dumps(
                    model.recommend(args.user, logs, tags, args.top),
                    ensure_ascii=False,
                    indent=2,
                )
            )
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Error: {exc}\n")
