"""Train and compare the three models on one common time split."""

import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from data.load import load_retailrocket, load_wpp, split_time
from data.demo import make_demo
from models.common import ModelData
from models.v1_baseline import Baseline
from models.v2_neural import NeuralTwoTower
from models.v3_sequence import SequenceAttention
from analytics.evaluate import evaluate
from analytics.eda import summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["demo", "retailrocket", "wpp"], required=True)
    parser.add_argument("--path", default="")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-users", type=int, default=300, help="users sampled per evaluation cohort")
    parser.add_argument("--max-train-pairs", type=int, default=300000)
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    if args.source != "demo" and not args.path:
        parser.error("--path is required for Retailrocket and WPP")
    events, catalog = (make_demo() if args.source == "demo" else
                       load_retailrocket(args.path, args.max_events) if args.source == "retailrocket" else load_wpp(args.path))
    train, valid, test = split_time(events)
    # The candidate catalog is frozen before validation starts.
    catalog = pd.concat([catalog, pd.DataFrame({"item_id": train["item_id"].unique(), "name": "", "category": ""})]).drop_duplicates("item_id")
    data = ModelData(train, catalog)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    for model in (Baseline(), NeuralTwoTower(epochs=args.epochs, max_pairs=args.max_train_pairs),
                  SequenceAttention(epochs=args.epochs, max_pairs=args.max_train_pairs)):
        print(f"Training {model.name}...", flush=True)
        model.fit(data)
        reports.append(evaluate(model, data, valid, max_users=args.max_users).assign(split="validation"))
        reports.append(evaluate(model, data, test, max_users=args.max_users).assign(split="test"))
    pd.concat(reports, ignore_index=True).to_csv(out / "comparison.csv", index=False)
    (out / "run.json").write_text(json.dumps({"source": args.source, "path": args.path,
        "train": summary(train, catalog), "validation_events": len(valid), "test_events": len(test),
        "epochs": args.epochs, "max_users_per_cohort": args.max_users,
        "max_train_pairs_per_neural_model": args.max_train_pairs,
        "neural_device": "cuda" if torch.cuda.is_available() and len(data.items) > 1000 else "cpu"}, indent=2), encoding="utf-8")
    print(f"Wrote {out / 'comparison.csv'}")


if __name__ == "__main__":
    main()
