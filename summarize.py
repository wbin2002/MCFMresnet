from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine experiment summary CSV files.")
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/all_results.csv"))
    args = parser.parse_args()
    files = sorted(args.outputs.rglob("summary_*.csv"))
    if not files:
        raise SystemExit(f"No summary CSV files found under {args.outputs}")
    combined = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    numeric = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    aggregate = combined.groupby(["experiment", "model"])[numeric].agg(["mean", "std"])
    aggregate.to_csv(args.output.with_name("aggregate_mean_sd.csv"))
    print(aggregate.to_string())


if __name__ == "__main__":
    main()

