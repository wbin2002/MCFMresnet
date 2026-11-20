from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize completed public benchmark runs.")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for metrics_path in sorted(args.output_root.glob("*/seed_*/metrics.json")):
        config_path = metrics_path.with_name("config.json")
        if not config_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": config["model"],
                "seed": config["training"]["seed"],
                "parameters": config["parameter_count"],
                **metrics,
                "run_dir": str(metrics_path.parent),
            }
        )
    if not rows:
        raise SystemExit(f"No completed runs found under {args.output_root}")

    frame = pd.DataFrame(rows).sort_values(["model", "seed"])
    frame.to_csv(args.output_root / "all_runs.csv", index=False)
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    grouped = frame.groupby("model")[metrics].agg(["mean", "std", "count"])
    grouped.columns = ["_".join(column) for column in grouped.columns]
    grouped.reset_index().to_csv(args.output_root / "summary_mean_sd.csv", index=False)
    print(frame.to_string(index=False))
    print(f"Saved: {args.output_root / 'all_runs.csv'}")
    print(f"Saved: {args.output_root / 'summary_mean_sd.csv'}")


if __name__ == "__main__":
    main()

