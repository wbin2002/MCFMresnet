from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import metrics_from_arrays


def paired_patient_bootstrap(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    iterations: int,
    seed: int,
) -> dict:
    keys = ["image_id", "patient_id", "y_true"]
    merged = reference[keys + ["y_pred"]].merge(
        comparison[keys + ["y_pred"]],
        on=keys,
        suffixes=("_reference", "_comparison"),
        validate="one_to_one",
    )
    if len(merged) != len(reference) or len(merged) != len(comparison):
        raise ValueError("Paired prediction files do not contain exactly the same test images")

    patient_ids = merged.patient_id.unique()
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(iterations):
        sampled = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        chunks = [merged[merged.patient_id == pid] for pid in sampled]
        sample = pd.concat(chunks, ignore_index=True)
        reference_f1 = metrics_from_arrays(
            sample.y_true.to_numpy(), sample.y_pred_reference.to_numpy()
        )["f1_macro"]
        comparison_f1 = metrics_from_arrays(
            sample.y_true.to_numpy(), sample.y_pred_comparison.to_numpy()
        )["f1_macro"]
        differences.append(comparison_f1 - reference_f1)

    observed_reference = metrics_from_arrays(
        merged.y_true.to_numpy(), merged.y_pred_reference.to_numpy()
    )["f1_macro"]
    observed_comparison = metrics_from_arrays(
        merged.y_true.to_numpy(), merged.y_pred_comparison.to_numpy()
    )["f1_macro"]
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "reference_f1": observed_reference,
        "comparison_f1": observed_comparison,
        "paired_difference": observed_comparison - observed_reference,
        "bootstrap_ci_95_low": float(low),
        "bootstrap_ci_95_high": float(high),
        "bootstrap_iterations": iterations,
        "resampling_unit": "patient_or_verified_group",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired patient-level bootstrap for two prediction files.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = paired_patient_bootstrap(
        pd.read_csv(args.reference),
        pd.read_csv(args.comparison),
        args.iterations,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

