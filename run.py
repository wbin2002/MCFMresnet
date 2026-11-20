from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from .data import five_folds, make_loaders, save_split_manifest, seed_split, split_summary
from .engine import set_seed, train_model
from .models import ModelSpec, build_model


DEFAULT_SEEDS = [2021, 2022, 2023, 2024, 2025]


def parse_contrasts(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def run_one(args, frame: pd.DataFrame, seed: int, fold: int | None, split, spec: ModelSpec) -> dict:
    set_seed(seed)
    model, three_channel, normalize = build_model(spec)
    train_loader, val_loader, test_loader = make_loaders(
        split,
        args.dataset,
        args.batch_size,
        args.num_workers,
        seed,
        three_channel,
        normalize,
    )
    tag = f"seed_{seed}" if fold is None else f"fold_{fold}_seed_{seed}"
    output_dir = args.output / args.experiment_name / spec.name / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    save_split_manifest(split, output_dir / "split_manifest.csv", seed, fold)
    config = {
        "experiment_name": args.experiment_name,
        "seed": seed,
        "fold": fold,
        "model": asdict(spec),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "checkpoint_criterion": "validation_macro_f1",
            "test_evaluations": 1,
        },
        "split": split_summary(split),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    metrics, predictions = train_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        output_dir,
        args.epochs,
        args.learning_rate,
        args.weight_decay,
        spec.num_classes,
        torch.device(args.device),
        args.amp,
    )
    return {
        "experiment": args.experiment_name,
        "model": spec.name,
        "seed": seed,
        "fold": fold,
        **metrics,
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-free MCFM experiments.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--mode", choices=["single", "five-seed", "five-fold"], default="single")
    parser.add_argument("--model", default="mcfm_resnet18")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--contrasts", type=parse_contrasts, default=(1.0, 1.3, 1.6, 2.0))
    parser.add_argument("--maps-per-contrast", type=int, default=3)
    parser.add_argument("--branch-activation", choices=["mish", "relu"], default="mish")
    parser.add_argument("--fusion", choices=["gccp", "gap", "concat", "se", "cbam"], default="gccp")
    parser.add_argument("--clip-contrast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-inferred-patient-ids",
        action="store_true",
        help="Allow filename-prefix inferred groups. Do not use for patient-level manuscript claims.",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    if (
        "patient_id_source" in frame.columns
        and frame.patient_id_source.astype(str).str.contains("inferred").any()
        and not args.allow_inferred_patient_ids
    ):
        raise SystemExit(
            "The manifest contains inferred patient IDs. Replace them with clinically verified IDs, "
            "or pass --allow-inferred-patient-ids only for exploratory runs."
        )
    spec = ModelSpec(
        name=args.model,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        contrasts=args.contrasts,
        maps_per_contrast=args.maps_per_contrast,
        branch_activation=args.branch_activation,
        fusion=args.fusion,
        clip_contrast=args.clip_contrast,
    )

    rows = []
    if args.mode == "single":
        split = seed_split(frame, args.seed)
        rows.append(run_one(args, frame, args.seed, None, split, spec))
    elif args.mode == "five-seed":
        for seed in DEFAULT_SEEDS:
            split = seed_split(frame, seed)
            rows.append(run_one(args, frame, seed, None, split, spec))
    else:
        for fold, split in five_folds(frame, args.seed):
            rows.append(run_one(args, frame, args.seed, fold, split, spec))

    summary_dir = args.output / args.experiment_name
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"summary_{spec.name}.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

