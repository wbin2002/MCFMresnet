from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import pandas as pd
import sklearn
import torch

try:
    import torchvision
except (ImportError, RuntimeError) as exc:
    message = str(exc)
    if "torchvision::nms" in message or "Couldn't load custom C++ ops" in message:
        raise SystemExit(
            "Incompatible PyTorch/torchvision binaries were detected. "
            "This benchmark requires the tested pair torch==2.5.1 and "
            "torchvision==0.20.1, installed from the same PyTorch wheel index "
            "(for example cu124, cu121, or cpu). See README_中文.md section "
            "'环境修复'. Original error: " + message
        ) from exc
    raise

from mcfm_repro.data import SplitFrames, make_loaders, split_summary
from mcfm_repro.engine import set_seed, train_model
from public_fetal_benchmark.models import build_public_model


def read_fixed_split(path: Path) -> SplitFrames:
    frame = pd.read_csv(path, dtype={"patient_id": str})
    required = {"image_id", "patient_id", "label", "class_name", "image_path", "subset"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Split manifest is missing columns: {sorted(missing)}")
    if set(frame["subset"].unique()) != {"train", "validation", "test"}:
        raise ValueError("Manifest must contain train, validation, and test subsets")

    parts = {
        name: frame.loc[frame["subset"] == name].copy().reset_index(drop=True)
        for name in ("train", "validation", "test")
    }
    patient_sets = {name: set(part["patient_id"]) for name, part in parts.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = patient_sets[left] & patient_sets[right]
        if overlap:
            raise RuntimeError(
                f"Patient leakage detected in supplied manifest ({left}/{right}): {sorted(overlap)[:10]}"
            )
    return SplitFrames(train=parts["train"], val=parts["validation"], test=parts["test"])


def package_versions() -> dict:
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "scikit_learn": sklearn.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        import timm

        versions["timm"] = timm.__version__
    except ImportError:
        versions["timm"] = None
    try:
        import mamba_ssm

        versions["mamba_ssm"] = getattr(mamba_ssm, "__version__", "unknown")
    except ImportError:
        versions["mamba_ssm"] = None
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one model on the fixed official FETAL_PLANES_DB split."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=["mcfm_resnet18", "resnet18", "convnextv2_base", "medmamba_t"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=0.0)
    parser.add_argument("--augmentation", choices=["none", "mild"], default="none")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--medmamba-repo", type=Path, default=None)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    split = read_fixed_split(args.split_manifest)
    set_seed(args.seed)
    model, three_channel, normalize, provenance = build_public_model(
        args.model, 6, args.pretrained, args.medmamba_repo
    )
    train_loader, validation_loader, test_loader = make_loaders(
        split,
        args.dataset_root.resolve(),
        args.batch_size,
        args.num_workers,
        args.seed,
        three_channel,
        normalize,
        args.augmentation,
    )

    output_dir = args.output_root / args.model / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    split_frame = pd.concat(
        [
            split.train.assign(subset="train"),
            split.val.assign(subset="validation"),
            split.test.assign(subset="test"),
        ],
        ignore_index=True,
    )
    split_frame.to_csv(output_dir / "split_manifest.csv", index=False)

    config = {
        "dataset": {
            "name": "FETAL_PLANES_DB",
            "version": "1.0",
            "doi": "10.5281/zenodo.3904280",
            "root": str(args.dataset_root.resolve()),
            "split_manifest": str(args.split_manifest.resolve()),
        },
        "model": args.model,
        "model_provenance": provenance,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "optimizer": args.optimizer,
            "scheduler": args.scheduler,
            "warmup_epochs": args.warmup_epochs,
            "label_smoothing": args.label_smoothing,
            "gradient_clip": args.gradient_clip,
            "augmentation": args.augmentation,
            "pretrained": args.pretrained,
            "amp": args.amp,
            "checkpoint_criterion": "validation_macro_f1",
            "test_evaluations": 1,
        },
        "preprocessing": {
            "resize": [224, 224],
            "grayscale": True,
            "range": [0.0, 1.0],
            "repeat_to_three_channels": three_channel,
            "imagenet_normalization": normalize,
        },
        "split": split_summary(split),
        "software": package_versions(),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    metrics, _ = train_model(
        model,
        train_loader,
        validation_loader,
        test_loader,
        output_dir,
        args.epochs,
        args.learning_rate,
        None,
        None,
        args.weight_decay,
        args.optimizer,
        args.scheduler,
        args.warmup_epochs,
        args.label_smoothing,
        args.gradient_clip,
        6,
        torch.device(args.device),
        args.amp,
    )
    print(json.dumps({"model": args.model, "seed": args.seed, **metrics}, indent=2))


if __name__ == "__main__":
    main()

