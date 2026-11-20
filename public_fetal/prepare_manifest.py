from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import pandas as pd
from PIL import Image


CLASS_TO_LABEL = {
    "Fetal abdomen": 0,
    "Fetal brain": 1,
    "Fetal femur": 2,
    "Fetal thorax": 3,
    "Maternal cervix": 4,
    "Other": 5,
}


def read_official_metadata(metadata_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(metadata_path, sep=";", dtype=str)
    frame.columns = [str(column).strip() for column in frame.columns]
    for column in frame.columns:
        frame[column] = frame[column].astype(str).str.strip()

    required = {"Image_name", "Patient_num", "Plane", "Train"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Official metadata is missing columns: {sorted(missing)}")
    if len(frame) != 12_400:
        raise ValueError(f"Expected 12,400 metadata rows, found {len(frame):,}")
    unknown = set(frame["Plane"].unique()) - set(CLASS_TO_LABEL)
    if unknown:
        raise ValueError(f"Unexpected classes in metadata: {sorted(unknown)}")
    invalid_flags = set(frame["Train"].unique()) - {"0", "1"}
    if invalid_flags:
        raise ValueError(f"Unexpected official Train values: {sorted(invalid_flags)}")
    if frame["Image_name"].duplicated().any():
        raise ValueError("Duplicate Image_name values found in official metadata")

    frame = frame.rename(
        columns={"Image_name": "image_id", "Patient_num": "patient_id", "Plane": "class_name"}
    )
    frame["patient_id"] = frame["patient_id"].astype(str).str.zfill(4)
    frame["label"] = frame["class_name"].map(CLASS_TO_LABEL).astype(int)
    frame["official_subset"] = frame["Train"].map({"1": "official_train", "0": "official_test"})
    frame["image_path"] = "Images/" + frame["image_id"] + ".png"
    return frame


def inspect_image(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    if path.stat().st_size == 0:
        return False, "zero_byte"
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:  # Pillow exposes several decoder-specific exception types.
        return False, f"unreadable:{type(exc).__name__}:{exc}"
    return True, "ok"


def assign_validation_patients(
    frame: pd.DataFrame, seed: int, validation_fraction: float
) -> set[str]:
    patients = sorted(
        frame.loc[frame["official_subset"] == "official_train", "patient_id"]
        .astype(str)
        .unique()
        .tolist()
    )
    random.Random(seed).shuffle(patients)
    validation_count = math.ceil(len(patients) * validation_fraction)
    return set(patients[:validation_count])


def subset_summary(frame: pd.DataFrame) -> dict:
    summary = {}
    for subset, part in frame.groupby("subset", sort=False):
        summary[str(subset)] = {
            "images": int(len(part)),
            "patients": int(part["patient_id"].nunique()),
            "class_counts": {
                str(name): int(count)
                for name, count in part["class_name"].value_counts().sort_index().items()
            },
        }
    return summary


def assert_no_patient_leakage(frame: pd.DataFrame) -> dict:
    patient_sets = {
        subset: set(part["patient_id"].astype(str))
        for subset, part in frame.groupby("subset")
    }
    overlaps = {}
    names = ["train", "validation", "test"]
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = patient_sets.get(left, set()) & patient_sets.get(right, set())
            overlaps[f"{left}_vs_{right}"] = len(overlap)
            if overlap:
                raise RuntimeError(
                    f"Patient leakage detected between {left} and {right}: {sorted(overlap)[:10]}"
                )
    return overlaps


def prepare(
    dataset_root: Path,
    output_dir: Path,
    validation_seed: int,
    validation_fraction: float,
) -> dict:
    dataset_root = dataset_root.resolve()
    metadata_path = dataset_root / "FETAL_PLANES_DB_data.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing official metadata: {metadata_path}")

    frame = read_official_metadata(metadata_path)
    official_train_patients = set(
        frame.loc[frame["official_subset"] == "official_train", "patient_id"]
    )
    official_test_patients = set(
        frame.loc[frame["official_subset"] == "official_test", "patient_id"]
    )
    official_overlap = official_train_patients & official_test_patients
    if official_overlap:
        raise RuntimeError(f"Official metadata contains patient overlap: {sorted(official_overlap)[:10]}")

    validation_patients = assign_validation_patients(
        frame, validation_seed, validation_fraction
    )
    frame["subset"] = "test"
    official_train_mask = frame["official_subset"] == "official_train"
    frame.loc[official_train_mask, "subset"] = "train"
    frame.loc[
        official_train_mask & frame["patient_id"].isin(validation_patients), "subset"
    ] = "validation"

    checks = [inspect_image(dataset_root / relative) for relative in frame["image_path"]]
    frame["usable"] = [usable for usable, _ in checks]
    frame["integrity_status"] = [reason for _, reason in checks]

    usable = frame.loc[frame["usable"]].copy().reset_index(drop=True)
    excluded = frame.loc[~frame["usable"]].copy().reset_index(drop=True)
    overlaps = assert_no_patient_leakage(usable)

    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "image_id",
        "patient_id",
        "class_name",
        "label",
        "image_path",
        "official_subset",
        "subset",
        "usable",
        "integrity_status",
    ]
    frame[columns].to_csv(output_dir / "official_manifest_all.csv", index=False)
    usable[columns].to_csv(output_dir / "split_manifest.csv", index=False)
    excluded[columns].to_csv(output_dir / "excluded_images.csv", index=False)

    official_counts = (
        frame.groupby("official_subset")
        .agg(images=("image_id", "size"), patients=("patient_id", "nunique"))
        .to_dict(orient="index")
    )
    official_class_counts = {
        str(subset): {
            str(name): int(count)
            for name, count in part["class_name"].value_counts().sort_index().items()
        }
        for subset, part in frame.groupby("official_subset")
    }
    summary = {
        "dataset": {
            "name": "FETAL_PLANES_DB",
            "version": "1.0",
            "doi": "10.5281/zenodo.3904280",
            "metadata": str(metadata_path),
        },
        "validation": {
            "seed": validation_seed,
            "fraction_of_official_training_patients": validation_fraction,
            "selection": "seeded random sample of official training patient IDs only",
        },
        "official_before_integrity_exclusions": official_counts,
        "official_class_counts_before_integrity_exclusions": official_class_counts,
        "integrity": {
            "metadata_rows": int(len(frame)),
            "usable_images": int(len(usable)),
            "excluded_images": int(len(excluded)),
            "exclusion_reasons": {
                str(reason): int(count)
                for reason, count in excluded["integrity_status"].value_counts().items()
            },
        },
        "final_split": subset_summary(usable),
        "patient_overlap_counts": {"official_train_vs_test": len(official_overlap), **overlaps},
        "class_to_label": CLASS_TO_LABEL,
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an auditable official FETAL_PLANES_DB split manifest."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-seed", type=int, default=2021)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise SystemExit("--validation-fraction must be between 0 and 1")
    summary = prepare(
        args.dataset_root,
        args.output_dir,
        args.validation_seed,
        args.validation_fraction,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

