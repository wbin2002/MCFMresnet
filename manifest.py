from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


CLASS_TO_LABEL = {
    "双肾横断": 0,
    "双肾矢状": 1,
    "脊柱": 2,
    "腹围": 3,
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def stable_hash(value: str, prefix: str) -> str:
    digest = hashlib.sha256((prefix + value).encode("utf-8")).hexdigest()
    return digest[:16]


def patient_key_and_source(path: Path) -> tuple[str, str]:
    """Read the shared patient index from the revised filename structure."""
    match = re.search(r"\((\d+)\)\s*$", path.stem)
    if match:
        return f"P{int(match.group(1)):04d}", "structured_parenthesized_index"
    return path.stem.split("_", 1)[0].strip(), "filename_prefix_inferred"


def build_manifest(dataset_dir: Path, output_csv: Path) -> pd.DataFrame:
    dataset_dir = dataset_dir.resolve()
    rows: list[dict] = []
    unknown_dirs: set[str] = set()

    for path in sorted(dataset_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        class_name = path.parent.name
        if class_name not in CLASS_TO_LABEL:
            unknown_dirs.add(class_name)
            continue
        relative = path.relative_to(dataset_dir).as_posix()
        raw_patient_key, patient_id_source = patient_key_and_source(path)
        rows.append(
            {
                "image_path": relative,
                "image_id": stable_hash(relative, "image:"),
                "patient_id": stable_hash(raw_patient_key, "patient:"),
                "label": CLASS_TO_LABEL[class_name],
                "class_name": class_name,
                "patient_id_source": patient_id_source,
            }
        )

    if unknown_dirs:
        raise ValueError(f"Unknown class directories: {sorted(unknown_dirs)}")
    if not rows:
        raise ValueError(f"No supported images found under {dataset_dir}")

    frame = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return frame


def audit_manifest(frame: pd.DataFrame) -> dict:
    required = {"image_path", "image_id", "patient_id", "label", "class_name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    duplicate_images = int(frame["image_id"].duplicated().sum())
    patient_labels: dict[str, set[int]] = defaultdict(set)
    patient_image_counts = Counter()
    for row in frame.itertuples(index=False):
        patient_labels[str(row.patient_id)].add(int(row.label))
        patient_image_counts[str(row.patient_id)] += 1

    complete_four_class = sum(
        count == 4 and patient_labels[pid] == {0, 1, 2, 3}
        for pid, count in patient_image_counts.items()
    )
    any_four_class = sum(labels == {0, 1, 2, 3} for labels in patient_labels.values())
    source_counts = (
        frame.get("patient_id_source", pd.Series(["unknown"] * len(frame)))
        .value_counts()
        .to_dict()
    )

    return {
        "images": int(len(frame)),
        "patients_or_inferred_groups": int(frame["patient_id"].nunique()),
        "class_counts": {
            str(int(k)): int(v) for k, v in frame["label"].value_counts().sort_index().items()
        },
        "duplicate_image_ids": duplicate_images,
        "groups_covering_all_four_classes": int(any_four_class),
        "groups_with_exactly_one_image_per_class": int(complete_four_class),
        "patient_id_source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "warning": (
            "Filename-derived patient IDs are an inference and must be replaced by a clinically "
            "verified patient manifest before making patient-level claims."
            if source_counts.get("filename_prefix_inferred", 0)
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and audit the internal dataset manifest.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("manifests/internal_manifest.csv"))
    parser.add_argument("--audit-json", type=Path, default=Path("manifests/internal_audit.json"))
    args = parser.parse_args()

    frame = build_manifest(args.dataset, args.output)
    audit = audit_manifest(frame)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

