from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .manifest import audit_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a verified image-to-patient mapping.")
    parser.add_argument("--inferred-manifest", type=Path, required=True)
    parser.add_argument(
        "--mapping",
        type=Path,
        required=True,
        help="CSV with image_id and verified_patient_id columns.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(args.inferred_manifest)
    mapping = pd.read_csv(args.mapping)
    required = {"image_id", "verified_patient_id"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Mapping missing columns: {sorted(missing)}")
    if mapping.image_id.duplicated().any():
        raise ValueError("Mapping contains duplicate image_id values")
    merged = manifest.merge(mapping[list(required)], on="image_id", how="left", validate="one_to_one")
    if merged.verified_patient_id.isna().any():
        count = int(merged.verified_patient_id.isna().sum())
        raise ValueError(f"Verified mapping is missing {count} images")
    merged["patient_id"] = merged.verified_patient_id.astype(str)
    merged["patient_id_source"] = "clinically_verified_mapping"
    merged = merged.drop(columns=["verified_patient_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False, encoding="utf-8-sig")
    audit = audit_manifest(merged)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

