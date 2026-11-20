from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a blank verified-patient mapping template.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest)
    template = frame[["image_id", "image_path", "class_name"]].copy()
    template["verified_patient_id"] = ""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(template)} rows to {args.output}")


if __name__ == "__main__":
    main()
