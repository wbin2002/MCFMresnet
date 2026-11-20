from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure clipping fractions for each contrast coefficient.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--contrasts", default="1.0,1.3,1.6,2.0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    split = pd.read_csv(args.split_manifest)[["image_id", "subset"]]
    frame = manifest.merge(split, on="image_id", how="inner", validate="one_to_one")
    if len(frame) != len(manifest):
        raise ValueError("Split manifest does not cover every image in the dataset manifest")
    coefficients = [float(item) for item in args.contrasts.split(",")]
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]
    )
    totals = {(subset, alpha): [0, 0, 0] for subset in frame.subset.unique() for alpha in coefficients}
    totals.update({("all", alpha): [0, 0, 0] for alpha in coefficients})

    for row in frame.itertuples(index=False):
        with Image.open(args.dataset / str(row.image_path)) as image:
            tensor = transform(image.convert("RGB"))
        mean = tensor.mean()
        for alpha in coefficients:
            unbounded = mean + alpha * (tensor - mean)
            below = int((unbounded < 0.0).sum().item())
            above = int((unbounded > 1.0).sum().item())
            count = int(unbounded.numel())
            for key in ((str(row.subset), alpha), ("all", alpha)):
                totals[key][0] += below
                totals[key][1] += above
                totals[key][2] += count

    rows = []
    for (subset, alpha), (below, above, count) in sorted(totals.items()):
        rows.append(
            {
                "subset": subset,
                "contrast": alpha,
                "pixels": count,
                "fraction_clipped_low": below / count,
                "fraction_clipped_high": above / count,
                "fraction_clipped_total": (below + above) / count,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

