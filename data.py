from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class ManifestDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        dataset_root: Path,
        three_channel: bool = False,
        imagenet_normalize: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.dataset_root = dataset_root
        ops: list = [transforms.Resize((224, 224)), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]
        if three_channel:
            ops.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
        if imagenet_normalize:
            if not three_channel:
                raise ValueError("ImageNet normalization requires three_channel=True")
            ops.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = self.dataset_root / str(row.image_path)
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return {
            "image": tensor,
            "label": torch.tensor(int(row.label), dtype=torch.long),
            "image_id": str(row.image_id),
            "patient_id": str(row.patient_id),
        }


def _check_disjoint(split: SplitFrames) -> None:
    train_ids = set(split.train.patient_id.astype(str))
    val_ids = set(split.val.patient_id.astype(str))
    test_ids = set(split.test.patient_id.astype(str))
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Patient leakage detected between train/validation/test subsets")


def seed_split(frame: pd.DataFrame, seed: int, test_fraction: float = 0.30, val_fraction_of_train_pool: float = 0.20) -> SplitFrames:
    groups = frame.patient_id.astype(str).to_numpy()
    outer = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    trainval_idx, test_idx = next(outer.split(frame, frame.label, groups))
    trainval = frame.iloc[trainval_idx].reset_index(drop=True)
    test = frame.iloc[test_idx].reset_index(drop=True)

    inner_groups = trainval.patient_id.astype(str).to_numpy()
    inner = GroupShuffleSplit(n_splits=1, test_size=val_fraction_of_train_pool, random_state=seed + 100_000)
    train_idx, val_idx = next(inner.split(trainval, trainval.label, inner_groups))
    split = SplitFrames(
        train=trainval.iloc[train_idx].reset_index(drop=True),
        val=trainval.iloc[val_idx].reset_index(drop=True),
        test=test,
    )
    _check_disjoint(split)
    return split


def five_folds(frame: pd.DataFrame, seed: int = 2021) -> Iterable[tuple[int, SplitFrames]]:
    groups = frame.patient_id.astype(str).to_numpy()
    outer = GroupKFold(n_splits=5)
    for fold, (trainval_idx, test_idx) in enumerate(outer.split(frame, frame.label, groups), start=1):
        trainval = frame.iloc[trainval_idx].reset_index(drop=True)
        test = frame.iloc[test_idx].reset_index(drop=True)
        inner_groups = trainval.patient_id.astype(str).to_numpy()
        inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed + fold)
        train_idx, val_idx = next(inner.split(trainval, trainval.label, inner_groups))
        split = SplitFrames(
            train=trainval.iloc[train_idx].reset_index(drop=True),
            val=trainval.iloc[val_idx].reset_index(drop=True),
            test=test,
        )
        _check_disjoint(split)
        yield fold, split


def save_split_manifest(split: SplitFrames, output_csv: Path, seed: int, fold: int | None) -> None:
    frames = []
    for subset, frame in (("train", split.train), ("validation", split.val), ("test", split.test)):
        part = frame[["image_id", "patient_id", "label", "class_name"]].copy()
        part.insert(0, "subset", subset)
        part["seed"] = seed
        part["fold"] = "" if fold is None else fold
        frames.append(part)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(output_csv, index=False)


def split_summary(split: SplitFrames) -> dict:
    summary = {}
    for name, frame in (("train", split.train), ("validation", split.val), ("test", split.test)):
        summary[name] = {
            "images": int(len(frame)),
            "patients_or_groups": int(frame.patient_id.nunique()),
            "class_counts": {str(int(k)): int(v) for k, v in frame.label.value_counts().sort_index().items()},
        }
    return summary


def make_loaders(
    split: SplitFrames,
    dataset_root: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    three_channel: bool,
    imagenet_normalize: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train = DataLoader(
        ManifestDataset(split.train, dataset_root, three_channel, imagenet_normalize),
        shuffle=True,
        generator=generator,
        **common,
    )
    val = DataLoader(
        ManifestDataset(split.val, dataset_root, three_channel, imagenet_normalize),
        shuffle=False,
        **common,
    )
    test = DataLoader(
        ManifestDataset(split.test, dataset_root, three_channel, imagenet_normalize),
        shuffle=False,
        **common,
    )
    return train, val, test

