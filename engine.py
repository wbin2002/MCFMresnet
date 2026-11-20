from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, num_classes: int, include_predictions: bool):
    model.eval()
    true, pred, probs = [], [], []
    image_ids, patient_ids = [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(images)
        probability = torch.softmax(logits, dim=1)
        true.extend(labels.cpu().tolist())
        pred.extend(probability.argmax(1).cpu().tolist())
        probs.extend(probability.cpu().tolist())
        image_ids.extend(batch["image_id"])
        patient_ids.extend(batch["patient_id"])
    y_true = np.asarray(true, dtype=int)
    y_pred = np.asarray(pred, dtype=int)
    metrics = metrics_from_arrays(y_true, y_pred)
    predictions = None
    if include_predictions:
        predictions = pd.DataFrame(
            {
                "image_id": image_ids,
                "patient_id": patient_ids,
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )
        probabilities = np.asarray(probs)
        for class_index in range(num_classes):
            predictions[f"prob_{class_index}"] = probabilities[:, class_index]
    return metrics, predictions


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    output_dir: Path,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    num_classes: int,
    device: torch.device,
    amp: bool,
) -> tuple[dict, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")
    best_f1 = float("-inf")
    best_epoch = -1
    checkpoint = output_dir / "best_validation_f1.pt"
    history_rows = []

    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * labels.shape[0]
            total_examples += labels.shape[0]

        val_metrics, _ = evaluate(model, val_loader, device, num_classes, include_predictions=False)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(1, total_examples),
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
        )
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint)

    pd.DataFrame(history_rows).to_csv(output_dir / "training_history.csv", index=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test_metrics, predictions = evaluate(model, test_loader, device, num_classes, include_predictions=True)
    result = {
        **test_metrics,
        "best_validation_f1": float(best_f1),
        "best_epoch": int(best_epoch),
        "test_evaluations": 1,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    return result, predictions

