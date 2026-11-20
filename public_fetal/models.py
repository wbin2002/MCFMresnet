from __future__ import annotations

import importlib.util
import hashlib
import types
from pathlib import Path

import torch.nn as nn

from mcfm_repro.models import ModelSpec, build_model


def _load_official_medmamba_source(repo: Path):
    source_path = repo / "MedMamba.py"
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Official MedMamba.py not found at {source_path}. "
            "Run public_fetal_benchmark/install_medmamba_cuda.sh first."
        )
    if importlib.util.find_spec("mamba_ssm") is None:
        raise RuntimeError(
            "mamba_ssm is not installed. Install the CUDA-matched mamba-ssm and "
            "causal-conv1d packages before running MedMamba."
        )

    source = source_path.read_text(encoding="utf-8")
    # The upstream file creates three CUDA models and runs a forward pass at import
    # time. Execute the class definitions only; the architecture itself is unchanged.
    marker = "medmamba_t = VSSM("
    if marker not in source:
        raise RuntimeError(
            "The official MedMamba source layout changed; refusing to guess the import boundary."
        )
    definitions = source.split(marker, maxsplit=1)[0]
    module = types.ModuleType("official_medmamba_model")
    module.__file__ = str(source_path)
    exec(compile(definitions, str(source_path), "exec"), module.__dict__)
    if "VSSM" not in module.__dict__:
        raise RuntimeError("VSSM class was not found in the official MedMamba source")
    if "selective_scan_fn" not in module.__dict__:
        raise RuntimeError(
            "The installed mamba_ssm does not expose selective_scan_fn at the API used by "
            "the official MedMamba implementation. Install a compatible CUDA build."
        )
    return module.VSSM


def build_public_model(
    name: str,
    num_classes: int,
    pretrained: bool,
    medmamba_repo: Path | None,
) -> tuple[nn.Module, bool, bool, dict]:
    normalized = name.lower().replace("-", "_")
    if normalized in {"mcfm_resnet18", "resnet18"}:
        spec = ModelSpec(
            name=normalized,
            num_classes=num_classes,
            pretrained=pretrained,
            contrasts=(1.0, 1.3, 1.6, 2.0),
            maps_per_contrast=3,
            branch_activation="mish",
            fusion="gccp",
            clip_contrast=True,
        )
        model, three_channel, normalize = build_model(spec)
        provenance = {
            "implementation": "mcfm_repro.models",
            "model_spec": spec.__dict__,
        }
        return model, three_channel, normalize, provenance

    if normalized == "convnextv2_base":
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("Install timm before running ConvNeXt-V2-Base") from exc
        timm_name = (
            "convnextv2_base.fcmae_ft_in1k" if pretrained else "convnextv2_base"
        )
        model = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=3,
        )
        return model, True, pretrained, {
            "implementation": "timm",
            "timm_model": timm_name,
        }

    if normalized == "medmamba_t":
        if pretrained:
            raise ValueError(
                "This benchmark does not silently load MedMamba weights; use --no-pretrained."
            )
        if medmamba_repo is None:
            raise ValueError("--medmamba-repo is required for model medmamba_t")
        VSSM = _load_official_medmamba_source(medmamba_repo.resolve())
        source_path = (medmamba_repo / "MedMamba.py").resolve()
        model = VSSM(
            depths=[2, 2, 4, 2],
            dims=[96, 192, 384, 768],
            num_classes=num_classes,
        )
        return model, True, False, {
            "implementation": "official YubiaoYue/MedMamba MedMamba.py",
            "source_path": str(source_path),
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "variant": "medmamba_t",
            "depths": [2, 2, 4, 2],
            "dims": [96, 192, 384, 768],
        }

    raise ValueError(f"Unsupported public benchmark model: {name}")

