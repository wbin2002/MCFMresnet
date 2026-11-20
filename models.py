from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights


@dataclass(frozen=True)
class ModelSpec:
    name: str
    num_classes: int = 4
    pretrained: bool = False
    contrasts: tuple[float, ...] = (1.0, 1.3, 1.6, 2.0)
    maps_per_contrast: int = 3
    branch_activation: str = "mish"
    fusion: str = "gccp"
    clip_contrast: bool = True


def apply_contrast(images: torch.Tensor, contrasts: Sequence[float], clip: bool) -> torch.Tensor:
    mean = images.mean(dim=(2, 3), keepdim=True)
    outputs = []
    for alpha in contrasts:
        transformed = mean + float(alpha) * (images - mean)
        if clip:
            transformed = torch.clamp(transformed, 0.0, 1.0)
        outputs.append(transformed)
    return torch.cat(outputs, dim=1)


class ShallowBranches(nn.Module):
    def __init__(self, groups: int, maps_per_group: int, activation: str) -> None:
        super().__init__()
        self.maps_per_group = maps_per_group
        self.activation = activation
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, maps_per_group, kernel_size=3, padding=1) for _ in range(groups)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for conv, group in zip(self.convs, torch.split(x, 1, dim=1)):
            value = conv(group)
            if self.activation == "mish":
                value = F.mish(value)
            elif self.activation == "relu":
                value = F.relu(value)
            else:
                raise ValueError(f"Unknown branch activation: {self.activation}")
            outputs.append(value)
        return torch.cat(outputs, dim=1)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        descriptor = x.mean(dim=(2, 3))
        weights = torch.sigmoid(self.fc2(F.relu(self.fc1(descriptor))))
        return x * weights[:, :, None, None]


class CBAMBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=False)
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.fc2(F.relu(self.fc1(F.adaptive_avg_pool2d(x, 1))))
        maximum = self.fc2(F.relu(self.fc1(F.adaptive_max_pool2d(x, 1))))
        x = x * torch.sigmoid(avg + maximum)
        spatial_input = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.spatial(spatial_input))


def _make_resnet(depth: int, pretrained: bool):
    if depth == 18:
        return models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    if depth == 34:
        return models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    raise ValueError(f"Unsupported ResNet depth: {depth}")


def _replace_conv_with_repeated_pretrained(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    replacement = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=False,
    )
    with torch.no_grad():
        source = conv.weight.detach()
        repeats = (in_channels + source.shape[1] - 1) // source.shape[1]
        tiled = source.repeat(1, repeats, 1, 1)[:, :in_channels]
        replacement.weight.copy_(tiled * (source.shape[1] / in_channels))
    return replacement


class MCFMResNet(nn.Module):
    def __init__(self, depth: int, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.groups = len(spec.contrasts)
        self.maps_per_group = spec.maps_per_contrast
        channels = self.groups * self.maps_per_group

        self.branches = ShallowBranches(self.groups, self.maps_per_group, spec.branch_activation)
        if spec.fusion == "gccp":
            self.fc1 = nn.Linear(self.groups, self.groups)
            self.fc2 = nn.Linear(self.groups, self.groups)
        elif spec.fusion == "gap":
            self.fc1 = nn.Linear(channels, channels)
            self.fc2 = nn.Linear(channels, channels)
        elif spec.fusion == "se":
            self.attention = SEBlock(channels)
        elif spec.fusion == "cbam":
            self.attention = CBAMBlock(channels)
        elif spec.fusion != "concat":
            raise ValueError(f"Unknown fusion: {spec.fusion}")

        backbone = _make_resnet(depth, spec.pretrained)
        if spec.pretrained:
            backbone.conv1 = _replace_conv_with_repeated_pretrained(backbone.conv1, channels)
        else:
            backbone.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.fc = nn.Linear(backbone.fc.in_features, spec.num_classes)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        contrasted = apply_contrast(x, self.spec.contrasts, self.spec.clip_contrast)
        features = self.branches(contrasted)
        batch, channels, height, width = features.shape

        if self.spec.fusion == "gccp":
            descriptor = features.view(
                batch, self.groups, self.maps_per_group, height, width
            ).mean(dim=(2, 3, 4))
            weights = torch.sigmoid(self.fc2(F.relu(self.fc1(descriptor))))
            grouped = features.view(batch, self.groups, self.maps_per_group, height, width)
            fused = (grouped * weights[:, :, None, None, None]).view(batch, channels, height, width)
        elif self.spec.fusion == "gap":
            descriptor = features.mean(dim=(2, 3))
            weights = torch.sigmoid(self.fc2(F.relu(self.fc1(descriptor))))
            fused = features * weights[:, :, None, None]
        elif self.spec.fusion in {"se", "cbam"}:
            fused = self.attention(features)
        else:
            fused = features
        return self.backbone(fused)


def build_model(spec: ModelSpec) -> tuple[nn.Module, bool, bool]:
    """Return model, three_channel_input, imagenet_normalize."""
    name = spec.name.lower()
    if name in {"mcfm_resnet18", "mcfm-resnet18"}:
        return MCFMResNet(18, spec), False, False
    if name in {"mcfm_resnet34", "mcfm-resnet34"}:
        return MCFMResNet(34, spec), False, False

    if name in {"resnet18", "resnet34"}:
        depth = 18 if name == "resnet18" else 34
        model = _make_resnet(depth, spec.pretrained)
        if spec.pretrained:
            three_channel, normalize = True, True
        else:
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            three_channel, normalize = False, False
        model.fc = nn.Linear(model.fc.in_features, spec.num_classes)
        return model, three_channel, normalize

    constructors = {
        "vgg11": (models.vgg11, models.VGG11_Weights.IMAGENET1K_V1),
        "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
        "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1),
        "swin_t": (models.swin_t, models.Swin_T_Weights.IMAGENET1K_V1),
        "convnext_base": (models.convnext_base, models.ConvNeXt_Base_Weights.IMAGENET1K_V1),
        "inception_v3": (models.inception_v3, models.Inception_V3_Weights.IMAGENET1K_V1),
    }
    if name not in constructors:
        raise ValueError(f"Unsupported model: {spec.name}")
    constructor, weights = constructors[name]
    kwargs = {"weights": weights if spec.pretrained else None}
    model = constructor(**kwargs)
    if name == "inception_v3":
        model.aux_logits = False
        model.AuxLogits = None
    if name == "vgg11":
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, spec.num_classes)
    elif name == "efficientnet_b0":
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, spec.num_classes)
    elif name == "vit_b_16":
        model.heads.head = nn.Linear(model.heads.head.in_features, spec.num_classes)
    elif name == "swin_t":
        model.head = nn.Linear(model.head.in_features, spec.num_classes)
    elif name == "convnext_base":
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, spec.num_classes)
    else:
        model.fc = nn.Linear(model.fc.in_features, spec.num_classes)
    return model, True, spec.pretrained

