from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from app.config import settings
from app.utils import multiclass_mask_to_overlay_base64, pil_to_normalized_array, read_image_as_pil
from models.architecture_felix import FreqDWTUNet
from models.architecture_unet import UNet


@dataclass
class PredictionResult:
    overlay_base64: str
    score: float


class ModelManager:
    def __init__(self) -> None:
        self.device = torch.device(settings.device)
        self.models: Dict[str, nn.Module] = {}
        self.metadata: Dict[str, dict] = {}

    def load_models(self) -> None:
        self.models["unet"], self.metadata["unet"] = self._load_model(
            settings.base_model_path,
            "unet",
        )
        self.models["felix"], self.metadata["felix"] = self._load_model(
            settings.felix_model_path,
            "felix",
        )

    def _load_model(self, checkpoint_path: Path, model_type: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        num_classes = checkpoint["num_classes"]
        base_channels = checkpoint.get("base_channels", 32)

        if model_type == "unet":
            model = UNet(in_channels=1, num_classes=num_classes, base=base_channels)
        elif model_type == "felix":
            model = FreqDWTUNet(in_channels=1, num_classes=num_classes, base=base_channels)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        state_dict = checkpoint["model_state_dict"]
        cleaned_state = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(cleaned_state, strict=False)
        model.to(self.device)
        model.eval()

        metadata = {
            "num_classes": num_classes,
            "img_size": checkpoint.get("img_size", settings.input_size),
            "normalize": checkpoint.get("normalize", {"mean": 0.5, "std": 0.25}),
            "base_channels": base_channels,
            "model_name": checkpoint.get("model_name", model.__class__.__name__),
        }

        return model, metadata

    @torch.inference_mode()
    def predict_from_bytes(self, image_bytes: bytes) -> dict:
        original = read_image_as_pil(image_bytes)

        target_size = int(self.metadata["unet"].get("img_size", settings.input_size))
        arr = pil_to_normalized_array(original, target_size)
        tensor = torch.from_numpy(arr).to(self.device)

        start = time.perf_counter()
        unet_pred = self._predict_one("unet", tensor, original)
        felix_pred = self._predict_one("felix", tensor, original)
        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "unet_score": unet_pred.score,
            "felix_score": felix_pred.score,
            "unet_mask_base64": unet_pred.overlay_base64,
            "felix_mask_base64": felix_pred.overlay_base64,
            "latency_ms": round(latency_ms, 2),
        }

    def _predict_one(self, model_name: str, tensor: torch.Tensor, original_image) -> PredictionResult:
        logits = self.models[model_name](tensor)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype("uint8")

        overlay = multiclass_mask_to_overlay_base64(original_image, pred)

        foreground_ratio = float((pred > 0).mean())
        return PredictionResult(
            overlay_base64=overlay,
            score=round(foreground_ratio, 4),
        )


model_manager = ModelManager()
