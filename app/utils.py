import base64
import io
from typing import Tuple

import numpy as np
from PIL import Image


def read_image_as_pil(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    return image


def pil_to_normalized_array(image: Image.Image, size: int) -> np.ndarray:
    image = image.resize((size, size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # [1, H, W]
    arr = np.expand_dims(arr, axis=0)  # [1, 1, H, W]
    return arr


def mask_to_overlay_base64(
    original: Image.Image,
    mask: np.ndarray,
    tint: Tuple[int, int, int],
    alpha: int = 110,
) -> str:
    base = original.convert("RGBA")
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_img = mask_img.resize(base.size, Image.NEAREST)

    overlay = Image.new("RGBA", base.size, (*tint, 0))
    overlay_alpha = np.asarray(mask_img, dtype=np.uint8)
    overlay_alpha = np.where(overlay_alpha > 0, alpha, 0).astype(np.uint8)
    overlay.putalpha(Image.fromarray(overlay_alpha, mode="L"))

    merged = Image.alpha_composite(base, overlay)
    buffer = io.BytesIO()
    merged.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"
