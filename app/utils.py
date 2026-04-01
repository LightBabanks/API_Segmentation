import base64
import io
from typing import Dict, Tuple

import numpy as np
from PIL import Image


CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),           # fond
    1: (142, 180, 255),     # bleu
    2: (255, 188, 214),     # rose
    3: (180, 255, 200),     # vert
    4: (255, 220, 150),     # orange clair
    5: (210, 190, 255),     # violet
    6: (255, 150, 150),     # rouge clair
    7: (150, 240, 240),     # cyan
}


def read_image_as_pil(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("L")


def pil_to_normalized_array(image: Image.Image, size: int) -> np.ndarray:
    image = image.resize((size, size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.25
    arr = np.expand_dims(arr, axis=0)  # [1, H, W]
    arr = np.expand_dims(arr, axis=0)  # [1, 1, H, W]
    return arr


def multiclass_mask_to_overlay_base64(
    original: Image.Image,
    mask: np.ndarray,
    alpha: int = 110,
) -> str:
    base = original.convert("RGBA")
    mask_img = Image.fromarray(mask.astype(np.uint8), mode="L")
    mask_img = mask_img.resize(base.size, Image.NEAREST)
    mask_arr = np.asarray(mask_img, dtype=np.uint8)

    overlay = np.zeros((base.size[1], base.size[0], 4), dtype=np.uint8)

    unique_labels = np.unique(mask_arr)
    for label in unique_labels:
        if label == 0:
            continue
        color = CLASS_COLORS.get(int(label), (255, 255, 255))
        region = mask_arr == label
        overlay[region, 0] = color[0]
        overlay[region, 1] = color[1]
        overlay[region, 2] = color[2]
        overlay[region, 3] = alpha

    overlay_img = Image.fromarray(overlay, mode="RGBA")
    merged = Image.alpha_composite(base, overlay_img)

    buffer = io.BytesIO()
    merged.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"
