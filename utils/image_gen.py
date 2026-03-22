"""Image generation and manipulation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw


def create_blank_image(
    width: int = 512,
    height: int = 512,
    color: Tuple[int, int, int] = (255, 255, 255),
    output_path: Optional[Union[str, Path]] = None,
) -> Image.Image:
    """Create a blank solid-colour image.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        color: RGB fill colour.
        output_path: If provided, save the image to this path.

    Returns:
        The created PIL Image.
    """
    img = Image.new("RGB", (width, height), color)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
    return img


def load_and_resize(
    path: Union[str, Path],
    target_size: Tuple[int, int] = (512, 512),
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> Image.Image:
    """Load an image and resize it to the target dimensions.

    Args:
        path: Path to the source image.
        target_size: ``(width, height)`` to resize to.
        resample: PIL resampling filter.

    Returns:
        The resized PIL Image in RGB mode.
    """
    img = Image.open(path).convert("RGB")
    if img.size != target_size:
        img = img.resize(target_size, resample)
    return img


def apply_noise(
    image: Image.Image,
    intensity: float = 0.05,
    noise_type: str = "gaussian",
) -> Image.Image:
    """Add random noise to an image.

    Args:
        image: Source PIL Image.
        intensity: Noise strength (standard deviation for Gaussian, or max
            magnitude for uniform), relative to the ``[0, 255]`` range.
        noise_type: ``"gaussian"`` or ``"uniform"``.

    Returns:
        Noisy copy of the image.
    """
    arr = np.array(image, dtype=np.float32)

    if noise_type == "gaussian":
        noise = np.random.normal(0, intensity * 255, arr.shape)
    elif noise_type == "uniform":
        noise = np.random.uniform(-intensity * 255, intensity * 255, arr.shape)
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def save_comparison(
    original: Union[str, Path, Image.Image],
    adversarial: Union[str, Path, Image.Image],
    output_path: Union[str, Path],
    labels: Tuple[str, str] = ("Original", "Adversarial"),
    padding: int = 20,
    label_height: int = 30,
    background_color: Tuple[int, int, int] = (240, 240, 240),
) -> Path:
    """Save a side-by-side comparison of the original and adversarial images.

    Args:
        original: Original image or path.
        adversarial: Adversarial image or path.
        output_path: Where to save the comparison.
        labels: Captions below each image.
        padding: Pixel padding around and between images.
        label_height: Vertical space reserved for labels.
        background_color: Canvas background colour.

    Returns:
        Path to the saved comparison image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(original, (str, Path)):
        original = Image.open(original).convert("RGB")
    if isinstance(adversarial, (str, Path)):
        adversarial = Image.open(adversarial).convert("RGB")

    # Resize adversarial to match original
    adversarial = adversarial.resize(original.size, Image.Resampling.LANCZOS)

    w, h = original.size
    canvas_w = 2 * w + 3 * padding
    canvas_h = h + 2 * padding + label_height

    canvas = Image.new("RGB", (canvas_w, canvas_h), background_color)
    canvas.paste(original, (padding, padding))
    canvas.paste(adversarial, (2 * padding + w, padding))

    draw = ImageDraw.Draw(canvas)
    for i, label in enumerate(labels):
        x = padding + i * (w + padding) + w // 2
        y = padding + h + 5
        draw.text((x, y), label, fill=(0, 0, 0), anchor="mt")

    canvas.save(output_path)
    return output_path
