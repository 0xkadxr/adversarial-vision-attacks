"""Typographic attacks on Vision-Language Models.

This module implements attacks that overlay text instructions directly onto images,
exploiting the tendency of VLMs to follow text visible in the image. These attacks
are effective because multimodal models often treat in-image text as instructions.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple, Union

from PIL import Image, ImageDraw, ImageFont

from utils.text_render import render_text, render_multiline, calculate_text_bbox


PositionType = Literal["top", "center", "bottom", "top-left", "top-right", "bottom-left", "bottom-right"]
BlendMode = Literal["normal", "multiply", "screen", "overlay", "soft-light"]


class TypographicAttack:
    """Generate typographic adversarial images that embed visible text instructions.

    Typographic attacks overlay human-readable text onto images. When a VLM processes
    the image, it reads the text and may follow the embedded instructions, overriding
    the user's actual prompt.

    Args:
        font_size: Default font size for rendered text.
        font_color: RGBA tuple for text color. Defaults to black.
        position: Default text placement position on the image.
        opacity: Text opacity from 0 (transparent) to 255 (opaque).
        font_path: Optional path to a .ttf/.otf font file.
    """

    def __init__(
        self,
        font_size: int = 40,
        font_color: Tuple[int, int, int, int] = (0, 0, 0, 255),
        position: PositionType = "center",
        opacity: int = 255,
        font_path: Optional[str] = None,
    ) -> None:
        self.font_size = font_size
        self.font_color = font_color
        self.position = position
        self.opacity = opacity
        self.font_path = font_path

    def _load_font(self, size: Optional[int] = None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Load the configured font at the given size."""
        sz = size or self.font_size
        if self.font_path:
            return ImageFont.truetype(self.font_path, sz)
        try:
            return ImageFont.truetype("arial.ttf", sz)
        except OSError:
            return ImageFont.load_default()

    def _resolve_position(
        self,
        image_size: Tuple[int, int],
        text_size: Tuple[int, int],
        position: Optional[PositionType] = None,
        margin: int = 10,
    ) -> Tuple[int, int]:
        """Calculate pixel coordinates for the given named position.

        Args:
            image_size: (width, height) of the target image.
            text_size: (width, height) of the rendered text bounding box.
            position: Named position. Falls back to ``self.position``.
            margin: Pixel margin from edges.

        Returns:
            (x, y) coordinates for the top-left of the text box.
        """
        pos = position or self.position
        iw, ih = image_size
        tw, th = text_size

        positions: dict[str, Tuple[int, int]] = {
            "top": ((iw - tw) // 2, margin),
            "center": ((iw - tw) // 2, (ih - th) // 2),
            "bottom": ((iw - tw) // 2, ih - th - margin),
            "top-left": (margin, margin),
            "top-right": (iw - tw - margin, margin),
            "bottom-left": (margin, ih - th - margin),
            "bottom-right": (iw - tw - margin, ih - th - margin),
        }
        return positions.get(pos, positions["center"])

    def generate(
        self,
        image_path: Union[str, Path],
        text: str,
        output_path: Union[str, Path],
        position: Optional[PositionType] = None,
        font_size: Optional[int] = None,
        font_color: Optional[Tuple[int, int, int, int]] = None,
    ) -> Path:
        """Overlay adversarial text instructions onto an image.

        Args:
            image_path: Path to the source image.
            text: Adversarial text to render on the image.
            output_path: Where to save the resulting image.
            position: Override default position for this generation.
            font_size: Override default font size.
            font_color: Override default font color.

        Returns:
            Path to the saved adversarial image.
        """
        image_path = Path(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(image_path).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        font = self._load_font(font_size)
        color = font_color or self.font_color
        color_with_opacity = (*color[:3], self.opacity)

        bbox = calculate_text_bbox(text, font)
        text_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        xy = self._resolve_position(img.size, text_size, position)

        draw.text(xy, text, fill=color_with_opacity, font=font)
        result = Image.alpha_composite(img, overlay).convert("RGB")
        result.save(output_path)
        return output_path

    def generate_grid(
        self,
        texts: Sequence[str],
        grid_size: Tuple[int, int] = (2, 2),
        output_path: Union[str, Path] = "grid_attack.png",
        cell_size: Tuple[int, int] = (400, 400),
        background_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> Path:
        """Create a grid image where each cell contains adversarial text.

        Useful for testing multiple adversarial instructions simultaneously.

        Args:
            texts: List of text strings, one per grid cell.
            grid_size: (columns, rows) of the grid.
            output_path: Where to save the grid image.
            cell_size: (width, height) of each grid cell in pixels.
            background_color: RGB background color for cells.

        Returns:
            Path to the saved grid image.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cols, rows = grid_size
        cw, ch = cell_size
        grid_img = Image.new("RGB", (cols * cw, rows * ch), background_color)
        draw = ImageDraw.Draw(grid_img)
        font = self._load_font()

        for idx, text in enumerate(texts[: cols * rows]):
            col = idx % cols
            row = idx // cols
            x_offset = col * cw
            y_offset = row * ch

            bbox = calculate_text_bbox(text, font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            x = x_offset + (cw - tw) // 2
            y = y_offset + (ch - th) // 2

            draw.text((x, y), text, fill=self.font_color[:3], font=font)

            # Draw cell borders
            draw.rectangle(
                [x_offset, y_offset, x_offset + cw - 1, y_offset + ch - 1],
                outline=(200, 200, 200),
                width=1,
            )

        grid_img.save(output_path)
        return output_path

    def generate_with_camouflage(
        self,
        image_path: Union[str, Path],
        text: str,
        blend_mode: BlendMode = "overlay",
        output_path: Union[str, Path] = "camouflage_attack.png",
        font_size: Optional[int] = None,
    ) -> Path:
        """Blend adversarial text into an image so it appears natural.

        The text is rendered and then composited using the specified blend mode,
        making it harder for humans to notice while still readable by VLMs.

        Args:
            image_path: Path to the source image.
            text: Adversarial text to embed.
            blend_mode: How to blend text with the background image.
            output_path: Where to save the result.
            font_size: Override default font size.

        Returns:
            Path to the saved image.
        """
        image_path = Path(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(image_path).convert("RGBA")
        text_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_layer)
        font = self._load_font(font_size)

        bbox = calculate_text_bbox(text, font)
        text_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        xy = self._resolve_position(img.size, text_size)

        draw.text(xy, text, fill=self.font_color, font=font)

        result = self._apply_blend(img, text_layer, blend_mode)
        result.convert("RGB").save(output_path)
        return output_path

    @staticmethod
    def _apply_blend(
        base: Image.Image,
        layer: Image.Image,
        mode: BlendMode,
    ) -> Image.Image:
        """Apply a blend mode between a base image and an overlay layer.

        Args:
            base: The background RGBA image.
            layer: The overlay RGBA image (text layer).
            mode: Blend mode to apply.

        Returns:
            Blended RGBA image.
        """
        import numpy as np

        base_arr = np.array(base, dtype=np.float32) / 255.0
        layer_arr = np.array(layer, dtype=np.float32) / 255.0

        alpha = layer_arr[:, :, 3:4]
        b_rgb = base_arr[:, :, :3]
        l_rgb = layer_arr[:, :, :3]

        if mode == "normal":
            blended = l_rgb
        elif mode == "multiply":
            blended = b_rgb * l_rgb
        elif mode == "screen":
            blended = 1.0 - (1.0 - b_rgb) * (1.0 - l_rgb)
        elif mode == "overlay":
            mask = b_rgb < 0.5
            blended = np.where(mask, 2 * b_rgb * l_rgb, 1.0 - 2 * (1.0 - b_rgb) * (1.0 - l_rgb))
        elif mode == "soft-light":
            blended = np.where(
                l_rgb < 0.5,
                b_rgb - (1.0 - 2 * l_rgb) * b_rgb * (1.0 - b_rgb),
                b_rgb + (2 * l_rgb - 1.0) * (np.sqrt(b_rgb) - b_rgb),
            )
        else:
            blended = l_rgb

        # Composite using alpha
        out_rgb = b_rgb * (1 - alpha) + blended * alpha
        out_alpha = base_arr[:, :, 3:4] + alpha * (1 - base_arr[:, :, 3:4])
        out = np.concatenate([out_rgb, out_alpha], axis=2)
        out = (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)
        return Image.fromarray(out, "RGBA")
