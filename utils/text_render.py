"""Text rendering utilities for placing text on images."""

from __future__ import annotations

from typing import Optional, Tuple, Union

from PIL import Image, ImageDraw, ImageFont


def _load_default_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Attempt to load a TrueType font, falling back to the built-in bitmap font."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_text(
    image: Image.Image,
    text: str,
    position: Tuple[int, int],
    font_size: int = 20,
    color: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (0, 0, 0),
    opacity: int = 255,
    font_path: Optional[str] = None,
) -> Image.Image:
    """Render a single line of text onto an image.

    Draws directly on the provided image (mutates in place) and also returns it
    for convenience.

    Args:
        image: Target PIL Image (RGB or RGBA).
        text: Text string to render.
        position: ``(x, y)`` pixel coordinates of the top-left corner.
        font_size: Font size in points.
        color: RGB or RGBA colour tuple.
        opacity: Alpha value (0--255). Only effective when the image is RGBA.
        font_path: Optional path to a ``.ttf`` / ``.otf`` font file.

    Returns:
        The same image with text drawn on it.
    """
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = _load_default_font(font_size)

    if len(color) == 3:
        fill = (*color, opacity)
    else:
        fill = color

    if image.mode == "RGBA":
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.text(position, text, fill=fill, font=font)
        image.paste(Image.alpha_composite(image, overlay))
    else:
        draw = ImageDraw.Draw(image)
        draw.text(position, text, fill=color[:3], font=font)

    return image


def render_multiline(
    image: Image.Image,
    text: str,
    bbox: Tuple[int, int, int, int],
    font_size: int = 16,
    color: Union[Tuple[int, int, int], Tuple[int, int, int, int]] = (0, 0, 0),
    line_spacing: int = 4,
    font_path: Optional[str] = None,
) -> Image.Image:
    """Render multi-line text within a bounding box, wrapping as needed.

    Text is word-wrapped to fit within the horizontal extent of *bbox*. Lines
    that would exceed the vertical extent are silently clipped.

    Args:
        image: Target PIL Image.
        text: The full text to render (newlines in the string are honoured).
        bbox: ``(x1, y1, x2, y2)`` bounding rectangle.
        font_size: Font size in points.
        color: RGB or RGBA colour.
        line_spacing: Extra vertical pixels between lines.
        font_path: Optional TrueType font path.

    Returns:
        The image with text rendered.
    """
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = _load_default_font(font_size)

    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = bbox
    max_width = x2 - x1

    paragraphs = text.split("\n")
    lines: list[str] = []

    for para in paragraphs:
        words = para.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            test = f"{current} {word}"
            tw = draw.textlength(test, font=font)
            if tw <= max_width:
                current = test
            else:
                lines.append(current)
                current = word
        lines.append(current)

    y = y1
    for line in lines:
        line_bbox = draw.textbbox((0, 0), line, font=font)
        line_height = line_bbox[3] - line_bbox[1]
        if y + line_height > y2:
            break
        draw.text((x1, y), line, fill=color[:3] if len(color) >= 3 else color, font=font)
        y += line_height + line_spacing

    return image


def calculate_text_bbox(
    text: str,
    font: Union[ImageFont.FreeTypeFont, ImageFont.ImageFont, int] = 20,
) -> Tuple[int, int, int, int]:
    """Calculate the bounding box of rendered text without drawing it.

    Args:
        text: The text string to measure.
        font: A PIL font object, or an integer font size (which loads the
            default font at that size).

    Returns:
        ``(x0, y0, x1, y1)`` bounding box tuple.
    """
    if isinstance(font, int):
        font = _load_default_font(font)

    # Use a temporary drawing surface just for measuring
    tmp = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(tmp)
    return draw.textbbox((0, 0), text, font=font)
