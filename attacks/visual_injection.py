"""Visual prompt injection attacks.

Embeds hidden or semi-hidden instructions into images that override or
manipulate the behaviour of Vision-Language Models when the image is provided
as part of a prompt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.text_render import render_text, render_multiline

InjectionMethod = Literal["border_text", "watermark", "metadata", "overlay"]


class VisualPromptInjection:
    """Inject hidden instructions into images targeting VLMs.

    Provides several techniques for embedding adversarial instructions that
    are difficult for humans to notice but are read and followed by multimodal
    models.

    Args:
        font_size: Default font size for injected text.
        font_path: Optional path to a TrueType font file.
    """

    def __init__(
        self,
        font_size: int = 16,
        font_path: Optional[str] = None,
    ) -> None:
        self.font_size = font_size
        self.font_path = font_path

    def _load_font(self, size: Optional[int] = None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        sz = size or self.font_size
        if self.font_path:
            return ImageFont.truetype(self.font_path, sz)
        try:
            return ImageFont.truetype("arial.ttf", sz)
        except OSError:
            return ImageFont.load_default()

    # ------------------------------------------------------------------ #
    # Core injection methods
    # ------------------------------------------------------------------ #

    def inject_instruction(
        self,
        image_path: Union[str, Path],
        instruction: str,
        method: InjectionMethod = "border_text",
        output_path: Union[str, Path] = "injected.png",
        **kwargs,
    ) -> Path:
        """Inject a hidden instruction into an image using the given method.

        Args:
            image_path: Source image path.
            instruction: The adversarial instruction to embed.
            method: Injection technique to use.
            output_path: Destination for the modified image.
            **kwargs: Extra arguments forwarded to the specific method.

        Returns:
            Path to the output image.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dispatch = {
            "border_text": self._inject_border,
            "watermark": self._inject_watermark,
            "metadata": self._inject_metadata,
            "overlay": self._inject_overlay,
        }

        handler = dispatch.get(method)
        if handler is None:
            raise ValueError(
                f"Unknown injection method '{method}'. "
                f"Choose from: {list(dispatch.keys())}"
            )

        return handler(image_path, instruction, output_path, **kwargs)

    def create_adversarial_document(
        self,
        text_content: str,
        hidden_instruction: str,
        output_path: Union[str, Path] = "adversarial_doc.png",
        page_size: Tuple[int, int] = (800, 1100),
        background_color: Tuple[int, int, int] = (255, 255, 255),
        text_color: Tuple[int, int, int] = (30, 30, 30),
        hidden_color: Optional[Tuple[int, int, int]] = None,
        margin: int = 50,
    ) -> Path:
        """Create a document-style image with visible text and a hidden instruction.

        The hidden instruction is rendered in a colour very close to the
        background, making it invisible to casual inspection but readable by
        VLMs that process the raw pixel data.

        Args:
            text_content: Visible body text for the document.
            hidden_instruction: Instruction to hide in the document.
            output_path: Destination path.
            page_size: (width, height) in pixels.
            background_color: Background RGB colour.
            text_color: Visible text RGB colour.
            hidden_color: Colour for the hidden text. Defaults to a shade very
                close to *background_color*.
            margin: Page margin in pixels.

        Returns:
            Path to the saved document image.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if hidden_color is None:
            # Very slight offset from background -- invisible to humans
            hidden_color = tuple(min(c + 3, 255) for c in background_color)

        img = Image.new("RGB", page_size, background_color)

        # Render visible text
        body_font = self._load_font(self.font_size)
        bbox = (margin, margin, page_size[0] - margin, page_size[1] - margin)
        render_multiline(img, text_content, bbox, self.font_size, text_color)

        # Render hidden instruction at the bottom
        hidden_font = self._load_font(max(10, self.font_size - 4))
        hidden_y = page_size[1] - margin - 20
        render_text(img, hidden_instruction, (margin, hidden_y), self.font_size - 4, hidden_color, 255)

        img.save(output_path)
        return output_path

    def qr_injection(
        self,
        image_path: Union[str, Path],
        payload: str,
        output_path: Union[str, Path] = "qr_injected.png",
        block_size: int = 2,
        position: Tuple[int, int] = (10, 10),
    ) -> Path:
        """Embed an instruction encoded as a QR-like binary pattern.

        Each character of *payload* is converted to an 8-bit binary sequence
        and rendered as a grid of tiny black/white blocks on the image. VLMs
        with OCR capability may decode this pattern.

        Args:
            image_path: Source image path.
            payload: Text to encode as a binary pattern.
            output_path: Destination path.
            block_size: Pixel size of each binary block.
            position: (x, y) top-left placement of the pattern.

        Returns:
            Path to the output image.
        """
        image_path = Path(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(image_path).convert("RGB")
        pixels = np.array(img)

        bits = "".join(format(ord(ch), "08b") for ch in payload)
        cols = 64  # bits per row
        rows = (len(bits) + cols - 1) // cols

        ox, oy = position
        for i, bit in enumerate(bits):
            row = i // cols
            col = i % cols
            colour = 0 if bit == "1" else 255
            y_start = oy + row * block_size
            x_start = ox + col * block_size
            y_end = min(y_start + block_size, pixels.shape[0])
            x_end = min(x_start + block_size, pixels.shape[1])
            pixels[y_start:y_end, x_start:x_end] = colour

        Image.fromarray(pixels).save(output_path)
        return output_path

    # ------------------------------------------------------------------ #
    # Private injection implementations
    # ------------------------------------------------------------------ #

    def _inject_border(
        self,
        image_path: Union[str, Path],
        instruction: str,
        output_path: Path,
        border_width: int = 30,
        border_color: Tuple[int, int, int] = (255, 255, 255),
        text_color: Tuple[int, int, int] = (245, 245, 245),
        font_size: Optional[int] = None,
    ) -> Path:
        """Add a border with near-invisible text around the image."""
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        new_w = w + 2 * border_width
        new_h = h + 2 * border_width

        bordered = Image.new("RGB", (new_w, new_h), border_color)
        bordered.paste(img, (border_width, border_width))

        font = self._load_font(font_size or max(10, self.font_size - 4))
        draw = ImageDraw.Draw(bordered)
        draw.text((4, 4), instruction, fill=text_color, font=font)

        bordered.save(output_path)
        return output_path

    def _inject_watermark(
        self,
        image_path: Union[str, Path],
        instruction: str,
        output_path: Path,
        opacity: int = 8,
        angle: float = -30.0,
        font_size: Optional[int] = None,
    ) -> Path:
        """Overlay instruction as a very faint repeating watermark."""
        img = Image.open(image_path).convert("RGBA")
        watermark = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)
        font = self._load_font(font_size or self.font_size)

        bbox = draw.textbbox((0, 0), instruction, font=font)
        tw = bbox[2] - bbox[0] + 40
        th = bbox[3] - bbox[1] + 40

        for y in range(0, img.size[1], th):
            for x in range(0, img.size[0], tw):
                draw.text((x, y), instruction, fill=(128, 128, 128, opacity), font=font)

        watermark = watermark.rotate(angle, expand=False, resample=Image.BICUBIC)
        watermark = watermark.crop((0, 0, img.size[0], img.size[1]))

        result = Image.alpha_composite(img, watermark).convert("RGB")
        result.save(output_path)
        return output_path

    def _inject_metadata(
        self,
        image_path: Union[str, Path],
        instruction: str,
        output_path: Path,
    ) -> Path:
        """Write instruction into EXIF/PNG metadata fields.

        Note: Most VLMs do not read EXIF data, so this is mainly useful for
        research into metadata-based attack vectors.
        """
        from PIL.PngImagePlugin import PngInfo

        img = Image.open(image_path).convert("RGB")
        metadata = PngInfo()
        metadata.add_text("Description", instruction)
        metadata.add_text("Comment", instruction)
        metadata.add_text("UserComment", instruction)

        img.save(output_path, pnginfo=metadata)
        return output_path

    def _inject_overlay(
        self,
        image_path: Union[str, Path],
        instruction: str,
        output_path: Path,
        opacity: int = 15,
        font_size: Optional[int] = None,
    ) -> Path:
        """Place a semi-transparent text overlay across the image."""
        img = Image.open(image_path).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        font = self._load_font(font_size or self.font_size + 10)

        bbox = draw.textbbox((0, 0), instruction, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = (img.size[0] - tw) // 2
        y = (img.size[1] - th) // 2

        draw.text((x, y), instruction, fill=(0, 0, 0, opacity), font=font)

        result = Image.alpha_composite(img, overlay).convert("RGB")
        result.save(output_path)
        return output_path
