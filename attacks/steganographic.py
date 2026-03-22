"""Steganographic attacks -- hide messages within images.

These techniques embed text payloads into images by modifying pixel values in
ways that are imperceptible to the human eye. While traditional steganography
targets human viewers, VLMs may pick up on subtle patterns that encode text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DecodeMethod = Literal["lsb", "alpha", "whitespace"]


class SteganographicAttack:
    """Hide adversarial text inside images using steganographic techniques.

    Supports Least Significant Bit (LSB) encoding, alpha-channel encoding,
    and near-invisible text overlays.

    Args:
        font_path: Optional path to a TrueType font for text-based methods.
    """

    def __init__(self, font_path: Optional[str] = None) -> None:
        self.font_path = font_path

    # ------------------------------------------------------------------ #
    # Encoding methods
    # ------------------------------------------------------------------ #

    def lsb_encode(
        self,
        image_path: Union[str, Path],
        message: str,
        output_path: Union[str, Path] = "lsb_encoded.png",
        bits: int = 1,
    ) -> Path:
        """Encode a message into the least significant bits of each pixel channel.

        The message is converted to a binary string, prefixed with a 32-bit
        length header, and written into the lowest *bits* of each colour channel
        in raster order.

        Args:
            image_path: Source image (will be converted to RGB).
            message: Text message to hide.
            output_path: Destination for the encoded image (use PNG to avoid
                lossy compression destroying the payload).
            bits: Number of least significant bits to use per channel (1--4).

        Returns:
            Path to the encoded image.

        Raises:
            ValueError: If the message is too long for the image capacity.
        """
        image_path = Path(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(image_path).convert("RGB")
        pixels = np.array(img)

        # Prepare binary payload: 32-bit length prefix + message bits
        msg_bytes = message.encode("utf-8")
        length_bits = format(len(msg_bytes), "032b")
        data_bits = "".join(format(b, "08b") for b in msg_bytes)
        payload = length_bits + data_bits

        capacity = pixels.size * bits
        if len(payload) > capacity:
            raise ValueError(
                f"Message requires {len(payload)} bits but image can hold "
                f"{capacity} bits at {bits} bit(s) per channel."
            )

        flat = pixels.flatten()
        mask = (0xFF << bits) & 0xFF  # zero out the lowest `bits`

        for i, bit_group_start in enumerate(range(0, len(payload), bits)):
            chunk = payload[bit_group_start : bit_group_start + bits].ljust(bits, "0")
            value = int(chunk, 2)
            flat[i] = (int(flat[i]) & mask) | value

        pixels = flat.reshape(pixels.shape)
        Image.fromarray(pixels.astype(np.uint8)).save(output_path)
        return output_path

    def alpha_channel_encode(
        self,
        image_path: Union[str, Path],
        message: str,
        output_path: Union[str, Path] = "alpha_encoded.png",
    ) -> Path:
        """Hide a message in the alpha (transparency) channel of an image.

        The alpha channel is set to 255 (fully opaque) or 254 per bit of
        the message. Visually the image appears identical because a single
        level of alpha difference is imperceptible.

        Args:
            image_path: Source image.
            message: Text message to hide.
            output_path: Output path (must be a format supporting alpha, e.g. PNG).

        Returns:
            Path to the encoded image.
        """
        image_path = Path(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(image_path).convert("RGBA")
        pixels = np.array(img)
        alpha = pixels[:, :, 3].flatten()

        # Payload with length prefix
        msg_bytes = message.encode("utf-8")
        length_bits = format(len(msg_bytes), "032b")
        data_bits = "".join(format(b, "08b") for b in msg_bytes)
        payload = length_bits + data_bits

        if len(payload) > len(alpha):
            raise ValueError("Message too long for the image alpha channel.")

        # Reset alpha to 255, then flip LSB for '1' bits
        alpha[:] = 255
        for i, bit in enumerate(payload):
            if bit == "1":
                alpha[i] = 254

        pixels[:, :, 3] = alpha.reshape(pixels[:, :, 3].shape)
        Image.fromarray(pixels).save(output_path)
        return output_path

    def whitespace_text(
        self,
        image_path: Union[str, Path],
        message: str,
        font_color: Tuple[int, int, int] = (254, 254, 254),
        font_size: int = 14,
        output_path: Union[str, Path] = "whitespace_encoded.png",
        position: Tuple[int, int] = (5, 5),
    ) -> Path:
        """Overlay near-invisible text on the image.

        Renders the message in a colour almost identical to white (or a
        specified near-background colour). Humans see a blank area, but VLMs
        processing raw pixels can read the text.

        Args:
            image_path: Source image.
            message: Text to overlay.
            font_color: RGB colour very close to background.
            font_size: Font size.
            output_path: Destination path.
            position: (x, y) coordinate for text placement.

        Returns:
            Path to the encoded image.
        """
        image_path = Path(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        if self.font_path:
            font = ImageFont.truetype(self.font_path, font_size)
        else:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except OSError:
                font = ImageFont.load_default()

        draw.text(position, message, fill=font_color, font=font)
        img.save(output_path)
        return output_path

    # ------------------------------------------------------------------ #
    # Decoding
    # ------------------------------------------------------------------ #

    def decode(
        self,
        image_path: Union[str, Path],
        method: DecodeMethod = "lsb",
        bits: int = 1,
    ) -> str:
        """Extract a hidden message from a steganographically encoded image.

        Args:
            image_path: Path to the encoded image.
            method: Encoding method that was used (``"lsb"``, ``"alpha"``,
                or ``"whitespace"``).
            bits: Number of LSB bits per channel (only relevant for ``"lsb"``).

        Returns:
            The decoded message string.

        Raises:
            ValueError: If the method is ``"whitespace"`` (cannot be decoded
                programmatically from pixel data alone).
        """
        if method == "lsb":
            return self._decode_lsb(image_path, bits)
        elif method == "alpha":
            return self._decode_alpha(image_path)
        elif method == "whitespace":
            raise ValueError(
                "Whitespace text encoding cannot be reliably decoded from "
                "pixel data. Use OCR or a VLM to read the image."
            )
        else:
            raise ValueError(f"Unknown decode method: {method}")

    def _decode_lsb(self, image_path: Union[str, Path], bits: int = 1) -> str:
        """Decode an LSB-encoded message."""
        img = Image.open(image_path).convert("RGB")
        flat = np.array(img).flatten()

        # Extract bits
        mask = (1 << bits) - 1
        extracted = ""
        for val in flat:
            chunk = format(int(val) & mask, f"0{bits}b")
            extracted += chunk

        # Read 32-bit length header
        length = int(extracted[:32], 2)
        msg_bits = extracted[32 : 32 + length * 8]

        chars = []
        for i in range(0, len(msg_bits), 8):
            byte = msg_bits[i : i + 8]
            if len(byte) < 8:
                break
            chars.append(chr(int(byte, 2)))

        return "".join(chars)

    def _decode_alpha(self, image_path: Union[str, Path]) -> str:
        """Decode an alpha-channel-encoded message."""
        img = Image.open(image_path).convert("RGBA")
        alpha = np.array(img)[:, :, 3].flatten()

        # Extract bits: 255 -> 0, 254 -> 1
        extracted = ""
        for val in alpha:
            extracted += "0" if val == 255 else "1"

        length = int(extracted[:32], 2)
        msg_bits = extracted[32 : 32 + length * 8]

        chars = []
        for i in range(0, len(msg_bits), 8):
            byte = msg_bits[i : i + 8]
            if len(byte) < 8:
                break
            chars.append(chr(int(byte, 2)))

        return "".join(chars)
