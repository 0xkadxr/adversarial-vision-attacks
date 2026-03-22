#!/usr/bin/env python3
"""Demo: Steganographic attacks -- hide messages in images.

Demonstrates LSB encoding, alpha-channel encoding, and near-invisible
text overlay techniques.

Usage:
    python examples/steganographic_demo.py --method lsb --message "hidden payload"
    python examples/steganographic_demo.py --method alpha --message "secret" -o stego.png
    python examples/steganographic_demo.py --decode stego.png --method lsb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attacks.steganographic import SteganographicAttack
from utils.image_gen import create_blank_image, save_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Steganographic message encoding/decoding in images.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run.")

    # Encode sub-command
    enc = subparsers.add_parser("encode", help="Encode a message into an image.")
    enc.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Source image. A blank image is created if omitted.",
    )
    enc.add_argument(
        "--message", "-m",
        type=str,
        default="This is a hidden adversarial instruction.",
        help="Message to hide in the image.",
    )
    enc.add_argument(
        "--method",
        type=str,
        default="lsb",
        choices=["lsb", "alpha", "whitespace"],
        help="Encoding method.",
    )
    enc.add_argument(
        "--output", "-o",
        type=str,
        default="output/steganographic.png",
        help="Output image path.",
    )
    enc.add_argument(
        "--compare",
        action="store_true",
        help="Save a side-by-side comparison with the original.",
    )

    # Decode sub-command
    dec = subparsers.add_parser("decode", help="Decode a hidden message from an image.")
    dec.add_argument(
        "image",
        type=str,
        help="Path to the encoded image.",
    )
    dec.add_argument(
        "--method",
        type=str,
        default="lsb",
        choices=["lsb", "alpha"],
        help="Decoding method.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attack = SteganographicAttack()

    if args.command == "decode":
        message = attack.decode(args.image, method=args.method)
        print(f"Decoded message: {message}")
        return

    if args.command is None or args.command == "encode":
        # Default to encode if no subcommand given
        if not hasattr(args, "method"):
            args.method = "lsb"
            args.message = "This is a hidden adversarial instruction."
            args.output = "output/steganographic.png"
            args.image = None
            args.compare = False

        # Prepare source image
        if args.image:
            source = args.image
        else:
            source = "output/_tmp_blank.png"
            create_blank_image(512, 512, (220, 220, 240), source)

        if args.method == "lsb":
            result = attack.lsb_encode(source, args.message, args.output)
        elif args.method == "alpha":
            result = attack.alpha_channel_encode(source, args.message, args.output)
        elif args.method == "whitespace":
            result = attack.whitespace_text(source, args.message, output_path=args.output)
        else:
            print(f"Unknown method: {args.method}")
            sys.exit(1)

        print(f"Encoded image saved to {result}")

        if hasattr(args, "compare") and args.compare:
            comp = Path(args.output).with_name("stego_comparison.png")
            save_comparison(source, result, comp)
            print(f"Comparison saved to {comp}")

        # Verify by decoding (skip whitespace -- requires OCR)
        if args.method in ("lsb", "alpha"):
            decoded = attack.decode(result, method=args.method)
            print(f"Verification decode: {decoded}")


if __name__ == "__main__":
    main()
