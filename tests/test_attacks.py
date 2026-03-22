"""Tests for adversarial attack modules."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attacks.typographic import TypographicAttack
from attacks.steganographic import SteganographicAttack
from attacks.visual_injection import VisualPromptInjection
from attacks.composite import CompositeAttack, AttackStep
from utils.image_gen import create_blank_image, apply_noise, save_comparison
from utils.text_render import render_text, render_multiline, calculate_text_bbox
from utils.evaluation import compare_responses


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def sample_image(tmp_dir: Path) -> Path:
    """Create a simple test image."""
    path = tmp_dir / "sample.png"
    img = Image.new("RGB", (256, 256), (100, 150, 200))
    img.save(path)
    return path


@pytest.fixture
def white_image(tmp_dir: Path) -> Path:
    """Create a white test image."""
    path = tmp_dir / "white.png"
    Image.new("RGB", (256, 256), (255, 255, 255)).save(path)
    return path


# ---------------------------------------------------------------------------
# TypographicAttack tests
# ---------------------------------------------------------------------------

class TestTypographicAttack:

    def test_generate_creates_file(self, sample_image: Path, tmp_dir: Path) -> None:
        attack = TypographicAttack(font_size=20)
        output = tmp_dir / "typo_out.png"
        result = attack.generate(sample_image, "Test text", output)
        assert result.exists()
        img = Image.open(result)
        assert img.size == (256, 256)

    def test_generate_grid(self, tmp_dir: Path) -> None:
        attack = TypographicAttack(font_size=16)
        output = tmp_dir / "grid.png"
        texts = ["A", "B", "C", "D"]
        result = attack.generate_grid(texts, (2, 2), output, cell_size=(128, 128))
        assert result.exists()
        img = Image.open(result)
        assert img.size == (256, 256)

    def test_camouflage_blend(self, sample_image: Path, tmp_dir: Path) -> None:
        attack = TypographicAttack(font_size=24)
        for mode in ("normal", "multiply", "screen", "overlay", "soft-light"):
            output = tmp_dir / f"camo_{mode}.png"
            result = attack.generate_with_camouflage(sample_image, "hidden", mode, output)
            assert result.exists()

    def test_position_variants(self, sample_image: Path, tmp_dir: Path) -> None:
        attack = TypographicAttack(font_size=14)
        positions = ["top", "center", "bottom", "top-left", "top-right", "bottom-left", "bottom-right"]
        for pos in positions:
            output = tmp_dir / f"pos_{pos}.png"
            result = attack.generate(sample_image, "test", output, position=pos)
            assert result.exists()


# ---------------------------------------------------------------------------
# SteganographicAttack tests
# ---------------------------------------------------------------------------

class TestSteganographicAttack:

    def test_lsb_roundtrip(self, white_image: Path, tmp_dir: Path) -> None:
        attack = SteganographicAttack()
        message = "Hello, steganography!"
        encoded = tmp_dir / "lsb.png"
        attack.lsb_encode(white_image, message, encoded)
        decoded = attack.decode(encoded, method="lsb")
        assert decoded == message

    def test_alpha_roundtrip(self, white_image: Path, tmp_dir: Path) -> None:
        attack = SteganographicAttack()
        message = "Alpha secret"
        encoded = tmp_dir / "alpha.png"
        attack.alpha_channel_encode(white_image, message, encoded)
        decoded = attack.decode(encoded, method="alpha")
        assert decoded == message

    def test_whitespace_creates_file(self, white_image: Path, tmp_dir: Path) -> None:
        attack = SteganographicAttack()
        output = tmp_dir / "ws.png"
        result = attack.whitespace_text(white_image, "ghost text", output_path=output)
        assert Path(result).exists()

    def test_lsb_message_too_long(self, tmp_dir: Path) -> None:
        # Tiny image can't hold a long message
        tiny = tmp_dir / "tiny.png"
        Image.new("RGB", (4, 4), (255, 255, 255)).save(tiny)
        attack = SteganographicAttack()
        with pytest.raises(ValueError, match="too long\\b|requires"):
            attack.lsb_encode(tiny, "A" * 1000, tmp_dir / "fail.png")

    def test_whitespace_decode_raises(self, white_image: Path) -> None:
        attack = SteganographicAttack()
        with pytest.raises(ValueError, match="Whitespace"):
            attack.decode(white_image, method="whitespace")


# ---------------------------------------------------------------------------
# VisualPromptInjection tests
# ---------------------------------------------------------------------------

class TestVisualPromptInjection:

    def test_border_injection(self, sample_image: Path, tmp_dir: Path) -> None:
        inj = VisualPromptInjection()
        output = tmp_dir / "border.png"
        result = inj.inject_instruction(sample_image, "injected", "border_text", output)
        assert Path(result).exists()
        # Border adds pixels, so output should be larger
        img = Image.open(result)
        assert img.size[0] > 256

    def test_watermark_injection(self, sample_image: Path, tmp_dir: Path) -> None:
        inj = VisualPromptInjection()
        output = tmp_dir / "watermark.png"
        result = inj.inject_instruction(sample_image, "watermark text", "watermark", output)
        assert Path(result).exists()

    def test_overlay_injection(self, sample_image: Path, tmp_dir: Path) -> None:
        inj = VisualPromptInjection()
        output = tmp_dir / "overlay.png"
        result = inj.inject_instruction(sample_image, "overlay text", "overlay", output)
        assert Path(result).exists()

    def test_metadata_injection(self, sample_image: Path, tmp_dir: Path) -> None:
        inj = VisualPromptInjection()
        output = tmp_dir / "meta.png"
        result = inj.inject_instruction(sample_image, "meta instruction", "metadata", output)
        assert Path(result).exists()

    def test_adversarial_document(self, tmp_dir: Path) -> None:
        inj = VisualPromptInjection()
        output = tmp_dir / "doc.png"
        result = inj.create_adversarial_document(
            "Visible content here.",
            "Hidden instruction here.",
            output,
        )
        assert Path(result).exists()
        img = Image.open(result)
        assert img.size == (800, 1100)

    def test_qr_injection(self, sample_image: Path, tmp_dir: Path) -> None:
        inj = VisualPromptInjection()
        output = tmp_dir / "qr.png"
        result = inj.qr_injection(sample_image, "payload", output)
        assert Path(result).exists()

    def test_unknown_method_raises(self, sample_image: Path, tmp_dir: Path) -> None:
        inj = VisualPromptInjection()
        with pytest.raises(ValueError, match="Unknown injection method"):
            inj.inject_instruction(sample_image, "x", "nonexistent", tmp_dir / "x.png")


# ---------------------------------------------------------------------------
# CompositeAttack tests
# ---------------------------------------------------------------------------

class TestCompositeAttack:

    def test_empty_pipeline_raises(self, sample_image: Path, tmp_dir: Path) -> None:
        comp = CompositeAttack()
        with pytest.raises(RuntimeError, match="no steps"):
            comp.execute(sample_image, tmp_dir / "out.png")

    def test_single_step(self, sample_image: Path, tmp_dir: Path) -> None:
        comp = CompositeAttack()
        comp.add_step("typographic", "generate", {"text": "hello"})
        result = comp.execute(sample_image, tmp_dir / "composite.png")
        assert result.exists()

    def test_multi_step(self, sample_image: Path, tmp_dir: Path) -> None:
        comp = CompositeAttack()
        comp.add_step("typographic", "generate", {"text": "step1"})
        comp.add_step("steganographic", "lsb_encode", {"message": "step2"})
        result = comp.execute(sample_image, tmp_dir / "multi.png")
        assert result.exists()

    def test_build_attack(self, sample_image: Path, tmp_dir: Path) -> None:
        comp = CompositeAttack()
        comp.build_attack([
            {"attack_type": "typographic", "method": "generate", "params": {"text": "built"}},
        ])
        assert len(comp.steps) == 1
        result = comp.execute(sample_image, tmp_dir / "built.png")
        assert result.exists()

    def test_yaml_roundtrip(self, tmp_dir: Path) -> None:
        comp = CompositeAttack()
        comp.add_step("typographic", "generate", {"text": "yaml test"})
        comp.add_step("steganographic", "lsb_encode", {"message": "secret"})

        yaml_path = tmp_dir / "pipeline.yaml"
        comp.to_yaml(yaml_path)
        assert yaml_path.exists()

        loaded = CompositeAttack.from_yaml(yaml_path)
        assert len(loaded.steps) == 2
        assert loaded.steps[0].attack_type == "typographic"
        assert loaded.steps[1].params["message"] == "secret"

    def test_unknown_attack_type_raises(self, sample_image: Path, tmp_dir: Path) -> None:
        comp = CompositeAttack()
        comp.add_step("nonexistent", "method", {})
        with pytest.raises(ValueError, match="Unknown attack type"):
            comp.execute(sample_image, tmp_dir / "fail.png")


# ---------------------------------------------------------------------------
# Utils tests
# ---------------------------------------------------------------------------

class TestUtils:

    def test_create_blank_image(self, tmp_dir: Path) -> None:
        path = tmp_dir / "blank.png"
        img = create_blank_image(100, 200, (128, 64, 32), path)
        assert img.size == (100, 200)
        assert path.exists()

    def test_apply_noise(self) -> None:
        img = Image.new("RGB", (64, 64), (128, 128, 128))
        noisy = apply_noise(img, intensity=0.1)
        assert noisy.size == img.size
        # Noise should change at least some pixels
        orig = np.array(img)
        nsy = np.array(noisy)
        assert not np.array_equal(orig, nsy)

    def test_save_comparison(self, sample_image: Path, tmp_dir: Path) -> None:
        comp_path = tmp_dir / "comp.png"
        result = save_comparison(sample_image, sample_image, comp_path)
        assert result.exists()

    def test_render_text(self) -> None:
        img = Image.new("RGB", (200, 100), (255, 255, 255))
        result = render_text(img, "hello", (10, 10), 16, (0, 0, 0))
        assert result.size == (200, 100)

    def test_calculate_text_bbox(self) -> None:
        bbox = calculate_text_bbox("Test", 20)
        assert len(bbox) == 4
        assert bbox[2] > bbox[0]  # width > 0

    def test_compare_responses(self) -> None:
        result = compare_responses(
            "A photo of a cat sitting on a mat",
            "The image shows a dog running in a park",
        )
        assert "similarity_score" in result
        assert 0 <= result["similarity_score"] <= 1
        assert result["substantially_different"] is True

    def test_compare_identical_responses(self) -> None:
        result = compare_responses("same text here", "same text here")
        assert result["similarity_score"] == 1.0
        assert result["substantially_different"] is False
