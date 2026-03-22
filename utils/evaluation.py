"""Evaluation utilities for measuring adversarial attack effectiveness.

Provides helpers to query a VLM API with original and adversarial images,
then compare the responses to assess whether the attack succeeded.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import httpx
from PIL import Image


@dataclass
class EvaluationResult:
    """Result of evaluating a single adversarial image.

    Attributes:
        original_response: Model response to the original image.
        adversarial_response: Model response to the adversarial image.
        attack_success: Whether the adversarial response differs significantly.
        similarity_score: Cosine or textual similarity between the two responses
            (lower means more effective attack). Range [0, 1].
        details: Additional metadata.
    """

    original_response: str
    adversarial_response: str
    attack_success: bool
    similarity_score: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelAPI:
    """Configuration for a VLM API endpoint.

    Supports OpenAI-compatible vision APIs. Adjust ``headers`` and
    ``payload_builder`` for other providers.

    Attributes:
        base_url: API base URL (e.g. ``"https://api.openai.com/v1"``).
        api_key: Bearer token or API key.
        model: Model identifier (e.g. ``"gpt-4o"``).
        prompt: Default prompt sent alongside the image.
        max_tokens: Maximum response tokens.
    """

    base_url: str
    api_key: str
    model: str = "gpt-4o"
    prompt: str = "Describe this image in detail."
    max_tokens: int = 300


def _image_to_base64(image_path: Union[str, Path]) -> str:
    """Read an image file and return its Base64-encoded content."""
    data = Path(image_path).read_bytes()
    return base64.b64encode(data).decode("utf-8")


def _query_model(image_path: Union[str, Path], api: ModelAPI, prompt: Optional[str] = None) -> str:
    """Send an image to the configured VLM API and return the text response.

    Args:
        image_path: Path to the image to send.
        api: ``ModelAPI`` configuration.
        prompt: Override the default prompt.

    Returns:
        The model's text response.
    """
    b64 = _image_to_base64(image_path)
    ext = Path(image_path).suffix.lstrip(".").lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")

    payload = {
        "model": api.model,
        "max_tokens": api.max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or api.prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            }
        ],
    }

    headers = {
        "Authorization": f"Bearer {api.api_key}",
        "Content-Type": "application/json",
    }

    response = httpx.post(
        f"{api.base_url}/chat/completions",
        json=payload,
        headers=headers,
        timeout=60.0,
    )
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]


def _text_similarity(a: str, b: str) -> float:
    """Compute a simple word-overlap (Jaccard) similarity between two strings.

    This is a lightweight heuristic; for production use, consider embedding-
    based similarity.

    Args:
        a: First text.
        b: Second text.

    Returns:
        Jaccard similarity in [0, 1].
    """
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a and not words_b:
        return 1.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union) if union else 1.0


def evaluate_attack(
    original_image: Union[str, Path],
    adversarial_image: Union[str, Path],
    model_api: ModelAPI,
    prompt: Optional[str] = None,
    success_threshold: float = 0.5,
) -> EvaluationResult:
    """Evaluate whether an adversarial image fools the target model.

    Sends both the original and adversarial images to the VLM API, compares
    the responses, and decides whether the attack was successful based on
    how different the two responses are.

    Args:
        original_image: Path to the clean image.
        adversarial_image: Path to the adversarial image.
        model_api: API configuration for the target VLM.
        prompt: Override the default evaluation prompt.
        success_threshold: If similarity drops below this value, the attack
            is considered successful.

    Returns:
        An ``EvaluationResult`` with the comparison data.
    """
    orig_resp = _query_model(original_image, model_api, prompt)
    adv_resp = _query_model(adversarial_image, model_api, prompt)

    similarity = _text_similarity(orig_resp, adv_resp)
    success = similarity < success_threshold

    return EvaluationResult(
        original_response=orig_resp,
        adversarial_response=adv_resp,
        attack_success=success,
        similarity_score=similarity,
        details={
            "model": model_api.model,
            "prompt": prompt or model_api.prompt,
            "threshold": success_threshold,
        },
    )


def compare_responses(
    response_original: str,
    response_adversarial: str,
) -> Dict[str, Any]:
    """Analyse the difference between model responses to original and adversarial images.

    Args:
        response_original: Model response to the original image.
        response_adversarial: Model response to the adversarial image.

    Returns:
        Dictionary with similarity score, word-level diff statistics, and
        a boolean indicating whether the responses are substantially different.
    """
    similarity = _text_similarity(response_original, response_adversarial)

    words_orig = set(response_original.lower().split())
    words_adv = set(response_adversarial.lower().split())

    return {
        "similarity_score": similarity,
        "words_only_in_original": sorted(words_orig - words_adv),
        "words_only_in_adversarial": sorted(words_adv - words_orig),
        "shared_words": len(words_orig & words_adv),
        "original_word_count": len(response_original.split()),
        "adversarial_word_count": len(response_adversarial.split()),
        "substantially_different": similarity < 0.5,
    }


def batch_evaluate(
    image_pairs: Sequence[Tuple[Union[str, Path], Union[str, Path]]],
    model_api: ModelAPI,
    prompt: Optional[str] = None,
    success_threshold: float = 0.5,
) -> List[EvaluationResult]:
    """Evaluate multiple original/adversarial image pairs.

    Args:
        image_pairs: List of ``(original_path, adversarial_path)`` tuples.
        model_api: API configuration for the target VLM.
        prompt: Override the default evaluation prompt.
        success_threshold: Similarity threshold for success determination.

    Returns:
        List of ``EvaluationResult`` objects, one per pair.
    """
    results: List[EvaluationResult] = []
    for orig, adv in image_pairs:
        result = evaluate_attack(orig, adv, model_api, prompt, success_threshold)
        results.append(result)
    return results
