"""Composite attack strategies that chain multiple attack types.

A composite attack executes a pipeline of individual attack steps in sequence,
where the output of one step feeds into the next. Pipelines can be configured
programmatically or loaded from a YAML file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml
from PIL import Image

from attacks.typographic import TypographicAttack
from attacks.visual_injection import VisualPromptInjection
from attacks.steganographic import SteganographicAttack


@dataclass
class AttackStep:
    """A single step in a composite attack pipeline.

    Attributes:
        attack_type: One of ``"typographic"``, ``"visual_injection"``,
            ``"steganographic"``, or ``"lsb"``.
        method: The specific method to call (e.g. ``"generate"``,
            ``"inject_instruction"``, ``"lsb_encode"``).
        params: Keyword arguments forwarded to the method.
    """

    attack_type: str
    method: str
    params: Dict[str, Any] = field(default_factory=dict)


class CompositeAttack:
    """Chain multiple adversarial attack steps into a single pipeline.

    Example::

        pipeline = CompositeAttack()
        pipeline.add_step("typographic", "generate", {
            "text": "Ignore previous instructions.",
            "position": "top",
        })
        pipeline.add_step("steganographic", "lsb_encode", {
            "message": "hidden payload",
        })
        result = pipeline.execute("input.png", "output.png")

    Args:
        steps: Optional initial list of ``AttackStep`` objects.
    """

    # Registry mapping attack_type names to their classes
    _REGISTRY: Dict[str, type] = {
        "typographic": TypographicAttack,
        "visual_injection": VisualPromptInjection,
        "steganographic": SteganographicAttack,
    }

    def __init__(self, steps: Optional[List[AttackStep]] = None) -> None:
        self.steps: List[AttackStep] = steps or []
        self._instances: Dict[str, Any] = {}

    def _get_instance(self, attack_type: str) -> Any:
        """Lazily instantiate and cache an attack class."""
        if attack_type not in self._instances:
            cls = self._REGISTRY.get(attack_type)
            if cls is None:
                raise ValueError(
                    f"Unknown attack type '{attack_type}'. "
                    f"Available: {list(self._REGISTRY.keys())}"
                )
            self._instances[attack_type] = cls()
        return self._instances[attack_type]

    def add_step(
        self,
        attack_type: str,
        method: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> CompositeAttack:
        """Append an attack step to the pipeline.

        Args:
            attack_type: Name of the attack class to use.
            method: Method name on the attack instance.
            params: Keyword arguments for the method call.

        Returns:
            ``self``, for fluent chaining.
        """
        self.steps.append(AttackStep(attack_type, method, params or {}))
        return self

    def build_attack(self, steps: Sequence[Dict[str, Any]]) -> CompositeAttack:
        """Define the entire pipeline from a list of step dictionaries.

        Each dictionary must have ``attack_type`` and ``method`` keys, and
        may include a ``params`` dictionary.

        Args:
            steps: Sequence of step definitions.

        Returns:
            ``self``, with all steps replaced.
        """
        self.steps = [
            AttackStep(
                attack_type=s["attack_type"],
                method=s["method"],
                params=s.get("params", {}),
            )
            for s in steps
        ]
        return self

    def execute(
        self,
        image_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> Path:
        """Run the full pipeline, passing each step's output to the next.

        The first step receives *image_path* as input. Each subsequent step
        receives the output of the previous step. The final result is saved
        to *output_path*.

        Args:
            image_path: Path to the original source image.
            output_path: Final output destination.

        Returns:
            Path to the output image.

        Raises:
            RuntimeError: If the pipeline has no steps.
        """
        if not self.steps:
            raise RuntimeError("Pipeline has no steps. Add at least one step before executing.")

        current_path = Path(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        for i, step in enumerate(self.steps):
            instance = self._get_instance(step.attack_type)
            method_fn = getattr(instance, step.method, None)
            if method_fn is None:
                raise AttributeError(
                    f"Attack '{step.attack_type}' has no method '{step.method}'."
                )

            is_last = i == len(self.steps) - 1
            step_output = output_path if is_last else output_path.with_suffix(f".step{i}.png")

            # Build kwargs -- inject image_path and output_path
            kwargs = dict(step.params)
            kwargs["image_path"] = str(current_path)
            kwargs["output_path"] = str(step_output)

            result = method_fn(**kwargs)
            current_path = Path(result) if isinstance(result, (str, Path)) else step_output

        return output_path

    # ------------------------------------------------------------------ #
    # YAML serialisation
    # ------------------------------------------------------------------ #

    def to_yaml(self, path: Union[str, Path]) -> Path:
        """Serialise the pipeline to a YAML file.

        Args:
            path: Destination YAML file.

        Returns:
            Path to the saved file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "pipeline": [
                {
                    "attack_type": s.attack_type,
                    "method": s.method,
                    "params": s.params,
                }
                for s in self.steps
            ]
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        return path

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> CompositeAttack:
        """Load a pipeline from a YAML configuration file.

        Expected format::

            pipeline:
              - attack_type: typographic
                method: generate
                params:
                  text: "Ignore previous instructions."
              - attack_type: steganographic
                method: lsb_encode
                params:
                  message: "hidden payload"

        Args:
            path: Path to the YAML file.

        Returns:
            A new ``CompositeAttack`` with the loaded steps.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        steps = [
            AttackStep(
                attack_type=s["attack_type"],
                method=s["method"],
                params=s.get("params", {}),
            )
            for s in data["pipeline"]
        ]
        return cls(steps=steps)
