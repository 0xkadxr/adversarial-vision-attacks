"""Gradient-based adversarial perturbation attacks.

Implements classic adversarial ML techniques (FGSM, PGD, adversarial patches)
adapted for attacking Vision-Language Models. These perturbations are typically
imperceptible to humans but can significantly alter model outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

NormType = Literal["linf", "l2"]


@dataclass
class PerturbationResult:
    """Container for an adversarial perturbation result.

    Attributes:
        adversarial_image: The perturbed image as a torch tensor (C, H, W).
        perturbation: The raw perturbation tensor (C, H, W).
        original_image: The original clean image tensor.
        norm: The measured norm of the perturbation.
        iterations: Number of optimisation iterations used.
    """

    adversarial_image: torch.Tensor
    perturbation: torch.Tensor
    original_image: torch.Tensor
    norm: float
    iterations: int


class AdversarialPerturbation:
    """Generate gradient-based adversarial perturbations targeting VLMs.

    Supports Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD),
    and adversarial patch attacks. These methods require a differentiable model
    or a surrogate loss function to compute gradients.

    Args:
        epsilon: Maximum perturbation magnitude (L-inf bound, in [0, 1] range).
        steps: Default number of PGD iterations.
        alpha: Step size per iteration for PGD.
        norm: Norm constraint type, either ``"linf"`` or ``"l2"``.
        device: Torch device to use. Defaults to CUDA if available.
    """

    def __init__(
        self,
        epsilon: float = 8 / 255,
        steps: int = 40,
        alpha: float = 2 / 255,
        norm: NormType = "linf",
        device: Optional[str] = None,
    ) -> None:
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha
        self.norm = norm
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- public helpers ----------

    @staticmethod
    def image_to_tensor(image_path: Union[str, Path]) -> torch.Tensor:
        """Load an image file and return a normalised float32 tensor.

        Args:
            image_path: Path to an image file.

        Returns:
            Tensor of shape (1, C, H, W) in [0, 1].
        """
        img = Image.open(image_path).convert("RGB")
        tensor = transforms.ToTensor()(img)  # (C, H, W) in [0, 1]
        return tensor.unsqueeze(0)

    @staticmethod
    def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
        """Convert a (1, C, H, W) or (C, H, W) tensor back to a PIL image.

        Args:
            tensor: Float tensor in [0, 1].

        Returns:
            PIL RGB Image.
        """
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        tensor = tensor.clamp(0, 1)
        return transforms.ToPILImage()(tensor.cpu())

    # ---------- core attacks ----------

    def fgsm(
        self,
        image: torch.Tensor,
        target_text: str,
        model: Callable[..., torch.Tensor],
        targeted: bool = False,
    ) -> PerturbationResult:
        """Fast Gradient Sign Method (Goodfellow et al., 2014).

        Computes a single-step perturbation in the direction of the gradient of
        the loss with respect to the input image.

        Args:
            image: Input image tensor of shape (1, C, H, W) in [0, 1].
            target_text: Text target for the loss function. Passed to *model*.
            model: A callable ``model(image, text) -> loss`` that returns a
                scalar differentiable loss tensor.
            targeted: If ``True``, minimise the loss toward *target_text*
                (targeted attack). Otherwise, maximise the loss (untargeted).

        Returns:
            A ``PerturbationResult`` with the adversarial image and metadata.
        """
        image = image.clone().to(self.device).requires_grad_(True)

        loss = model(image, target_text)
        loss.backward()

        grad = image.grad.data
        if targeted:
            perturbation = -self.alpha * grad.sign()
        else:
            perturbation = self.alpha * grad.sign()

        perturbation = self._project(perturbation)
        adv_image = (image.data + perturbation).clamp(0, 1)

        return PerturbationResult(
            adversarial_image=adv_image.detach(),
            perturbation=perturbation.detach(),
            original_image=image.data.detach(),
            norm=self._measure_norm(perturbation),
            iterations=1,
        )

    def pgd(
        self,
        image: torch.Tensor,
        target_text: str,
        model: Callable[..., torch.Tensor],
        iterations: Optional[int] = None,
        targeted: bool = False,
        random_start: bool = True,
    ) -> PerturbationResult:
        """Projected Gradient Descent (Madry et al., 2018).

        An iterative variant of FGSM that applies small perturbation steps and
        projects back onto the epsilon-ball after each step.

        Args:
            image: Input image tensor of shape (1, C, H, W) in [0, 1].
            target_text: Text target for the loss function.
            model: A callable ``model(image, text) -> loss``.
            iterations: Number of PGD steps. Defaults to ``self.steps``.
            targeted: If ``True``, run a targeted attack.
            random_start: Initialise with uniform random noise inside the
                epsilon-ball.

        Returns:
            A ``PerturbationResult`` with the adversarial image and metadata.
        """
        iters = iterations or self.steps
        image = image.clone().to(self.device)
        original = image.data.clone()

        if random_start:
            delta = torch.empty_like(image).uniform_(-self.epsilon, self.epsilon)
            delta = self._project(delta)
            image = (image + delta).clamp(0, 1)

        for _ in range(iters):
            image = image.clone().requires_grad_(True)
            loss = model(image, target_text)
            loss.backward()

            grad = image.grad.data
            if targeted:
                step = -self.alpha * grad.sign()
            else:
                step = self.alpha * grad.sign()

            image = image.data + step
            # Project back onto epsilon ball around original
            perturbation = image - original
            perturbation = self._project(perturbation)
            image = (original + perturbation).clamp(0, 1)

        perturbation = image - original
        return PerturbationResult(
            adversarial_image=image.detach(),
            perturbation=perturbation.detach(),
            original_image=original.detach(),
            norm=self._measure_norm(perturbation),
            iterations=iters,
        )

    def patch_attack(
        self,
        image: torch.Tensor,
        patch_size: Tuple[int, int],
        target_text: str,
        model: Optional[Callable[..., torch.Tensor]] = None,
        location: Optional[Tuple[int, int]] = None,
        iterations: int = 500,
    ) -> PerturbationResult:
        """Adversarial patch attack (Brown et al., 2017).

        Optimises a small patch that, when placed on the image, causes the
        model to produce the target output. Unlike full-image perturbations,
        patches are bounded spatially but unconstrained in magnitude.

        Args:
            image: Input image tensor of shape (1, C, H, W) in [0, 1].
            patch_size: (height, width) of the adversarial patch.
            target_text: Desired model output text.
            model: Optional differentiable model. If ``None``, generates a
                random patch (useful for testing the pipeline).
            location: (y, x) top-left corner for patch placement. Defaults to
                center of the image.
            iterations: Optimisation iterations.

        Returns:
            A ``PerturbationResult`` with the patched image.
        """
        image = image.clone().to(self.device)
        _, c, h, w = image.shape
        ph, pw = patch_size

        # Default location: center
        if location is None:
            ly = (h - ph) // 2
            lx = (w - pw) // 2
        else:
            ly, lx = location

        # Initialise patch
        patch = torch.rand(1, c, ph, pw, device=self.device, requires_grad=True)

        if model is not None:
            optimiser = torch.optim.Adam([patch], lr=1e-2)
            for _ in range(iterations):
                optimiser.zero_grad()
                patched = image.clone()
                patched[:, :, ly: ly + ph, lx: lx + pw] = patch.clamp(0, 1)
                loss = model(patched, target_text)
                loss.backward()
                optimiser.step()
        else:
            # Without a model, create a random high-contrast patch
            patch = torch.rand(1, c, ph, pw, device=self.device)

        adv_image = image.clone()
        adv_image[:, :, ly: ly + ph, lx: lx + pw] = patch.clamp(0, 1).detach()
        perturbation = adv_image - image

        return PerturbationResult(
            adversarial_image=adv_image.detach(),
            perturbation=perturbation.detach(),
            original_image=image.detach(),
            norm=self._measure_norm(perturbation),
            iterations=iterations if model else 0,
        )

    # ---------- projection / norms ----------

    def _project(self, perturbation: torch.Tensor) -> torch.Tensor:
        """Project perturbation onto the epsilon-ball under the configured norm."""
        if self.norm == "linf":
            return perturbation.clamp(-self.epsilon, self.epsilon)
        elif self.norm == "l2":
            norms = perturbation.view(perturbation.shape[0], -1).norm(dim=1, keepdim=True)
            norms = norms.view(-1, 1, 1, 1)
            factor = torch.min(torch.ones_like(norms), self.epsilon / (norms + 1e-12))
            return perturbation * factor
        else:
            raise ValueError(f"Unsupported norm: {self.norm}")

    def _measure_norm(self, perturbation: torch.Tensor) -> float:
        """Compute the norm of a perturbation tensor."""
        flat = perturbation.view(-1)
        if self.norm == "linf":
            return flat.abs().max().item()
        return flat.norm(p=2).item()

    # ---------- convenience I/O ----------

    def save_perturbation(
        self,
        result: PerturbationResult,
        output_path: Union[str, Path],
        amplify: float = 10.0,
    ) -> Path:
        """Save the perturbation as a visible image (amplified for inspection).

        Args:
            result: A ``PerturbationResult`` from one of the attack methods.
            output_path: Destination file path.
            amplify: Multiplier to make the perturbation visible.

        Returns:
            Path to the saved file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        vis = (result.perturbation.squeeze(0) * amplify + 0.5).clamp(0, 1)
        transforms.ToPILImage()(vis.cpu()).save(output_path)
        return output_path
