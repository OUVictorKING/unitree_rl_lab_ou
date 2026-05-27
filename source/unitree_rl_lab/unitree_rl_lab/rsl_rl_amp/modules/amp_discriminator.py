# Copyright (c) 2025, Unitree Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""AMP discriminator network.

The discriminator distinguishes transitions coming from expert motion data
(``real``) from transitions collected by the policy during rollouts (``fake``).

It consumes a *transition feature* by default: the concatenation of the AMP
observation at time ``t`` and at time ``t + 1``. This mirrors the setup used
in Peng et al., "AMP: Adversarial Motion Priors for Stylized Physics-Based
Character Control" (2021).

The module is intentionally small and generic:

- It does not assume any specific robot.
- It operates on a pre-computed AMP observation tensor of shape ``(B, D)``.
  The semantics of that tensor (joint_pos_rel + joint_vel_rel for Basic AMP,
  richer body-level features for Soft AMP, etc.) are defined by the
  environment / dataset converter.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Sequence

from rsl_rl.utils import resolve_nn_activation


class AmpDiscriminator(nn.Module):
    """Configurable MLP discriminator for AMP-style imitation.

    Args:
        amp_obs_dim: Dimension of a *single* AMP observation vector. The
            discriminator input is ``2 * amp_obs_dim`` by default because it
            consumes transitions ``(amp_obs_t, amp_obs_{t+1})``.
        hidden_dims: Hidden layer sizes.
        activation: Activation name (resolved via ``resolve_nn_activation``).
        dropout: Dropout probability applied between hidden layers.
        reward_style: Formulation of the style reward. One of:

            * ``"log"`` (default): ``r = -log(1 - sigmoid(logit) + eps)``.
              Pulls the generator toward the "real" class, similar to GAIL.
            * ``"amp"``: the AMP paper form
              ``r = max(0, 1 - 0.25 * (logit - 1) ** 2)``.

        reward_eps: Numerical epsilon used inside the log reward.
        reward_clip: Optional positive value; if provided, the style reward
            is clamped to ``[-reward_clip, reward_clip]``.
        use_transition_input: If ``True`` (default), the discriminator expects
            pre-concatenated transition features of shape ``(B, 2*amp_obs_dim)``.
            If ``False`` it expects a single state of shape ``(B, amp_obs_dim)``.
    """

    _REWARD_STYLES = ("log", "amp")

    def __init__(
        self,
        amp_obs_dim: int,
        hidden_dims: Sequence[int] = (512, 256),
        activation: str = "elu",
        dropout: float = 0.0,
        reward_style: str = "log",
        reward_eps: float = 1.0e-4,
        reward_clip: float | None = None,
        use_transition_input: bool = True,
    ) -> None:
        super().__init__()

        if reward_style not in self._REWARD_STYLES:
            raise ValueError(
                f"Unknown reward_style '{reward_style}'. Expected one of {self._REWARD_STYLES}."
            )

        self.amp_obs_dim = int(amp_obs_dim)
        self.use_transition_input = bool(use_transition_input)
        self.input_dim = 2 * self.amp_obs_dim if self.use_transition_input else self.amp_obs_dim

        self.reward_style = reward_style
        self.reward_eps = float(reward_eps)
        self.reward_clip = None if reward_clip is None else float(reward_clip)

        act_cls = resolve_nn_activation(activation)

        layers: list[nn.Module] = []
        prev_dim = self.input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, int(h)))
            layers.append(act_cls)
            if dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev_dim = int(h)
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    def _assemble_input(
        self, amp_obs: torch.Tensor, next_amp_obs: torch.Tensor | None
    ) -> torch.Tensor:
        """Assemble the discriminator input based on the configured mode."""
        if self.use_transition_input:
            if next_amp_obs is None:
                raise ValueError(
                    "Discriminator was configured with use_transition_input=True but "
                    "next_amp_obs is None."
                )
            if amp_obs.shape != next_amp_obs.shape:
                raise ValueError(
                    f"Shape mismatch between amp_obs {tuple(amp_obs.shape)} and "
                    f"next_amp_obs {tuple(next_amp_obs.shape)}."
                )
            return torch.cat([amp_obs, next_amp_obs], dim=-1)
        return amp_obs

    def forward(
        self, amp_obs: torch.Tensor, next_amp_obs: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the scalar logit of the discriminator.

        Output shape: ``(B, 1)``. The logit is positive for inputs the
        discriminator believes are real (i.e., from the motion dataset).
        """
        x = self._assemble_input(amp_obs, next_amp_obs)
        h = self.trunk(x)
        return self.head(h)

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------
    def bce_loss(
        self,
        real_logits: torch.Tensor,
        fake_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross-entropy loss with targets ``1`` for real, ``0`` for fake.

        Uses the numerically stable ``BCEWithLogitsLoss``.
        """
        bce = nn.functional.binary_cross_entropy_with_logits
        real_targets = torch.ones_like(real_logits)
        fake_targets = torch.zeros_like(fake_logits)
        loss_real = bce(real_logits, real_targets)
        loss_fake = bce(fake_logits, fake_targets)
        return 0.5 * (loss_real + loss_fake)

    def gradient_penalty(
        self, amp_obs: torch.Tensor, next_amp_obs: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute the gradient penalty on the discriminator input.

        Penalizes the squared L2 norm of ``d logit / d input`` evaluated on
        real samples. This is the "R1"-style regularization commonly used in
        AMP implementations to stabilize the discriminator.
        """
        x = self._assemble_input(amp_obs, next_amp_obs)
        x = x.detach().clone().requires_grad_(True)
        h = self.trunk(x)
        logit = self.head(h)
        grad = torch.autograd.grad(
            outputs=logit.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return grad.pow(2).sum(dim=-1).mean()

    # ------------------------------------------------------------------
    # Style reward
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_reward(
        self, amp_obs: torch.Tensor, next_amp_obs: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute the AMP style reward for a batch of transitions.

        The output has shape ``(B, 1)`` with positive values indicating the
        discriminator thinks the transition is close to expert motion.

        Formulations:

        - ``reward_style="log"``:
            ``r = -log(1 - sigmoid(logit) + eps)``. This is a log-probability
            style reward with ``eps`` preventing ``log(0)``.

        - ``reward_style="amp"``:
            ``r = max(0, 1 - 0.25 * (logit - 1) ** 2)``. Form from the AMP
            paper. Not a log; kept as an option for parity.
        """
        logit = self.forward(amp_obs, next_amp_obs)
        if self.reward_style == "log":
            prob_fake = 1.0 - torch.sigmoid(logit)
            reward = -torch.log(prob_fake.clamp(min=self.reward_eps))
        else:  # "amp"
            reward = (1.0 - 0.25 * (logit - 1.0) ** 2).clamp(min=0.0)
        if self.reward_clip is not None:
            reward = reward.clamp(min=-self.reward_clip, max=self.reward_clip)
        return reward
