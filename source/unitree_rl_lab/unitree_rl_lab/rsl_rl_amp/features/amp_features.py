# Copyright (c) 2025, Unitree Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""Unified AMP feature construction — single source of truth.

The ``build_amp_frame_from_state`` function produces a single-frame style
feature ``phi_t`` in a fixed layout. ``build_amp_window`` stacks ``stack_k``
consecutive frames into the AMP observation consumed by the discriminator.

Both the environment (live simulation state) and the motion dataset (npz
replay state) call into these two functions — there is exactly one
implementation of the AMP feature layout in the codebase.

Frame layout (Penguin V1, J=23 joints, 2 feet, 2 hands) — fixed order.
Each row is gated by the corresponding ``include_*`` flag on
:class:`AmpObsSpec`; a disabled block contributes 0 to ``frame_dim``.
The V1 default turns every *implemented* block on for ``frame_dim=80``;
downstream specs (e.g. the V2 / trimmed Penguin variant) can disable
individual rows via the ``include_*`` flags.

Feet / hand flags are split so you can see at a glance whether a spec
includes **position only**, **orientation only**, **linear velocity
only**, or **angular velocity only** (no ambiguity between 3-dim
position vs. 4-dim orientation vs. 6-dim pose). The orientation /
angular-velocity flags are placeholders: :class:`AmpObsSpec` rejects
them with :exc:`NotImplementedError` until :class:`AmpObsState`,
:func:`build_amp_frame_from_state`, and :class:`MotionDataset` are
extended to provide foot/hand quaternions and angular velocities.

====  ===========================================================  ====
  i     field                                                       dim
====  ===========================================================  ====
  1    ``root_height``  (pelvis world Z)                             1
  2    ``projected_gravity_b``  (body frame)                         3
  3    ``root_lin_vel_h``  (heading frame, yaw-only world→robot)     3
  4    ``root_ang_vel_b``  (body frame)                              3
  5    ``joint_pos_rel``  (q − q_default)                           23
  6    ``joint_vel``                                                23
  7    ``foot_position_rel_pelvis_b``         (L+R, pelvis body)     6
  8    ``foot_orientation_rel_pelvis_b``      (L+R, quat)   [N/A]    8  *
  9    ``foot_linear_velocity_rel_pelvis_b``  (L+R)                  6
 10    ``foot_angular_velocity_rel_pelvis_b`` (L+R)         [N/A]    6  *
 11    ``hand_position_rel_pelvis_b``         (L+R, pelvis body)     6
 12    ``hand_orientation_rel_pelvis_b``      (L+R, quat)   [N/A]    8  *
 13    ``hand_linear_velocity_rel_pelvis_b``  (L+R)                  6
 14    ``hand_angular_velocity_rel_pelvis_b`` (L+R)         [N/A]    6  *
 --    ``total (currently implemented flags on)``                   80
====  ===========================================================  ====

Rows marked ``[N/A]`` are unimplemented placeholders — enabling them
raises :exc:`NotImplementedError` at spec construction time. Their
``dim`` columns document the intended size for future implementers.

The window is ``amp_obs_t = concat(phi_{t-k+1}, ..., phi_t)`` with oldest
frame first and newest frame last. ``amp_obs_dim = frame_dim * stack_k``.
"""

from __future__ import annotations

import dataclasses
import torch
from typing import Mapping, Sequence

from isaaclab.utils.math import yaw_quat

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:  # pragma: no cover -- IsaacLab < 2.1.0
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse


# =========================================================================
# Spec + state container
# =========================================================================


@dataclasses.dataclass
class AmpObsSpec:
    """Static metadata describing the AMP feature layout.

    The spec is plain metadata; it does not know about any tensor. It is
    shared by the environment and the motion dataset so both sides produce
    identical feature vectors.

    Parameters
    ----------
    joint_names:
        Ordered joint names. The live environment must expose these joints
        in the same order; the npz dataset must have ``joint_pos``/
        ``joint_vel`` in the same order. Length gives ``num_joints``.
    pelvis_body_name:
        Name of the pelvis/root link. Used as anchor for the body-frame.
    foot_body_names:
        ``(left, right)`` foot link names. Must have length 2.
    hand_body_names:
        ``(left, right)`` hand/wrist end-effector link names. Must have
        length 2.
    stack_k:
        Number of frames stacked into a single AMP observation. Phase-1
        default is 4.
    include_root_height:
        Whether to include the pelvis world Z in the frame. Always True
        in V1 but exposed as a flag so future specs can omit it.
    include_projected_gravity:
        Include the 3-dim gravity projected into the body frame.
    include_root_lin_vel_heading:
        Include the root linear velocity in the heading frame.
    include_root_ang_vel_body:
        Include the root angular velocity in the body frame.
    include_joint_pos_rel:
        Include joint positions relative to the default pose.
    include_joint_vel:
        Include joint velocities (raw).
    include_feet_position:
        Include left+right foot **positions** (3 dim per foot, 6 total)
        in the pelvis body frame.
    include_feet_orientation:
        **[UNIMPLEMENTED]** Include left+right foot **orientations** (4
        dim quaternion per foot, 8 total) in the pelvis body frame.
        Setting this to ``True`` raises :exc:`NotImplementedError`
        because :class:`AmpObsState` does not yet carry foot
        quaternions and neither :func:`build_amp_frame_from_state` nor
        :class:`MotionDataset` emit them.
    include_feet_linear_velocity:
        Include left+right foot **linear velocities** (3 dim per foot,
        6 total) in the pelvis body frame.
    include_feet_angular_velocity:
        **[UNIMPLEMENTED]** Include left+right foot **angular
        velocities** (3 dim per foot, 6 total) in the pelvis body
        frame. Setting this to ``True`` raises
        :exc:`NotImplementedError` for the same reason as
        ``include_feet_orientation``.
    include_hand_position:
        Include left+right hand/wrist **positions** (3 dim per hand, 6
        total) in the pelvis body frame.
    include_hand_orientation:
        **[UNIMPLEMENTED]** Left+right hand **orientations** — see
        ``include_feet_orientation``.
    include_hand_linear_velocity:
        Include left+right hand/wrist **linear velocities** (3 dim per
        hand, 6 total) in the pelvis body frame.
    include_hand_angular_velocity:
        **[UNIMPLEMENTED]** Left+right hand **angular velocities** —
        see ``include_feet_angular_velocity``.
    """

    joint_names: Sequence[str]
    pelvis_body_name: str = "pelvis"
    foot_body_names: Sequence[str] = ("left_ankle_roll_link", "right_ankle_roll_link")
    hand_body_names: Sequence[str] = (
        "left_wrist_roll_rubber_hand",
        "right_wrist_roll_rubber_hand",
    )
    stack_k: int = 4

    # -- root / joints (existing flags, unchanged semantics) --
    include_root_height: bool = True
    include_projected_gravity: bool = True
    include_root_lin_vel_heading: bool = True
    include_root_ang_vel_body: bool = True
    include_joint_pos_rel: bool = True
    include_joint_vel: bool = True

    # -- feet: position / orientation / linear velocity / angular velocity --
    #    ``position`` and ``linear_velocity`` are currently implemented;
    #    ``orientation`` and ``angular_velocity`` are placeholders (default
    #    False; __post_init__ rejects True).
    include_feet_position: bool = True
    include_feet_orientation: bool = False
    include_feet_linear_velocity: bool = True
    include_feet_angular_velocity: bool = False

    # -- hands: same split as feet --
    include_hand_position: bool = True
    include_hand_orientation: bool = False
    include_hand_linear_velocity: bool = True
    include_hand_angular_velocity: bool = False

    def __post_init__(self) -> None:
        """Reject enabled-but-unimplemented feature flags with a clear error.

        Foot/hand orientation and angular velocity require extending
        :class:`AmpObsState` (foot_quat_w, foot_ang_vel_w, hand_quat_w,
        hand_ang_vel_w), :func:`build_amp_frame_from_state` (emit the
        body-frame quats and body-frame angular velocities), and
        :class:`MotionDataset` (load/replay these columns from the npz
        clips). Until that extension lands, flipping these flags True
        would silently produce the wrong frame dim.
        """
        unimplemented = [
            name
            for name in (
                "include_feet_orientation",
                "include_feet_angular_velocity",
                "include_hand_orientation",
                "include_hand_angular_velocity",
            )
            if getattr(self, name)
        ]
        if unimplemented:
            raise NotImplementedError(
                f"AmpObsSpec flag(s) {unimplemented} are not yet wired through "
                "AmpObsState / build_amp_frame_from_state / MotionDataset. Leave "
                "them at their False default, or extend those three sites to "
                "emit foot/hand quaternions and angular velocities."
            )

    # ------------------------------------------------------------------
    # Derived
    # ------------------------------------------------------------------
    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def frame_dim(self) -> int:
        """Dimension of a single ``phi_t`` vector."""
        dim = 0
        if self.include_root_height:
            dim += 1
        if self.include_projected_gravity:
            dim += 3
        if self.include_root_lin_vel_heading:
            dim += 3
        if self.include_root_ang_vel_body:
            dim += 3
        if self.include_joint_pos_rel:
            dim += self.num_joints
        if self.include_joint_vel:
            dim += self.num_joints
        # -- feet (4 split flags) --
        if self.include_feet_position:
            dim += 2 * 3  # position xyz (L, R)
        if self.include_feet_orientation:  # [UNIMPLEMENTED — __post_init__ rejects]
            dim += 2 * 4  # quaternion wxyz (L, R)
        if self.include_feet_linear_velocity:
            dim += 2 * 3  # linear velocity (L, R)
        if self.include_feet_angular_velocity:  # [UNIMPLEMENTED]
            dim += 2 * 3  # angular velocity (L, R)
        # -- hands (same 4-way split) --
        if self.include_hand_position:
            dim += 2 * 3
        if self.include_hand_orientation:  # [UNIMPLEMENTED]
            dim += 2 * 4
        if self.include_hand_linear_velocity:
            dim += 2 * 3
        if self.include_hand_angular_velocity:  # [UNIMPLEMENTED]
            dim += 2 * 3
        return dim

    @property
    def amp_obs_dim(self) -> int:
        """Dimension of a full stacked AMP observation."""
        return self.frame_dim * int(self.stack_k)

    @property
    def discriminator_input_dim(self) -> int:
        """Discriminator input is ``(amp_obs_t, amp_obs_{t+1})`` concatenated."""
        return 2 * self.amp_obs_dim

    # ------------------------------------------------------------------
    # Human-readable order description
    # ------------------------------------------------------------------
    def frame_layout(self) -> list[tuple[str, int]]:
        """Return an ordered list of ``(field_name, dim)`` tuples."""
        out: list[tuple[str, int]] = []
        if self.include_root_height:
            out.append(("root_height", 1))
        if self.include_projected_gravity:
            out.append(("projected_gravity_b", 3))
        if self.include_root_lin_vel_heading:
            out.append(("root_lin_vel_h", 3))
        if self.include_root_ang_vel_body:
            out.append(("root_ang_vel_b", 3))
        if self.include_joint_pos_rel:
            out.append(("joint_pos_rel", self.num_joints))
        if self.include_joint_vel:
            out.append(("joint_vel", self.num_joints))
        # -- feet (4 split flags) --
        if self.include_feet_position:
            out.append(("foot_position_rel_pelvis_b_lr", 6))
        if self.include_feet_orientation:  # [UNIMPLEMENTED]
            out.append(("foot_orientation_rel_pelvis_b_lr", 8))
        if self.include_feet_linear_velocity:
            out.append(("foot_linear_velocity_rel_pelvis_b_lr", 6))
        if self.include_feet_angular_velocity:  # [UNIMPLEMENTED]
            out.append(("foot_angular_velocity_rel_pelvis_b_lr", 6))
        # -- hands (same 4-way split) --
        if self.include_hand_position:
            out.append(("hand_position_rel_pelvis_b_lr", 6))
        if self.include_hand_orientation:  # [UNIMPLEMENTED]
            out.append(("hand_orientation_rel_pelvis_b_lr", 8))
        if self.include_hand_linear_velocity:
            out.append(("hand_linear_velocity_rel_pelvis_b_lr", 6))
        if self.include_hand_angular_velocity:  # [UNIMPLEMENTED]
            out.append(("hand_angular_velocity_rel_pelvis_b_lr", 6))
        return out


@dataclasses.dataclass
class AmpObsState:
    """Per-frame state consumed by ``build_amp_frame_from_state``.

    All tensors have leading batch dimension ``B`` (number of envs on the
    sim side, or number of motion frames on the dataset side). All
    quantities live in world frame unless the name says otherwise.
    ``root_*`` refers to the pelvis link.

    Field shapes
    ------------
    - ``root_pos_w``:      ``(B, 3)``
    - ``root_quat_w``:     ``(B, 4)`` — wxyz (IsaacLab convention)
    - ``root_lin_vel_w``:  ``(B, 3)``
    - ``root_ang_vel_w``:  ``(B, 3)``
    - ``joint_pos``:       ``(B, J)``
    - ``joint_vel``:       ``(B, J)``
    - ``default_joint_pos``: ``(J,)`` or ``(B, J)``
    - ``foot_pos_w``:      ``(B, 2, 3)`` — (left, right)
    - ``foot_lin_vel_w``:  ``(B, 2, 3)``
    - ``hand_pos_w``:      ``(B, 2, 3)``
    - ``hand_lin_vel_w``:  ``(B, 2, 3)``

    Shapes/ordering MUST be identical between env and dataset callers.
    """

    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    default_joint_pos: torch.Tensor
    foot_pos_w: torch.Tensor
    foot_lin_vel_w: torch.Tensor
    hand_pos_w: torch.Tensor
    hand_lin_vel_w: torch.Tensor


# =========================================================================
# Core feature construction
# =========================================================================

_GRAVITY_W: torch.Tensor | None = None  # lazily created per-device


def _gravity_like(ref: torch.Tensor) -> torch.Tensor:
    """Return ``(B, 3)`` gravity vector in world frame, matching ``ref`` dtype/device."""
    shape = (ref.shape[0], 3)
    g = torch.zeros(shape, dtype=ref.dtype, device=ref.device)
    g[..., 2] = -1.0
    return g


def build_amp_frame_from_state(state: AmpObsState, spec: AmpObsSpec) -> torch.Tensor:
    """Compute the single-frame AMP feature ``phi_t`` from a state snapshot.

    Parameters
    ----------
    state:
        Raw per-frame state — all quantities in world frame; quaternions
        are wxyz. See :class:`AmpObsState` for field shapes.
    spec:
        Feature layout. The output dimension is ``spec.frame_dim``.

    Returns
    -------
    phi_t : torch.Tensor
        Shape ``(B, frame_dim)``, dtype matches ``state.root_pos_w``.

    Notes
    -----
    - All expensive operations are batched; the function is safe to call
      inside the rollout loop.
    - Inputs are treated as read-only; no in-place ops on the state.
    """
    root_pos_w = state.root_pos_w
    root_quat_w = state.root_quat_w
    root_lin_vel_w = state.root_lin_vel_w
    root_ang_vel_w = state.root_ang_vel_w

    parts: list[torch.Tensor] = []

    # 1) Root height (pelvis world Z).
    if spec.include_root_height:
        parts.append(root_pos_w[..., 2:3])

    # 2) Projected gravity in the body frame.
    if spec.include_projected_gravity:
        g_w = _gravity_like(root_pos_w)
        parts.append(quat_apply_inverse(root_quat_w, g_w))

    # 3) Root linear velocity in the heading frame (yaw-only world).
    if spec.include_root_lin_vel_heading:
        yaw_q = yaw_quat(root_quat_w)
        parts.append(quat_apply_inverse(yaw_q, root_lin_vel_w))

    # 4) Root angular velocity in the body frame.
    if spec.include_root_ang_vel_body:
        parts.append(quat_apply_inverse(root_quat_w, root_ang_vel_w))

    # 5) Joint positions relative to default.
    default_jp = state.default_joint_pos
    if default_jp.dim() == 1:
        default_jp = default_jp.unsqueeze(0).expand_as(state.joint_pos)
    if spec.include_joint_pos_rel:
        parts.append(state.joint_pos - default_jp)

    # 6) Raw joint velocities.
    if spec.include_joint_vel:
        parts.append(state.joint_vel)

    # 7) Feet in pelvis body frame (position + linear velocity; orientation
    #    and angular velocity are unimplemented and rejected by
    #    AmpObsSpec.__post_init__).
    if spec.include_feet_position or spec.include_feet_linear_velocity:
        # foot_pos_w: (B, 2, 3); pelvis: (B, 1, 3)
        B = root_quat_w.shape[0]
        q2 = root_quat_w.unsqueeze(1).expand(B, 2, 4).reshape(B * 2, 4)
        if spec.include_feet_position:
            foot_rel_w = state.foot_pos_w - root_pos_w.unsqueeze(1)
            foot_rel_b = quat_apply_inverse(q2, foot_rel_w.reshape(B * 2, 3)).reshape(
                B, 6
            )
            parts.append(foot_rel_b)
        if spec.include_feet_linear_velocity:
            foot_vel_rel_w = state.foot_lin_vel_w - root_lin_vel_w.unsqueeze(1)
            foot_vel_b = quat_apply_inverse(
                q2, foot_vel_rel_w.reshape(B * 2, 3)
            ).reshape(B, 6)
            parts.append(foot_vel_b)

    # 8) Hands in pelvis body frame (same split as feet).
    if spec.include_hand_position or spec.include_hand_linear_velocity:
        B = root_quat_w.shape[0]
        q2 = root_quat_w.unsqueeze(1).expand(B, 2, 4).reshape(B * 2, 4)
        if spec.include_hand_position:
            hand_rel_w = state.hand_pos_w - root_pos_w.unsqueeze(1)
            hand_rel_b = quat_apply_inverse(q2, hand_rel_w.reshape(B * 2, 3)).reshape(
                B, 6
            )
            parts.append(hand_rel_b)
        if spec.include_hand_linear_velocity:
            hand_vel_rel_w = state.hand_lin_vel_w - root_lin_vel_w.unsqueeze(1)
            hand_vel_b = quat_apply_inverse(
                q2, hand_vel_rel_w.reshape(B * 2, 3)
            ).reshape(B, 6)
            parts.append(hand_vel_b)

    phi_t = torch.cat(parts, dim=-1)
    if phi_t.shape[-1] != spec.frame_dim:
        raise RuntimeError(
            f"build_amp_frame_from_state produced dim {phi_t.shape[-1]}, "
            f"expected frame_dim={spec.frame_dim}. Feature flags may be out of sync with the spec."
        )
    return phi_t


def build_amp_window(frames: torch.Tensor, stack_k: int) -> torch.Tensor:
    """Flatten a ``(..., K, frame_dim)`` tensor into ``(..., K*frame_dim)``.

    Convention: the ``K`` axis is *oldest-first, newest-last*. A caller
    that wants ``amp_obs_t`` should pass ``frames[..., t-K+1:t+1, :]``
    (or an equivalent circular-buffer arrangement with the same order).

    Parameters
    ----------
    frames:
        Tensor of shape ``(..., K, D)``. ``K`` must equal ``stack_k``.
    stack_k:
        Window size — pass it explicitly so shape mismatches fail loudly.

    Returns
    -------
    ``(..., K * D)`` tensor.
    """
    if frames.shape[-2] != stack_k:
        raise ValueError(
            f"build_amp_window expected the second-to-last axis to be stack_k={stack_k}, "
            f"got shape {tuple(frames.shape)}."
        )
    return frames.reshape(*frames.shape[:-2], frames.shape[-2] * frames.shape[-1])


def concat_frame_history(
    new_frame: torch.Tensor, history: torch.Tensor, stack_k: int
) -> torch.Tensor:
    """Advance a (oldest-first) circular-ish history by one frame.

    Given a ``(B, K, D)`` history buffer and a new frame ``(B, D)``,
    return a new ``(B, K, D)`` buffer with the oldest frame dropped and
    the new frame appended at the end.

    This is the helper the environment uses each step to update its
    K-frame buffer before reading out ``amp_obs_t = flatten(history)``.
    """
    if history.shape[1] != stack_k:
        raise ValueError(f"history.shape[1]={history.shape[1]} != stack_k={stack_k}")
    if (
        new_frame.shape[0] != history.shape[0]
        or new_frame.shape[-1] != history.shape[-1]
    ):
        raise ValueError(
            f"shape mismatch: new_frame={tuple(new_frame.shape)}, "
            f"history={tuple(history.shape)}"
        )
    # Shift: drop oldest, append newest.
    return torch.cat([history[:, 1:], new_frame.unsqueeze(1)], dim=1)


# =========================================================================
# Mapping resolvers — name → articulation index
# =========================================================================


def resolve_indices(
    requested: Sequence[str], available: Sequence[str], context: str
) -> list[int]:
    """Resolve each name in ``requested`` to its index in ``available``.

    Raises a ``ValueError`` with a clear message listing both the missing
    names and the available names if any requested name is absent. No
    silent fallback.
    """
    avail_list = list(available)
    lookup = {name: i for i, name in enumerate(avail_list)}
    missing = [n for n in requested if n not in lookup]
    if missing:
        raise ValueError(
            f"[{context}] Missing names: {missing}. Available ({len(avail_list)}): {avail_list}"
        )
    return [lookup[n] for n in requested]


def resolve_spec_indices(
    spec: AmpObsSpec,
    available_joint_names: Sequence[str],
    available_body_names: Sequence[str],
) -> Mapping[str, list[int]]:
    """Resolve spec names against the env's articulation naming.

    Returns a mapping with keys ``joints``, ``pelvis``, ``feet``, ``hands``
    — each a list of indices into the provided name lists.
    """
    return {
        "joints": resolve_indices(
            spec.joint_names, available_joint_names, "amp.joints"
        ),
        "pelvis": resolve_indices(
            [spec.pelvis_body_name], available_body_names, "amp.pelvis"
        ),
        "feet": resolve_indices(spec.foot_body_names, available_body_names, "amp.feet"),
        "hands": resolve_indices(
            spec.hand_body_names, available_body_names, "amp.hands"
        ),
    }
