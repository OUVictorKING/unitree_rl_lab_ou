from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


DEFAULT_EXPERT_ROOT = (
    Path(__file__).resolve().parents[6]
    / "motion_datasets"
    / "pingpong"
    / "humanoid_data"
    / "final"
    / "expert"
)


def yaw_from_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Return yaw angle from a wxyz quaternion tensor."""
    w, x, y, z = quat.unbind(dim=-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def quat_slerp_wxyz(q0: torch.Tensor, q1: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Batched quaternion slerp for wxyz convention."""
    q0 = torch.nn.functional.normalize(q0, dim=-1)
    q1 = torch.nn.functional.normalize(q1, dim=-1)
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.abs(dot).clamp(max=0.9995)

    alpha = alpha.view(*alpha.shape, *([1] * (q0.dim() - alpha.dim())))
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = torch.sin(theta)

    s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0.clamp_min(1e-8)
    s1 = sin_theta / sin_theta_0.clamp_min(1e-8)
    slerped = s0 * q0 + s1 * q1
    lerped = q0 + alpha * (q1 - q0)
    out = torch.where(sin_theta_0 > 1e-4, slerped, lerped)
    return torch.nn.functional.normalize(out, dim=-1)


@dataclass
class PingpongRefState:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_pos_w: torch.Tensor
    body_quat_w: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_ang_vel_w: torch.Tensor
    pelvis_pos_w: torch.Tensor
    pelvis_quat_w: torch.Tensor
    pelvis_lin_vel_w: torch.Tensor
    pelvis_ang_vel_w: torch.Tensor
    ref_frame_f: torch.Tensor


class PingpongMotionClip:
    """Single expert clip with float-frame interpolation."""

    def __init__(self, path: str | Path, tracked_body_names: list[str], device: str):
        self.path = str(path)
        data = np.load(self.path, allow_pickle=True)
        self.name = str(data["clip_name"][0]) if "clip_name" in data else Path(path).stem
        self.fps = int(data["fps"][0])
        self.impact_frame = int(data["impact_frame"][0])
        self.body_names = [str(x) for x in data["body_names"].tolist()]
        self.body_ids = [self.body_names.index(name) for name in tracked_body_names]
        self.pelvis_body_id = self.body_names.index("pelvis")
        self.blade_body_id = self.body_names.index("right_paddle_blade")

        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self.body_pos_w_all = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self.body_quat_w_all = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self.body_lin_vel_w_all = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self.body_ang_vel_w_all = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self.length = int(self.joint_pos.shape[0])

        self.body_pos_w = self.body_pos_w_all[:, self.body_ids]
        self.body_quat_w = self.body_quat_w_all[:, self.body_ids]
        self.body_lin_vel_w = self.body_lin_vel_w_all[:, self.body_ids]
        self.body_ang_vel_w = self.body_ang_vel_w_all[:, self.body_ids]

        imp = self.impact_frame
        pelvis_pos = self.body_pos_w_all[imp, self.pelvis_body_id]
        blade_pos = self.body_pos_w_all[imp, self.blade_body_id]
        pelvis_quat = self.body_quat_w_all[imp, self.pelvis_body_id]
        yaw = yaw_from_wxyz(pelvis_quat)
        diff = blade_pos[:2] - pelvis_pos[:2]
        c = torch.cos(-yaw)
        s = torch.sin(-yaw)
        self.expert_offset_base = torch.stack((c * diff[0] - s * diff[1], s * diff[0] + c * diff[1]))
        self.pre_duration = float(self.impact_frame / self.fps)
        self.post_duration = float((self.length - 1 - self.impact_frame) / self.fps)

    def frame_from_step(self, cur_step: torch.Tensor, t_pre: torch.Tensor, t_post: torch.Tensor, dt: float) -> torch.Tensor:
        sim_t = cur_step.to(torch.float32) * dt
        pre = t_pre.clamp_min(dt)
        post = t_post.clamp_min(dt)
        impact = float(self.impact_frame)
        last = float(self.length - 1)

        pre_frame = (sim_t / pre).clamp(0.0, 1.0) * impact
        post_frame = impact + ((sim_t - pre) / post).clamp(0.0, 1.0) * (last - impact)
        return torch.where(sim_t <= pre, pre_frame, post_frame).clamp(0.0, last)

    def sample(self, frame_f: torch.Tensor, env_origins: torch.Tensor) -> PingpongRefState:
        frame_f = frame_f.clamp(0.0, float(self.length - 1))
        lo = torch.floor(frame_f).long()
        hi = torch.clamp(lo + 1, max=self.length - 1)
        alpha = frame_f - lo.to(frame_f.dtype)

        def lerp(data: torch.Tensor) -> torch.Tensor:
            return data[lo] * (1.0 - alpha).view(-1, *([1] * (data.ndim - 1))) + data[hi] * alpha.view(
                -1, *([1] * (data.ndim - 1))
            )

        joint_pos = lerp(self.joint_pos)
        joint_vel = lerp(self.joint_vel)
        body_pos = lerp(self.body_pos_w) + env_origins[:, None, :]
        body_quat = quat_slerp_wxyz(self.body_quat_w[lo], self.body_quat_w[hi], alpha)
        body_lin = lerp(self.body_lin_vel_w)
        body_ang = lerp(self.body_ang_vel_w)

        pelvis_pos = lerp(self.body_pos_w_all[:, self.pelvis_body_id]) + env_origins
        pelvis_quat = quat_slerp_wxyz(
            self.body_quat_w_all[lo, self.pelvis_body_id], self.body_quat_w_all[hi, self.pelvis_body_id], alpha
        )
        pelvis_lin = lerp(self.body_lin_vel_w_all[:, self.pelvis_body_id])
        pelvis_ang = lerp(self.body_ang_vel_w_all[:, self.pelvis_body_id])

        return PingpongRefState(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos,
            body_quat_w=body_quat,
            body_lin_vel_w=body_lin,
            body_ang_vel_w=body_ang,
            pelvis_pos_w=pelvis_pos,
            pelvis_quat_w=pelvis_quat,
            pelvis_lin_vel_w=pelvis_lin,
            pelvis_ang_vel_w=pelvis_ang,
            ref_frame_f=frame_f,
        )


class PingpongMotionLoader:
    """Two-clip expert loader used by the HITTER pingpong task."""

    def __init__(
        self,
        forward_file: str | Path,
        backward_file: str | Path,
        tracked_body_names: list[str],
        device: str,
    ):
        self.tracked_body_names = tracked_body_names
        self.clips = {
            "forehand": PingpongMotionClip(forward_file, tracked_body_names, device),
            "backhand": PingpongMotionClip(backward_file, tracked_body_names, device),
        }
        self.swing_to_index = {"forehand": 0, "backhand": 1}
        self.index_to_swing = {0: "forehand", 1: "backhand"}

    @property
    def expert_offset_base(self) -> torch.Tensor:
        return torch.stack(
            (self.clips["forehand"].expert_offset_base, self.clips["backhand"].expert_offset_base), dim=0
        )

    def sample(
        self,
        swing_type: torch.Tensor,
        cur_step: torch.Tensor,
        t_pre: torch.Tensor,
        t_post: torch.Tensor,
        dt: float,
        env_origins: torch.Tensor,
    ) -> PingpongRefState:
        num_envs = swing_type.shape[0]
        device = swing_type.device

        ref = None
        for swing_idx, name in self.index_to_swing.items():
            ids = torch.nonzero(swing_type == swing_idx, as_tuple=False).flatten()
            if len(ids) == 0:
                continue
            clip = self.clips[name]
            frame_f = clip.frame_from_step(cur_step[ids], t_pre[ids], t_post[ids], dt)
            sub = clip.sample(frame_f, env_origins[ids])
            if ref is None:
                ref = self._empty_state(num_envs, sub, device)
            self._assign_state(ref, sub, ids)

        if ref is None:
            raise RuntimeError("PingpongMotionLoader.sample called with no environments")
        return ref

    @staticmethod
    def _empty_state(num_envs: int, template: PingpongRefState, device: torch.device) -> PingpongRefState:
        def zeros_like_tail(x: torch.Tensor) -> torch.Tensor:
            return torch.zeros((num_envs, *x.shape[1:]), dtype=x.dtype, device=device)

        quat = torch.zeros((num_envs, *template.body_quat_w.shape[1:]), dtype=template.body_quat_w.dtype, device=device)
        quat[..., 0] = 1.0
        pelvis_quat = torch.zeros((num_envs, 4), dtype=template.pelvis_quat_w.dtype, device=device)
        pelvis_quat[:, 0] = 1.0
        return PingpongRefState(
            joint_pos=zeros_like_tail(template.joint_pos),
            joint_vel=zeros_like_tail(template.joint_vel),
            body_pos_w=zeros_like_tail(template.body_pos_w),
            body_quat_w=quat,
            body_lin_vel_w=zeros_like_tail(template.body_lin_vel_w),
            body_ang_vel_w=zeros_like_tail(template.body_ang_vel_w),
            pelvis_pos_w=zeros_like_tail(template.pelvis_pos_w),
            pelvis_quat_w=pelvis_quat,
            pelvis_lin_vel_w=zeros_like_tail(template.pelvis_lin_vel_w),
            pelvis_ang_vel_w=zeros_like_tail(template.pelvis_ang_vel_w),
            ref_frame_f=torch.zeros(num_envs, dtype=template.ref_frame_f.dtype, device=device),
        )

    @staticmethod
    def _assign_state(dst: PingpongRefState, src: PingpongRefState, ids: torch.Tensor) -> None:
        for name in dst.__dataclass_fields__:
            getattr(dst, name)[ids] = getattr(src, name)
