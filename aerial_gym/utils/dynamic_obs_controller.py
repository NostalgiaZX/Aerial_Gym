import torch

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("dynamic_obs_controller")


class DynamicObsController:
    """基于均匀 Catmull-Rom 样条 + P 控制器的动态障碍物速度控制器。

    每 env 维护 4 个控制点 (P0, P1, P2, P3)，全部采样自固定边界
    [min_position, max_position]；样条经过 P1→P2 段，段参数 t 每步推进
    1/segment_length，t 跨过 1.0 时把控制点向前滑一格并重采样新的 P3。

    为了让运动方向和速度随时都在变化，跨段时会同时重采样：
        * 段长 segment_length ∈ [min_segment_steps, max_segment_steps]
        * 段内速度上限 segment_max_speed ∈ [min_velocity, max_velocity]

    get_twist 的思路是位置跟踪：
        1. 用样条算出参考位置 desired = catmull_rom(t)，再夹到边界内；
        2. 用 (desired - current) * position_gain 得到期望速度；
        3. EMA 平滑 + 可选噪声 + 按当前段的 segment_max_speed 保方向裁剪；
        4. 硬约束：若 current + vel*dt 会越界，把该轴速度截到刚好贴边。
    """

    _EPS = 1e-6

    def __init__(
        self,
        min_position,
        max_position,
        num_envs,
        device,
        dt,
        min_velocity=1.5,
        max_velocity=3.5,
        min_segment_steps=10,
        max_segment_steps=40,
        smoothing_factor=0.2,
        noise_scale=0.2,
        position_gain=4.0,
    ):
        self.num_envs = int(num_envs)
        self.device = device
        self.dt = float(dt)

        self.min_velocity = float(min_velocity)
        self.max_velocity = float(max_velocity)
        assert self.max_velocity >= self.min_velocity >= 0.0

        self.min_segment_steps = max(1, int(min_segment_steps))
        self.max_segment_steps = max(self.min_segment_steps, int(max_segment_steps))

        self.smoothing_factor = float(smoothing_factor)
        self.noise_scale = float(noise_scale)
        self.position_gain = float(position_gain)

        self.min_position = self._to_bounds(min_position)
        self.max_position = self._to_bounds(max_position)

        self.control_points = torch.zeros(
            (self.num_envs, 4, 3), device=self.device
        )
        self.segment_t = torch.zeros(self.num_envs, device=self.device)
        # 每个 env 独立的段长（以 1/steps 存放，方便直接加到 segment_t）。
        self.segment_t_step = torch.zeros(self.num_envs, device=self.device)
        self.segment_max_speed = torch.zeros(self.num_envs, device=self.device)
        self.prev_velocity = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self.desired_position = torch.zeros(
            (self.num_envs, 3), device=self.device
        )

        self.reset()

    # -------------------- public API --------------------

    def update_bounds(self, min_position, max_position):
        self.min_position = self._to_bounds(min_position)
        self.max_position = self._to_bounds(max_position)

    def reset(self, initial_positions=None, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(self.device).long()
        if env_ids.numel() == 0:
            return

        if initial_positions is not None:
            p1 = initial_positions.to(self.device)[env_ids]
        else:
            p1 = self._sample_points(env_ids)
        p1 = torch.clamp(p1, self.min_position[env_ids], self.max_position[env_ids])

        p2 = self._sample_points(env_ids)
        p3 = self._sample_points(env_ids)
        # P0 反向外推，保 C¹ 连续；再夹一次防越界。
        p0 = torch.clamp(
            2.0 * p1 - p2,
            self.min_position[env_ids],
            self.max_position[env_ids],
        )

        self.control_points[env_ids, 0] = p0
        self.control_points[env_ids, 1] = p1
        self.control_points[env_ids, 2] = p2
        self.control_points[env_ids, 3] = p3
        self.segment_t[env_ids] = 0.0
        self.segment_t_step[env_ids] = self._sample_segment_t_step(env_ids)
        self.segment_max_speed[env_ids] = self._sample_segment_speed(env_ids)
        self.prev_velocity[env_ids] = 0.0
        self.desired_position[env_ids] = p1

    def reset_idx(self, env_ids, initial_positions=None):
        self.reset(initial_positions=initial_positions, env_ids=env_ids)

    def get_twist(self, current_positions):
        self._advance_segment()

        p0 = self.control_points[:, 0]
        p1 = self.control_points[:, 1]
        p2 = self.control_points[:, 2]
        p3 = self.control_points[:, 3]
        t = self.segment_t.unsqueeze(-1)
        t2 = t * t
        t3 = t2 * t

        # 均匀 Catmull-Rom 位置公式。
        q = 0.5 * (
            2.0 * p1
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
        # 样条在段端点附近可能有轻微 overshoot，硬夹到边界内。
        q = torch.clamp(q, self.min_position, self.max_position)
        self.desired_position[:] = q

        current = current_positions.to(self.device)
        desired_velocity = (q - current) * self.position_gain

        alpha = self.smoothing_factor
        velocity = alpha * self.prev_velocity + (1.0 - alpha) * desired_velocity

        # 噪声放在平滑之后，避免大部分随机被 EMA 低通掉。
        if self.noise_scale > 0.0:
            velocity = velocity + (
                self.noise_scale * self.max_velocity
            ) * torch.randn_like(velocity)

        # 按当前段的速度上限保方向缩放（每段速度都不同 → 前进速度会变）。
        speed = torch.linalg.norm(velocity, dim=-1, keepdim=True)
        seg_cap = self.segment_max_speed.unsqueeze(-1)
        scale = torch.clamp(seg_cap / (speed + self._EPS), max=1.0)
        velocity = velocity * scale

        # 硬边界：让 current + vel*dt 严格落在盒子里。
        velocity = self._enforce_bounds(current, velocity)

        self.prev_velocity[:] = velocity

        twist = torch.zeros((self.num_envs, 6), device=self.device)
        twist[:, 0:3] = velocity
        return twist

    # -------------------- internals --------------------

    def _advance_segment(self):
        self.segment_t += self.segment_t_step
        crossed = self.segment_t >= 1.0
        while crossed.any():
            idx = crossed.nonzero(as_tuple=True)[0]
            self.control_points[idx, 0] = self.control_points[idx, 1]
            self.control_points[idx, 1] = self.control_points[idx, 2]
            self.control_points[idx, 2] = self.control_points[idx, 3]
            self.control_points[idx, 3] = self._sample_points(idx)
            self.segment_t[idx] -= 1.0
            # 每段都重新采一次段长和速度上限 → 方向和快慢随时变。
            self.segment_t_step[idx] = self._sample_segment_t_step(idx)
            self.segment_max_speed[idx] = self._sample_segment_speed(idx)
            crossed = self.segment_t >= 1.0

    def _enforce_bounds(self, current, velocity):
        dt = self.dt
        if dt <= 0.0:
            return velocity
        max_step_vel = (self.max_position - current) / dt
        min_step_vel = (self.min_position - current) / dt
        # 已在盒子外时只允许向内速度：clamp 到有向的可行域。
        max_allow = torch.clamp(max_step_vel, min=0.0)
        min_allow = torch.clamp(min_step_vel, max=0.0)
        return torch.clamp(velocity, min=min_allow, max=max_allow)

    def _sample_points(self, env_ids):
        count = env_ids.numel()
        rand = torch.rand((count, 3), device=self.device)
        lo = self.min_position[env_ids]
        hi = self.max_position[env_ids]
        return lo + rand * (hi - lo)

    def _sample_segment_t_step(self, env_ids):
        count = env_ids.numel()
        span = float(self.max_segment_steps - self.min_segment_steps)
        lengths = float(self.min_segment_steps) + torch.rand(
            count, device=self.device
        ) * span
        return 1.0 / lengths

    def _sample_segment_speed(self, env_ids):
        count = env_ids.numel()
        span = self.max_velocity - self.min_velocity
        return self.min_velocity + torch.rand(count, device=self.device) * span

    def _to_bounds(self, bounds):
        if not isinstance(bounds, torch.Tensor):
            bounds = torch.as_tensor(bounds, device=self.device, dtype=torch.float32)
        else:
            bounds = bounds.to(self.device).float()
        if bounds.ndim == 1:
            bounds = bounds.unsqueeze(0).expand(self.num_envs, -1).contiguous()
        return bounds
