from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.dynamic_obs_controller import DynamicObsController
import torch
from aerial_gym.utils.helpers import get_args

if __name__ == "__main__":
    args = get_args()
    logger.warning(
        "Battle UAV example: dynamic obstacle driven by Catmull-Rom spline controller."
    )
    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="dynamic_uav_env",
        robot_name="base_quadrotor",
        controller_name="lee_position_control",
        args=None,
        device="cuda:0",
        num_envs=args.num_envs,
        headless=args.headless,
        use_warp=args.use_warp,
    )
    num_assets_in_env = env_manager.IGE_env.num_assets_per_env - 1
    env_manager.reset()

    obs_dict = env_manager.get_obs()
    uav_index = env_manager.get_assets_index("dynamic_uav")

    # 固定盒子：所有 env 随机边界的公共内框，确保样条 waypoint 不会打到边界外。
    obs_bounds_min = [-1.0, -2.5, 0.5]
    obs_bounds_max = [9.0, 2.5, 4.0]
    dynamic_obs_controller = DynamicObsController(
        min_position=obs_bounds_min,
        max_position=obs_bounds_max,
        num_envs=env_manager.num_envs,
        device="cuda:0",
        dt=float(obs_dict["dt"]),
        min_velocity=1.0,
        max_velocity=1.4,
        min_segment_steps=5,
        max_segment_steps=20,
        smoothing_factor=0.4,
        noise_scale=0.4,
        position_gain=4.0,
    )
    # 用真实初始位置重置控制器，避免样条从 0 起步跳变。
    dynamic_obs_controller.reset(
        initial_positions=env_manager.get_obs_position()[:, uav_index[0], :]
    )

    asset_twist = torch.zeros(
        (env_manager.num_envs, num_assets_in_env, 6), device="cuda:0"
    )
    actions = torch.zeros((env_manager.num_envs, 4), device="cuda:0")
    for i in range(10000):
        if i % 500 == 0:
            logger.info(f"Step {i}")
        euler_angles = env_manager.get_obs_euler()
        position = env_manager.get_obs_position()

        controller_twist = dynamic_obs_controller.get_twist(
            position[:, uav_index[0], :]
        )
        asset_twist.zero_()
        asset_twist[:, uav_index[0]] = controller_twist

        # lee_position_control 直接跟随目标当前位姿。
        actions[:, 0:3] = position[:, uav_index[0], 0:3]
        actions[:, 3] = euler_angles[:, uav_index[0], 2]

        env_manager.step(actions=actions, env_actions=asset_twist)
