from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.sim.sim_builder import SimBuilder
import torch
from aerial_gym.utils.helpers import get_args

if __name__ == "__main__":
    args = get_args()
    logger.warning("This example demonstrates the use of geometric controllers for a quadrotor.")
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
    num_assets_in_env = (
            env_manager.IGE_env.num_assets_per_env - 1
    )
    actions = torch.zeros((env_manager.num_envs,num_assets_in_env, 6)).to("cuda:0")
    env_manager.reset()
    asset_twist=torch.zeros((env_manager.num_envs,env_manager.asset_manager. 4)).to("cuda:0")
    for i in range(10000):
        if i % 500 == 0:
            logger.info(f"Step {i}, changing target setpoint.")
            # actions[:, 0:3] = 2.0 * (torch.rand_like(actions[:, 0:3]) * 2 - 1)
            # actions[:, 3] = torch.pi * (torch.rand_like(actions[:, 3]) * 2 - 1)
            env_manager.reset()
        asset_twist[:, :, 0] = torch.sin(0.2 * i * torch.ones_like(asset_twist[:, :, 0]))
        asset_twist[:, :, 1] = torch.cos(0.2 * i * torch.ones_like(asset_twist[:, :, 1]))
        asset_twist[:, :, 2] = 0.0
        env_manager.step(actions=actions)
