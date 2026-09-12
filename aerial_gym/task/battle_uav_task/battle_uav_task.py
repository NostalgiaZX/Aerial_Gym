from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
import torch
import numpy as np

from aerial_gym.utils.math import *
from aerial_gym.utils.dynamic_obs_controller import DynamicObsController

from aerial_gym.utils.logging import CustomLogger

import gymnasium as gym
from gym.spaces import Dict, Box

logger = CustomLogger("battle_uav_task")


def dict_to_class(dict):
    return type("ClassFromDict", (object,), dict)


class BattleUavTask(BaseTask):
    def __init__(
            self, task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None
    ):
        # overwrite the params if user has provided them
        if seed is not None:
            task_config.seed = seed
        if num_envs is not None:
            task_config.num_envs = num_envs
        if headless is not None:
            task_config.headless = headless
        if device is not None:
            task_config.device = device
        if use_warp is not None:
            task_config.use_warp = use_warp

        super().__init__(task_config)
        self.device = self.task_config.device
        # set the each of the elements of reward parameter to a torch tensor
        for key in self.task_config.reward_parameters.keys():
            self.task_config.reward_parameters[key] = torch.tensor(
                self.task_config.reward_parameters[key], device=self.device
            )
        logger.info("Building environment for battle uav task.")
        logger.info(
            "\nSim Name: {},\nEnv Name: {},\nRobot Name: {}, \nController Name: {}".format(
                self.task_config.sim_name,
                self.task_config.env_name,
                self.task_config.robot_name,
                self.task_config.controller_name,
            )
        )
        logger.info(
            "\nNum Envs: {},\nUse Warp: {},\nHeadless: {}".format(
                self.task_config.num_envs,
                self.task_config.use_warp,
                self.task_config.headless,
            )
        )

        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            args=self.task_config.args,
            device=self.device,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless,
        )

        self.actions = torch.zeros(
            (self.sim_env.num_envs, self.task_config.action_space_dim),
            device=self.device,
            requires_grad=False,
        )
        self.prev_actions = torch.zeros_like(self.actions)
        self.counter = 0
        self.uav_index = self.sim_env.get_assets_index("dynamic_uav")
        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )
        self.target_velocity = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )
        # Get the dictionary once from the environment and use it to get the observations later.
        # This is to avoid constant retuning of data back anf forth across functions as the tensors update and can be read in-place.
        self.obs_dict = self.sim_env.get_obs()
        self.obs_dict["num_obstacles_in_env"] = self.sim_env.num_obs_in_env
        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.truncations.shape[0], device=self.device)

        self.observation_space = Dict(
            {"observations": Box(low=-1.0, high=1.0, shape=(self.task_config.observation_space_dim,), dtype=np.float32)}
        )
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.task_config.action_space_dim,),
            dtype=np.float32,
        )
        # self.action_transformation_function = self.sim_env.robot_manager.robot.action_transformation_function

        self.num_envs = self.sim_env.num_envs

        # Currently only the "observations" are sent to the actor and critic.
        # The "priviliged_obs" are not handled so far in sample-factory

        self.task_obs = {
            "observations": torch.zeros(
                (self.sim_env.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            "priviliged_obs": torch.zeros(
                (
                    self.sim_env.num_envs,
                    self.task_config.privileged_observation_space_dim,
                ),
                device=self.device,
                requires_grad=False,
            ),
            "collisions": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
            "rewards": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
        }

        self.obs_bounds_min = torch.tensor(
            getattr(self.task_config, "obs_bounds_min", [-1.0, -2.5, 0.5]),
            device=self.device,
            dtype=torch.float32,
        )
        self.obs_bounds_max = torch.tensor(
            getattr(self.task_config, "obs_bounds_max", [9.0, 2.5, 4.0]),
            device=self.device,
            dtype=torch.float32,
        )
        self.obs_twist = torch.zeros(
            (
                self.sim_env.num_envs,
                self.sim_env.IGE_env.num_assets_per_env - 1,
                6,
            ),
            device=self.device,
        )
        self.dynamic_obs_controller = DynamicObsController(
            min_position=self.obs_bounds_min,
            max_position=self.obs_bounds_max,
            num_envs=self.sim_env.num_envs,
            device=self.device,
            dt=float(self.obs_dict["dt"]),
            min_velocity=float(getattr(self.task_config, "obs_min_velocity", 1.5)),
            max_velocity=float(getattr(self.task_config, "obs_max_velocity", 3.5)),
            min_segment_steps=int(
                getattr(self.task_config, "obs_min_segment_steps", 10)
            ),
            max_segment_steps=int(
                getattr(self.task_config, "obs_max_segment_steps", 40)
            ),
            smoothing_factor=float(
                getattr(self.task_config, "obs_smoothing_factor", 0.2)
            ),
            noise_scale=float(getattr(self.task_config, "obs_noise_scale", 0.2)),
            position_gain=float(getattr(self.task_config, "obs_position_gain", 4.0)),
        )

    def close(self):
        self.sim_env.delete_env()

    def reset(self):
        self.infos = {}
        self.sim_env.reset()
        self.update_obs_state()
        self._reset_dynamic_obs_controller()
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        self.infos = {}
        self.sim_env.reset_idx(env_ids)
        self.update_obs_state()
        self._reset_dynamic_obs_controller(env_ids)
        return

    def render(self):
        return None

    def step(self, actions):
        self.counter += 1
        self.prev_actions[:] = self.actions
        self.actions[:] = actions

        # this uses the action, gets observations
        # calculates rewards, returns tuples
        # In this case, the episodes that are terminated need to be
        # first reset, and the first obseration of the new episode
        # needs to be returned.
        self.compute_obs_next_action()
        self.sim_env.step(actions=self.actions, env_actions=self.obs_twist)
        self.update_obs_state()
        # This step must be done since the reset is done after the reward is calculated.
        # This enables the robot to send back an updated state, and an updated observation to the RL agent after the reset.
        # This is important for the RL agent to get the correct state after the reset.
        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)

        if self.task_config.return_state_before_reset == True:
            return_tuple = self.get_return_tuple()

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps, 1, 0
        )
        self.sim_env.post_reward_calculation_step()
        self.update_obs_state()

        self.infos = {}  # self.obs_dict["infos"]

        if self.task_config.return_state_before_reset == False:
            return_tuple = self.get_return_tuple()

        return return_tuple

    def get_return_tuple(self):
        self.process_obs_for_task()
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )
    def process_obs_for_task(self):
        self.task_obs["observations"][:, 0:3] = (
                self.target_position - self.obs_dict["robot_position"]
        )
        self.task_obs["observations"][:, 3:7] = self.obs_dict["robot_orientation"]
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"]
        self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_body_angvel"]
        self.task_obs["observations"][:, 13:16] = self.target_velocity
        self.task_obs["rewards"] = self.rewards
        self.task_obs["terminations"] = self.terminations
        self.task_obs["truncations"] = self.truncations

    def compute_rewards_and_crashes(self, obs_dict):
        robot_position = obs_dict["robot_position"]
        target_position = self.target_position
        robot_linvel = obs_dict["robot_linvel"]
        robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]
        robot_orientation = obs_dict["robot_orientation"]
        target_orientation = torch.zeros_like(robot_orientation, device=self.device)
        target_orientation[:, 3] = 1.0
        angular_velocity = obs_dict["robot_body_angvel"]
        root_quats = obs_dict["robot_orientation"]

        pos_error_vehicle_frame = quat_apply_inverse(
            robot_vehicle_orientation, (target_position - robot_position)
        )
        return compute_reward(
            pos_error_vehicle_frame,
            robot_linvel,
            root_quats,
            angular_velocity,
            obs_dict["crashes"],
            1.0,  # obs_dict["curriculum_level_multiplier"],
            self.actions,
            self.prev_actions,
            self.task_config.reward_parameters,
        )

    def compute_obs_next_action(self):
        self.obs_twist.zero_()
        controller_twist = self.dynamic_obs_controller.get_twist(self.target_position)
        # dynamic_uav 在障碍列表里的槽位对所有 env 相同，直接向量化赋值。
        self.obs_twist[:, self.uav_index[0]] = controller_twist

    def _reset_dynamic_obs_controller(self, env_ids=None):
        # 边界是固定盒子，不需要在 reset 时刷新；只重播样条控制点。
        self.dynamic_obs_controller.reset(
            initial_positions=self.target_position, env_ids=env_ids
        )

    def update_obs_state(self):
        target_position_all = self.sim_env.get_obs_position()
        target_velocity_all = self.sim_env.get_obs_linvel()
        self.target_position[:] = target_position_all[:, self.uav_index[0], :]
        self.target_velocity[:] = target_velocity_all[:, self.uav_index[0], :]

@torch.jit.script
def exp_func(x, gain, exp):
    # type: (Tensor, float, float) -> Tensor
    return gain * torch.exp(-exp * x * x)



@torch.jit.script
def exp_penalty_func(x, gain, exp):
    # type: (Tensor, float, float) -> Tensor
    return gain * (torch.exp(-exp * x * x) - 1)

@torch.jit.script
def compute_reward(
        pos_error,
        lin_vels,
        robot_quats,
        robot_angvels,
        crashes,
        curriculum_level_multiplier,
        current_action,
        prev_actions,
        parameter_dict,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Dict[str, Tensor]) -> Tuple[Tensor, Tensor]

    dist = torch.norm(pos_error, dim=1)

    pos_reward = exp_func(dist, 3.0, 8.0) + exp_func(dist, 2.0, 4.0) + exp_func(dist, 5.0, 100.0)

    dist_reward = (20 - dist) / 8.0

    ups = quat_axis(robot_quats, 2)
    tiltage = torch.abs(1 - ups[..., 2])
    up_reward = 0.2 / (0.1 + tiltage * tiltage)

    spinnage = torch.norm(robot_angvels, dim=1)
    ang_vel_reward = (1.0 / (1.0 + spinnage * spinnage)) * 3

    total_reward = (
            pos_reward + dist_reward + 0.1 * (up_reward + ang_vel_reward) - 3.0
    )
    total_reward[:] = curriculum_level_multiplier * total_reward

    physical_collision = crashes > 0
    hit_target = physical_collision & (dist < 0.8)
    hit_other = physical_collision & (dist >= 0.8)
    too_far = dist > 15.0

    total_reward[:] = torch.where(hit_target, torch.full_like(total_reward, 500.0), total_reward)
    total_reward[:] = torch.where(hit_other, torch.full_like(total_reward, -20.0), total_reward)
    total_reward[:] = torch.where(too_far, torch.full_like(total_reward, -20.0), total_reward)

    crashes[:] = torch.where(
        hit_target | hit_other | too_far,
        torch.ones_like(crashes),
        crashes,
    )

    return total_reward, crashes
