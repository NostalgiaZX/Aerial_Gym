import torch


class task_config:
    seed = 1
    sim_name = "base_sim"
    env_name = "dynamic_uav_env"
    robot_name = "base_quadrotor"
    controller_name = "lee_velocity_control"
    args = {}
    num_envs = 4096
    use_warp = False
    headless = False
    device = "cuda:0"
    observation_space_dim = 16
    privileged_observation_space_dim = 0
    action_space_dim = 4
    episode_len_steps = 500  # real physics time for simulation is this value multiplied by sim.dt
    return_state_before_reset = False

    # 动态障碍物（dynamic_uav）活动的固定盒子，选在所有 env 随机边界的公共内框里。
    obs_bounds_min = [-1.0, -2.5, 0.5]
    obs_bounds_max = [9.0, 2.5, 4.0]
    # 每段速度上限从 [min, max] 采样，段与段之间前进速度不同。
    obs_min_velocity = 1.0
    obs_max_velocity = 1.5
    # 每段步数从 [min, max] 采样（dt≈0.01 时约对应 0.1–0.4s 换一次方向）。
    obs_min_segment_steps = 5
    obs_max_segment_steps = 20
    obs_smoothing_factor = 0.4
    obs_position_gain = 4.0
    # 每步在速度输出上叠加 noise_scale * max_velocity 的高斯扰动。
    obs_noise_scale = 0.4
    reward_parameters = {
        "pos_error_gain1": [2.0, 2.0, 2.0],
        "pos_error_exp1": [1 / 3.5, 1 / 3.5, 1 / 3.5],
        "pos_error_gain2": [2.0, 2.0, 2.0],
        "pos_error_exp2": [2.0, 2.0, 2.0],
        "dist_reward_coefficient": 7.5,
        "max_dist": 15.0,
        "action_diff_penalty_gain": [1.0, 1.0, 1.0],
        "absolute_action_reward_gain": [2.0, 2.0, 2.0],
        "crash_penalty": -100,
    }
