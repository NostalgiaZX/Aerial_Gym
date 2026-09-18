import torch
from aerial_gym import AERIAL_GYM_DIRECTORY


class task_config:
    seed = 1
    sim_name = "base_sim"
    env_name = "hard_dynamic_uav_env"
    robot_name = "lmf2"
    controller_name = "lmf2_velocity_control"
    args = {}
    num_envs = 1024
    use_warp = True
    headless = False
    device = "cuda:0"
    observation_space_dim = 84
    privileged_observation_space_dim = 0
    action_space_dim = 4
    episode_len_steps = 500  # real physics time for simulation is this value multiplied by sim.dt
    return_state_before_reset = False

    class vae_config:
        use_vae = True
        latent_dims = 64
        model_file = (
            AERIAL_GYM_DIRECTORY
            + "/aerial_gym/utils/vae/weights/ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth"
        )
        model_folder = AERIAL_GYM_DIRECTORY
        image_res = (270, 480)
        interpolation_mode = "nearest"
        return_sampled_latent = True

    # 动态障碍物（hard_dynamic_uav）活动的固定盒子，选在所有 env 随机边界的公共内框里。
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
        "x_action_diff_penalty_magnitude": 0.25,
        "x_action_diff_penalty_exponent": 3.333,
        "z_action_diff_penalty_magnitude": 0.25,
        "z_action_diff_penalty_exponent": 5.0,
        "yawrate_action_diff_penalty_magnitude": 0.25,
        "yawrate_action_diff_penalty_exponent": 3.33,
        "x_absolute_action_penalty_magnitude": 0.05,
        "x_absolute_action_penalty_exponent": 0.3,
        "z_absolute_action_penalty_magnitude": 0.4,
        "z_absolute_action_penalty_exponent": 1.0,
        "yawrate_absolute_action_penalty_magnitude": 0.4,
        "yawrate_absolute_action_penalty_exponent": 2.0,
    }
