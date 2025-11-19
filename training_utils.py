import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import hydra
from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.info import ReachGoal, Collision, Timeout, EpisodeInfo, Nothing, Danger
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.ballbot import BallBot
from tqdm import tqdm
import yaml, json


def load_wmpc_config(mpc_param_name: str = "set_tgt1", training_param: str = "default", use_nn: str = "no_use") -> DictConfig:
    exp_config_name = "experiment_param"
    config_path = "config"

    hydra.core.global_hydra.GlobalHydra.instance().clear()  # hydraの内部error対策
    hydra.initialize(config_path=config_path, version_base=None)
    dict_config = hydra.compose(
        config_name=exp_config_name,
        overrides=[
            "params/mpc_param=" + mpc_param_name,
            "params/training_param=" + training_param,
            "params/mpc_param/use_nn=" + use_nn
        ]
    )
    if not (dict_config.sim.robot_policy in ["wnum_mpc", "orca", "vanilla_mpc", "mean_mpc"]):
        raise ValueError("robot_policy in experimental_param.yaml is not wnum_mpc")

    print("[INFO] loading WNumMPC config")
    print("[INFO] exp_config is {}".format(exp_config_name))
    print("[INFO] mpc_config is {}".format(mpc_param_name))
    print("[INFO] training_param is {}".format(training_param))
    print("[INFO] load configs -> done")
    return dict_config


def load_cadrl_config(cadrl_param_name: str = "default") -> DictConfig:
    exp_config_name = "experiment_param"
    config_path = "config"

    hydra.core.global_hydra.GlobalHydra.instance().clear()  # hydraの内部error対策
    hydra.initialize(config_path=config_path, version_base=None)
    dict_config = hydra.compose(
        config_name=exp_config_name,
        overrides=[
            "cadrl_param=" + cadrl_param_name,
        ]
    )
    if dict_config.sim.robot_policy != "cadrl":
        raise ValueError("robot_policy in experimental_param.yaml is not cadrl")

    print("[INFO] loading CADRL config")
    print("[INFO] cadrl_param is {}".format(cadrl_param_name))
    print("[INFO] load configs -> done")
    return dict_config


def print_episode_info(episode_info):
    if isinstance(episode_info, ReachGoal):
        print('Success')
    elif isinstance(episode_info, Collision):
        print('Collision')
    elif isinstance(episode_info, Timeout):
        print('Time out')
    elif isinstance(episode_info, Nothing):
        print("Nothing")
    elif isinstance(episode_info, Danger):
        print("Danger")
    else:
        raise ValueError('Invalid end signal from environment: {}'.format(type(episode_info)))


def set_vizualize(env: CrowdSim) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel('x(m)', fontsize=16)
    ax.set_ylabel('y(m)', fontsize=16)
    plt.ion()
    plt.show()
    env.render_axis = ax


def print_result(result: dict) -> None:
    print("ave_reward: ", result["ave_reward"])
    print("ave_wnum_reward: ", result["ave_wnum_reward"])
    print("ave_nav_time: ", result["ave_nav_time"])
    print("path_length_ave: ", result["path_length_ave"])
    print("CHC: ", result["CHC"])
    print("ave_wnum_percent: ", result["ave_wnum_percent"])
    print("ave_extra_time_to_goals: ", result["ave_extra_time_to_goals"])
    print("percentiles of extra_time_to_goals (50,75,90)")
    print(" -> ", np.percentile(result["extra_time_to_goals"], [50, 75, 90]))
    print("========================")
    print("success_rate: ", result["success_rate"])
    print("collision_rate: ", result["collision_rate"])
    print(" -> collision case: ", result["collision_case"])
    print("timeout_rate: ", result["timeout_rate"])
    print(" -> timeout case: ", result["timeout_cases"])


def trial_single_episode(env: CrowdSim, visualize: bool = False, seed=None):
    obs, observed_ids = env.reset(seed=seed)
    done, episode_info = False, None
    rewards, wnum_rews, w_percents = [], [], []
    ave_w_reward = 0.0

    # for evaluation
    last_poses = [env.robot.get_full_state().get_position()] + [hum.get_full_state().get_position() for hum in
                                                                env.humans]
    last_angles = [env.robot.get_full_state().get_heading()] + [hum.get_full_state().get_heading() for hum in
                                                                    env.humans]
    CHCs = [0.0 for _ in range(env.human_num + 1)]
    paths = [0.0 for _ in range(env.human_num + 1)]
    global_time = 0.0
    extra_time_to_goals: list[float | None] = []

    while not done:
        if visualize:
            env.render()
        if type(env.robot) is BallBot:
            action = env.robot.act(obs, observed_ids, env.is_goaled_list)
        else:
            action: ActionXY = env.robot.act(obs)
        obs, rew, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        observed_ids: list[int] = info["observed_ids"]
        episode_info: EpisodeInfo = info["episode_info"]
        wnum_reward: float | None = info["wnum_reward"]
        wnum_percent: float = info["wnum_percent"]
        wnum_rewards_all: list = info["wnum_reward_all"]

        if len(wnum_rewards_all) != 0 and len(list(filter(None, wnum_rewards_all))) != 0:
            ave_w_reward += np.sum(list(filter(None, wnum_rewards_all))) / float(len(wnum_rewards_all))
        if wnum_reward is not None:
            wnum_rews.append(wnum_reward)
            w_percents.append(wnum_percent)
        if rew is not None:
            rewards.append(rew)

        # for evaluation
        global_time = env.global_time
        if done:
            extra_time_to_goals = info["extra_time_to_goals"]
        for i in range(env.human_num + 1):
            if i == 0:
                agent = env.robot
            else:
                agent = env.humans[i - 1]
            paths[i] = paths[i] + np.linalg.norm(agent.get_full_state().get_position() - last_poses[i])
            cur_angle = agent.get_full_state().get_heading()
            CHCs[i] = CHCs[i] + abs(cur_angle - last_angles[i])
            last_poses[i] = agent.get_full_state().get_position()
            last_angles[i] = cur_angle

    all_agent_info = {}
    path = paths[0]
    chc = CHCs[0]
    if env.config.sim.done_if_all_agents_reached:
        all_agent_info["mean_path"] = np.mean(paths)
        all_agent_info["mean_CHC"] = np.mean(CHCs)
        all_agent_info["ave_w_rewards_all"] = ave_w_reward
    return sum(rewards), sum(wnum_rews), episode_info, path, chc, global_time, np.average(w_percents), all_agent_info, extra_time_to_goals


def trial(env: CrowdSim, episode_num: int, print_info: bool = False, visualize: bool = False) -> dict:
    if visualize:
        set_vizualize(env)
    if isinstance(env.robot, BallBot) and env.robot.policy.model_predictor.wnum_selector is not None:
        env.robot.policy.model_predictor.wnum_selector.eval()  # eval

    # for evaluation
    chc_total, path_lengths, episode_rewards, wnum_rews, w_pers, extra_time_to_goals = [], [], [], [], [], []
    success_times, collision_times, timeout_times = [], [], []
    collision_cases, timeout_cases = [], []

    if hasattr(env.config.policy_config, "result_file"):
        result_file = env.config.policy_config.result_file
    else:
        result_file = None

    for k in tqdm(range(episode_num), desc="[EVAL]", disable=not print_info):
        rews, wrews, episode_info, path, chc, global_time, w_per, all_agent_info, etg_datas\
            = trial_single_episode(env, visualize, seed=k)
        # for evaluation
        path_lengths.append(all_agent_info["mean_path"] if any(all_agent_info) else path)
        chc_total.append(all_agent_info["mean_CHC"] if any(all_agent_info) else chc)
        wnum_rews.append(all_agent_info["ave_w_rewards_all"] if any(all_agent_info) else wrews)
        if any(all_agent_info):
            extra_time_to_goals += etg_datas
        else:
            extra_time_to_goals.append(etg_datas[env.robot.id])
        episode_rewards.append(rews)
        w_pers.append(w_per)

        if isinstance(episode_info, ReachGoal):
            success_times.append(global_time)
        elif isinstance(episode_info, Collision):
            collision_cases.append(k)
            collision_times.append(global_time)
            success_times.append(np.nan)
        elif isinstance(episode_info, Timeout):
            timeout_cases.append(k)
            timeout_times.append(env.time_limit)
            success_times.append(np.nan)
        #elif isinstance(episode_info, Nothing) or isinstance(episode_info, Danger):
        #    pass
        else:
            raise ValueError('Invalid end signal from environment: {}'.format(type(episode_info)))

    success_rate = np.sum(~np.isnan(np.array(success_times))) / episode_num
    collision_rate = len(collision_times) / episode_num
    timeout_rate = len(timeout_times) / episode_num

    if result_file != None:
        data = locals()
        result = {name: data[name] for name in [
            'chc_total', 'path_lengths', 'episode_rewards', 'wnum_rews',
            'w_pers', 'extra_time_to_goals',
            'success_times', 'collision_times', 'timeout_times',
            'collision_cases', 'timeout_cases', 'success_rate', 'collision_rate', 'timeout_rate']
        }
        with open(result_file, 'w') as file:
            json.dump(result, file)
    
    success_times = np.array(success_times)
    avg_nav_time = np.nanmean(success_times) if any(~np.isnan(success_times)) else env.time_limit
    result: dict = {
        "CHC": float(sum(chc_total) / episode_num),
        "path_length_ave": float(sum(path_lengths) / episode_num),
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "timeout_rate": timeout_rate,
        "collision_case": collision_cases, "timeout_cases": timeout_cases, "success_times": success_times,
        "ave_nav_time": float(avg_nav_time),
        "ave_reward": float(np.average(episode_rewards)),
        "ave_wnum_reward": float(np.average(wnum_rews)),
        "ave_wnum_percent": float(np.average(w_pers)),
        "ave_extra_time_to_goals": float(np.average(extra_time_to_goals)),
        "extra_time_to_goals": extra_time_to_goals
    }

    if isinstance(env.robot, BallBot) and env.robot.policy.model_predictor.wnum_selector is not None:
        env.robot.policy.model_predictor.wnum_selector.enable_train()  # train
    return result


def get_setting() -> tuple[str, int]:
    with open("./config/experiment_param.yaml", "r") as yml:
        config = yaml.safe_load(yml)
        return config["sim"]["robot_policy"], config["sim"]["human_num"]
