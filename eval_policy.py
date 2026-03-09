import numpy as np
from crowd_sim.envs.crowd_sim import CrowdSim
from config.config import Config as NavConfig
from omegaconf import DictConfig
from training_utils import load_wmpc_config, trial, print_result, load_cadrl_config
import torch
from torchrl.envs.utils import set_exploration_type, ExplorationType
from crowd_nav.policy.cadrl import CADRL
from crowd_nav.policy.wnum_mpc import WNumMPC
from training_utils import get_setting


def eval_policy(dict_config: DictConfig, visualize: bool, network_path: str | None = None) -> None:
    config: NavConfig = NavConfig(dict_config)

    np.random.seed(0)
    torch.manual_seed(0)
    env = CrowdSim(seed=0)

    # configure network
    env.configure(config)
    env.set_phase("test")

    # setting parameters
    if network_path is not None:
        if isinstance(env.robot.policy, CADRL):
            env.robot.policy.model.load_state_dict(torch.load(network_path, map_location=torch.device("cpu")))
            env.robot.policy.set_device('cpu')
            env.sync_policy_setting()
        elif isinstance(env.robot.policy, WNumMPC):
            env.robot.policy.model_predictor.wnum_selector.model.load_state_dict(torch.load(network_path))

    # rollout episodes
    episode_num: int = config.env.eval_episodes
    if isinstance(env.robot.policy, WNumMPC) and network_path is not None:
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            result: dict = trial(env, episode_num=episode_num, print_info=True, visualize=visualize)
    else:
        with torch.no_grad():
            result: dict = trial(env, episode_num=episode_num, print_info=True, visualize=visualize)

    #env.reset()
    print_result(result)
    del env


def main():
    policy_setting, human_num = get_setting()
    if policy_setting == "cadrl":  # CADRL
        network_path: str = "./models/CADRL/human_{}/rl_model_best.pth".format(human_num)
        d_conf: DictConfig = load_cadrl_config(cadrl_param_name="default")

    elif policy_setting == "vanilla_mpc" or policy_setting == "orca":  # Vanilla MPC or ORCA
        network_path: str | None = None
        mpc_param: str = "vanilla_mpc_H{}".format(human_num)
        d_conf: DictConfig = load_wmpc_config(mpc_param_name=mpc_param, training_param="default", use_nn="no_use")

    elif policy_setting == "mean_mpc":  # T-MPC
        network_path: str | None = None
        mpc_param: str = "mean_mpc_H{}".format(human_num)
        d_conf: DictConfig = load_wmpc_config(mpc_param_name=mpc_param, training_param="default", use_nn="no_use")

    else:  # WNumMPC
        split_num = 5 if human_num <= 4 else 3
        mpc_param: str = "wnum_mpc_H{}".format(human_num)
        training_param: str = "h128"
        network_path: str = "./models/ww_human_{}/WNumPPO_{}_mean/best.pth".format(human_num, training_param)

        use_nn: str = "no_use" if network_path is None else "use"
        d_conf: DictConfig = load_wmpc_config(mpc_param_name=mpc_param, training_param=training_param, use_nn=use_nn)

    eval_policy(d_conf, False, network_path)


def run_evaluation(policy_setting: str, human_num: int, eval_state: str = None, result_file: str = None):
    if policy_setting == "cadrl":  # CADRL
        network_path: str = "./models/CADRL/human_{}/rl_model_best.pth".format(human_num)
        d_conf: DictConfig = load_cadrl_config(cadrl_param_name="default")

    elif policy_setting == "vanilla_mpc" or policy_setting == "orca":  # Vanilla MPC or ORCA
        network_path: str | None = None
        mpc_param: str = "vanilla_mpc_H{}".format(human_num)
        d_conf: DictConfig = load_wmpc_config(mpc_param_name=mpc_param, training_param="default", use_nn="no_use")

    elif policy_setting == "mean_mpc":  # T-MPC
        network_path: str | None = None
        mpc_param: str = "mean_mpc_H{}".format(human_num)
        d_conf: DictConfig = load_wmpc_config(mpc_param_name=mpc_param, training_param="default", use_nn="no_use")

    else:  # WNumMPC
        mpc_param: str = "wnum_mpc_H{}".format(human_num)
        training_param: str = "h128"
        network_path: str = "./models/ww_human_{}/WNumPPO_{}_mean/best.pth".format(human_num, training_param)
        use_nn: str = "no_use" if network_path is None else "use"
        d_conf: DictConfig = load_wmpc_config(mpc_param_name=mpc_param, training_param=training_param, use_nn=use_nn)

    d_conf["sim"]["robot_policy"] = policy_setting
    d_conf["sim"]["human_num"] = human_num
    if eval_state is not None and result_file is not None:
        d_conf["eval_states"] = eval_state
        d_conf["result_file"] = result_file

    eval_policy(d_conf, False, network_path)


if __name__ == "__main__":
    main()
