import copy
import numpy as np
from crowd_sim.envs.crowd_sim import CrowdSim
from config.config import Config as NavConfig
from omegaconf import DictConfig
from training_utils import load_wmpc_config, trial
import torch
import optuna
import yaml
import os
import multiprocess as mp
from omegaconf import OmegaConf


def f_trial(config: NavConfig) -> dict:
    np.random.seed(0)
    torch.manual_seed(0)
    env = CrowdSim(seed=0)
    env.configure(config)

    result: dict = trial(env, episode_num=config.env.eval_episodes, print_info=False, visualize=False)
    del env
    return result


def objective(dict_config: DictConfig, optuna_trial: optuna.Trial) -> float:
    nav_config: NavConfig = NavConfig(dict_config)
    q_obs = optuna_trial.suggest_float('q_obs', 0.0, 20.0)  # xの範囲指定
    q_goal = optuna_trial.suggest_float('q_goal', 0.0, 20.0)  # yの範囲指定
    nav_config.policy_config["params"]["mpc_param"]['cost']['q']['obs'] = q_obs
    nav_config.policy_config["params"]["mpc_param"]['cost']['q']['goal'] = q_goal

    if nav_config.policy_config.params.mpc_param.set_target_winding_num:
        q_goal_s = optuna_trial.suggest_float('q_goal_s', 0.0, 20.0)  # yの範囲指定
        q_obs_s = optuna_trial.suggest_float('q_obs_s', 0.0, 20.0)  # yの範囲指定
        nav_config.policy_config["params"]["mpc_param"]['cost']['q_select']["obs"] = q_obs_s
        nav_config.policy_config["params"]["mpc_param"]['cost']['q_select']["goal"] = q_goal_s

    result: dict = f_trial(nav_config)
    score = -result["success_rate"] * 400.0 + (result["ave_nav_time"] + result["path_length_ave"]) / 4.0
    return score


def optimize(dict_config: DictConfig, study_name: str, storage: str, trial_num: int) -> None:  # 並列化用の関数
    optuna_study: optuna.Study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    obj = lambda x: objective(dict_config, x)
    optuna_study.optimize(func=obj, n_trials=trial_num)


def optimize_param(mpc_param: str, train_param: str) -> None:
    wmpc_config: DictConfig = load_wmpc_config(mpc_param_name=mpc_param, training_param=train_param)
    n_trials = wmpc_config["optuna_trials"]

    study_name: str = "{}".format(mpc_param)

    # log保存用のディレクトリを作成
    os.makedirs("./datas/optuna/" + study_name, exist_ok=True)
    with open("./datas/optuna/{}/config.yaml".format(study_name), "w") as file:
        OmegaConf.save(wmpc_config, file)

    if not wmpc_config["multi_thread"]:
        study = optuna.create_study()
        obj = lambda x: objective(copy.deepcopy(wmpc_config), x)
        study.optimize(func=obj, n_trials=n_trials)

    else:  # 並列化
        concurrency = mp.cpu_count()  # max_cpuより使うCPUの数が多くないことを確認
        n_trials_per_cpu = n_trials // concurrency

        detabase_name = "optuna_{}.db".format(study_name)
        DATABASE_URI = 'sqlite:///' + detabase_name

        if os.path.exists("./" + detabase_name):  # dbが存在する場合
            os.remove("./" + detabase_name)  # 削除
            print("[INFO] remove previous detabase")

        # dbの作成
        optuna.create_study(study_name=study_name, storage=DATABASE_URI)
        print("[INFO] create detabase {}".format(detabase_name))

        # 並列化
        mp.set_start_method("spawn", force=True)
        workers = [mp.Process(target=optimize, args=(wmpc_config, study_name, DATABASE_URI, n_trials_per_cpu)) for _ in range(concurrency)]
        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()
        study = optuna.load_study(study_name=study_name, storage=DATABASE_URI)

    print("best params: {}".format(study.best_params))
    print("best value: {}".format(study.best_value))
    print("best trial: {}".format(study.best_trial))

    data: dict = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "best_trials": study.best_trials,
    }

    file_name = './datas/optuna/{}/optuna_result.yaml'.format(study_name)

    with open(file_name, 'w') as file:
        yaml.dump(data, file)

    fig = optuna.visualization.plot_contour(study, params=["q_obs", "q_goal"])
    fig.write_image("./datas/optuna/{}/optuna_result.png".format(study_name))

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image('./datas/optuna/{}/optimization_history.png'.format(study_name))

    fig = optuna.visualization.plot_slice(study)
    fig.write_image('./datas/optuna/{}/slice.png'.format(study_name))


if __name__ == '__main__':
    optimize_param("mean_mpc_H8", "default")

