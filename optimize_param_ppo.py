import copy
from omegaconf import DictConfig
from training_utils import load_wmpc_config, get_setting
from train_ppo import train_ppo
import optuna
import yaml
import os
import multiprocessing
from omegaconf import OmegaConf


def f_trial(dict_config: DictConfig, trial_id: int) -> dict:
    policy_setting = dict_config["sim"]["robot_policy"]
    human_num = dict_config["sim"]["human_num"]
    if policy_setting != "wnum_mpc":
        raise ValueError("set robot_policy as wnum_mpc in experiment_param.yaml")
    training_param: str = "h32" if human_num <= 4 else "h64"

    ## train wnum_mpc by PPO
    result_data = train_ppo(model_name="tmp_optuna", wmpc_config=dict_config, training_param_name=training_param, logging=False, trial_id=trial_id)
    return result_data


def objective(dict_config: DictConfig, optuna_trial: optuna.Trial) -> float:
    dict_config["params"]["training_param"]["ppo_param"]["n_iter"] = 20000

    # optuna params
    learning_rate = optuna_trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    entropy_eps = optuna_trial.suggest_float("entropy_eps", 1e-4, 1e-2, log=True)
    if dict_config["sim"]["human_num"] >= 6:
        batch_size = optuna_trial.suggest_categorical("batch_size", [1024, 2048, 4096])
    else:
        batch_size = optuna_trial.suggest_categorical("batch_size", [512, 1024, 2048])

    # setting
    dict_config["params"]["training_param"]["ppo_param"]["learning_rate"] = learning_rate
    dict_config["params"]["training_param"]["ppo_param"]["batch_size"] = batch_size
    dict_config["params"]["training_param"]["ppo_param"]["entropy_eps"] = entropy_eps

    trial_id: int = optuna_trial.number
    result: dict = f_trial(dict_config, trial_id)
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


def optimize_param() -> None:
    ## load default param
    policy_setting, human_num = get_setting()
    training_param: str = "h32" if human_num <= 4 else "h64"
    mpc_param: str = "wnum_mpc_H{}".format(human_num)
    wmpc_config: DictConfig = load_wmpc_config(mpc_param_name=mpc_param, training_param=training_param, use_nn="use")

    ## study setting
    study_name: str = "{}".format(mpc_param)
    rew_lambda: float = wmpc_config["params"]["training_param"]["reward_param"]["rew_lambda"]
    if rew_lambda != 0.0:
        study_name += "_lambda{}".format(int(rew_lambda*100))
    n_trials = wmpc_config["optuna_trials"]

    # log保存用のディレクトリを作成
    os.makedirs("./datas/optuna/ppo/" + study_name, exist_ok=True)
    with open("./datas/optuna/ppo/{}/config.yaml".format(study_name), "w") as file:
        OmegaConf.save(wmpc_config, file)

    if not wmpc_config["multi_thread"]:
        study = optuna.create_study()
        obj = lambda x: objective(copy.deepcopy(wmpc_config), x)
        study.optimize(func=obj, n_trials=n_trials)

    else:  # 並列化
        n_processes: int = 2
        n_trials_per_process = n_trials // n_processes

        detabase_name = "optuna_{}.db".format(study_name)
        DATABASE_URI = 'sqlite:///' + detabase_name

        if os.path.exists("./" + detabase_name):  # dbが存在する場合
            os.remove("./" + detabase_name)  # 削除
            print("[INFO] remove previous detabase")

        # dbの作成
        optuna.create_study(study_name=study_name, storage=DATABASE_URI)
        print("[INFO] create detabase {}".format(detabase_name))

        # 並列化
        workers = [
            multiprocessing.Process(target=optimize, args=(wmpc_config, study_name, DATABASE_URI, n_trials_per_process)) for _ in range(n_processes)
        ]

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

    file_name = './datas/optuna/ppo/{}/optuna_result.yaml'.format(study_name)
    with open(file_name, 'w') as file:
        yaml.dump(data, file)

    # plot result
    param_names = study.best_params.keys()
    num_params = len(param_names)
    base_size: int = 500

    fig = optuna.visualization.plot_contour(study)
    fig.write_image("./datas/optuna/ppo/{}/optuna_result.png".format(study_name), width=base_size*4, height=base_size*4, scale=2)

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image('./datas/optuna/ppo/{}/optimization_history.png'.format(study_name), width=base_size*4, height=base_size*3, scale=2)

    fig = optuna.visualization.plot_slice(study)
    slice_w, slice_h = 700, 800
    fig.write_image('./datas/optuna/ppo/{}/slice.png'.format(study_name),  width=slice_w*num_params, height=slice_h, scale=2)

    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_image('./datas/optuna/ppo/{}/parallel_coordinate.png'.format(study_name), width=base_size*4, height=base_size*3, scale=2)

    print("finish optimization")


if __name__ == '__main__':
    optimize_param()

