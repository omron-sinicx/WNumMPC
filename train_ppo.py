import os
import copy
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from tensordict import TensorDict
from torchrl.objectives import ClipPPOLoss
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from typing import List, Optional
import torch
import torch.nn as nn
import tensordict
import numpy as np
from tqdm import tqdm
from crowd_sim.envs.crowd_sim import CrowdSim
from config.config import Config as NavConfig
from crowd_nav.policy.wnum_mpc import WNumMPC
from omegaconf import DictConfig
from crowd_nav.policy.wnum_mpc_utils.wnum_utils import convert_trajectory
from training_utils import load_wmpc_config, print_result, trial
from torchrl.data import BoundedTensorSpec, SamplerWithoutReplacement, ReplayBuffer
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tensordict.nn import TensorDictModule
from training_utils import get_setting
import multiprocess as mp
from torch.utils.tensorboard import SummaryWriter
from torchrl.data import LazyMemmapStorage
from crowd_nav.policy.wnum_mpc_utils.nn_module import WNumNetworkCritic
from torchrl.objectives.value import GAE
from torchrl.collectors.utils import split_trajectories



class Collector:
    def __init__(self, config: NavConfig, model_state_dict, episode_num: int):
        self.env: CrowdSim = CrowdSim(seed=0)
        self.env.configure(config)
        if not isinstance(self.env.robot.policy, WNumMPC):
            raise ValueError("This is not WNumMPC")

        self.env.robot.policy.model_predictor.wnum_selector.model.load_state_dict(
            copy.deepcopy(model_state_dict)
        )
        self.env.robot.policy.model_predictor.wnum_selector.model.to("cpu")
        self.episode_num: int = episode_num

    def __call__(self, seed: int, data):
        obs, observed_ids = self.env.reset(seed=seed)
        for _ in range(self.episode_num):
            while True:
                with set_exploration_type(ExplorationType.RANDOM):
                    action = self.env.robot.act(obs, observed_ids, self.env.is_goaled_list)
                    next_obs, rew, terminated, truncated, info = self.env.step(action)

                obs = next_obs
                if terminated or truncated:
                    obs, observed_ids = self.env.reset()
                    break

        history: List[TensorDict] = []  # (T1 + T2 + ..., td(dim))
        for tmp in self.env.rb:  # historyの結合 (agent別になってる)
            history += tmp

        for h in history:
            h["observation"] = convert_trajectory(h["observation"])
            h["next"]["observation"] = convert_trajectory(h["next"]["observation"])

        # history: List[TensorDict] = []  # (Agent, td(Ti, dim))
        # for tmp in self.env.rb:  # historyの結合 (agent別になってる)
        #     for t in tmp:
        #         t["observation"] = convert_trajectory(t["observation"])
        #         t["next"]["observation"] = convert_trajectory(t["next"]["observation"])
        #
        #     tmp_dict: TensorDict = tensordict.dense_stack_tds(tmp, 0)
        #     history.append(tmp_dict)

        if data is not None:
            data.put(history)
            del self.env
            return None

        else:
            return history


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight)
        torch.nn.init.uniform_(m.bias, -1e-2, 1e-2)


def trial_network(config: NavConfig, episode_num: int, network_model, print_info: bool) -> dict:
    trial_env = CrowdSim(seed=0)
    trial_env.configure(config)
    trial_env.robot.policy.model_predictor.wnum_selector.model = copy.deepcopy(network_model)

    result: dict = trial(trial_env, episode_num=episode_num, print_info=print_info, visualize=False)
    del trial_env
    return result


def train_ppo(model_name: str, wmpc_config: DictConfig, training_param_name: str, logging: bool, trial_id: Optional[int]=None) -> dict[str, float | int]:
    np.random.seed(0)
    torch.manual_seed(0)

    wmpc_config.eval_episodes = 30  # for training
    config: NavConfig = NavConfig(wmpc_config)
    base_env: CrowdSim = CrowdSim(seed=0)
    base_env.configure(config, [])
    if not isinstance(base_env.robot.policy, WNumMPC):
        raise ValueError("This is not WNumMPC")

    human_num: int = base_env.human_num
    rew_lambda: float = wmpc_config["params"]["training_param"]["reward_param"]["rew_lambda"]

    if logging:
        dir_path = "./datas/tensorboard/ppo/ww_human_{}/".format(human_num) + "{}/".format(model_name) + "/" + training_param_name
        if rew_lambda != 0.0:
            dir_path += "_lambda{}".format(int(rew_lambda * 100))

        os.makedirs(dir_path, exist_ok=True)
        writer = SummaryWriter(log_dir=dir_path)

    elif trial_id is not None:
        dir_path = "./datas/tensorboard/ppo/optuna/ww_human_{}/".format(human_num) + "{}/".format(model_name) + "/" + training_param_name
        if rew_lambda != 0.0:
            dir_path += "_lambda{}".format(int(rew_lambda * 100))
        dir_path += "/trial{}".format(trial_id)

        os.makedirs(dir_path, exist_ok=True)
        writer = SummaryWriter(log_dir=dir_path)
    else:
        writer = None

    base_env.robot.policy.model_predictor.wnum_selector.enable_train()
    # base_env.robot.policy.model_predictor.wnum_selector.model.apply(init_weights)

    # params
    num_envs: int = 36
    training_span: int = num_envs

    eval_span: int = training_span * 35  # training_spanの倍数にすること
    save_span: int = eval_span  # eval_spanの倍数にすること

    ppo_parm = wmpc_config["params"]["training_param"]["ppo_param"]
    n_iter = ppo_parm["n_iter"]
    num_epoch = ppo_parm["num_epoch"]
    batch_size = ppo_parm["batch_size"]

    clip_epsilon = ppo_parm["clip_epsilon"]
    entropy_eps = ppo_parm["entropy_eps"]
    gamma = ppo_parm["gamma"]
    lmbda = ppo_parm["lmbda"]

    max_grad_norm = ppo_parm["max_grad_norm"]
    learning_rate = ppo_parm["learning_rate"]

    dev = torch.device("cpu")

    # Buffer
    buffer: ReplayBuffer = ReplayBuffer(
        storage=LazyMemmapStorage(max_size=batch_size * num_epoch * 100),
        batch_size=batch_size,  # 後から上書き可能
        sampler=SamplerWithoutReplacement(),
    )

    # Critic
    critic_input_size: int = base_env.robot.policy.model_predictor.wnum_selector.input_size
    in_key = "observation"

    critic_hidden_size: int = config.policy_config.params.training_param.nn_param.hidden_size
    critic = WNumNetworkCritic(input_size=critic_input_size, hidden_size=critic_hidden_size).to(dev)
    value_module = TensorDictModule(critic, [in_key], ["value"])
    critic_module = ValueOperator(
        module=critic,
        in_keys=[in_key],
    )

    # Loss Functions
    advantage_gae = GAE(
        gamma=gamma,
        lmbda=lmbda,
        value_network=critic_module,
        average_gae=False,
        # average_gae=True,
        time_dim=1
    )

    # PPO module
    loss_module = ClipPPOLoss(
        actor_network=base_env.robot.policy.model_predictor.wnum_selector.model,
        critic_network=critic_module,
        clip_epsilon=clip_epsilon,
        log_explained_variance=True,
        normalize_advantage=True,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )
    loss_module.to(dev)

    optim = torch.optim.Adam(loss_module.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, n_iter // num_envs, 0.0
    )

    max_reward = 0.0
    max_success = 0.0

    rollout_datas: list[TensorDict] = []

    for i in tqdm(range(n_iter // num_envs), desc="[TRAIN]"):  # training loop
        datas = mp.Queue()
        base_env.rb = [[] for _ in range(human_num + 1)]
        tmp_model_state_dict = copy.deepcopy(base_env.robot.policy.model_predictor.wnum_selector.model.state_dict())
        jobs = [
            mp.Process(target=Collector(config, tmp_model_state_dict, training_span // num_envs), args=(env_id, datas))
            for env_id in range(num_envs * i, num_envs * (i + 1))
        ]
        for job in jobs:
            job.daemon = True
            job.start()

        for _ in range(num_envs):
            rollout_datas.extend(datas.get())

        for job in jobs:
            job.join()

        if i % (training_span // num_envs) == 0:
            if False:
                if logging:
                    print("data is not enough, {}/{}".format(len(rollout_datas), batch_size))
            else:
                history_tmp: TensorDict = tensordict.dense_stack_tds(rollout_datas, dim=0)

                # padding and get mask
                padded_history = split_trajectories(history_tmp, done_key=("done"))
                mask_key = ("collector", "mask") if ("collector", "mask") in padded_history.keys(True, True) else "mask"
                flatten_mask = padded_history.get(mask_key).reshape(-1)

                # calc GAE
                with torch.no_grad():
                    advantage_gae(
                        padded_history,
                        params=loss_module.critic_network_params,
                        target_params=loss_module.target_critic_network_params,
                    )

                # get valid history (remove padding)
                padded_history = padded_history.reshape(-1)
                history = padded_history[flatten_mask]
                buffer.extend(history)

                loss_dicts: list[TensorDict] = []
                ave_loss = []

                # training loop
                for _ in range(num_epoch):
                    for batch in buffer:
                        loss_val_dict: TensorDict = loss_module(batch)

                        loss_value = loss_val_dict["loss_objective"] + loss_val_dict["loss_critic"] + loss_val_dict["loss_entropy"]
                        loss_dicts.append(loss_val_dict.clone().detach())
                        ave_loss.append(loss_value.clone().detach())

                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)

                        optim.step()
                        optim.zero_grad()

                if writer is not None:
                    loss_dict_mean: TensorDict = tensordict.dense_stack_tds(loss_dicts, dim=0).mean(dim=0)
                    writer.add_scalar("loss", torch.mean(torch.Tensor(ave_loss)), i * training_span)
                    writer.add_scalar("ppo/loss", torch.mean(torch.Tensor(ave_loss)), i * training_span)
                    writer.add_scalar("ppo/loss_objective", loss_dict_mean["loss_objective"], i * training_span)
                    writer.add_scalar("ppo/loss_critic", loss_dict_mean["loss_critic"], i * training_span)
                    writer.add_scalar("ppo/loss_entropy", loss_dict_mean["loss_entropy"], i * training_span)
                    writer.add_scalar("ppo/ESS", loss_dict_mean["ESS"], i * training_span)
                    writer.add_scalar("ppo/entropy", loss_dict_mean["entropy"], i * training_span)
                    writer.add_scalar("ppo/clip_fraction", loss_dict_mean["clip_fraction"], i * training_span)
                    writer.add_scalar("ppo/lr", optim.param_groups[0]["lr"], i * training_span)

                buffer.empty()  # bufferのclear
                rollout_datas = []
                scheduler.step()

        result = None
        if (writer is not None and (i * num_envs) % eval_span == 0) or (i == (n_iter // num_envs) - 1):
            with set_exploration_type(ExplorationType.MODE), torch.no_grad():
                episode_num: int = config.env.eval_episodes
                if trial_id is not None and (not (i == (n_iter // num_envs) - 1)):
                    episode_num = min(episode_num, 20)

                result: dict = trial_network(config, episode_num, base_env.robot.policy.model_predictor.wnum_selector.model, logging)

            if logging:
                print_result(result)

                if rew_lambda != 0.0:
                    model_path = "./models/ww_human_{}/{}_lambda{}".format(base_env.human_num, model_name, int(rew_lambda*100))
                else:
                    model_path = "./models/ww_human_{}/{}".format(base_env.human_num, model_name)
                os.makedirs(model_path, exist_ok=True)

                if result["success_rate"] >= 1.0 or (i * num_envs) % save_span == 0:
                    if logging:
                        torch.save(base_env.robot.policy.model_predictor.wnum_selector.model.state_dict(),
                                   model_path + "/epi{}.pth".format(i * training_span))
                    print("ave reward: {}, model saved!".format(result["ave_reward"]))

                if max_success < result["success_rate"] or (
                        max_reward < result["ave_wnum_reward"] and max_success == result["success_rate"]):
                    if logging:
                        torch.save(base_env.robot.policy.model_predictor.wnum_selector.model.state_dict(),
                                   model_path + "/best.pth")
                    max_reward = result["ave_wnum_reward"]
                    max_success = result["success_rate"]
                    print("[MAX] ave reward all: {}, model saved!".format(result["ave_wnum_reward"]))

            if writer is not None:
                writer.add_scalar("eval/success_rate", result["success_rate"], i * training_span)
                writer.add_scalar("eval/ave_reward", result["ave_reward"], i * training_span)
                writer.add_scalar("eval/ave_nav_time", result["ave_nav_time"], i * training_span)
                writer.add_scalar("eval/path_length_ave", result["path_length_ave"], i * training_span)
                writer.add_scalar("eval/ave_wnum_reward", result["ave_wnum_reward"], i * training_span)
                writer.add_scalar("eval/CHC", result["CHC"], i * training_span)
                writer.add_scalar("eval/ave_wnum_percent", result["ave_wnum_percent"], i * training_span)
                writer.add_scalar("eval/ave_extra_time_to_goals", result["ave_extra_time_to_goals"], i * training_span)

    if logging:
        writer.close()

    return result




if __name__ == "__main__":
    policy_setting, human_num = get_setting()

    if policy_setting != "wnum_mpc":
        raise ValueError("set robot_policy as wnum_mpc in experiment_param.yaml")

    # params setting
    # n_iter: int = 40000
    # training_param: str = "h32" if human_num <= 4 else "h64"
    # mpc_param: str = "rot_real_wnum_mpc_H{}_margin".format(human_num)
    # batch_size: int = 128 if human_num >= 4 else 96
    # model_name: str = "WNumPPO_{}_margin".format(training_param)

    training_param: str = "h32" if human_num <= 4 else "h64"
    mpc_param: str = "rot_real_wnum_mpc_H{}".format(human_num)
    model_name: str = "WNumPPO_{}".format(training_param)

    # train wnum_mpc by PPO
    # wmpc_config: DictConfig = load_wmpc_config(mpc_param_name=mpc_param, training_param=training_param, use_nn="use")
    # wmpc_config.eval_episodes = 30  # for time saving
    #wmpc_config.eval_states = "./datas/eval_states/train-{}-30.npy".format(human_num+1)
    # if hasattr(wmpc_config, "eval_states"):
    #     del wmpc_config.eval_states

    # train wnum_mpc by PPO
    # train_ppo(
    #     n_iter,
    #     batch_size,
    #     15,
    #     model_name=model_name,
    #     wmpc_config=wmpc_config
    # )

    wmpc_config: DictConfig = load_wmpc_config(mpc_param_name=mpc_param, training_param=training_param, use_nn="use")
    if hasattr(wmpc_config, "eval_states"):
        del wmpc_config.eval_states

    result_data = train_ppo(model_name=model_name, wmpc_config=wmpc_config, training_param_name=training_param, logging=True)

