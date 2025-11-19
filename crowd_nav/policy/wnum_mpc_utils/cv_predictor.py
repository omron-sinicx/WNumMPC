import copy
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional
from omegaconf import DictConfig
from crowd_nav.policy.wnum_mpc_utils.nn_module import WNumNNSelector
from crowd_nav.policy.wnum_mpc_utils.wnum_utils import to_numpy, WNumPolicyObservation, to_WnumPolicyObservation
from config.config import Config
from crowd_sim.envs.utils.state import BallbotState


class Target:
    def __init__(self, target_wnums: np.ndarray | None):
        self.wnums = target_wnums

    def is_None(self) -> bool:
        return self.wnums is None


class CV(object):
    def __init__(self):
        self.wnum_selector: WNumNNSelector | None = None
        self.observed_state_used: WNumPolicyObservation = None  # wnum決定時に用いたtrajectoryとstate
        self.observed_state: WNumPolicyObservation = None
        self.dt = None
        self.prediction_horizon = None
        self.wrap = np.vectorize(self._wrap)
        self.target: Target = Target(None)
        self.wnum_id: torch.Tensor = None
        self.flag_change_target_wnum: bool = False  # 次のstepでwinding numberを変更するか否か
        self.w_score_percents: list[float] = []  # winding numberの評価値の割合
        self.self_states: list[np.ndarray] = []  # stateのリスト
        self.sample_log_probs: torch.Tensor = torch.Tensor([0.0])

    def set_params(self, config: Config) -> None:
        dict_conf: DictConfig = config.policy_config['params']
        self.dt = dict_conf['dt']
        self.prediction_horizon = dict_conf['prediction_horizon']
        self.rollout_steps = int(np.ceil(self.prediction_horizon / self.dt))
        self.prediction_length = int(np.ceil(self.prediction_horizon / self.dt)) + 1
        self.history_length = dict_conf['history_length']

        self.q_obs = dict_conf["mpc_param"]['cost']['q']['obs']
        self.q_goal = dict_conf["mpc_param"]['cost']['q']['goal']
        self.q_wind = dict_conf["mpc_param"]['cost']['q']['wind']

        self.sigma_h = dict_conf["mpc_param"]['cost']['sigma']['h']
        self.sigma_s = dict_conf["mpc_param"]['cost']['sigma']['s']
        self.sigma_r = dict_conf["mpc_param"]['cost']['sigma']['r']
        self.sigma_w = self.sigma_s

        # Enviornment size
        self.min_x = config.min_x
        self.max_x = config.max_x
        self.min_y = config.min_y
        self.max_y = config.max_y

        # Normalization factors for Q weights
        self.q_goal_norm = np.square(2 / float(self.prediction_horizon))

        # # Empirically found
        q_wind_norm = 0.1 * np.deg2rad(350)
        self.q_wind_norm = np.square(q_wind_norm)
        self.q_obs_norm = np.square(0.5)

        # Normalized weights
        self.Q_obs = self.q_obs / self.q_obs_norm
        self.Q_goal = self.q_goal / self.q_goal_norm
        self.Q_discrete = self.q_wind / self.q_wind_norm
        self.Q_dev = 0

        self.log_cost = dict_conf['log_cost']
        self.discrete_cost_type = dict_conf["mpc_param"]['cost']['discrete_cost_type']  # TODO: remove (only "winding")

        self.select_obs = dict_conf["mpc_param"]['cost']['q_select']['obs'] / self.q_obs_norm
        self.select_goal = dict_conf["mpc_param"]['cost']['q_select']['goal'] / self.q_goal_norm
        self.select_wind = dict_conf["mpc_param"]['cost']['q_select']['wind'] / self.q_wind_norm
        self.viz_percent = dict_conf["viz_percent"]
        self.plot_obs_cost = dict_conf["obs_plot"]

        self.set_target_winding_num: bool = dict_conf["mpc_param"]["set_target_winding_num"]  # 事前にwinding numberを決め打ちするか否か

        self.split_num: int = dict_conf["mpc_param"]["split_num"]
        self.select_span: int = dict_conf["mpc_param"]["select_span"]
        self.randomize_select_step: bool = dict_conf["mpc_param"]["randomize_select_step"]

        self.step_count: int = 0
        if self.randomize_select_step:
            self.step_count: int = np.random.randint(0, self.select_span)

        self.use_NN: bool = dict_conf["mpc_param"]["use_nn"].flag

        self.wnum_selector: WNumNNSelector = None
        self.wnum_id: torch.Tensor = None

        if self.use_NN:
            self.human_num = config.sim.human_num
            input_size: int = (config.sim.human_num * 7 + 5)  # (H*7+5)
            output_size: int = 2 * config.sim.human_num  # output_shape: (2*H)
            self.wnum_selector: WNumNNSelector = WNumNNSelector(
                dict_conf["training_param"],
                input_size=input_size,
                out_size=output_size,
                human_num=config.sim.human_num
            )

    def reset(self, rng: Optional[np.random.Generator]=None) -> None:
        self.target = Target(None)
        self.w_score_percents = []
        self.self_states = []
        if self.randomize_select_step:
            if rng is not None:
                self.step_count: int = rng.integers(0, self.select_span)
            else:
                self.step_count: int = np.random.randint(0, self.select_span)
        else:
            self.step_count: int = 0

    def get_predictions(self, trajectory, actions):  # actionの情報使ってない (最後にrepeatしてるだけ)
        # trajectory: episode_num x (H+1) x 5 (px, py, vx, vy, r)
        velocity = trajectory[None, -1, 0:, 2:4]  # 1 x H x 2  # (vx, vy)
        init_pos = trajectory[None, -1, 0:, 0:2]  # 1 x H x 2  # (px, py)
        steps = 1 + np.arange(self.prediction_length, dtype=float)[:, None, None]  # T' x 1 x 1

        steps = np.multiply(velocity, steps) * self.dt  # T' x H x 2
        steps = (init_pos + steps)[None, None]  # N x S x T' x H x 2
        steps = np.concatenate((steps[:, :, :-1], self.predict_velocity(steps)), axis=-1)  # N x S x T' x (1+H) x 4
        # [px, py, vx, vy]
        steps = np.repeat(steps, actions.shape[0], axis=0)
        return steps

    def predict_velocity(self, steps):
        return (steps[:, :, 1:] - steps[:, :, :-1]) / self.dt

    def _eval_actions(self, predictions, state, actions, goal, target: Target, select_flag: bool):
        # evaluate actions
        c_goal = self.goal_cost(state, actions, goal, select_flag)
        c_obs = self.obstacle_cost(actions, predictions, select_flag)
        c_discrete, wnums = self.discrete_cost(state, actions, predictions, target, select_flag)
        c_predictor = self.predictor_cost(state, actions, predictions)
        c_total = c_goal + c_obs + c_discrete + c_predictor
        return c_total, [c_obs, c_goal, c_discrete], wnums

    def select_action_from_target(self, prediction, state, actions, goal, target: Target):
        scores, each_scores, wnums = self._eval_actions(prediction, state, actions, goal, target, True)

        optimal_id = np.argmin(np.mean(scores, axis=-1))
        action = actions[optimal_id].reshape(1, actions.shape[1], actions.shape[2])

        return action, each_scores, scores

    def select_target_winding_number(self, predictions, state, actions, goal):  # 全探索して最適なwinding numberを選択
        H = predictions.shape[-2]  # human num

        def base_n(num: int, n=3) -> list[int]:
            ans = []
            while num:
                ans.append(num % n)
                num //= n
            while len(ans) < H:
                ans.append(0)
            return ans[::-1]

        def num_to_w(num: int, n=3) -> list[float]:
            tmp = base_n(num, n)
            vals = np.linspace(-1.0, 1.0, num=n)
            return [vals[tp] for tp in tmp]

        tgt_wnums = [np.array(num_to_w(bit, self.split_num)).reshape(1, -1) for bit in range(self.split_num ** H)]
        targets, scores, each_scores, scores_list = [], [], [], []
        for target_wnums in tgt_wnums:  # 回転数の組み合わせを全探索
            target = Target(target_wnums)
            action, _, _ = self.select_action_from_target(predictions, state, actions, goal, target)
            c_total, _, _ = self._eval_actions(predictions[0:1], state, action, goal, Target(None), False)  # actionの評価
            scores.append(c_total[0][0])
            targets.append(target_wnums)

        optimal_id = np.argmin(np.array(scores))
        return targets[optimal_id]  # target_wnums

    def plot_obs_field(self, state: np.ndarray, predictions: np.ndarray, file_name: str):
        def scalar_field(x, y) -> float:
            T = 1
            dummy_actions = np.repeat(np.array([x, y, 0.0, 0.0]).reshape(1, -1), T).reshape(1, T, 4)
            dummy_state = copy.deepcopy(state)
            dummy_state[0] = np.array([x, y, 0.0, 0.0, state[0, -1]])
            obs_cost = self.obstacle_cost(dummy_actions, predictions[0:1, :, 0:T, :, :], False)
            return obs_cost.sum()

        # プロット範囲の設定
        x_range = np.linspace(-3, 3, 50)
        y_range = np.linspace(-3, 3, 60)

        # メッシュグリッドの生成
        X, Y = np.meshgrid(x_range, y_range)

        # スカラー場の値を計算
        Z = np.array([[scalar_field(X[i, j], Y[i, j]) for j in range(X.shape[1])] for i in range(X.shape[0])])

        # 等高線プロット
        fig, ax = plt.subplots(figsize=(14, 11))
        pcm = ax.pcolormesh(X, Y, Z, cmap='viridis', shading='auto')

        # グラフの装飾（必要に応じて変更）
        plt.axis('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('obs cost')
        plt.colorbar(pcm, ax=ax, label='obs cost')

        robot_info = [(state[0][0], state[0][1], state[0][-1])]
        human_info = [(s[0], s[1], s[-1]) for s in state[1:]]
        human_r = state[1][-1]
        human_infos_pred = np.array([predictions[0, 0, :, i, 0:2] for i in range(predictions.shape[-2])])
        # 円を描く
        for x, y, radius in robot_info:
            circle = plt.Circle((x, y), radius, fill=False, color='b')
            ax.add_patch(circle)

        for x, y, radius in human_info:
            circle = plt.Circle((x, y), radius, fill=False, color='r')
            ax.add_patch(circle)

        for human_pred in human_infos_pred:
            for pred in human_pred:
                circle = plt.Circle((pred[0], pred[1]), human_r, fill=False, color='grey')
                ax.add_patch(circle)

        # 画像として保存
        plt.savefig(file_name)
        plt.close(fig)

    def predict(self, trajectory: np.ndarray, state: np.ndarray, self_state: BallbotState, actions: np.ndarray, goal,
                plot_info: dict):
        # (T x (1+H) x 5), ((1+H) x 5), (N x T' x 2), (2, )
        # (trajectory: traj of all agents, actions: action set, state: states of all agents)
        predictions = self.get_predictions(trajectory, actions)  # N x S x T' x (H+1) x 4

        self.self_states.append(to_numpy(self_state))
        predictions = predictions[:, :, :, 1:, :]  # N x S x T' x H x 4
        self.observed_state = to_WnumPolicyObservation(np.array(self.self_states), trajectory, 1)

        # select action and eval each action
        if self.set_target_winding_num:
            if self.target.wnums is None or (self.step_count % self.select_span) == 0:
                # select target winding number
                self.w_score_percents = []
                self.observed_state_used = copy.deepcopy(self.observed_state)
                if self.use_NN:
                    self.target.wnums, self.wnum_id, self.sample_log_probs = self.wnum_selector.select_target_winding_number(self.observed_state_used)
                else:
                    self.target.wnums = self.select_target_winding_number(predictions, state, actions, goal)[0]
            _target = Target(np.array([self.target.wnums]))
            # select action from target winding number
            _, each_scores, c_total = self.select_action_from_target(predictions, state, actions, goal, _target)
        else:
            # select action and eval each action without target winding number
            c_total, each_scores, _ = self._eval_actions(predictions, state, actions, goal, Target(None), False)

        # costs -> best action
        best_action_idx = np.argmin(np.mean(c_total, axis=-1))
        best_action = actions[best_action_idx][0]

        o_scores, g_scores, w_scores = (np.mean(each_scores[0], axis=-1), np.mean(each_scores[1], axis=-1)
                                        , np.abs(np.mean(each_scores[2], axis=-1)))
        best_score = o_scores[best_action_idx] + g_scores[best_action_idx] + w_scores[best_action_idx]
        score_percent = "{} | {:.4g}, {:.4g}, {:.4g}".format(best_action_idx,
                                                             o_scores[best_action_idx] / best_score,
                                                             g_scores[best_action_idx] / best_score,
                                                             w_scores[best_action_idx] / best_score)
        best_action_o = actions[np.argmin(o_scores)][0]
        best_action_w = actions[np.argmin(w_scores)][0]
        best_action_g = actions[np.argmin(g_scores)][0]
        self.w_score_percents.append(w_scores[best_action_idx] / best_score)

        if self.plot_obs_cost and plot_info["plot_flag"]:
            directory = "./plot_data/obs_cost/episode{}".format(plot_info["eps_id"])
            os.makedirs(directory, exist_ok=True)  # log保存用のディレクトリを作成
            file_name = directory + "/obs_step{}".format(self.step_count)
            self.plot_obs_field(state, predictions, file_name)

        if self.viz_percent:
            print(score_percent)
        self.step_count += 1
        self.flag_change_target_wnum = (self.step_count % self.select_span == 0)

        info: dict = {
            "best_action": best_action,
            "best_action_idx": best_action_idx,
            "score_percent": score_percent,
            "w_scores": w_scores.reshape(-1, 1),
            "g_scores": g_scores.reshape(-1, 1),
            "o_scores": o_scores.reshape(-1, 1),
            "target_wnums": None if not self.set_target_winding_num else self.target.wnums,
            "best_action_o": best_action_o,
            "best_action_w": best_action_w,
            "best_action_g": best_action_g,
        }

        return predictions, c_total, actions, info

    def predictor_cost(self, state, actions, predictions):
        return np.array([0.0])

    def goal_cost(self, state, actions, goal, select: bool):
        """ゴールとの距離の変化量"""
        dist = np.linalg.norm(goal[None, None] - actions[:, :, :2], axis=-1)  # (N x T')

        states = np.ones(dist.shape[0]) * np.linalg.norm(goal - state[0, :2])
        diff = np.diff(np.concatenate([states[:, None], dist], axis=-1), axis=-1)
        diff_sum = np.abs(np.sum(np.sign(diff), axis=-1))

        for i in range(len(diff_sum)):
            if diff_sum[i] != (dist.shape[-1]):
                min_dist = np.min(dist[i])
                min_index = np.argmin(dist[i])
                for j in range(min_index, len(dist[i])):
                    dist[i][j] = min_dist

        cost = np.sum(dist, axis=-1)[:, None]
        cost *= 4
        if cost.shape[0] != 1:
            cost = (cost - np.min(cost))

        if select:
            return self.select_goal * cost
        else:
            return self.Q_goal * cost

    def obstacle_cost(self, actions, predictions, select: bool):
        """
        Cost using 2D Gaussian around obstacles
        """
        # actions.shape : N x T x 4 (px, py, vx, vy)
        dx = actions[:, None, :, None, 0] - predictions[:, :, :, :, 0]  # N x S x T' x H
        dy = actions[:, None, :, None, 1] - predictions[:, :, :, :, 1]  # N x S x T' x H

        obs_theta = np.arctan2(predictions[:, :, :, :, 3], predictions[:, :, :, :, 2])  # N x S x T' x H
        # Checking for static obstacles
        static_obs = (np.linalg.norm(predictions[:, :, :, :, 2:4], axis=-1) < -0.01)  # N x S x T' x H
        # Alpha calculates whether ego agent is in front or behind "other agent"
        alpha = self.wrap(np.arctan2(dy, dx) - obs_theta + np.pi / 2.0) <= 0  # N x S x T' x H

        # Sigma values used to create 2D gaussian around obstacles for cost penalty
        sigma = np.where(alpha, self.sigma_r, self.sigma_h)
        sigma = static_obs + np.multiply(1 - static_obs, sigma)  # N x S x T' x H
        sigma_s = 1.0 * static_obs + self.sigma_s * (1 - static_obs)  # N x S x T' x H

        # Variables used in cost_obs function based on sigma and obs_theta
        a = np.cos(obs_theta) ** 2 / (2 * sigma ** 2) + np.sin(obs_theta) ** 2 / (2 * sigma_s ** 2)
        b = np.sin(2 * obs_theta) / (4 * sigma ** 2) - np.sin(2 * obs_theta) / (4 * sigma_s ** 2)
        c = np.sin(obs_theta) ** 2 / (2 * sigma ** 2) + np.cos(obs_theta) ** 2 / (2 * sigma_s ** 2)

        cost = np.exp(-(a * (dx ** 2) + (2 * b * dx * dy) + c * (dy ** 2)))  # N x S x T' x H
        cost = np.mean(cost, axis=3)

        # Add boundary cost
        H = predictions.shape[-1]
        mx = np.minimum(actions[:,:,0]-self.min_x, self.max_x - actions[:,:,0]) # N x T
        cost += np.exp(-2*mx*abs(mx)/self.sigma_w**2)[:,None,:]/H
        my = np.minimum(actions[:,:,1]-self.min_y, self.max_y - actions[:,:,1]) # N x T
        cost += np.exp(-2*my*abs(my)/self.sigma_w**2)[:,None,:]/H

        cost = np.sum(cost, axis=-1)

        if select:
            return self.select_obs * (cost ** 2)
        else:
            return self.Q_obs * (cost ** 2)

    def discrete_cost(self, state, actions, predictions, target: Target, select: bool):
        # costs of winding number, (1+H) x 5, N x T' x 4, N x S x T' x H x 4
        #assert target.wnums.shape[0] == 1 and target.wnums.shape[1] == 2 * self.human_num
        if self.use_NN:
            target_wnum = target.wnums[:,:self.human_num]
        else:
            target_wnum = target.wnums

        N = actions.shape[0]  # N: actions num (discrete num)
        S = predictions.shape[1]
        state_ = np.tile(state[None, None, None, None, 0, :2] - state[None, None, None, 1:, :2], (N, S, 1, 1, 1))
        predictions = predictions[0:N]  # N x S x T' x H x 2になってる
        dxdy = np.concatenate((state_, actions[:, None, :, None, :2] - predictions[:, :, :, :, :2]), axis=2)
        # human_vel = np.linalg.norm(dxdy[:, :, 0, :, 0:2], axis=-1)  # N x S x T' x H
        thetas = np.arctan2(dxdy[:, :, :, :, 1], dxdy[:, :, :, :, 0])  # N x S x T' x H
        d_thetas = self.wrap(thetas[:, :, 1:] - thetas[:, :, :-1])
        # H: human num, T': prediction num (horizon num)

        if target_wnum is not None and target_wnum[0] is not None:
            w_nums_pre_human = (np.sum(d_thetas, axis=2)) / (2 * math.pi)  # N x S x H
            target_wnum = np.repeat(target_wnum, S, axis=0).reshape(1, S, -1)  # S x H
            target_wnum = np.repeat(target_wnum, N, axis=0)  # N x S x H

            w_nums = w_nums_pre_human - target_wnum
            w_nums = w_nums ** 2

            if self.use_NN:
                wnum_weight = target.wnums[0,self.human_num:]
                wnum_weight = 0.5 * wnum_weight + 0.5
                winding_nums = np.mean(w_nums * wnum_weight[None,None,:], axis=-1)  # N x S
            else:
                winding_nums = np.mean(w_nums, axis=-1)  # N x S
            winding_nums = winding_nums - np.min(winding_nums)

            if select:
                return self.select_wind * winding_nums, target_wnum
            else:
                return self.Q_discrete * winding_nums, target_wnum

        w_nums_pre_human = np.abs(np.sum(d_thetas, axis=2)) / (2 * math.pi)  # N x S x H
        wnums = np.sum(d_thetas, axis=2) / (2 * math.pi)  # N x S x H
        wnums = (wnums > 0.0).astype(float) * 2.0 - 1.0

        # 各actionについてのwinding_number
        winding_nums = w_nums_pre_human ** 2
        winding_nums = np.mean(winding_nums, axis=-1)  # N x S

        if select:
            return -self.select_wind * winding_nums, wnums
        else:
            return -self.Q_discrete * winding_nums, wnums

    @staticmethod
    def _wrap(angle):  # keep angle between [-pi, pi]
        while angle >= np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
