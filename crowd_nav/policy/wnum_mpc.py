import numpy as np
import copy
from typing import Optional
from crowd_nav.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY
from crowd_nav.policy.wnum_mpc_utils.cv_predictor import CV
from crowd_sim.envs.utils.state import JointState, FullState, BallbotState, ObservableState
from config.config import Config

import os
import matplotlib.pyplot as plt


class WNumMPC(Policy):
    def __init__(self, config: Config, agent_id: int):
        super(WNumMPC, self).__init__(config, agent_id)
        self.indexed_trajectory = None
        self.prediction = None
        self.model_predictor: CV = CV()

        self.trajectory = None
        self.model_predictor.vpref = config.robot.v_pref
        self.model_predictor.dt = config.env.time_step
        # Action Setting
        self.action_params = config.policy_config['params']['action']
        self.plot_cost_data: bool = config.policy_config["params"]['plot_cost_data']
        self.span = np.deg2rad(self.action_params['span'])
        self.n_actions = self.action_params["n_actions"]
        self.n_vel = self.action_params["n_velocity"]
        self.max_v_ratio = self.action_params["max_v_ratio"]

        if self.span == 2 * np.pi:
            self.span = 2 * np.pi - ((2 * np.pi) / self.n_actions)
        self.model_predictor.set_params(config)
        self.name = 'wnum_mpc'

        self.step_counter = 0
        self.episode_counter = 0
        self.observed_ids: list[int] | None = None  # ids of observed agents
        self.obs_state: np.ndarray = None

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        self.trajectory: np.ndarray = None  # trajectories of all agents (robot, humans)
        self.indexed_trajectory = None
        self.prediction = None
        # self.logger.reset()
        self.model_predictor.reset(rng=rng)
        self.step_counter = 0
        self.episode_counter += 1

    @staticmethod
    def flatten_state(state: BallbotState | ObservableState | FullState) -> np.ndarray:
        px, py = state.get_position()
        vx, vy = state.get_velocity()
        return np.array([px, py, vx, vy, state.radius])

    def update_trajectory(self, state: JointState) -> np.ndarray:
        arr_pre: list[np.ndarray] = ([self.flatten_state(state.self_state)]
                                     + [self.flatten_state(state_tmp) for state_tmp in state.human_states])

        arr: np.ndarray = np.stack(arr_pre, axis=0)

        agent_num: int = len(state.human_states) + 1
        agent_id: int = [i for i in range(agent_num) if i not in self.observed_ids][0]  # agent_id: observed_idsに含まれないid
        present_index: list[int] = [agent_id] + self.observed_ids
        idx_arr = np.stack(arr_pre, axis=0)[np.argsort(present_index)]

        self.trajectory = np.concatenate((self.indexed_trajectory[:, present_index], arr[None]), axis=0) if self.indexed_trajectory is not None else arr[None]
        self.indexed_trajectory = np.concatenate((self.indexed_trajectory, idx_arr[None]), axis=0) if self.indexed_trajectory is not None else idx_arr[None]
        return arr

    def predict(self, state: JointState, observed_ids: list[int] | None = None, goaled_ids: list[int] | None = None) -> ActionXY:
        self.observed_ids = observed_ids
        self.obs_state = self.update_trajectory(state)  # array_state

        goal = np.array(state.self_state.get_goal())
        action_set: np.ndarray = self.generate_action_set(copy.deepcopy(state.self_state), goal)
        # (N x S x T' x H x 2), (N x S), (2 x 1)
        plot_info = {"eps_id": self.episode_counter, "plot_flag": self.step_counter % 5 == 2}

        # select action
        predictions, costs, action_set, info = self.model_predictor.predict(self.trajectory, self.obs_state,
                                                                            state.self_state, action_set, goal, plot_info)
        self.prediction = predictions
        best_action = info["best_action"]
        # self.logger.add_predictions(array_state, action_set, predictions, costs, goal)
        action: ActionXY = self.action_post_processing(best_action[2:])

        if self.reach_destination(state):
            return self.action_post_processing([0, 0])

        if self.step_counter % 5 == 2 and self.plot_cost_data:
            directory = "./plot_data/cost_plot/episode{}".format(plot_info["eps_id"])
            os.makedirs(directory, exist_ok=True)  # log保存用のディレクトリを作成
            file_name = directory + "/cost_step{}".format(self.step_counter)
            self.plot_env_states(state, action_set, costs, best_action[2:], self.trajectory, predictions, info,
                                 file_name)

        self.step_counter += 1
        return action

    def action_post_processing(self, action) -> ActionXY:
        global_action_xy = ActionXY(action[0], action[1])
        # Keep in global frame ActionXY
        action_xy = global_action_xy

        return action_xy

    def generate_action_set(self, state: BallbotState, goal) -> np.ndarray:
        pos = np.array(state.get_position())
        vel = state.get_velocity()
        theta = state.get_theta()
        theta_dot = state.get_theta_dot()

        sim_heading = state.get_heading()
        thetas = [sim_heading - (self.span / 2.0) + i * self.span / (self.n_actions - 1) for i in range(self.n_actions)]
        # 向かってる方向に対して-span/2度した所からn_action個に等間隔区分したangleを生成
        thetas = thetas if len(thetas) > 1 else np.arctan2(goal - pos)
        vels = np.linspace(0.0, self.max_v_ratio * state.get_vpref(), self.n_vel)  # 速度の絶対値の候補
        cv_rollout = []

        for v_tmp in vels:
            goals = pos[:, None] + (v_tmp * self.model_predictor.prediction_horizon * 10) * np.stack(
                (np.cos(thetas), np.sin(thetas)), axis=0)  # (2, N)
            state = np.array([theta[0], theta_dot[0], vel[0], theta[1], theta_dot[1], vel[1]])  # (6, )
            tmp_rollout = self.generate_cv_rollout(pos, state, goals, v_tmp, self.model_predictor.rollout_steps)
            cv_rollout.append(tmp_rollout)

        ans = np.concatenate(np.array(cv_rollout), axis=0)
        return ans

    def generate_cv_rollout(self, position, state, goals, vpref, length):
        """To get particular rollout"""
        rollouts = []
        state = np.repeat(state[:, None], goals.shape[1], axis=1)  # (6, N)
        position = np.repeat(position[:, None], goals.shape[1], axis=1)  # (2, N)
        for _ in range(length):
            ref_velocities = self.generate_cv_action(position, goals, vpref)  # (2, N)
            state, position, _ = self.step_dynamics(position, state, ref_velocities)
            rollouts.append(np.concatenate((position, ref_velocities), axis=0).transpose((1, 0)))
        rollouts = np.stack(rollouts, axis=1)  # N x T x 4
        return rollouts

    def generate_cv_action(self, position, goal, vpref):
        """To get particular action"""
        dxdy = goal - position  # (2, N)
        thetas = np.arctan2(dxdy[1], dxdy[0])  # (N, )
        return np.stack((np.cos(thetas), np.sin(thetas)), axis=0) * vpref  # (2, N)

    def step_dynamics(self, position, state, action):  # (2, N), (6, N), (2, N)
        # Reference input (global)
        N = action.shape[1]
        U = np.concatenate((np.zeros((2, N)), action[0, None], np.zeros((2, N)), action[1, None]), axis=0)  # (6, N)

        # Integrate with ballbot dynamics
        next_state = self.integrator(state, U)  # (6, N)

        velocity = next_state[[2, 5]]  # (2, N)
        position = position + velocity * self.model_predictor.dt  # (2, N)
        return next_state, position, velocity

    def integrator(self, S, U):
        dt_ = float(self.model_predictor.dt)
        S_next = np.array(S)
        S_next += dt_ * U
        return S_next

    def plot_env_states(self, state, action_set: np.array, costs: np.array, best_action, trajectory, predictions, info,
                        file_name):
        score_percent, w_scores, o_scores = info["score_percent"], info["w_scores"], info["o_scores"]
        g_scores = info["g_scores"]

        fig = plt.figure(figsize=(14, 11))
        ax1 = plt.subplot2grid((2, 6), (0, 0), colspan=3)
        ax2 = plt.subplot2grid((2, 6), (0, 3), colspan=3)
        ax3 = plt.subplot2grid((2, 6), (1, 0), colspan=2)
        ax4 = plt.subplot2grid((2, 6), (1, 2), colspan=2)
        ax5 = plt.subplot2grid((2, 6), (1, 4), colspan=2)

        # 軌道とstate、actionをplotする
        self._plot_trajectory(ax1, state, best_action, self.indexed_trajectory, predictions, info)

        # actionの評価のplot
        self._plot_color_mesh(fig, ax2, costs, action_set, best_action, "total cost, " + score_percent, w_scores)
        self._plot_color_mesh(fig, ax3, o_scores, action_set, info["best_action_o"][2:], "obs cost", None)
        self._plot_color_mesh(fig, ax4, w_scores, action_set, info["best_action_w"][2:], "winding cost", None)
        self._plot_color_mesh(fig, ax5, g_scores, action_set, info["best_action_g"][2:], "goal cost", None)

        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(file_name)
        plt.close(fig)

    def _plot_trajectory(self, ax, state: np.array, best_action, trajectory: np.array, predictions: np.ndarray, info):
        robot_state = state.self_state
        human_states = state.human_states

        circle_info = [(robot_state.px, robot_state.py, robot_state.radius)]  # (x, y, radius)のリスト
        vector_info = [
            (robot_state.px, robot_state.py, best_action[0], best_action[1]),
            (robot_state.px, robot_state.py, robot_state.vx, robot_state.vy),
        ]  # (x, y, vx, vy)のリスト

        original_human_index = self.observed_ids
        human_states = [human_states[i-1] for i in original_human_index]

        for human_state in human_states:
            circle_info.append((human_state.px, human_state.py, robot_state.radius))
            vector_info.append((human_state.px, human_state.py, human_state.vx, human_state.vy))

        # 円を描く
        for x, y, radius in circle_info:
            circle = plt.Circle((x, y), radius, fill=False, color='b')
            ax.add_patch(circle)

        # 矢印を描く
        for x, y, vx, vy in vector_info[0:1]:
            ax.quiver(x, y, vx, vy, angles='xy', scale_units='xy', scale=1, color='b')

        for i, vec_info in enumerate(vector_info[1:]):
            x, y, vx, vy = vec_info
            ax.quiver(x, y, vx, vy, angles='xy', scale_units='xy', scale=1, color='r')
            ax.text(x, y, str(i))
            if i != 0 and info["target_wnums"] is not None:
                ax.text(x, y + 0.2, str(info["target_wnums"][i - 1]), color="b")

        # ここからはtrajectoryのplot
        pred = np.mean(predictions, axis=1)  # N x T' x H x 4
        best_action_idx = info["best_action_idx"]

        pred = pred[best_action_idx]  # T' x H x 4
        rs = np.full((pred.shape[0], pred.shape[1], 1), trajectory[0][1][-1])
        pred = np.concatenate([pred, rs], axis=-1)  # N x T' x H x 5

        for i in range(trajectory.shape[1]):  # H
            color = "red"
            if i == 0:
                color = "cyan"
            self._plot_one_trajectory(ax, trajectory[::5, i, :], color)

        for i in range(pred.shape[1]):  # H
            self._plot_one_trajectory(ax, pred[:, i, :], "grey")

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        ax.set_title("step{}".format(self.step_counter))

    def _plot_one_trajectory(self, ax, trajectory: np.ndarray, color1):
        # 1体のtrajectoryをplot, [px, py, vx, vy, state.radius]
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        vx = trajectory[:, 2]
        vy = trajectory[:, 3]
        r = trajectory[:, 4]

        # 各点(x, y)に半径rの円を描く
        for xi, yi, ri in zip(x, y, r):
            circle = plt.Circle((xi, yi), ri, fill=False, linestyle='dashed', color=color1)
            ax.add_patch(circle)

        # 配列順に各点(x, y)同士を点線で結ぶ
        for i in range(len(x) - 1):
            ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], linestyle='--', color=color1)

    def _plot_color_mesh(self, fig, ax2, costs, action_set, best_action, title, w_scores):
        # 右側のグラフにpcolormeshを描く
        # action_set: N x T x 4 (px, py, vx, vy)
        # costs: N
        dummy_v = None
        xx = action_set[:, 0, 2]
        yy = action_set[:, 0, 3]
        value = costs[:, 0]
        v_map = dict()
        for i in range(xx.shape[0]):
            v_map[(xx[i], yy[i])] = value[i]

        xs, ys = np.meshgrid(xx, yy)
        values = np.zeros(xs.shape)
        for i in range(xs.shape[0]):
            for j in range(xs.shape[1]):
                if v_map.get((xs[i][j], ys[i][j])) is None:
                    values[i][j] = dummy_v
                else:
                    values[i][j] = v_map[(xs[i][j], ys[i][j])]

        im = ax2.pcolormesh(xs, ys, values, cmap='viridis')

        circle = plt.Circle((best_action[0], best_action[1]), 0.1, fill=False, color='r')
        ax2.add_patch(circle)
        if w_scores is not None:
            w_opt_id = np.argmin(w_scores, axis=0)
            circle = plt.Circle((xx[w_opt_id], yy[w_opt_id]), 0.08, fill=False, color='orange')
            ax2.add_patch(circle)

        ax2.set_aspect('equal')
        ax2.set_xlabel('VX')
        ax2.set_ylabel('VY')
        ax2.set_title(title)
        fig.colorbar(im, ax=ax2)

