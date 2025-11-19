import torch
import numpy as np
from tensordict import TensorDict
from crowd_sim.envs.utils.state import BallbotState
import math
import copy


def to_numpy(ballbot_state: BallbotState) -> np.ndarray:
    return np.array([
        ballbot_state.px,
        ballbot_state.py,
        ballbot_state.vx,
        ballbot_state.vy,
        ballbot_state.radius,
        ballbot_state.gx,
        ballbot_state.gy,
        ballbot_state.theta
    ])


WNumPolicyObservation = TensorDict


def to_WnumPolicyObservation(self_states: np.ndarray, trajectory: np.ndarray, length: int) -> TensorDict:
    obs_states = self_states[-min(length, self_states.shape[0]):]
    if obs_states.shape[0] < length:
        tmp_states = np.repeat(obs_states[0:1], length, axis=0)
        obs_states = np.concatenate([tmp_states, obs_states], axis=0)[-length:]

    obs_traj = trajectory[-min(length, trajectory.shape[0]):]
    if obs_traj.shape[0] < length:
        tmp_traj = np.repeat(obs_traj[0:1], length, axis=0)
        obs_traj = np.concatenate([tmp_traj, obs_traj], axis=0)[-length:]

    return WNumPolicyObservation({
        "self_states": copy.deepcopy(obs_states),
        "trajectory": copy.deepcopy(obs_traj),
        "length": length
    }, [])


def rotate(state: np.ndarray, angle: float) -> np.ndarray:
    # state: (px, py, vx, vy, r)
    new_state: np.ndarray = np.array([
        state[0] * math.cos(angle) - state[1] * math.sin(angle),
        state[0] * math.sin(angle) + state[1] * math.cos(angle),
        state[2] * math.cos(angle) - state[3] * math.sin(angle),
        state[2] * math.sin(angle) + state[3] * math.cos(angle),
        state[4]
    ])
    return new_state


def convert_human_state(human_state: np.ndarray, self_state: np.ndarray) -> np.ndarray:  # s2
    # human stateの変換 (相対座標に)
    human_state_tmp = copy.deepcopy(human_state)
    human_state_tmp[0] -= self_state[0]
    human_state_tmp[1] -= self_state[1]

    # human stateの回転
    rotation_angle = - math.atan2(self_state[-2] - self_state[1], self_state[-3] - self_state[0])  # radian
    ans = rotate(human_state_tmp, rotation_angle)
    theta = math.atan2(ans[3], ans[2])  # radian

    # 要素の追加 (dist, theta)
    dist: float = np.linalg.norm(ans[:2])
    ans = np.append(ans, dist)
    ans = np.append(ans, theta)  # radian
    return ans  # (7,) : (px, py, v_x, v_y, r, dist, theta)


def convert_robot_state(self_state: np.ndarray, rotation_angle=None) -> np.ndarray:  # s2
    # robot stateの回転
    if rotation_angle is None:
        rotation_angle = - math.atan2(self_state[-2] - self_state[1], self_state[-3] - self_state[0])  # radian
    robot_state: np.ndarray = np.array([
        self_state[2] * math.cos(rotation_angle) - self_state[3] * math.sin(rotation_angle),
        self_state[2] * math.sin(rotation_angle) + self_state[3] * math.cos(rotation_angle),
        0.0,  # radian
        self_state[4],
        np.linalg.norm([self_state[-3] - self_state[0], self_state[-2] - self_state[1]])
    ])
    robot_state[2] = math.atan2(robot_state[1], robot_state[0])

    return robot_state  # (5,) : (vx, vy, phi, r, dist_g)


def convert_trajectory(observation: WNumPolicyObservation) -> torch.Tensor:
    # trajectory: episode_num x (H+1) x 5 (px, py, vx, vy, r)
    # predictions: # N x S x T' x (H+1) x 4 (px, py, vx, vy)
    # state: [self state, other state, ...]
    self_states: np.ndarray = observation["self_states"]
    trajectories: np.ndarray = observation["trajectory"]

    ans_traj = []
    for traj, self_state in zip(trajectories, self_states):
        tmp_single_traj: np.ndarray = convert_robot_state(self_state).flatten()  # robot stateの変換
        human_states = np.concatenate([[convert_human_state(tmp, self_state)] for tmp in traj[1:]], axis=-2).flatten()  # human stateの変換
        ans_traj.append(np.concatenate([tmp_single_traj, human_states]))

    return torch.from_numpy(np.array(ans_traj).flatten()).to(torch.float32)  # episode_num * (5+H*7)

