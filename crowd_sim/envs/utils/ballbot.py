import logging
import math

import numpy as np
from omegaconf import DictConfig

from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import BallbotState, JointState
from config.config import Config
from crowd_sim.envs.utils.action import ActionXY, ActionRot


class BallBot(Agent):
    def __init__(self, config: Config, section: str, agent_id: int) -> None:
        super().__init__(config, section, agent_id)
        self.mpc_config: DictConfig = config.policy_config
        self.max_lean = 0.25
        self.lean_theta = np.array([0.0, 0.0], dtype='float64')
        self.lean_theta_dot = np.array([0.0, 0.0], dtype='float64')
        self.isObstacle = False  # always False (ball bot is not static obstacle)

    def print_info(self):
        logging.info('Ballbot: Agent is {} and has {} kinematic constraint'.format(
            'visible' if self.visible else 'invisible', self.kinematics))

    def act(self, ob, observed_ids: list[int] | None = None, goaled_ids: list[float] = None) -> list[float]:
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        if self.policy.name == "wnum_mpc":
            action = self.policy.predict(state, observed_ids, goaled_ids)
        else:
            action = self.policy.predict(state)
        return action

    def get_full_state(self) -> BallbotState:
        return BallbotState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta,
                            self.lean_theta, self.lean_theta_dot)

    def get_full_state_list(self) -> list[float]:
        return [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta,
                self.lean_theta, self.lean_theta_dot]

    def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None):
        super().set(px, py, gx, gy, vx, vy, theta, radius, v_pref)
        self.lean_theta = np.array([0.0, 0.0], dtype='float64')
        self.lean_theta_dot = np.array([0.0, 0.0], dtype='float64')

    def step(self, action):
        """
        Perform an action and update the state
        """
        self.check_validity(action)
        if self.kinematics == 'holonomic' or self.kinematics == "ballbot":
            vx, vy = (action.vx + self.vx) / 2.0, (action.vy + self.vy) / 2.0  # mean
            if np.linalg.norm([vx, vy]) < 1e-4:
                vx, vy = 0.0, 0.0
            cliped_action = ActionXY(vx, vy)
            vel_for_calc = ActionXY((self.vx + cliped_action.vx) / 2.0, (self.vy + cliped_action.vy) / 2.0)
            pos = self.compute_position(vel_for_calc, self.time_step)
            self.px, self.py = pos
            self.vx = cliped_action.vx
            self.vy = cliped_action.vy
            self.theta = np.arctan2(self.vy, self.vx)
        else:
            pos = self.compute_position(action, self.time_step)
            self.px, self.py = pos
            self.theta = (self.theta + action.r * self.time_step) % (2 * np.pi)
            self.vx = action.v * np.cos(self.theta)
            self.vy = action.v * np.sin(self.theta)

