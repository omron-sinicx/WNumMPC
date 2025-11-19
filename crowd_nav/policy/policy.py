import abc

import gym
import numpy as np
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from config.config import Config


class Policy(object):
    def __init__(self, config: Config, agent_id: int) -> None:
        """
        Base class for all policies, has an abstract method predict().
        """
        self.trainable = False
        self.phase: str = None
        self.model = None
        self.device: str = "cpu"
        self.last_state = None
        self.time_step = None
        # if agent is assumed to know the dynamics of real world
        self.env: gym.Env = None
        self.config: Config = config
        self.id: int = agent_id

    @abc.abstractmethod
    def predict(self, state) -> ActionXY | ActionRot:
        """
        Policy takes state as input and output an action

        """
        return

    @staticmethod
    def reach_destination(state):
        self_state = state.self_state
        if np.linalg.norm((self_state.py - self_state.gy, self_state.px - self_state.gx)) < self_state.radius:
            return True
        else:
            return False

    def set_env(self, env: gym.Env) -> None:
        self.env = env

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def set_device(self, device: str) -> None:
        self.device = device

    def get_model(self):
        return self.model
