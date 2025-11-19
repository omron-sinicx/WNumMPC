import numpy as np
from numpy.linalg import norm
import math
import abc
import logging
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from crowd_sim.envs.utils.state import ObservableState, FullState, ObservableState_noV, JointState
from config.config import Config
from crowd_nav.policy.policy import Policy
from crowd_sim.envs.utils.agent import Agent
from maru.software.control.python.maru import osx001Driver

class MaruAgent(Agent):
    def __init__(self, config: Config, section: str, agent_id: int, driver: osx001Driver, target_ids : list[int]) -> None:
        """
        Base class for robot and human. Have the physical attributes of an agent.

        """
        super().__init__(config, section, agent_id)

        self.driver = driver
        self.targetId = target_ids[agent_id]
        
        self.wheel_const = config.wheel_const
        self.scale = config.scale
        self.time_scale = config.time_scale

        self.v = 0

    def print_info(self):
        logging.info('Agent is {} and has {} kinematic constraint'.format(
            'visible' if self.visible else 'invisible', self.kinematics))

    def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None):
        assert False

    def set_list(self, px, py, vx, vy, radius, gx, gy, v_pref, theta):
        assert False

    def set_start_position(self):
        self.sx, self.sy = self.px, self.py

    def update_state(self, status):
        self.px, self.py, degree, voltage, yaw, pitch, roll = status[self.targetId]
        self.theta = degree / 180 * np.pi
        self.px /= self.scale
        self.py /= self.scale
        self.vx = self.v * np.cos(self.theta)
        self.vy = self.v * np.sin(self.theta)
        
    def set_position(self, position):
        assert False

    def set_velocity(self, velocity):
        assert False

    def set_goal_position(self, gx, gy):
        self.gx, self.gy = gx, gy
        self.stop()
    
    def act(self, ob, observed_ids: list[int] | None = None, goaled_ids: list[float] = None) -> list[float]:
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        if self.policy.name == "wnum_mpc":
            action = self.policy.predict(state, observed_ids, goaled_ids)
        else:
            action = self.policy.predict(state)
        return action

    '''
    def volt_poly(self, v):
        sg = np.sign(v)
        a = abs(v)
        return sg * (38.4354 * a * a - 24.4467 * a + 23.1063)
    '''
    
    def step(self, action):
        """
        Perform an action and update the state
        """
        self.check_validity(action)
        #diff_v = action.r / (6.3 + 0.3 * abs(action.r)) * 1.2
        #rightSpeed = action.v + diff_v
        #leftSpeed = action.v - diff_v
        #rightw = self.volt_poly(rightSpeed)
        #leftw = self.volt_poly(leftSpeed)
        action_v = action.v / self.time_scale
        action_r = action.r / self.time_scale 
        if self.targetId == 15:
            mean_w = (48 + 9.0*action_r**2) * action_v
            diff_w = (5.2 - 3*action_v) * action_r
        else:
            mean_w = (37 + 3.7*action_r**2) * action_v
            diff_w = (5.2 - 3*action_v) * action_r
        #mean_w = (43 + 2.0*action_r**2 + 60 * (0.6-abs(action_v))) * action_v
        #diff_w = (6.2 - 2.0*action_v) * action_r
        #print(f"targetId={self.targetId}, action_v={action_v}, action_r={action_r}, mean_w={mean_w}, diff_w={diff_w}")
        rightw = mean_w + diff_w
        leftw = mean_w - diff_w
        self.driver.writeMotorSpeed(self.targetId, rightw, leftw)
        self.v = action.v
        self.vx = action.v * np.cos(self.theta)
        self.vy = action.v * np.sin(self.theta)

    def stop(self):
        self.driver.writeMotorSpeed(self.targetId, 0, 0)
        self.v = self.vx = self.vy = 0
