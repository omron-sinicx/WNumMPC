import copy
import logging
import gym
import torch
import math
from numpy.linalg import norm
from crowd_nav.policy.cadrl import CADRL
from crowd_nav.policy.wnum_mpc import WNumMPC
from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.ballbot import BallBot
from crowd_sim.envs.utils.maru_agent import MaruAgent
from crowd_sim.envs.utils.info import *
from crowd_nav.policy.orca import ORCA
from crowd_sim.envs.utils.state import *
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from crowd_sim.envs.utils.agent import dist_agents
from crowd_sim.envs.logger import Logger
from config.config import Config
from omegaconf import DictConfig
from tensordict import TensorDict
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches
import os
import atexit
from time import time, sleep

from maru.software.control.python.maru import osx001Driver

def flush(driver):
    driver.flush_buffer()
    print("flush!!")

class CrowdSim(gym.Env):
    def __init__(self, seed: int, n_env: int = 0):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        self.time_limit = None
        self.time_step = None
        self.robot: Robot | BallBot | MaruAgent = None  # a Robot instance representing the robot
        self.humans = None  # a list of Human instances, representing all humans in the environment
        self.global_time = None

        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_dist_front = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config: Config = None
        self.training_config: DictConfig = None

        self.circle_radius = None
        self.human_num = None

        self.action_space = None
        self.observation_space = None

        self.done_if_all_agents_reached: bool = False  # 全てのAgentがgoalに到達したらdoneにするかどうか
        self.is_goaled_list: list[bool] = []  # 各Agentがgoalに到達したかどうか
        # falseの場合は、robotがgoalに到達したらdoneにする

        # limit FOV
        self.robot_fov = None
        self.human_fov = None

        self.dummy_human = None
        self.dummy_robot = None

        # seed
        self.thisSeed = seed  # the seed will be set when the env is created

        # nenv
        self.nenv = n_env  # the number of env will be set when the env is created.
        self._rng: np.random.Generator = np.random.default_rng(seed=self.thisSeed)

        self.phase: str | None = None  # set the phase to be train, val or test
        self.test_case = None  # the test case ID, which will be used to calculate a seed to generate a human crossing case

        # for render
        self.render_axis = None

        self.humans: list[Human] | list[BallBot] | list[MaruAgent] = []

        self.storage: bool = False
        self.logger: Logger = None
        self._global_obs_pre: list[np.ndarray] = []  # 各Agentに入力したglobal_obsのリスト
        self._extra_time_to_goals: list[float | None] = []  # 各Agentのgoalまでの余分な時間のリスト
        self.trajectory: list[list[FullState]] = None  # 各Agentの軌跡のリスト

    def getStatusFromDrivers(self):
        rs = [{}] * len(self.drivers)
        current = time()
        while True:
            for driver_id, driver in enumerate(self.drivers):
                res = driver.get_id_position()
                if len(res) >= len(rs[driver_id]):
                    rs[driver_id] = res
            if sum([len(r) for r in rs]) >= self.human_num + 1:
                break
            if time() - current > 10 * self.time_step * self.config.time_scale:
                print("connection failed!")
                exit(1)
        result = {}
        for r in rs:
            result.update(r)
        return result
    
    def configure(self, config: Config, buffer: list | None = None) -> None:
        self.config: Config = config
        if self.config.humans.policy in ["wnum_mpc", "vanilla_mpc", "mean_mpc"]:
            self.training_config: DictConfig = config.policy_config["params"]["training_param"]
        self.time_limit = config.env.time_limit
        self.time_step = config.env.time_step

        self.success_reward = config.reward.success_reward
        self.collision_penalty = config.reward.collision_penalty
        self.discomfort_dist = config.reward.discomfort_dist_back
        self.discomfort_dist_front = config.reward.discomfort_dist_front
        self.discomfort_penalty_factor = config.reward.discomfort_penalty_factor
        self.done_if_all_agents_reached = config.sim.done_if_all_agents_reached

        if self.config.humans.policy in ["orca", "social_force", "wnum_mpc", "cadrl", "vanilla_mpc", "mean_mpc"]:
            self.circle_radius = config.sim.circle_radius
            self.circle_radius_x = config.sim.circle_radius_x
            self.circle_radius_y = config.sim.circle_radius_y
            self.human_num = config.sim.human_num

        else:
            raise NotImplementedError

        logging.info('human number: {}'.format(self.human_num))
        logging.info('Circle width: {}'.format(self.circle_radius))

        self.robot_fov = np.pi * config.robot.FOV
        self.human_fov = np.pi * config.humans.FOV
        logging.info('robot FOV %f', self.robot_fov)
        logging.info('humans FOV %f', self.human_fov)

        # set dummy human and dummy robot
        # dummy humans, used if any human is not in view of other agents
        self.dummy_human = Human(self.config, 'humans', 0)
        # if a human is not in view, set its state to (px = 100, py = 100, vx = 0, vy = 0, theta = 0, radius = 0)
        self.dummy_human.set(7, 7, 7, 7, 0, 0, 0)  # (7, 7, 7, 7, 0, 0, 0)
        self.dummy_human.time_step = config.env.time_step

        self.dummy_robot = Robot(self.config, 'robot', 0)
        self.dummy_robot.set(7, 7, 7, 7, 0, 0, 0)
        self.dummy_robot.time_step = config.env.time_step
        self.dummy_robot.kinematics = 'holonomic'
        self.dummy_robot.policy = ORCA(config, agent_id=-1)

        # configure noise in state
        self.add_noise = config.noise.add_noise
        if self.add_noise:
            self.noise_type = config.noise.type
            self.noise_magnitude = config.noise.magnitude

        self.last_human_states = np.zeros((self.human_num, 5))

        self.episode_counter: int = 0
        
        # set robot for this envs
        config.use_maru = config.policy_config.use_maru
        if config.use_maru:
            self.drivers = []
            for serial_port in config.serial_port:
                driver = osx001Driver(serial_port)
                atexit.register(flush, driver)
                self.drivers.append(driver)
            ids = [[]] * len(self.drivers)
            while True:
                for driver_id, driver in enumerate(self.drivers):
                    res = list(driver.get_id_position().keys())
                    if len(res) >= len(ids[driver_id]):
                        ids[driver_id] = res
                if sum([len(l) for l in ids]) >= self.human_num + 1:
                    break
            composed_ids = sorted(sum([[(i, id) for id in ids[i]] for i in range(len(ids))], []))
            self.driver_ids = [p[0] for p in composed_ids]
            self.target_ids = [p[1] for p in composed_ids]
            #self.episode_counter = config.policy_config.episode_starts

            self.collision_radius = config.policy_config.collision_radius
            
            rob_RL = MaruAgent(config, 'robot', 0, self.drivers[self.driver_ids[0]], self.target_ids)
        elif config.action_space.kinematics == "ballbot" or config.action_space.kinematics == "diff_wheel":
            rob_RL: BallBot = BallBot(config, 'robot', 0)
        else:
            rob_RL: Robot = Robot(config, 'robot', 0)

        if hasattr(config.policy_config, "eval_states"):
            self.eval_states = np.load(config.policy_config.eval_states)

        self.set_robot(rob_RL)
        self.step_counter: int = 0

        # logger
        if self.config.humans.policy in ["wnum_mpc", "vanilla_mpc", "mean_mpc"]:
            self.logger: Logger = Logger(logger_size=config.policy_config.params.mpc_param.select_span)
        else:
            self.logger: Logger = Logger(logger_size=5)
        self.rb: [] = buffer
        self.rb = [[] for _ in range(self.human_num + 1)]
        self.plot_trajectory: bool = self.config.policy_config["plot_trajectory"]
        self.plot_animation: bool = self.config.policy_config["plot_animation"]
        if self.plot_animation:
            self.animation_dir = self.config.policy_config["animation_dir"]
        self.save_trajectory: bool = self.config.policy_config["save_trajectory"]
        if self.save_trajectory:
            self.trajectory_dir = self.config.policy_config["trajectory_dir"]
            self.saved_trajectories = []
        self._global_obs_pre = [self.generate_global_ob() for _ in range(self.human_num + 1)]
        self._extra_time_to_goals = [None for _ in range(self.human_num + 1)]

        if self.plot_animation:
            self.anim_images: list = []
            self.fig = plt.figure(figsize=(8, 8), dpi=100)
        return

    def set_phase(self, phase: str) -> None:
        self.phase = phase
        for agent in [self.robot] + self.humans:
            agent.policy.phase = phase

    def set_robot(self, robot: Robot | BallBot) -> None:
        self.robot: Robot | BallBot = robot

    def generate_random_human_position(self, human_num: int) -> None:
        """
        Generate human position: generate start position on a circle, goal position is at the opposite side
        :param human_num:
        :return:
        """
        # initial min separation distance to avoid danger penalty at beginning
        for i in range(human_num):
            self.humans.append(self.generate_circle_crossing_human(i + 1))

    def generate_circle_crossing_human(self, agent_id: int) -> Human | BallBot | MaruAgent:
        if self.config.humans.policy in ["wnum_mpc", "vanilla_mpc", "mean_mpc"]:  # wnum_mpcの場合はballbotを使用
            tmp = self.config.policy_config['params']["obs_plot"]
            tmp2 = self.config.policy_config['params']["plot_cost_data"]
            self.config.policy_config['params']["obs_plot"] = False
            self.config.policy_config['params']["plot_cost_data"] = False
            if self.config.use_maru:
                human = MaruAgent(self.config, 'humans', agent_id, self.drivers[self.driver_ids[agent_id]], self.target_ids)
            else:
                human: BallBot = BallBot(self.config, 'humans', agent_id)
            self.config.policy_config['params']["obs_plot"] = tmp
            self.config.policy_config['params']["plot_cost_data"] = tmp2
        else:
            human: Human = Human(self.config, 'humans', agent_id)

        if self.config.use_maru:
            gx, gy = self.eval_states[self.episode_counter][human.id]
            human.set_goal_position(gx, gy)
        elif hasattr(self, "eval_states"):
            if self.eval_states.ndim == 4:
                sx, sy = self.eval_states[self.episode_counter][0][human.id]
                gx, gy = self.eval_states[self.episode_counter][1][human.id]
            else:
                sx, sy = self.eval_states[self.episode_counter][human.id]
                gx, gy = self.eval_states[self.episode_counter+1][human.id]
            human.set(sx, sy, gx, gy, 0, 0, math.atan2(gy-sy, gx-sx))            
        else:
            ct = 0
            while True:
                angle = self._rng.random() * np.pi * 2
                # add some noise to simulate all the possible cases robot could meet with human
                v_pref = 1.0 if human.v_pref == 0 else human.v_pref
                px_noise = (self._rng.random() - 0.5) * v_pref
                py_noise = (self._rng.random() - 0.5) * v_pref
                px = self.circle_radius_x * np.cos(angle) + px_noise
                py = self.circle_radius_y * np.sin(angle) + py_noise
                collide = False
    
                for i, agent in enumerate([self.robot] + self.humans):
                    # keep human at least 3 meters away from robot
                    if self.robot.kinematics == 'unicycle' and i == 0:
                        min_dist = self.circle_radius / 2  # Todo: if circle_radius <= 4, it will get stuck here
                    else:
                        min_dist = self.config.sim.min_dist
                    if norm((px + agent.gx, py + agent.gy)) < min_dist:
                        collide = True
                        break
                if not collide:
                    break
                ct+=1
    
            if self.episode_counter == 0:
                human.set(px, py, -px, -py, 0, 0, math.atan2(-py, -px))
            else:
                human.set(self.last_gx[agent_id], self.last_gy[agent_id], -px, -py, 0, 0, math.atan2(-py, -px))
        return human

    # add noise according to env.config to state
    def apply_noise(self, ob):
        if isinstance(ob[0], ObservableState):
            for i in range(len(ob)):
                if self.noise_type == 'uniform':
                    noise = self._rng.uniform(-self.noise_magnitude, self.noise_magnitude, 5)
                elif self.noise_type == 'gaussian':
                    noise = self._rng.normal(size=5)
                else:
                    print('noise type not defined')
                ob[i].px = ob[i].px + noise[0]
                ob[i].py = ob[i].px + noise[1]
                ob[i].vx = ob[i].px + noise[2]
                ob[i].vy = ob[i].px + noise[3]
                ob[i].radius = ob[i].px + noise[4]
            return ob
        else:
            if self.noise_type == 'uniform':
                noise = self._rng.uniform(-self.noise_magnitude, self.noise_magnitude, len(ob))
            elif self.noise_type == 'gaussian':
                noise = self._rng.normal(size=len(ob))
            else:
                print('noise type not defined')
                noise = [0] * len(ob)

            return ob + noise

    # update the robot belief of human states
    # if a human is visible, its state is updated to its current ground truth state
    # else we assume it keeps going in a straight line with last observed velocity
    def update_last_human_states(self, human_visibility, reset):
        """
        update the self.last_human_states array
        human_visibility: list of booleans returned by get_human_in_fov (e.x. [T, F, F, T, F])
        reset: True if this function is called by reset, False if called by step
        :return:
        """
        # keep the order of 5 humans at each timestep
        for i in range(self.human_num):
            if human_visibility[i]:
                humanS = np.array(self.humans[i].get_observable_state_list())
                self.last_human_states[i, :] = humanS

            else:
                if reset:
                    humanS = np.array([15., 15., 0., 0., 0.3])
                    self.last_human_states[i, :] = humanS

                else:
                    px, py, vx, vy, r = self.last_human_states[i, :]
                    # Plan A: linear approximation of human's next position
                    px = px + vx * self.time_step
                    py = py + vy * self.time_step
                    self.last_human_states[i, :] = np.array([px, py, vx, vy, r])

                    # Plan B: assume the human doesn't move, use last observation
                    # self.last_human_states[i, :] = np.array([px, py, 0., 0., r])

    # set robot initial state and generate all humans for reset function
    # for crowd nav: human_num == self.human_num
    # for leader follower: human_num = self.human_num - 1
    def generate_robot_humans(self, phase, human_num=None):
        if human_num is None:
            human_num = self.human_num

        # for FoV environment
        if True:
            if self.robot.kinematics == 'unicycle':
                angle = self._rng.uniform(0, np.pi * 2)
                px = self.circle_radius * np.cos(angle)
                py = self.circle_radius * np.sin(angle)
                while True:
                    gx, gy = self._rng.uniform(-self.circle_radius, self.circle_radius, 2)
                    if np.linalg.norm([px - gx, py - gy]) >= 6:  # 1 was 6
                        break
                self.robot.set(px, py, gx, gy, 0, 0, self._rng.uniform(0, 2 * np.pi))  # randomize init orientation

            # randomize starting position and goal position
            elif self.config.use_maru:
                gx, gy = self.eval_states[self.episode_counter][self.robot.id]
                self.robot.set_goal_position(gx, gy)
            elif hasattr(self, "eval_states"):
                if self.eval_states.ndim == 4:
                    sx, sy = self.eval_states[self.episode_counter][0][self.robot.id]
                    gx, gy = self.eval_states[self.episode_counter][1][self.robot.id]
                else:
                    sx, sy = self.eval_states[self.episode_counter][self.robot.id]
                    gx, gy = self.eval_states[self.episode_counter+1][self.robot.id]
                self.robot.set(sx, sy, gx, gy, 0, 0, math.atan2(gy-sy, gx-sx))
            else:
                while True:
                    # px, py, gx, gy = self._rng.uniform(-self.circle_radius, self.circle_radius, 4)
                    #px, py = self._rng.uniform(-self.circle_radius, self.circle_radius, 2)
                    angle = self._rng.random() * np.pi * 2
                    v_pref = 1.0 if self.robot.v_pref == 0 else self.robot.v_pref
                    px_noise = (self._rng.random() - 0.5) * v_pref
                    py_noise = (self._rng.random() - 0.5) * v_pref
                    px = self.circle_radius_x * np.cos(angle) + px_noise
                    py = self.circle_radius_y * np.sin(angle) + py_noise
                    gx, gy = -px, -py
                    if np.linalg.norm([px - gx, py - gy]) >= (self.circle_radius * 2.0 * 0.9):
                        break
                if self.episode_counter == 0:
                    self.robot.set(px, py, gx, gy, 0, 0, math.atan2(gy - py, gx - px))
                else:
                    self.robot.set(self.last_gx[0], self.last_gy[0], -px, -py, 0, 0, math.atan2(-py, -px))

            # generate humans
            self.generate_random_human_position(human_num=human_num)  # Humanの生成

        if self.config.use_maru:
            self.update_robot_human_state()
            self.robot.set_start_position()
            for human in self.humans:
                human.set_start_position()
        else:
            self.last_gx, self.last_gy = {}, {}
            self.last_gx[0], self.last_gy[0] = self.robot.gx, self.robot.gy
            #print(f"id={self.robot.id}, gx = {self.robot.gx}, gy={self.robot.gy}")
            for human in self.humans:
                self.last_gx[human.id], self.last_gy[human.id] = human.gx, human.gy
                #print(f"id={human.id}, gx = {human.gx}, gy={human.gy}")

    def update_robot_human_state(self):
        status = self.getStatusFromDrivers()
        self.robot.update_state(status)
        for human in self.humans:
            human.update_state(status)

    def reset(self, phase='train', test_case=None, seed=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """

        if self.phase is not None:
            phase = self.phase
        if self.test_case is not None:
            test_case = self.test_case

        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']

        self.global_time = 0

        self.trajectory = [[] for _ in range(self.human_num + 1)]
        self.humans = []

        if seed is not None:
            self._rng = np.random.default_rng(seed=seed)

        self.generate_robot_humans(phase)
        self.is_goaled_list = [False for _ in range(len(self.humans) + 1)]

        # get current observation
        ob, observed_ids = self.generate_ob(reset=True)

        # setting trajectory
        for agent in [self.robot] + self.humans:
            self.trajectory[agent.id].append(agent.get_full_state())

        self.logger.clear()

        if isinstance(self.robot.policy, WNumMPC):
            self.robot.policy.reset()

        self.sync_policy_setting()

        self.step_counter = 0
        self.episode_counter += 1
        self._global_obs_pre = [self.generate_global_ob() for _ in range(self.human_num + 1)]
        self._extra_time_to_goals = [None for _ in range(self.human_num + 1)]

        if self.plot_animation and len(self.anim_images) != 0:
            ani = animation.ArtistAnimation(self.fig, self.anim_images, interval=40, repeat=False, blit=True)
            os.makedirs(self.animation_dir, exist_ok=True)
            ani.save(self.animation_dir + f'/episode{self.episode_counter-1}.gif', writer='pillow')
            self.fig.clf()
            self.anim_images = []

        self.previous_time = None
            
        return ob, observed_ids

    def sync_policy_setting(self) -> None:
        for human in self.humans:
            if isinstance(human.policy, WNumMPC):
                if (human.policy.model_predictor.wnum_selector is not None
                        and self.robot.policy.model_predictor.wnum_selector is not None):
                    human.policy.model_predictor.wnum_selector.model = copy.deepcopy(
                        self.robot.policy.model_predictor.wnum_selector.model
                    )
                human.policy.reset()
            elif isinstance(human.policy, CADRL):
                human.policy.model = copy.deepcopy(self.robot.policy.model)

        for agent in [self.robot] + self.humans:
            agent.policy.set_env(self)
            agent.policy.set_device("cpu")
            if isinstance(agent.policy, CADRL):
                agent.policy.set_epsilon(self.robot.policy.epsilon)
            if self.phase is not None:
                agent.policy.phase = self.phase

    def set_policy(self, policy: str) -> None:
        self.config.humans.policy = policy
        self.config.robot.policy = policy

        # set robot for this envs
        if self.config.use_maru:
            rob_RL = MaruAgent(self.config, 'robot', 0, self.drivers[self.driver_ids[0]], self.target_ids)
        elif self.config.action_space.kinematics == "ballbot":
            rob_RL: BallBot = BallBot(self.config, 'robot', 0)
        else:
            rob_RL: Robot = Robot(self.config, 'robot', 0)
        self.set_robot(rob_RL)
        self.reset()

    # Caculate whether agent2 is in agent1's FOV
    # Not the same as whether agent1 is in agent2's FOV!!!!
    # arguments:
    # state1, state2: can be agent instance OR state instance
    # robot1: is True if state1 is robot, else is False
    # return value:
    # return True if state2 is visible to state1, else return False
    def detect_visible(self, state1, state2, robot1=False, custom_fov=None):
        if self.robot.kinematics == 'holonomic' or self.robot.kinematics == "ballbot":
            real_theta = np.arctan2(state1.vy, state1.vx)
        else:
            real_theta = state1.theta
        # angle of center line of FOV of agent1
        v_fov = [np.cos(real_theta), np.sin(real_theta)]

        # angle between agent1 and agent2
        v_12 = [state2.px - state1.px, state2.py - state1.py]
        # angle between center of FOV and agent 2

        v_fov = v_fov / np.linalg.norm(v_fov)
        v_12 = v_12 / np.linalg.norm(v_12)

        offset = np.arccos(np.clip(np.dot(v_fov, v_12), a_min=-1, a_max=1))
        if custom_fov:
            fov = custom_fov
        else:
            if robot1:
                fov = self.robot_fov
            else:
                fov = self.human_fov

        if np.abs(offset) <= fov / 2:
            return True
        else:
            return False

    # for robot:
    # return only visible humans to robot and number of visible humans and visible humans' ids (0 to 4)
    def get_num_human_in_fov(self):
        human_ids = []
        humans_in_view = []
        num_humans_in_view = 0

        for i in range(self.human_num):
            visible = self.detect_visible(self.robot, self.humans[i], robot1=True)
            if visible:
                humans_in_view.append(self.humans[i])
                num_humans_in_view = num_humans_in_view + 1
                human_ids.append(True)
            else:
                human_ids.append(False)

        return humans_in_view, num_humans_in_view, human_ids

    def last_human_states_obj(self):
        '''
        convert self.last_human_states to a list of observable state objects for old algorithms to use
        '''
        humans = []
        for i in range(self.human_num):
            h = ObservableState(*self.last_human_states[i])
            humans.append(h)
        return humans

    def check_goaled(self, agents: list[Agent], agent_id: int) -> bool:
        return norm(np.array(agents[agent_id].get_position()) - np.array(agents[agent_id].get_goal_position())) < \
            agents[agent_id].radius * self.config.policy_config.goal_threshold

    def calc_reward(self, agents: list[BallBot], agent_id: int,
                    agent_obs: list[ObservableState] = None):  # for w-num mpc
        dmin: float = float('inf')

        agent_index = -1  # agent_idに対応するagentsのindex
        for i, agent in enumerate(agents):
            if agent.id == agent_id:
                agent_index = i
        if agent_index < 0:
            raise ValueError("agent_id {} is not in agents".format(agent_id))

        danger_dists = []
        collision = False

        if (not hasattr(self.config.policy_config, "detect_collisions")) or self.config.policy_config.detect_collisions:
        
            agent_dist: list[float] = [dist_agents(agents[agent_index], agent) for agent in agents]  # agent間の距離
            if agent_obs is not None:
                agent_dist: list[float] = [agents[agent_index].get_observable_state().dist(agent_ob) for agent_ob in
                                           agent_obs]
    
            for i, agent in enumerate(agents):  # calc reward
                if i == agent_index:
                    continue
                closest_dist: float = agent_dist[i] - 2 * self.collision_radius if hasattr(self, "collision_radius") else  agent_dist[i] - agent.radius - agents[agent_index].radius  # Agent間の間隔
                if closest_dist < self.discomfort_dist:
                    danger_dists.append(closest_dist)
                if closest_dist < 0:  # check collision between agent[agent_id] and others
                    collision = True
                    break
                elif closest_dist < dmin:
                    dmin = closest_dist
    
            # check collision with walls
            px, py = agents[agent_index].get_position()
            rad = self.collision_radius if hasattr(self, "collision_radius") else agents[agent_index].radius
            if px < self.config.min_x + rad or self.config.max_x - rad < px \
                or py < self.config.min_y + rad or self.config.max_y + rad < py:
                collision = True
            
        # check if reaching the goal (agent[agent_id])
        reaching_goal = self.check_goaled(agents, agent_index)
        truncated = False  # 中断

        if self.global_time >= self.time_limit - 1:  # timeout
            reward = 0.0
            truncated = True
            terminated = False
            episode_info = Timeout()
        elif collision:
            reward = self.collision_penalty
            terminated = True
            episode_info = Collision()
        elif reaching_goal:
            reward = self.success_reward
            terminated = True
            episode_info = ReachGoal()

        elif dmin < self.discomfort_dist:  # entering discomfort zone
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor
            terminated = False
            episode_info = Danger(dmin)

        else:
            reward = 0.0
            terminated = False
            episode_info = Nothing()

        if not isinstance(agents[agent_index], BallBot) or (
                not self.robot.policy.model_predictor.set_target_winding_num):
            return reward, 0.0, terminated, truncated, 0.0, episode_info

        # wnum reward
        rew_lambda: float = self.training_config.reward_param.rew_lambda
        rew_wnums: list[float] = []
        correct_wnums: list = []

        target_wnms: dict = {}
        for i, observed_id in enumerate(agents[agent_index].policy.observed_ids):
            target_wnms["{}".format(observed_id)] = agents[agent_index].policy.model_predictor.target.wnums[i]

        for i, their_agent in enumerate(agents):
            if i == agent_index:
                continue
            if not isinstance(their_agent.policy, WNumMPC):
                raise ValueError("human policy is not WNumMPC")

            wnum_id = their_agent.policy.observed_ids.index(agent_id)
            their_target_wnum: float = their_agent.policy.model_predictor.target.wnums[wnum_id]
            target_wnum: float = target_wnms["{}".format(their_agent.id)]
            if not their_agent.reached_destination():
                rew_wnums.append(-float(abs(their_target_wnum - target_wnum)) * math.exp(-agent_dist[i]))
            correct_wnums.append(float(abs(their_target_wnum - target_wnum) < 0.1))

        rew_wnum = rew_lambda * np.mean(rew_wnums) if len(rew_wnums) > 0 else 0.0
        wnum_percent = np.average(correct_wnums)
        return reward, rew_wnum, terminated, truncated, wnum_percent, episode_info

    # compute the observation
    def generate_ob(self, reset):
        visible_human_states, num_visible_humans, human_visibility = self.get_num_human_in_fov()

        if type(self.robot) == BallBot or type(self.robot) == MaruAgent:
            return self.get_observation_for_agent(self.robot.id)

        observed_ids = []  # visibleなhumanのidを格納
        for i in range(self.human_num):
            if human_visibility[i]:
                observed_ids.append(self.humans[i].id)

        self.update_last_human_states(human_visibility, reset=reset)
        if self.robot.policy.name in ['lstm_ppo', 'srnn']:
            ob = [num_visible_humans]
            # append robot's state
            robotS = np.array(self.robot.get_full_state_list())
            ob.extend(list(robotS))

            ob.extend(list(np.ravel(self.last_human_states)))
            ob = np.array(ob)

        else:  # for orca and sf
            ob = self.last_human_states_obj()

        if self.add_noise:
            ob = self.apply_noise(ob)

        return ob, observed_ids

    def get_observation_for_agent(self, agent_id: int) -> tuple[list, list]:
        # 他のAgentの状態を取得し、距離が近い順にソート
        observations, obs_ids = [], []
        dists = []
        for agent in [self.robot] + self.humans:
            if agent.id != agent_id:
                observations.append(agent.get_observable_state())
                obs_ids.append(agent.id)
                dists.append(dist_agents(([self.robot] + self.humans)[agent_id], agent))

        # 距離が近い順にソート
        sorted_indices = np.argsort(dists)
        sorted_observations = [observations[i] for i in sorted_indices]
        sorted_obs_ids = [obs_ids[i] for i in sorted_indices]

        return sorted_observations, sorted_obs_ids

    def get_human_actions(self) -> list[ActionXY] | list[ActionRot]:
        # step all humans
        human_actions = []  # a list of all humans' actions
        for i, human in enumerate(self.humans):
            # observation for humans is always coordinates
            if type(human) != BallBot and type(human) != MaruAgent:
                ob, observed_ids = [], []
                for other_human in self.humans:
                    if other_human != human:
                        # detectable humans are always observable to each other
                        if self.detect_visible(human, other_human):
                            ob.append(other_human.get_observable_state())
                            observed_ids.append(other_human.id)
                        else:
                            ob.append(self.dummy_human.get_observable_state())

                if self.robot.visible:
                    # Else human will always see visible robots
                    if self.detect_visible(human, self.robot):
                        ob += [self.robot.get_observable_state()]
                        observed_ids.append(self.robot.id)
                    else:
                        ob += [self.dummy_robot.get_observable_state()]

            else:
                ob, observed_ids = self.get_observation_for_agent(human.id)
                global_obs = self.generate_global_ob()
                human_actions.append(human.act(ob, observed_ids, self.is_goaled_list))
        return human_actions

    def calc_wnum_reward(self, agent: BallBot) -> float:
        assert isinstance(agent, BallBot)

        rew_gamma: float = self.training_config.reward_param.rew_gamma
        prev_logs: list[dict] = self.logger.get_data_old_order(self.config.policy_config.params.mpc_param.select_span)
        rews = [log["{}".format(agent.id)]["reward"] for log in prev_logs]
        rews_wnum = [log["{}".format(agent.id)]["rew_wnum"] for log in prev_logs]
        w_rews_each_step = [rew + w_rew for rew, w_rew in zip(rews, rews_wnum)]
        wnum_reward = sum([rew * (rew_gamma ** i) for i, rew in enumerate(w_rews_each_step)]) / 10.0
        return wnum_reward

    def generate_global_ob(self) -> np.ndarray:
        # globalなobservationを生成
        obs = []
        for agent in [self.robot] + self.humans:
            obs.extend(agent.get_full_state_list()[0:7])
        ans: np.ndarray = np.array(obs)  # ((H+1)x7,)
        return ans

    def onestep_lookahead(self, action: ActionXY, agent_id: int) -> (list, float):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """

        actions: list[ActionXY] = []
        dists: list[float] = []
        agent_index: int = None
        for i, agent in enumerate([self.robot] + self.humans):
            if agent.id == agent_id:
                actions.append(action)
                agent_index = i
            else:
                actions.append(ActionXY(agent.vx, agent.vy))

        for agent in [self.robot] + self.humans:
            dists.append(dist_agents(([self.robot] + self.humans)[agent_index], agent))
        sorted_indices = np.argsort(dists)

        obs_pre = [agent.get_next_observable_state(action) for agent, action in
                   zip([self.robot] + self.humans, actions)]
        reward, _, _, _, _, _ = self.calc_reward([self.robot] + self.humans, agent_id, obs_pre)
        obs = [obs_pre[i] for i in sorted_indices]

        return obs, reward

    def step(self, action: ActionXY | ActionRot, update=True, store_actions=False) -> (list, float | int, bool, dict):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """

        log_data: dict = {}  # for logging  {id, {state, action, reward, w_nums}, {id2, ...}}
        human_actions = self.get_human_actions()

        # init flags
        is_goaled_list_next = copy.deepcopy(self.is_goaled_list)
        collision_flags = [False for _ in range(len(self.humans) + 1)]
        terminated_list: list[bool] = []

        # apply action and update all agents
            
        for agent, tmp_action in zip([self.robot] + self.humans, [action] + human_actions):
            agent.step(tmp_action)  # step human
        
        if type(self.robot) is MaruAgent:
            current = time()
            if self.previous_time != None:
                step_time = self.previous_time + self.time_step * self.config.time_scale
                if current > step_time:
                    print("time delayed!")
                    self.previous_time = current
                else:
                    sleep(step_time - current)
                    self.previous_time = step_time
            else:
                self.previous_time = current
            self.update_robot_human_state()
        for agent, tmp_action in zip([self.robot] + self.humans, [action] + human_actions):
            tmp_rew, tmp_rew_wnum, tmp_terminated, tmp_truncated, tmp_per, tmp_info \
                = self.calc_reward([self.robot] + self.humans, agent.id)
            if self.is_goaled_list[agent.id] and not isinstance(tmp_info, Timeout):
                tmp_info = ReachGoal()
                tmp_truncated = True
            terminated_list.append(tmp_terminated)

            if tmp_terminated and isinstance(tmp_info, Collision):  # 衝突
                collision_flags[agent.id] = True
                minimal_time = np.linalg.norm(np.array([agent.gx - agent.sx, agent.gy - agent.sy])) / agent.v_pref
                self._extra_time_to_goals[agent.id] = self.time_limit - minimal_time
            elif tmp_terminated and isinstance(tmp_info, ReachGoal) and not self.is_goaled_list[agent.id]:  # ゴール
                is_goaled_list_next[agent.id] = True
                minimal_time = np.linalg.norm(np.array([agent.gx - agent.sx, agent.gy - agent.sy])) / agent.v_pref
                self._extra_time_to_goals[agent.id] = self.global_time + self.time_step - minimal_time

            log_data["{}".format(agent.id)] = {
                "state": agent.get_full_state_list(),
                "action": tmp_action,
                "reward": tmp_rew,
                "rew_wnum": tmp_rew_wnum,
                "w_nums": agent.policy.model_predictor.target.wnums if isinstance(agent.policy, WNumMPC) else None,
                "wnum_percent": tmp_per,
                "episode_info": tmp_info,
            }

        self.logger.append(log_data)
        episode_info = log_data["{}".format(self.robot.id)]["episode_info"]  # episode info of this step
        for agent in [self.robot] + self.humans:
            if isinstance(log_data["{}".format(agent.id)]["episode_info"], Collision):
                episode_info = Collision()
                break

        for agent in [self.robot] + self.humans:
            self.trajectory[agent.id].append(agent.get_full_state())

        ts_dict: TensorDict | None = None

        w_rewards_all = [None for _ in range(self.human_num + 1)]
        global_obs: np.ndarray = self.generate_global_ob()
        for agent in [self.robot] + self.humans:
            goal_flag: bool = is_goaled_list_next[agent.id]
            collision_flag: bool = collision_flags[agent.id]
            terminated_flag: bool = goal_flag or collision_flag
            truncated_flag: bool = (not terminated_flag) and (
                        isinstance(episode_info, Timeout) or isinstance(episode_info, Collision))

            if isinstance(agent, BallBot) and agent.policy.model_predictor.set_target_winding_num:
                if agent.policy.model_predictor.flag_change_target_wnum or terminated_flag or truncated_flag:
                    if self.is_goaled_list[agent.id]:
                        continue
                    w_rewards_all[agent.id] = self.calc_wnum_reward(agent)
                    if self.rb is not None:
                        tmp_td: TensorDict = TensorDict({
                            "global_obs": torch.Tensor(self._global_obs_pre[agent.id]),
                            "observation": agent.policy.model_predictor.observed_state_used,
                            "action": agent.policy.model_predictor.target.wnums,
                            "sample_log_prob": agent.policy.model_predictor.sample_log_probs,
                            ("next", "reward"): torch.Tensor([w_rewards_all[agent.id]]).to(torch.float32),
                            ("next", "observation"): agent.policy.model_predictor.observed_state,
                            ("next", "global_obs"): torch.Tensor(global_obs),
                            ("next", "done"): [truncated_flag or terminated_flag],
                            ("next", "terminated"): [terminated_flag],
                            ("next", "truncated"): [truncated_flag],
                        }, [])
                        if agent == self.robot:
                            ts_dict = tmp_td
                        self.rb[agent.id].append(tmp_td)
                    self._global_obs_pre[agent.id] = global_obs

        # returnする情報の計算
        self.global_time += self.time_step  # max episode length=time_limit/time_step
        self.step_counter += 1
        ob, observed_ids = self.generate_ob(reset=False)

        info: dict = {
            "episode_info": episode_info,
            "observed_ids": observed_ids,
            "wnum_reward": w_rewards_all[self.robot.id],
            "wnum_percent": log_data["{}".format(self.robot.id)]["wnum_percent"] if not self.is_goaled_list[
                self.robot.id] else None,
            "tensor_dict": ts_dict,
            "wnum_reward_all": w_rewards_all,
            "extra_time_to_goals": self._extra_time_to_goals,
        }

        reward = log_data["{}".format(self.robot.id)]["reward"] if not self.is_goaled_list[self.robot.id] else None
        if self.done_if_all_agents_reached:
            terminated = all(is_goaled_list_next) or collision_flags[self.robot.id]
        else:
            terminated = is_goaled_list_next[self.robot.id] or collision_flags[self.robot.id]
        truncated = (not terminated) and (isinstance(episode_info, Timeout) or isinstance(episode_info, Collision))
        assert not isinstance(episode_info, Nothing) or not (terminated or truncated)

        for agent in [self.robot] + self.humans:  # extra time to goalsの更新
            if self._extra_time_to_goals[agent.id] is None:
                minimal_time = np.linalg.norm(np.array([agent.gx - agent.sx, agent.gy - agent.sy])) / agent.v_pref
                self._extra_time_to_goals[agent.id] = self.time_limit - minimal_time

        if self.plot_animation:
            ax = self.fig.add_subplot(111, aspect='equal')
            self._plot_trajectory(ax, self.trajectory)
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.axis("off")
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            self.fig.tight_layout()
            self.anim_images.append(ax.get_children())

        if self.save_trajectory:
            self._save_trajectory(self.trajectory)

        if (not self.plot_animation) and self.plot_trajectory and (self.step_counter % 5 == 2 or terminated or truncated):
            self.plot_env_states()  # plot and save trajectories
        self.is_goaled_list = is_goaled_list_next

        return ob, reward, terminated, truncated, info

    def plot_env_states(self):
        fig, axes = plt.subplots(1, 1, figsize=(14, 14))
        ax1 = axes
        self._plot_trajectory(ax1, self.trajectory)

        plt.axis('equal')
        plt.axis("off")
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('episode{}:step{}'.format(self.episode_counter, self.step_counter))
        plt.tight_layout()
        os.makedirs("./plot_data/episode_{}".format(self.episode_counter), exist_ok=True)
        plt.savefig("./plot_data/episode_{}/step{}.png".format(self.episode_counter, self.step_counter))
        plt.close(fig)

    def _save_trajectory(self, trajectorys: list[list[FullState]]) -> None:
        trajs_np = np.array([
            [[state.px, state.py, state.vx, state.vy, state.radius, state.gx, state.gy] for state in traj] for traj in
            trajectorys
        ])
        self.saved_trajectories.append(trajs_np)

    def _plot_trajectory(self, ax, trajectorys: list[list[FullState]]) -> None:
        trajs_np = np.array([
            [[state.px, state.py, state.vx, state.vy, state.radius, state.gx, state.gy] for state in traj] for traj in
            trajectorys
        ])

        for i, traj in enumerate(trajs_np):
            self._plot_one_trajectory(ax, traj, "C{}".format(i))
            x, y, vx, vy, r, gx, gy = traj[-1]
            ax.add_patch(patches.RegularPolygon((gx, gy), 6, color="C{}".format(i), radius=r * 1.5))
            ax.add_patch(patches.Circle((x, y), r, fill=True, color="C{}".format(i)))
            ax.add_patch(patches.Circle((x, y), r, fill=False, color="black"))
            ax.quiver(x, y, vx, vy, angles='xy', scale_units='xy', scale=1, color='r')
            # if isinstance(self.robot.policy, WNumMPC) and i != 0:  # plot wnum
            #     for j in range(len(self.robot.policy.observed_ids)):
            #         if i == self.robot.policy.observed_ids[j]:
            #             id_tmp = j
            #             break
            #     wnum_tmp = self.robot.policy.model_predictor.target.wnums[id_tmp]
            #     ax.text(x, y+0.15, "{:.2f}".format(wnum_tmp), fontsize=25)
            ax.text(x, y, str(i+1), fontsize=25)

    def _plot_one_trajectory(self, ax, trajectory: np.ndarray, color1):
        # 1体のtrajectoryをplot, [px, py, vx, vy, state.radius]
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        r = trajectory[:, 4]

        for xi, yi, ri in zip(x, y, r):
            circle = plt.Circle((xi, yi), ri, fill=False, linestyle='dashed', color=color1)
            ax.add_patch(circle)

        for i in range(len(x) - 1):
            ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], linestyle='-', color=color1)

    def render(self, mode='human'):
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        from matplotlib import patches

        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        robot_color = 'yellow'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        def calcFOVLineEndPoint(ang, point, extendFactor):
            # choose the extendFactor big enough
            # so that the endPoints of the FOVLine is out of xlim and ylim of the figure
            FOVLineRot = np.array([[np.cos(ang), -np.sin(ang), 0],
                                   [np.sin(ang), np.cos(ang), 0],
                                   [0, 0, 1]])
            point.extend([1])
            # apply rotation matrix
            newPoint = np.matmul(FOVLineRot, np.reshape(point, [3, 1]))
            # increase the distance between the line start point and the end point
            newPoint = [extendFactor * newPoint[0, 0], extendFactor * newPoint[1, 0], 1]
            return newPoint

        ax = self.render_axis
        artists = []

        # add goal
        goal = mlines.Line2D([self.robot.gx], [self.robot.gy], color=goal_color, marker='*', linestyle='None',
                             markersize=15, label='Goal')
        ax.add_artist(goal)
        artists.append(goal)

        # add robot
        robotX, robotY = self.robot.get_position()

        robot = plt.Circle((robotX, robotY), self.robot.radius, fill=True, color=robot_color)
        ax.add_artist(robot)
        artists.append(robot)

        plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)

        # compute orientation in each step and add arrow to show the direction
        radius = self.robot.radius
        arrowStartEnd = []

        robot_theta = self.robot.theta if self.robot.kinematics == 'unicycle' else np.arctan2(self.robot.vy,
                                                                                              self.robot.vx)

        arrowStartEnd.append(
            ((robotX, robotY), (robotX + radius * np.cos(robot_theta), robotY + radius * np.sin(robot_theta))))

        for i, human in enumerate(self.humans):
            theta = np.arctan2(human.vy, human.vx)
            arrowStartEnd.append(
                ((human.px, human.py), (human.px + radius * np.cos(theta), human.py + radius * np.sin(theta))))

        arrows = [patches.FancyArrowPatch(*arrow, color=arrow_color, arrowstyle=arrow_style)
                  for arrow in arrowStartEnd]
        for arrow in arrows:
            ax.add_artist(arrow)
            artists.append(arrow)

        # draw FOV for the robot
        # add robot FOV
        if self.robot_fov < np.pi * 2:
            FOVAng = self.robot_fov / 2
            FOVLine1 = mlines.Line2D([0, 0], [0, 0], linestyle='--')
            FOVLine2 = mlines.Line2D([0, 0], [0, 0], linestyle='--')

            startPointX = robotX
            startPointY = robotY
            endPointX = robotX + radius * np.cos(robot_theta)
            endPointY = robotY + radius * np.sin(robot_theta)

            # transform the vector back to world frame origin, apply rotation matrix, and get end point of FOVLine
            # the start point of the FOVLine is the center of the robot
            FOVEndPoint1 = calcFOVLineEndPoint(FOVAng, [endPointX - startPointX, endPointY - startPointY],
                                               20. / self.robot.radius)
            FOVLine1.set_xdata(np.array([startPointX, startPointX + FOVEndPoint1[0]]))
            FOVLine1.set_ydata(np.array([startPointY, startPointY + FOVEndPoint1[1]]))
            FOVEndPoint2 = calcFOVLineEndPoint(-FOVAng, [endPointX - startPointX, endPointY - startPointY],
                                               20. / self.robot.radius)
            FOVLine2.set_xdata(np.array([startPointX, startPointX + FOVEndPoint2[0]]))
            FOVLine2.set_ydata(np.array([startPointY, startPointY + FOVEndPoint2[1]]))

            ax.add_artist(FOVLine1)
            ax.add_artist(FOVLine2)
            artists.append(FOVLine1)
            artists.append(FOVLine2)

        # add humans and change the color of them based on visibility
        human_circles = [plt.Circle(human.get_position(), human.radius, fill=False) for human in self.humans]

        for i in range(len(self.humans)):
            ax.add_artist(human_circles[i])
            artists.append(human_circles[i])

            # green: visible; red: invisible
            if self.detect_visible(self.robot, self.humans[i], robot1=True):
                human_circles[i].set_color(c='g')
            else:
                human_circles[i].set_color(c='r')
            plt.text(self.humans[i].px - 0.1, self.humans[i].py - 0.1, str(i), color='black', fontsize=12)

        plt.pause(0.1)
        for item in artists:
            item.remove()  # there should be a better way to do this. For example,
            # initially use add_artist and draw_artist later on
        for t in ax.texts:
            t.set_visible(False)
