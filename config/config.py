from omegaconf import DictConfig


class BaseConfig(object):
    def __init__(self):
        pass


class Config(object):
    env = BaseConfig()
    env.env_name = 'CrowdSimDict-v0'  # name of the environment
    env.time_step = 0.1

    reward = BaseConfig()
    # reward.success_reward = 10
    # reward.collision_penalty = -10
    reward.success_reward = 1  # r~
    reward.collision_penalty = -1  # r~
    # discomfort distance for the front half of the robot
    reward.discomfort_dist_front = 0.25
    # discomfort distance for the back half of the robot
    reward.discomfort_dist_back = 0.25
    reward.discomfort_penalty_factor = 1.5
    reward.gamma = 0.95  # discount factor for rewards

    sim = BaseConfig()
    sim.render = False  # show GUI for visualization
    sim.train_val_sim = "circle_crossing"
    sim.test_sim = "circle_crossing"
    square_size = 10.0
    max_x =  square_size
    min_x = -square_size
    max_y = square_size
    min_y = -square_size

    # Human settings
    humans = BaseConfig()
    humans.visible = True  # a human is visible to other humans and the robot
    humans.radius = 0.15
    humans.v_pref = 0.8
    humans.sensor = "coordinates"
    humans.FOV = 2.  # FOV = this values * PI

    # Robot settings
    robot = BaseConfig()
    robot.visible = True  # the robot is visible to humans
    robot.radius = 0.15
    robot.v_pref = 0.8
    robot.sensor = "coordinates"
    robot.FOV = 2.  # FOV = this values * PI

    noise = BaseConfig()
    noise.add_noise = False
    # uniform, gaussian
    noise.type = "uniform"
    noise.magnitude = 0.1

    action_space = BaseConfig()
    action_space.kinematics = "ballbot"

    # config for ORCA
    orca = BaseConfig()
    orca.neighbor_dist = 10
    orca.safety_space = 0.15
    orca.time_horizon = 5
    orca.time_horizon_obst = 5

    # social force
    sf = BaseConfig()
    sf.A = 2.
    sf.B = 1
    sf.KI = 1

    # mpc
    policy_config: DictConfig

    def __dict__(self):
        return {
            "orca.neighbor_dist": self.orca.neighbor_dist,
            "orca.safety_space": self.orca.safety_space,
            "orca.time_horizon": self.orca.time_horizon,
            "orca.time_horizon_obst": self.orca.time_horizon_obst,
            "sf.A": self.sf.A,
            "sf.B": self.sf.B,
            "sf.KI": self.sf.KI,
            "noise.add_noise": self.noise.add_noise,
            "noise.type": self.noise.type,
            "noise.magnitude": self.noise.magnitude,
        }

    def __init__(self, wmpc_config: DictConfig) -> None:
        self.policy_config: DictConfig = wmpc_config

        self.sim.human_num = wmpc_config.sim.human_num
        self.sim.square_width = wmpc_config.sim.position.rectangle.w
        self.sim.circle_radius = wmpc_config.sim.position.circle.r
        self.sim.done_if_all_agents_reached = wmpc_config.sim.done_if_all_agents_reached
        self.env.time_limit = wmpc_config.sim.time_limit
        self.env.eval_episodes = wmpc_config.eval_episodes

        self.humans.policy = wmpc_config.sim.robot_policy
        self.robot.policy = wmpc_config.sim.robot_policy
        if not (self.robot.policy in ["wnum_mpc", "vanilla_mpc", "mean_mpc"]):
            self.action_space.kinematics = "holonomic"

