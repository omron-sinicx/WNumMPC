from omegaconf import DictConfig


class BaseConfig(object):
    def __init__(self):
        pass


class Config(object):
    env = BaseConfig()
    env.env_name = 'CrowdSimDict-v0'  # name of the environment
    env.time_step = 0.10

    reward = BaseConfig()
    # reward.success_reward = 10
    # reward.collision_penalty = -10
    reward.success_reward = 1
    reward.collision_penalty = -1
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

    # Real environment settings
    #wheel_cos= 0.935048188903213
    #wheel_dist= 0.22565681908597401
    scale= 100
    time_scale = 1.0
    serial_port = ["/dev/ttyUSB0", "/dev/ttyUSB1"]
    max_speed = 60.0
    #max_speed = 75.0
    wheel_const = 7.5 # max_rotation_speed / max_speed
    #wheel_const = 8.295
    #max_speed = 1600.0
    max_x =  360/scale
    min_x = -400/scale
    max_y = 260/scale
    min_y = -260/scale

    # Human settings
    humans = BaseConfig()
    humans.visible = True  # a human is visible to other humans and the robot
    humans.radius = 0.20
    humans.v_pref = max_speed / scale * time_scale
    humans.sensor = "coordinates"
    humans.FOV = 2.  # FOV = this values * PI

    # Robot settings
    robot = BaseConfig()
    robot.visible = True  # the robot is visible to humans
    robot.radius = humans.radius
    robot.v_pref = humans.v_pref
    robot.sensor = "coordinates"
    robot.FOV = 2.  # FOV = this values * PI

    noise = BaseConfig()
    noise.add_noise = False
    # uniform, gaussian
    noise.type = "uniform"
    noise.magnitude = 0.1

    action_space = BaseConfig()
    # action_space.kinematics = "ballbot"
    action_space.kinematics = "diff_wheel"

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
        self.sim.circle_radius_x = wmpc_config.sim.position.circle.rx
        self.sim.circle_radius_y = wmpc_config.sim.position.circle.ry
        self.sim.min_dist = wmpc_config.sim.position.min_dist
        self.sim.done_if_all_agents_reached = wmpc_config.sim.done_if_all_agents_reached
        self.env.time_limit = wmpc_config.sim.time_limit
        self.env.eval_episodes = wmpc_config.eval_episodes

        self.humans.policy = wmpc_config.sim.robot_policy
        self.robot.policy = wmpc_config.sim.robot_policy
        if not (self.robot.policy in ["wnum_mpc", "vanilla_mpc", "mean_mpc"]):
            self.action_space.kinematics = "holonomic"

