import numpy as np


class FullState(object):
    def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref, theta):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.gx = gx
        self.gy = gy
        self.v_pref = v_pref
        self.theta = theta

        self.position = (self.px, self.py)
        self.goal_position = (self.gx, self.gy)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy,
                                          self.v_pref, self.theta]])

    def get_position(self) -> np.ndarray:
        return np.array((self.px, self.py))

    def get_velocity(self) -> tuple[float, float]:
        return self.vx, self.vy

    def get_goal(self) -> tuple[float, float]:
        return self.gx, self.gy

    def get_theta(self) -> np.ndarray | float:
        return self.theta

    def get_vpref(self):
        return self.v_pref

    def get_heading(self) -> float:
        return np.arctan2(self.vy, self.vx)


class ObservableState(object):
    def __init__(self, px, py, vx, vy, radius):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius

        self.position = (self.px, self.py)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius]])

    def get_position(self) -> np.ndarray:
        return np.array((self.px, self.py))

    def get_velocity(self) -> tuple[float, float]:
        return self.vx, self.vy

    def dist(self, other: "ObservableState") -> float:
        return np.linalg.norm(np.array(self.position) - np.array(other.position))


class BallbotState(FullState):
    def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref, theta, lean_theta, lean_theta_dot):
        super().__init__(px, py, vx, vy, radius, gx, gy, v_pref, theta)
        self.lean_theta = lean_theta
        self.lean_theta_dot = lean_theta_dot
        self._heading: float = np.arctan2(self.vy, self.vx)
        self.theta = theta

    def get_theta(self) -> float:
        return self.lean_theta

    def get_theta_dot(self) -> float:
        return self.lean_theta_dot

    def get_heading(self) -> float:
        return self._heading

    def set_sim_heading(self, heading: float):
        self._heading = heading


class JointState(object):
    def __init__(self, self_state: FullState | BallbotState, human_states: list[ObservableState]):
        assert isinstance(self_state, FullState)
        for human_state in human_states:
            assert isinstance(human_state, ObservableState)

        self.self_state: FullState | BallbotState = self_state
        self.human_states: list[ObservableState] = human_states


class ObservableState_noV(object):
    def __init__(self, px, py, radius):
        self.px = px
        self.py = py
        self.radius = radius

        self.position = (self.px, self.py)

    def __add__(self, other):
        return other + (self.px, self.py, self.radius)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.radius]])


class JointState_noV(object):
    def __init__(self, self_state, human_states):
        assert isinstance(self_state, FullState)
        for human_state in human_states:
            assert isinstance(human_state, ObservableState_noV)

        self.self_state = self_state
        self.human_states = human_states
