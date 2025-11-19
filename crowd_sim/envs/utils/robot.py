from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState, JointState_noV
from config.config import Config


class Robot(Agent):
    def __init__(self, config: Config, section: str, agent_id: int) -> None:
        super().__init__(config, section, agent_id)

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')

        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    def act_noV(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState_noV(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    def actWithJointState(self, ob):
        action = self.policy.predict(ob)
        return action
