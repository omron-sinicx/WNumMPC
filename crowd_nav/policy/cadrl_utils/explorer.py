import logging
import copy
import torch
from crowd_sim.envs.utils.info import *
from tqdm import tqdm

from training_utils import print_episode_info


class Explorer(object):
    def __init__(self, env, device, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.target_model = None

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    # @profile
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None,
                       print_failure=False, progress_bar=False):
        self.env.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        timeout_cases = []
        self.env.set_phase(phase)

        if progress_bar:
            bar = tqdm(total=k)
            bar.set_description("Running Episodes".format(phase))
        for i in range(k):
            ob, ob_ids = self.env.reset(phase)
            done = False
            states = []
            actions = []
            rewards = []
            while not done:
                action = self.env.robot.act(ob)
                ob, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                states.append(self.env.robot.policy.last_state)
                actions.append(action)
                rewards.append(reward)
                episode_info: EpisodeInfo = info["episode_info"]

                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)

            if isinstance(episode_info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(episode_info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(episode_info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                print(episode_info)
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                if isinstance(episode_info, ReachGoal) or isinstance(episode_info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(states, actions, rewards, imitation_learning)
            cumulative_rewards.append(sum([pow(self.gamma, t * self.env.robot.time_step * self.env.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))
            if progress_bar:
                bar.update(1)

        success_rate = success / k
        collision_rate = collision / k
        timeout_rate = timeout / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'.
                     format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
                            average(cumulative_rewards)))
        if phase in ['val', 'test']:
            num_step = sum(success_times + collision_times + timeout_times) / self.env.robot.time_step
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         too_close / num_step, average(min_dist))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

        return success_rate, collision_rate, timeout_rate, avg_nav_time, average(cumulative_rewards)

    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                # value = pow(self.gamma, (len(states) - 1 - i) * self.robot.time_step * self.robot.v_pref)
                value = sum([pow(self.gamma, max(t - i, 0) * self.env.robot.time_step * self.env.robot.v_pref) * reward
                             * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.gamma, self.env.robot.time_step * self.env.robot.v_pref)
                    value = reward + gamma_bar * self.target_model(next_state[0].unsqueeze(0)).data.item()
            value = torch.Tensor([value]).to(self.device)

            # # transform state of different human_num into fixed-size tensor
            # if len(state.size()) == 1:
            #     human_num = 1
            #     feature_size = state.size()[0]
            # else:
            #     human_num, feature_size = state.size()
            # if human_num != 5:
            #     padding = torch.zeros((5 - human_num, feature_size))
            #     state = torch.cat([state, padding])
            self.memory.push((state, value))


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
