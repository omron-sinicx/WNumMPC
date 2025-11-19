import logging
import os
import torch
from omegaconf import DictConfig
from crowd_nav.policy.cadrl_utils.trainer import Trainer
from crowd_nav.policy.cadrl_utils.memory import ReplayMemory
from crowd_nav.policy.cadrl_utils.explorer import Explorer
from crowd_sim.envs import CrowdSim
from training_utils import load_cadrl_config, trial
from config.config import Config as NavConfig
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import pickle


def eval_model(model, config, device):

    episode_num = 30
    
    trial_env = CrowdSim(seed=0)
    config.sim.done_if_all_agents_reached = True
    trial_env.configure(config)
    trial_env.set_policy("cadrl")
    trial_env.set_phase("test")
    trial_env.robot.policy.model.load_state_dict(model.state_dict())
    trial_env.robot.policy.set_device(device)

    result: dict = trial(trial_env, episode_num=episode_num, print_info = True, visualize=False)
    del trial_env

    return result

def main():
    resume = False
    common_param = "default"
    device = "cuda:0"
    eval_episode_num: int = 1

    # configure environment
    dict_config: DictConfig = load_cadrl_config(cadrl_param_name=common_param)
    config: NavConfig = NavConfig(dict_config)
    np.random.seed(0)
    torch.manual_seed(0)
    env = CrowdSim(seed=0)

    config.sim.done_if_all_agents_reached = False
    env.configure(config)
    env.set_phase("test")

    # configure paths
    output_dir = './models/CADRL/human_{}/'.format(env.human_num)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    il_weight_file = os.path.join(output_dir, 'il_model.pth')
    memory_file = os.path.join(output_dir, 'memory.pkl')
    rl_weight_file = os.path.join(output_dir, 'rl_model_final.pth')
    best_rl_weight_file = os.path.join(output_dir, 'rl_model_best.pth')
    

    # Summary Writer
    dir_path = "./datas/tensorboard/CADRL/human_{}/".format(env.human_num)
    os.makedirs(dir_path, exist_ok=True)
    writer = SummaryWriter(log_dir=dir_path)

    # read training parameters
    cadrl_param: DictConfig = config.policy_config.cadrl_param
    rl_learning_rate = cadrl_param.train.rl_learning_rate
    train_batches = cadrl_param.train.train_batches
    train_episodes = cadrl_param.train.train_episodes
    sample_episodes = cadrl_param.train.sample_episodes
    target_update_interval = cadrl_param.train.target_update_interval
    evaluation_interval = cadrl_param.train.evaluation_interval
    capacity = cadrl_param.train.capacity

    epsilon_start = cadrl_param.train.epsilon_start
    epsilon_end = cadrl_param.train.epsilon_end
    epsilon_decay = cadrl_param.train.epsilon_decay
    checkpoint_interval = cadrl_param.train.checkpoint_interval

    # configure trainer and explorer
    memory = ReplayMemory(capacity)
    model = env.robot.policy.get_model()
    model = model.to(device)
    # batch_size = train_config.getint('trainer', 'batch_size')
    batch_size = cadrl_param.trainer.batch_size
    trainer = Trainer(model, memory, device, batch_size)
    explorer = Explorer(env, device, memory, env.robot.policy.gamma, target_policy=env.robot.policy)

    # imitation learning
    if resume:
        if not os.path.exists(rl_weight_file):
            logging.error('RL weights does not exist')
        model.load_state_dict(torch.load(rl_weight_file))
        rl_weight_file = os.path.join(output_dir, 'resumed_rl_model.pth')
        logging.info('Load reinforcement learning trained weights. Resume training')
    elif env.human_num > 2:
        model.load_state_dict(torch.load('./models/CADRL/human_2/rl_model_best.pth'))
        logging.info('Load weights trained in three agents.')        
    elif os.path.exists(il_weight_file):
        model.load_state_dict(torch.load(il_weight_file))
        logging.info('Load imitation learning trained weights.')
    else:
        il_episodes = cadrl_param.imitation_learning.il_episodes
        il_policy = cadrl_param.imitation_learning.il_policy
        il_epochs = cadrl_param.imitation_learning.il_epochs
        il_learning_rate = cadrl_param.imitation_learning.il_learning_rate
        trainer.set_learning_rate(il_learning_rate)

        env.set_policy(il_policy)
        if os.path.exists(memory_file):
            with open(memory_file, "rb") as f:
                trainer.memory = pickle.load(f)
        else:
            explorer.run_k_episodes(il_episodes, 'train', update_memory=True, imitation_learning=True, progress_bar=True)
            with open(memory_file, "wb") as f:
                pickle.dump(trainer.memory, f)
        trainer.optimize_epoch(il_epochs, writer)
        #torch.save(model.state_dict(), il_weight_file)
        logging.info('Finish imitation learning. Weights saved.')
        logging.info('Experience set size: %d/%d', len(memory), memory.capacity)
        exit(0)
    explorer.update_target_model(model)

    # reinforcement learning
    env.configure(config)
    env.set_phase("train")
    env.set_policy("cadrl")
    env.robot.print_info()
    trainer.set_learning_rate(rl_learning_rate)

    # fill the memory pool with some RL experience
    if resume:
        env.robot.policy.set_epsilon(epsilon_end)
        explorer.run_k_episodes(eval_episode_num, 'train', update_memory=True, episode=0)
        logging.info('Experience set size: %d/%d', len(memory), memory.capacity)

    max_success = -1.0
    min_etg= float('inf')
    
    for episode in tqdm(range(train_episodes), desc='Training episodes'):
        env.robot.policy.model.load_state_dict(model.state_dict())
        env.sync_policy_setting()
        if resume:
            epsilon = epsilon_end
        else:
            if episode < epsilon_decay:
                epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
            else:
                epsilon = epsilon_end
        env.robot.policy.set_epsilon(epsilon)

        # evaluate the model
        if episode % evaluation_interval == 0:
            #success_rate, _, _, avg_nav_time, ave_reward = explorer.run_k_episodes(eval_episode_num, 'val', episode=episode)
            #writer.add_scalar("success_rate", success_rate, episode)
            #writer.add_scalar("avg_nav_time", avg_nav_time, episode)
            #writer.add_scalar("ave_reward", ave_reward, episode)
            result = eval_model(model, config, device)
            writer.add_scalar("success_rate", result["success_rate"], episode)
            writer.add_scalar("ave_nav_time", result["ave_nav_time"], episode)
            writer.add_scalar("ave_extra_time_to_goals", result["ave_extra_time_to_goals"], episode)
            if max_success < result["success_rate"] or (
	            min_etg > result["ave_extra_time_to_goals"] and max_success == result["success_rate"]):
                torch.save(model.state_dict(), best_rl_weight_file)
                min_etg = result["ave_extra_time_to_goals"]
                max_success = result["success_rate"]
                print("model saved!")

        # sample k episodes into memory and optimize over the generated memory
        explorer.run_k_episodes(sample_episodes, 'train', update_memory=True, episode=episode)
        while not len(memory) >= sample_episodes:
            explorer.run_k_episodes(1, 'train', update_memory=True, episode=episode)

        average_loss = trainer.optimize_batch(train_batches)

        if episode % target_update_interval == 0:
            explorer.update_target_model(model)

        if episode != 0 and episode % checkpoint_interval == 0:
            torch.save(model.state_dict(), rl_weight_file)

        writer.add_scalar("epsilon", epsilon, episode)
        writer.add_scalar("average_loss", average_loss, episode)

    # final test
    explorer.run_k_episodes(eval_episode_num, 'test', print_failure=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
