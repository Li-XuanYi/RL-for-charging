
import numpy as np
from agent import Agents
from utils import RolloutWorker, ReplayBuffer
from SPM import SPM, MultiSPM
import matplotlib.pyplot as plt

from config import Config

conf = Config()
num_of_agent = 3
obs_shape = 3
state_shape = num_of_agent * obs_shape
actions = np.arange(-2, 7.5, 0.5)
n_actions = len(actions)
episode_limit = 50

def train():

    reward_record = []
    env = MultiSPM(num_of_agent, state_shape, obs_shape, n_actions, episode_limit, actions)
    env_info = env.get_env_info()
    conf.set_env_info(env_info)
    agents = Agents(conf)
    # agents.policy.load_model("QMIX_network 0.3 0.5 0.7 800")
    rollout_worker = RolloutWorker(env, agents, conf)
    buffer = ReplayBuffer(conf)
    #
    # reward = evaluate(rollout_worker)
    # print(reward)

    for epoch in range(conf.n_epochs):

        epsilon = 0.5 - 0.5 * (epoch + 1) / conf.n_epochs
        episode, episode_reward, _, _, _ = rollout_worker.generate_episode(epsilon)
        reward_record.append(episode_reward)
        print(f"这是第{epoch+1}次，奖励为{episode_reward}")

        buffer.store_episode(episode)
        if buffer.current_size >= 5:
            mini_batch = buffer.sample(min(buffer.current_size, conf.batch_size))
            agents.train(mini_batch)

        # if epoch % conf.evaluate_epoch == 0 and epoch != 0:
        #     episode_reward = evaluate(rollout_worker)
        #     print(f"评估奖励为{episode_reward}")

    agents.policy.save_model("QMIX_network 0.3 0.5 0.7 800 diff")

    return reward_record

def evaluate(rollout_worker):

    soc1 = []
    soc2 = []
    soc3 = []

    episode_rewards = 0
    for epoch in range(conf.evaluate_number):
        epsilon = 0
        _, episode_reward, soc1, soc2, soc3 = rollout_worker.generate_episode(epsilon)
        episode_rewards += episode_reward

    episodes = range(1, len(soc1) + 1)
    plt.plot(episodes, soc1, linestyle='-', color='orange', label="battery1")
    plt.plot(episodes, soc2, linestyle='-', color='b', label="battery2")
    plt.plot(episodes, soc3, linestyle='-', color='g', label="battery3")
    plt.show()

    return episode_rewards / conf.evaluate_number


if __name__ == "__main__":

    if conf.train:
        reward_record = train()
        np.savetxt("sum_reward 0.3 0.5 0.7 800 diff", reward_record)

    # data1 = np.loadtxt("sum_reward 0.3 0.5 0.7 800")
    # episodes = range(1, len(data1) + 1)
    # plt.plot(episodes, data1, linestyle='-', color='orange')
    # plt.show()

