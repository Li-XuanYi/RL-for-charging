"""Episode rollout and replay storage utilities."""

from __future__ import annotations

import threading

import numpy as np


class RolloutWorker:
    def __init__(self, env, agents, conf):
        self.conf = conf
        self.agents = agents
        self.env = env
        self.episode_limit = conf.episode_limit
        self.n_actions = conf.n_actions
        self.n_agents = conf.n_agents
        self.state_shape = conf.state_shape
        self.obs_shape = conf.obs_shape
        self.last_trace = []

    def generate_episode(self, epsilon, initial_socs=None):
        soc_history = [[], [], []]
        observations = []
        actions_buffer = []
        rewards = []
        states = []
        available_buffer = []
        action_onehot_buffer = []
        terminated_buffer = []
        padded = []

        self.env.reset(initial_socs=initial_socs)
        cells = (self.env.spm1, self.env.spm2, self.env.spm3)
        for index, cell in enumerate(cells):
            soc_history[index].append(cell.soc)

        terminated = False
        episode_reward = 0.0
        last_action = np.zeros((self.n_agents, self.n_actions))
        self.agents.policy.init_hidden(1)
        self.last_trace = []

        step = 0
        final_obs = self.env.get_obs()
        final_state = self.env.get_state(final_obs)
        while not terminated and step < self.episode_limit:
            obs = self.env.get_obs()
            state = self.env.get_state(obs)
            actions = []
            available_actions = []
            actions_onehot = []

            available_actions = [
                self.env.get_avail_agent_actions(agent_id)
                for agent_id in range(self.n_agents)
            ]
            if self.conf.algorithm == "dql":
                actions = self.agents.choose_joint_action(
                    state, available_actions, epsilon
                )
            else:
                actions = []
                for agent_id, available in enumerate(available_actions):
                    action = self.agents.choose_action(
                        obs[agent_id],
                        last_action[agent_id],
                        agent_id,
                        available,
                        epsilon,
                    )
                    actions.append(action)

            for agent_id, action in enumerate(actions):
                available = available_actions[agent_id]
                if not available[action]:
                    raise RuntimeError(
                        f"Selected unavailable action {action} for agent {agent_id}"
                    )
                onehot = np.zeros(self.n_actions)
                onehot[action] = 1.0
                actions_onehot.append(onehot)
                last_action[agent_id] = onehot

            reward, terminated = self.env.multi_step(actions, self.conf.beta)
            final_obs = self.env.get_obs()
            final_state = self.env.get_state(final_obs)
            for index, cell in enumerate(cells):
                soc_history[index].append(cell.soc)

            observations.append(obs)
            states.append(state)
            actions_buffer.append(np.reshape(actions, [self.n_agents, 1]))
            action_onehot_buffer.append(actions_onehot)
            available_buffer.append(available_actions)
            rewards.append([reward])
            terminated_buffer.append([terminated])
            padded.append([0.0])
            episode_reward += reward
            step += 1

            transition = dict(self.env.last_transition)
            transition["step"] = step
            transition["time_s"] = step * self.conf.sample_time
            self.last_trace.append(transition)

        observations.append(final_obs)
        states.append(final_state)
        next_observations = observations[1:]
        next_states = states[1:]
        observations = observations[:-1]
        states = states[:-1]

        final_available = [
            self.env.get_avail_agent_actions(agent_id)
            for agent_id in range(self.n_agents)
        ]
        available_buffer.append(final_available)
        next_available_buffer = available_buffer[1:]
        available_buffer = available_buffer[:-1]

        for _ in range(step, self.episode_limit):
            observations.append(np.zeros((self.n_agents, self.obs_shape)))
            actions_buffer.append(np.zeros([self.n_agents, 1]))
            states.append(np.zeros(self.state_shape))
            rewards.append([0.0])
            next_observations.append(np.zeros((self.n_agents, self.obs_shape)))
            next_states.append(np.zeros(self.state_shape))
            action_onehot_buffer.append(np.zeros((self.n_agents, self.n_actions)))
            available_buffer.append(np.zeros((self.n_agents, self.n_actions)))
            next_available_buffer.append(
                np.zeros((self.n_agents, self.n_actions))
            )
            padded.append([1.0])
            terminated_buffer.append([1.0])

        episode = {
            "o": observations,
            "s": states,
            "u": actions_buffer,
            "r": rewards,
            "o_": next_observations,
            "s_": next_states,
            "avail_u": available_buffer,
            "avail_u_": next_available_buffer,
            "u_onehot": action_onehot_buffer,
            "padded": padded,
            "terminated": terminated_buffer,
        }
        for key in episode:
            episode[key] = np.array([episode[key]])

        return episode, episode_reward, *soc_history


class ReplayBuffer:
    def __init__(self, conf):
        self.episode_limit = conf.episode_limit
        self.n_actions = conf.n_actions
        self.n_agents = conf.n_agents
        self.state_shape = conf.state_shape
        self.obs_shape = conf.obs_shape
        self.size = conf.buffer_size
        self.current_idx = 0
        self.current_size = 0
        self.buffers = {
            "o": np.empty(
                [self.size, self.episode_limit, self.n_agents, self.obs_shape]
            ),
            "u": np.empty([self.size, self.episode_limit, self.n_agents, 1]),
            "s": np.empty([self.size, self.episode_limit, self.state_shape]),
            "r": np.empty([self.size, self.episode_limit, 1]),
            "o_": np.empty(
                [self.size, self.episode_limit, self.n_agents, self.obs_shape]
            ),
            "s_": np.empty([self.size, self.episode_limit, self.state_shape]),
            "avail_u": np.empty(
                [self.size, self.episode_limit, self.n_agents, self.n_actions]
            ),
            "avail_u_": np.empty(
                [self.size, self.episode_limit, self.n_agents, self.n_actions]
            ),
            "u_onehot": np.empty(
                [self.size, self.episode_limit, self.n_agents, self.n_actions]
            ),
            "padded": np.empty([self.size, self.episode_limit, 1]),
            "terminated": np.empty([self.size, self.episode_limit, 1]),
        }
        self.lock = threading.Lock()

    def store_episode(self, episode_batch):
        batch_size = episode_batch["o"].shape[0]
        with self.lock:
            indices = self._get_storage_idx(inc=batch_size)
            for key in self.buffers:
                self.buffers[key][indices] = episode_batch[key]

    def sample(self, batch_size):
        indices = np.random.randint(0, self.current_size, batch_size)
        return {key: value[indices] for key, value in self.buffers.items()}

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_idx + inc <= self.size:
            indices = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            indices = np.concatenate(
                [np.arange(self.current_idx, self.size), np.arange(0, overflow)]
            )
            self.current_idx = overflow
        else:
            indices = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.size, self.current_size + inc)
        return indices[0] if inc == 1 else indices
