"""Shared recurrent agents for QMIX and VDN."""

from __future__ import annotations

import numpy as np
import torch

from policy import QMIX
from dql import CentralDQL


class Agents:
    def __init__(self, conf):
        self.conf = conf
        self.device = conf.device
        self.n_actions = conf.n_actions
        self.n_agents = conf.n_agents
        self.episode_limit = conf.episode_limit
        self.policy = CentralDQL(conf) if conf.algorithm == "dql" else QMIX(conf)

    def choose_joint_action(self, state, available_actions, epsilon) -> list[int]:
        if self.conf.algorithm != "dql":
            raise RuntimeError("Joint action selection is only defined for DQL")
        return self.policy.choose_joint_action(state, available_actions, epsilon)

    def choose_action(
        self, obs, last_action, agent_num, available_actions, epsilon
    ) -> int:
        available_indices = np.nonzero(available_actions)[0]
        if available_indices.size == 0:
            raise RuntimeError(f"No available action for agent {agent_num}")

        inputs = obs.copy()
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.0
        if self.conf.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.conf.reuse_network:
            inputs = np.hstack((inputs, agent_id))

        hidden_state = self.policy.eval_hidden[:, agent_num, :]
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(self.device)
        available_tensor = (
            torch.tensor(available_actions, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            q_value, next_hidden = self.policy.eval_drqn_net(inputs, hidden_state)
        self.policy.eval_hidden[:, agent_num, :] = next_hidden
        q_value[available_tensor == 0.0] = -float("inf")

        if np.random.uniform() < epsilon:
            return int(np.random.choice(available_indices))
        return int(torch.argmax(q_value).item())

    def _get_max_episode_len(self, batch) -> int:
        terminated = batch["terminated"]
        lengths = []
        for episode in terminated:
            terminal_indices = np.nonzero(episode[:, 0] == 1)[0]
            lengths.append(
                int(terminal_indices[0] + 1)
                if terminal_indices.size
                else self.episode_limit
            )
        return max(lengths)

    def train(self, batch, training_episode=None):
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch:
            batch[key] = batch[key][:, :max_episode_len]
        return self.policy.learn(batch, max_episode_len, training_episode)
