"""Matched centralized recurrent DQL baseline for the three-cell controller."""

from __future__ import annotations

import numpy as np
import torch

from NN import CentralDRQN


class CentralDQL:
    """Map the joint pack state to one Cartesian product of cell currents.

    This is a single-agent baseline: one recurrent Q-network observes the same
    global state available during QMIX centralized training and selects all
    per-cell currents simultaneously. The environment, action mask, reward,
    decision interval, and terminal rule are unchanged.
    """

    def __init__(self, conf):
        self.conf = conf
        self.device = conf.device
        self.n_actions = conf.n_actions
        self.n_agents = conf.n_agents
        self.n_joint_actions = self.n_actions ** self.n_agents

        self.eval_net = CentralDRQN(conf.state_shape, conf).to(self.device)
        self.target_net = CentralDRQN(conf.state_shape, conf).to(self.device)
        self._update_targets()
        self.eval_parameters = list(self.eval_net.parameters())
        if conf.optimizer != "RMS":
            raise ValueError(f"Unsupported optimizer: {conf.optimizer}")
        self.optimizer = torch.optim.RMSprop(
            self.eval_parameters, lr=conf.learning_rate
        )
        self.eval_hidden = None
        self.target_hidden = None
        self.learn_steps = 0

    def _update_targets(self) -> None:
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode [..., n_agents] per-cell action indices as one joint index."""
        encoded = torch.zeros(actions.shape[:-1], dtype=torch.long, device=actions.device)
        for agent_index in range(self.n_agents):
            encoded = encoded * self.n_actions + actions[..., agent_index].long()
        return encoded

    def decode_action(self, joint_action: int) -> list[int]:
        if not 0 <= joint_action < self.n_joint_actions:
            raise ValueError(f"joint action {joint_action} is out of range")
        actions = [0] * self.n_agents
        value = int(joint_action)
        for agent_index in range(self.n_agents - 1, -1, -1):
            actions[agent_index] = value % self.n_actions
            value //= self.n_actions
        return actions

    def _joint_availability(self, available: torch.Tensor) -> torch.Tensor:
        """Expand [batch, agents, actions] masks to [batch, joint_actions]."""
        mask = available[:, 0, :]
        for agent_index in range(1, self.n_agents):
            mask = (
                mask.unsqueeze(-1)
                * available[:, agent_index, :].unsqueeze(1)
            ).reshape(available.shape[0], -1)
        return mask

    def choose_joint_action(self, state, available_actions, epsilon: float) -> list[int]:
        available = torch.tensor(
            np.asarray(available_actions), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        joint_available = self._joint_availability(available)[0]
        available_indices = torch.nonzero(joint_available > 0.0).flatten()
        if available_indices.numel() == 0:
            raise RuntimeError("No jointly available DQL action")

        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        hidden = self.eval_hidden.to(self.device)
        with torch.no_grad():
            q_values, self.eval_hidden = self.eval_net(state_tensor, hidden)
        q_values[0, joint_available == 0.0] = -float("inf")
        if np.random.uniform() < epsilon:
            selected = int(
                available_indices[
                    np.random.randint(0, available_indices.numel())
                ].item()
            )
        else:
            selected = int(torch.argmax(q_values, dim=1).item())
        return self.decode_action(selected)

    def learn(self, batch, max_episode_len, training_episode=None):
        episode_num = batch["s"].shape[0]
        self.init_hidden(episode_num)
        tensors = {
            key: torch.tensor(
                value,
                dtype=torch.long if key == "u" else torch.float32,
                device=self.device,
            )
            for key, value in batch.items()
        }

        actions = tensors["u"].squeeze(-1)
        joint_actions = self.encode_actions(actions)
        rewards = tensors["r"]
        terminated = tensors["terminated"]
        mask = 1.0 - tensors["padded"]

        chosen_values = []
        target_values = []
        for transition_index in range(max_episode_len):
            q_eval, self.eval_hidden = self.eval_net(
                tensors["s"][:, transition_index], self.eval_hidden
            )
            with torch.no_grad():
                q_target, self.target_hidden = self.target_net(
                    tensors["s_"][:, transition_index], self.target_hidden
                )
                joint_available = self._joint_availability(
                    tensors["avail_u_"][:, transition_index]
                )
                q_target[joint_available == 0.0] = -1e9
                target_values.append(q_target.max(dim=1, keepdim=True).values)
            chosen_values.append(
                q_eval.gather(
                    1, joint_actions[:, transition_index].unsqueeze(1)
                )
            )

        q_eval = torch.stack(chosen_values, dim=1)
        q_target = torch.stack(target_values, dim=1)
        targets = rewards + self.conf.gamma * q_target * (1.0 - terminated)
        td_error = q_eval - targets
        loss = ((mask * td_error) ** 2).sum() / mask.sum().clamp_min(1.0)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.eval_parameters, self.conf.grad_norm_clip
        )
        self.optimizer.step()

        self.learn_steps += 1
        update_index = (
            self.learn_steps if training_episode is None else int(training_episode)
        )
        if update_index % self.conf.update_target_params == 0:
            self._update_targets()
        return float(loss.detach().cpu().item())

    def init_hidden(self, episode_num):
        shape = (episode_num, self.conf.drqn_hidden_dim)
        self.eval_hidden = torch.zeros(shape, device=self.device)
        self.target_hidden = torch.zeros(shape, device=self.device)

    def save_model(self, path):
        torch.save(
            {
                "eval_net": self.eval_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "learn_steps": self.learn_steps,
                "n_joint_actions": self.n_joint_actions,
            },
            path,
        )

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        if int(checkpoint["n_joint_actions"]) != self.n_joint_actions:
            raise ValueError("DQL checkpoint joint-action dimension mismatch")
        self.eval_net.load_state_dict(checkpoint["eval_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.learn_steps = int(checkpoint.get("learn_steps", 0))
