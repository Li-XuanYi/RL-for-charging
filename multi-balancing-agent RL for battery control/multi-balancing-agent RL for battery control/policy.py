"""QMIX/VDN policy implementation used by supplementary experiments."""

from __future__ import annotations

import torch

from NN import DRQN, QMIXNET, VDNMixer


class QMIX:
    def __init__(self, conf):
        self.conf = conf
        self.device = conf.device
        self.n_actions = conf.n_actions
        self.n_agents = conf.n_agents
        self.state_shape = conf.state_shape
        self.obs_shape = conf.obs_shape

        input_shape = self.obs_shape
        if conf.last_action:
            input_shape += self.n_actions
        if conf.reuse_network:
            input_shape += self.n_agents

        self.eval_drqn_net = DRQN(input_shape, conf).to(self.device)
        self.target_drqn_net = DRQN(input_shape, conf).to(self.device)
        mixer_type = QMIXNET if conf.mixer == "qmix" else VDNMixer
        self.eval_qmix_net = mixer_type(conf).to(self.device)
        self.target_qmix_net = mixer_type(conf).to(self.device)
        self._update_targets()

        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(
            self.eval_drqn_net.parameters()
        )
        if conf.optimizer != "RMS":
            raise ValueError(f"Unsupported optimizer: {conf.optimizer}")
        self.optimizer = torch.optim.RMSprop(
            self.eval_parameters, lr=conf.learning_rate
        )

        self.eval_hidden = None
        self.target_hidden = None
        self.learn_steps = 0

    def _update_targets(self) -> None:
        self.target_drqn_net.load_state_dict(self.eval_drqn_net.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

    def learn(self, batch, max_episode_len, training_episode=None):
        episode_num = batch["o"].shape[0]
        self.init_hidden(episode_num)
        for key in batch:
            dtype = torch.long if key == "u" else torch.float32
            batch[key] = torch.tensor(batch[key], dtype=dtype)

        states = batch["s"].to(self.device)
        actions = batch["u"].to(self.device)
        rewards = batch["r"].to(self.device)
        next_states = batch["s_"].to(self.device)
        next_available = batch["avail_u_"].to(self.device)
        terminated = batch["terminated"].to(self.device)
        mask = (1.0 - batch["padded"].float()).to(self.device)

        q_evals, q_targets = self.get_q_values(batch, max_episode_len)
        q_evals = torch.gather(q_evals, dim=3, index=actions).squeeze(3)
        q_targets[next_available == 0.0] = -1e9
        q_targets = q_targets.max(dim=3)[0]

        q_total_eval = self.eval_qmix_net(q_evals, states)
        q_total_target = self.target_qmix_net(q_targets, next_states)
        targets = rewards + self.conf.gamma * q_total_target * (1.0 - terminated)

        td_error = q_total_eval - targets.detach()
        masked_td_error = mask * td_error
        loss = (masked_td_error**2).sum() / mask.sum().clamp_min(1.0)

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

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch["o"].shape[0]
        q_evals = []
        q_targets = []
        for transition_idx in range(max_episode_len):
            inputs, next_inputs = self._get_inputs(batch, transition_idx)
            inputs = inputs.to(self.device)
            next_inputs = next_inputs.to(self.device)
            self.eval_hidden = self.eval_hidden.to(self.device)
            self.target_hidden = self.target_hidden.to(self.device)

            q_eval, self.eval_hidden = self.eval_drqn_net(inputs, self.eval_hidden)
            q_target, self.target_hidden = self.target_drqn_net(
                next_inputs, self.target_hidden
            )
            q_evals.append(q_eval.view(episode_num, self.n_agents, -1))
            q_targets.append(q_target.view(episode_num, self.n_agents, -1))

        return torch.stack(q_evals, dim=1), torch.stack(q_targets, dim=1)

    def _get_inputs(self, batch, transition_idx):
        observations = batch["o"][:, transition_idx]
        next_observations = batch["o_"][:, transition_idx]
        action_onehot = batch["u_onehot"]
        episode_num = observations.shape[0]

        inputs = [observations]
        next_inputs = [next_observations]
        if self.conf.last_action:
            previous = (
                torch.zeros_like(action_onehot[:, transition_idx])
                if transition_idx == 0
                else action_onehot[:, transition_idx - 1]
            )
            inputs.append(previous)
            next_inputs.append(action_onehot[:, transition_idx])

        if self.conf.reuse_network:
            agent_ids = torch.eye(self.n_agents).unsqueeze(0).expand(
                episode_num, -1, -1
            )
            inputs.append(agent_ids)
            next_inputs.append(agent_ids)

        inputs = torch.cat(
            [value.reshape(episode_num * self.n_agents, -1) for value in inputs],
            dim=1,
        )
        next_inputs = torch.cat(
            [
                value.reshape(episode_num * self.n_agents, -1)
                for value in next_inputs
            ],
            dim=1,
        )
        return inputs, next_inputs

    def init_hidden(self, episode_num):
        shape = (episode_num, self.n_agents, self.conf.drqn_hidden_dim)
        self.eval_hidden = torch.zeros(shape)
        self.target_hidden = torch.zeros(shape)

    def save_model(self, path):
        torch.save(
            {
                "eval_drqn_net": self.eval_drqn_net.state_dict(),
                "target_drqn_net": self.target_drqn_net.state_dict(),
                "eval_qmix_net": self.eval_qmix_net.state_dict(),
                "target_qmix_net": self.target_qmix_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "learn_steps": self.learn_steps,
            },
            path,
        )

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.eval_drqn_net.load_state_dict(checkpoint["eval_drqn_net"])
        self.target_drqn_net.load_state_dict(checkpoint["target_drqn_net"])
        self.eval_qmix_net.load_state_dict(checkpoint["eval_qmix_net"])
        self.target_qmix_net.load_state_dict(checkpoint["target_qmix_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.learn_steps = int(checkpoint.get("learn_steps", 0))
