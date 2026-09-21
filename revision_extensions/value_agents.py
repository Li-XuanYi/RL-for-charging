"""Discrete value-based policies for the revision experiments.

All currents entering this public interface are amperes, never action indices.
Requested actions label Q values; executed currents describe the next history.
Truncation is not termination.  Recurrent replay warms up with the complete
prefix and excludes padded transitions from the Bellman loss.

This module deliberately does not run experiments when imported.  Checkpoints
are trusted local PyTorch files; they do not include the caller's replay buffer,
environment, or experiment scheduler, and are not complete training resumes.
"""

from copy import deepcopy
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


N_ACTIONS = 16
N_FEATURES = 3
HIDDEN = 128
FORMAT_VERSION = 1
METHODS = {
    "qmix": ("qmix", True, True),
    "qmix_mlp": ("qmix", False, True),
    "vdn": ("vdn", True, True),
    "static_mixer": ("static", True, True),
    "iql_shared": ("none", True, True),
    "iql_independent": ("none", True, False),
    "dqn": ("none", False, False),
}


def _array(value, shape, name, dtype=np.float32):
    original = np.asarray(value)
    try:
        result = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if original.dtype.kind not in "biuf" or result.shape != shape:
        raise ValueError(f"{name} must be numeric with shape {shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains nonfinite values")
    return result


def _booleans(value, shape, name):
    result = _array(value, shape, name, np.float64)
    if not np.isin(result, (0, 1)).all():
        raise ValueError(f"{name} must contain only booleans or zero/one")
    return result.astype(bool)


def _currents(value, shape, name):
    result = _array(value, shape, name, np.float64)
    if np.any(result < -1e-7) or np.any(result > 7.5 + 1e-7):
        raise ValueError(f"{name} must lie in [0,7.5] A")
    # Only floating point boundary roundoff is clipped, never a physical action.
    return np.clip(result, 0.0, 7.5)


def _indices(value, shape, name):
    value = _currents(value, shape, name)
    scaled = 2.0 * value
    indices = np.rint(scaled)
    if not np.allclose(scaled, indices, atol=2e-6, rtol=0):
        raise ValueError(f"{name} must be on the 0.5 A discrete action grid")
    return indices.astype(np.int64)


def _executed_encoding(currents):
    """Encode actual amperage exactly, also after continuous power projection.

    Grid currents use the original one-hot encoding.  An off-grid current is
    represented by linear weights on its two adjacent bins; their weighted
    current equals the actual current.  It is not silently rounded to a label.
    The all-zero start-of-episode vector remains distinct from an executed 0 A.
    """
    scaled = np.asarray(currents, dtype=np.float64) * 2.0
    lower = np.floor(scaled).astype(np.int64)
    upper = np.minimum(lower + 1, N_ACTIONS - 1)
    fraction = scaled - lower
    eye = np.eye(N_ACTIONS, dtype=np.float32)
    return ((1.0 - fraction)[..., None] * eye[lower]
            + fraction[..., None] * eye[upper]).astype(np.float32)


class _CellNetwork(nn.Module):
    def __init__(self, input_size, recurrent):
        super().__init__()
        self.recurrent = recurrent
        self.fc1 = nn.Linear(input_size, HIDDEN)
        # The no-GRU ablation keeps hidden width 128, not equal parameter count.
        # Both counts are saved, so memory removal is not called a matched-size test.
        self.core = nn.GRUCell(HIDDEN, HIDDEN) if recurrent else nn.Linear(HIDDEN, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, N_ACTIONS)

    def forward(self, inputs, hidden):
        x = F.relu(self.fc1(inputs))
        state = self.core(x, hidden) if self.recurrent else F.relu(self.core(x))
        return self.fc2(state), state


class _Agents(nn.Module):
    def __init__(self, n_agents, recurrent, shared):
        super().__init__()
        self.n_agents, self.shared = n_agents, shared
        count = 1 if shared else n_agents
        self.cells = nn.ModuleList([
            _CellNetwork(N_FEATURES + N_ACTIONS + n_agents, recurrent)
            for _ in range(count)
        ])

    def forward(self, inputs, hidden):
        batch = inputs.shape[0]
        if self.shared:
            values, state = self.cells[0](inputs.reshape(batch * self.n_agents, -1),
                                           hidden.reshape(batch * self.n_agents, HIDDEN))
            return (values.reshape(batch, self.n_agents, N_ACTIONS),
                    state.reshape(batch, self.n_agents, HIDDEN))
        values, states = [], []
        for cell, network in enumerate(self.cells):
            q, h = network(inputs[:, cell], hidden[:, cell])
            values.append(q)
            states.append(h)
        return torch.stack(values, dim=1), torch.stack(states, dim=1)


class _CentralDQN(nn.Module):
    """Centralized feedforward DQN, with all 16**3 joint actions explicitly."""
    def __init__(self, n_agents):
        super().__init__()
        if n_agents != 3:
            raise ValueError("Central joint-action DQN is defined only for three cells (4096 actions)")
        self.n_agents = n_agents
        self.network = nn.Sequential(
            nn.Linear(n_agents * (N_FEATURES + N_ACTIONS), 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, N_ACTIONS ** n_agents),
        )

    def forward(self, inputs, hidden):
        # Cell order identifies the agent, so constant identity inputs add no data.
        x = inputs[..., :N_FEATURES + N_ACTIONS].reshape(inputs.shape[0], -1)
        return self.network(x), hidden


class _Mixer(nn.Module):
    def __init__(self, kind, n_agents):
        super().__init__()
        self.kind, self.n_agents = kind, n_agents
        state_size, width = n_agents * N_FEATURES, 256
        if kind == "qmix":
            self.hyper_w1 = nn.Sequential(nn.Linear(state_size, 64), nn.ReLU(),
                                          nn.Linear(64, n_agents * width))
            self.hyper_w2 = nn.Sequential(nn.Linear(state_size, 64), nn.ReLU(),
                                          nn.Linear(64, width))
            self.hyper_b1 = nn.Linear(state_size, width)
            self.hyper_b2 = nn.Sequential(nn.Linear(state_size, width), nn.ReLU(),
                                          nn.Linear(width, 1))
        elif kind == "static":
            # State-independent learned monotonic nonlinear mixer.  This removes
            # all state-conditioned hypernetworks, including their biases.
            self.w1 = nn.Parameter(torch.empty(n_agents, width))
            self.w2 = nn.Parameter(torch.empty(width, 1))
            self.b1 = nn.Parameter(torch.zeros(width))
            self.b2 = nn.Parameter(torch.zeros(1))
            nn.init.uniform_(self.w1, -1.0 / math.sqrt(n_agents), 1.0 / math.sqrt(n_agents))
            nn.init.uniform_(self.w2, -1.0 / math.sqrt(width), 1.0 / math.sqrt(width))
        elif kind not in ("none", "vdn"):
            raise ValueError(f"Unknown mixer {kind}")

    def forward(self, values, states):
        if self.kind == "none":
            return values
        if self.kind == "vdn":
            return values.sum(dim=-1, keepdim=True)
        if self.kind == "static":
            return F.elu(values @ self.w1.abs() + self.b1) @ self.w2.abs() + self.b2
        shape = values.shape[:-1]
        state = states.reshape(-1, self.n_agents * N_FEATURES)
        q = values.reshape(-1, 1, self.n_agents)
        w1 = self.hyper_w1(state).abs().reshape(-1, self.n_agents, 256)
        w2 = self.hyper_w2(state).abs().reshape(-1, 256, 1)
        hidden = F.elu(torch.bmm(q, w1) + self.hyper_b1(state).reshape(-1, 1, 256))
        return (torch.bmm(hidden, w2).reshape(-1, 1) + self.hyper_b2(state)).reshape(*shape, 1)


class ValuePolicy:
    learning_kind = "off_policy"
    action_kind = "discrete"

    def __init__(self, method: str, n_agents: int, seed: int, config: dict):
        if method not in METHODS:
            raise ValueError(f"Unknown value policy {method}; choose {tuple(METHODS)}")
        if isinstance(n_agents, bool) or not isinstance(n_agents, (int, np.integer)) or n_agents < 1:
            raise ValueError("n_agents must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0,2**63)")
        self.method, self.n_agents, self.seed = method, int(n_agents), int(seed)
        self.config = deepcopy(dict(config))
        kind, recurrent, shared = METHODS[method]
        self.kind = self.config.get("mixer", kind)
        self.recurrent = self.config.get("recurrent", recurrent)
        self.shared = self.config.get("shared", shared)
        if type(self.recurrent) is not bool or type(self.shared) is not bool:
            raise ValueError("recurrent and shared must be booleans")
        if method == "dqn" and (self.kind != "none" or self.recurrent or self.shared):
            raise ValueError("The central DQN baseline is feedforward without sharing or a mixer")
        self.architecture = {"method": method, "n_agents": self.n_agents,
                             "mixer": self.kind, "recurrent": self.recurrent,
                             "shared": self.shared, "agent_width": HIDDEN,
                             "mixer_width": 256, "hyper_width": 64,
                             "previous_action_encoding": "exact_barycentric_16_bins",
                             "identity_encoding": "cell_onehot_fixed_N"}
        self.gamma = float(self.config.get("gamma", 0.99))
        self.learning_rate = float(self.config.get("learning_rate", 2e-4))
        if not math.isfinite(self.gamma) or not 0 <= self.gamma <= 1:
            raise ValueError("gamma must lie in [0,1]")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        self.max_sequence_length = self.config.get("max_sequence_length", 80)
        self.burn_in = self.config.get("burn_in", 20)
        for key, value in (("max_sequence_length", self.max_sequence_length), ("burn_in", self.burn_in)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        self.double_q = self.config.get("double_q", False)
        if type(self.double_q) is not bool:
            raise ValueError("double_q must be boolean")
        self.device = torch.device(self.config.get("device", "cpu"))
        torch.manual_seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.agent = (_CentralDQN(self.n_agents) if method == "dqn" else
                      _Agents(self.n_agents, self.recurrent, self.shared)).to(self.device)
        self.mixer = _Mixer(self.kind, self.n_agents).to(self.device)
        self.target_agent = deepcopy(self.agent).eval().requires_grad_(False)
        self.target_mixer = deepcopy(self.mixer).eval().requires_grad_(False)
        self.params = list(self.agent.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.RMSprop(self.params, lr=self.learning_rate)
        self.updates = 0
        self.target_syncs = 0
        if method == "dqn":
            joint = np.indices((N_ACTIONS,) * self.n_agents).reshape(self.n_agents, -1).T.copy()
            self.joint_actions = torch.as_tensor(joint, dtype=torch.long, device=self.device)
        self.reset()

    def reset(self):
        self.hidden = torch.zeros(1, self.n_agents, HIDDEN, device=self.device)
        self.last = np.zeros((self.n_agents, N_ACTIONS), dtype=np.float32)

    def _joint_mask(self, mask):
        # Works for a BxNx16 mask and a BxTxNx16 mask without a 16**N tensor.
        result = torch.ones((*mask.shape[:-2], len(self.joint_actions)),
                            dtype=torch.bool, device=self.device)
        for cell in range(self.n_agents):
            result &= mask[..., cell, self.joint_actions[:, cell]]
        return result

    def act(self, obs, mask, training=False, epsilon=0.0):
        observation = _array(obs, (self.n_agents, N_FEATURES), "observation")
        available = _booleans(mask, (self.n_agents, N_ACTIONS), "action mask")
        if not available.any(axis=-1).all():
            raise ValueError("Every cell must allow at least one action")
        if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError("epsilon must lie in [0,1]")
        inputs = np.concatenate((observation, self.last, np.eye(self.n_agents, dtype=np.float32)), axis=-1)
        with torch.no_grad():
            q, hidden = self.agent(torch.as_tensor(inputs[None], device=self.device), self.hidden)
        if not torch.isfinite(q).all() or not torch.isfinite(hidden).all():
            raise FloatingPointError("Nonfinite policy output")
        if self.method == "dqn":
            joint_mask = self._joint_mask(torch.as_tensor(available[None], device=self.device))[0]
            joint_choice = int(q[0].masked_fill(~joint_mask, torch.finfo(q.dtype).min).argmax().item())
            choices = self.joint_actions[joint_choice].cpu().numpy().copy()
            if training and epsilon > 0 and (epsilon == 1 or self.rng.random() < epsilon):
                # Uniform independent allowed indices are uniform over the
                # Cartesian product, exactly the factorized valid joint set.
                choices = np.asarray([self.rng.choice(np.flatnonzero(row)) for row in available], dtype=np.int64)
        else:
            scores = q[0].cpu().numpy()
            scores[~available] = -np.inf
            choices = scores.argmax(axis=-1).astype(np.int64)
            if training and epsilon > 0:
                for cell in range(self.n_agents):
                    if epsilon == 1 or self.rng.random() < epsilon:
                        choices[cell] = int(self.rng.choice(np.flatnonzero(available[cell])))
        self.hidden = hidden.detach()
        currents = choices.astype(np.float64) * 0.5
        # Caller replaces this default immediately after a filtered transition.
        self.set_executed(currents)
        return currents, {"action_indices": choices.tolist(), "epsilon": float(epsilon if training else 0.0),
                          "method": self.method}

    def set_executed(self, currents_A):
        actual = _currents(currents_A, (self.n_agents,), "executed currents")
        self.last = _executed_encoding(actual)

    def _validate_episode(self, episode, number):
        prefix = f"episode {number}"
        required = ("obs", "masks", "actions", "executed_actions", "rewards", "done")
        if not isinstance(episode, dict) or any(key not in episode for key in required):
            raise ValueError(f"{prefix} requires {required}")
        actions = np.asarray(episode["actions"])
        if actions.ndim != 2 or actions.shape[1] != self.n_agents or len(actions) < 1:
            raise ValueError(f"{prefix} actions must have shape (T,{self.n_agents}), T>0")
        count = len(actions)
        obs = _array(episode["obs"], (count + 1, self.n_agents, N_FEATURES), prefix + " obs")
        masks = _booleans(episode["masks"], (count + 1, self.n_agents, N_ACTIONS), prefix + " masks")
        requested = _indices(actions, (count, self.n_agents), prefix + " requested currents")
        actual = _currents(episode["executed_actions"], (count, self.n_agents), prefix + " executed currents")
        rewards = _array(episode["rewards"], (count,), prefix + " rewards")
        done = _booleans(episode["done"], (count,), prefix + " done")
        if done[:-1].any():
            raise ValueError(f"{prefix} has transitions after an absorbing termination")
        if not masks.any(axis=-1).all():
            raise ValueError(f"{prefix} has an empty action mask")
        if not np.take_along_axis(masks[:-1], requested[..., None], axis=-1).all():
            raise ValueError(f"{prefix} contains a masked requested action")
        return count, obs, masks, requested, actual, rewards, done

    def learn(self, episodes):
        """One update over independently sampled complete-prefix learning windows.

        Each IQL cell minimizes its own Bellman error with the same team reward;
        it is not a centralized sum of Q values.  VDN sums Q values.  QMIX and
        static mixing preserve the monotonic decentralized argmax condition.
        """
        episodes = list(episodes)
        if not episodes:
            raise ValueError("learn requires at least one nonempty episode")
        checked = [self._validate_episode(ep, i) for i, ep in enumerate(episodes)]
        batch = len(checked)
        lengths = [min(ep[0], self.max_sequence_length) for ep in checked]
        starts = [int(self.rng.integers(0, ep[0] - length + 1)) if ep[0] > length else 0
                  for ep, length in zip(checked, lengths)]
        length, max_prefix = max(lengths), max(starts)
        size = N_FEATURES + N_ACTIONS + self.n_agents
        inputs = np.zeros((batch, length + 1, self.n_agents, size), dtype=np.float32)
        prefix = np.zeros((batch, max_prefix, self.n_agents, size), dtype=np.float32)
        actions = np.zeros((batch, length, self.n_agents), dtype=np.int64)
        rewards = np.zeros((batch, length, 1), dtype=np.float32)
        terminals = np.ones((batch, length, 1), dtype=bool)
        valid = np.zeros((batch, length, 1), dtype=bool)
        masks = np.zeros((batch, length + 1, self.n_agents, N_ACTIONS), dtype=bool)
        masks[..., 0] = True
        identity = np.eye(self.n_agents, dtype=np.float32)
        for b, (count, obs, ep_masks, requested, actual, reward, done) in enumerate(checked):
            start, span = starts[b], lengths[b]
            end = start + span
            full_input = np.zeros((count + 1, self.n_agents, size), dtype=np.float32)
            full_input[..., :N_FEATURES] = obs
            full_input[1:, :, N_FEATURES:N_FEATURES + N_ACTIONS] = _executed_encoding(actual)
            full_input[..., -self.n_agents:] = identity
            inputs[b, :span + 1] = full_input[start:end + 1]
            if start:
                prefix[b, :start] = full_input[:start]
            actions[b, :span] = requested[start:end]
            rewards[b, :span, 0] = reward[start:end]
            terminals[b, :span, 0] = done[start:end]
            valid[b, :span, 0] = True
            masks[b, :span + 1] = ep_masks[start:end + 1]
        inputs, prefix, actions, rewards, terminals, valid, masks = [
            torch.as_tensor(value, device=self.device)
            for value in (inputs, prefix, actions, rewards, terminals, valid, masks)
        ]
        states = inputs[..., :N_FEATURES].reshape(batch, length + 1, -1)
        hidden = torch.zeros(batch, self.n_agents, HIDDEN, device=self.device)
        target_hidden = torch.zeros_like(hidden)
        prefix_lengths = torch.as_tensor(starts, device=self.device)
        self.agent.train()
        self.mixer.train()
        if self.recurrent:
            with torch.no_grad():
                for chunk in range(0, max_prefix, self.burn_in):
                    for t in range(chunk, min(chunk + self.burn_in, max_prefix)):
                        _, candidate = self.agent(prefix[:, t], hidden)
                        _, target_candidate = self.target_agent(prefix[:, t], target_hidden)
                        active = (prefix_lengths > t).reshape(batch, 1, 1)
                        hidden = torch.where(active, candidate, hidden)
                        target_hidden = torch.where(active, target_candidate, target_hidden)
        online, targets = [], []
        for t in range(length + 1):
            q, hidden = self.agent(inputs[:, t], hidden)
            online.append(q)
            with torch.no_grad():
                tq, target_hidden = self.target_agent(inputs[:, t], target_hidden)
                targets.append(tq)
        online = torch.stack(online, dim=1)
        targets = torch.stack(targets, dim=1)
        floor = torch.finfo(targets.dtype).min
        if self.method == "dqn":
            multipliers = torch.as_tensor([N_ACTIONS ** i for i in range(self.n_agents - 1, -1, -1)],
                                          device=self.device)
            joint_labels = (actions * multipliers).sum(dim=-1, keepdim=True)
            total = online[:, :-1].gather(-1, joint_labels)
            with torch.no_grad():
                joint_next_mask = self._joint_mask(masks[:, 1:])
                selection = online[:, 1:].detach() if self.double_q else targets[:, 1:]
                next_labels = selection.masked_fill(~joint_next_mask, floor).argmax(dim=-1, keepdim=True)
                target_total = targets[:, 1:].gather(-1, next_labels)
        else:
            chosen = online[:, :-1].gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            total = self.mixer(chosen, states[:, :-1])
            with torch.no_grad():
                selection = online[:, 1:].detach() if self.double_q else targets[:, 1:]
                next_labels = selection.masked_fill(~masks[:, 1:], floor).argmax(dim=-1, keepdim=True)
                next_values = targets[:, 1:].gather(-1, next_labels).squeeze(-1)
                target_total = self.target_mixer(next_values, states[:, 1:])
        with torch.no_grad():
            continuation = torch.where(terminals, torch.zeros_like(target_total), target_total)
            td_target = rewards + self.gamma * continuation
        # IQL broadcasts the team reward to N local targets.  Mean over both
        # transitions and cells avoids making the effective step scale with N.
        errors = (total - td_target).masked_select(valid.expand_as(total))
        loss = errors.square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite value-policy Bellman loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.params, 10.0, error_if_nonfinite=True)
        self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in self.params):
            raise FloatingPointError("Nonfinite value-policy parameter after update")
        self.updates += 1
        return {"loss": float(loss.detach().cpu()), "gradient_norm": float(norm.detach().cpu()),
                "updates": self.updates, "batch_episodes": batch,
                "learning_transitions": int(sum(lengths)), "prefix_transitions": int(sum(starts)),
                "local_bellman_losses": bool(self.kind == "none" and self.method != "dqn")}

    def sync(self):
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.target_agent.eval()
        self.target_mixer.eval()
        self.target_syncs += 1

    def parameter_count(self):
        """Online trainable parameters; targets and optimizer slots are excluded."""
        return sum(p.numel() for p in self.params if p.requires_grad)

    def parameter_breakdown(self):
        return {"agent": sum(p.numel() for p in self.agent.parameters()),
                "mixer": sum(p.numel() for p in self.mixer.parameters()),
                "total_online": self.parameter_count(),
                "no_gru_matching": "same_hidden_width_not_same_parameter_count",
                "cross_N_policy_transfer": False}

    def save(self, path, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = {"numpy": deepcopy(self.rng.bit_generator.state), "torch_cpu": torch.get_rng_state()}
        if self.device.type == "cuda":
            rng["torch_cuda"] = torch.cuda.get_rng_state_all()
        value = {"format_version": FORMAT_VERSION, "architecture": self.architecture,
                 "config": self.config, "seed": self.seed,
                 "agent": self.agent.state_dict(), "mixer": self.mixer.state_dict(),
                 "target_agent": self.target_agent.state_dict(),
                 "target_mixer": self.target_mixer.state_dict(),
                 "optimizer": self.optimizer.state_dict(), "rng": rng,
                 "hidden": self.hidden.detach().cpu(), "last": self.last.copy(),
                 "updates": self.updates, "target_syncs": self.target_syncs,
                 "parameter_counts": self.parameter_breakdown(),
                 "metadata": deepcopy(metadata), "complete_training_resume": False}
        temporary = path.with_name(path.name + ".tmp")
        torch.save(value, temporary)
        os.replace(temporary, path)

    def load(self, path):
        """Load only a trusted checkpoint produced by these local experiments."""
        value = torch.load(Path(path), map_location=self.device, weights_only=False)
        if value.get("format_version") != FORMAT_VERSION or value.get("architecture") != self.architecture:
            raise ValueError("Checkpoint format or architecture does not match this policy")
        self.agent.load_state_dict(value["agent"])
        self.mixer.load_state_dict(value["mixer"])
        self.target_agent.load_state_dict(value["target_agent"])
        self.target_mixer.load_state_dict(value["target_mixer"])
        self.optimizer.load_state_dict(value["optimizer"])
        for module in (self.agent, self.mixer, self.target_agent, self.target_mixer):
            if any(not torch.isfinite(p).all() for p in module.parameters()):
                raise ValueError("Checkpoint contains nonfinite parameters")
        hidden = value["hidden"].to(self.device)
        if hidden.shape != (1, self.n_agents, HIDDEN) or not torch.isfinite(hidden).all():
            raise ValueError("Checkpoint has invalid rollout hidden state")
        last = _array(value["last"], (self.n_agents, N_ACTIONS), "checkpoint previous-action encoding")
        if np.any(last < 0) or np.any(last > 1):
            raise ValueError("Checkpoint previous-action encoding has invalid weights")
        sums = last.sum(axis=-1)
        if not np.all(np.isclose(sums, 0, atol=1e-6) | np.isclose(sums, 1, atol=1e-6)):
            raise ValueError("Checkpoint previous-action encoding must sum to zero or one")
        self.hidden, self.last = hidden.detach(), last.copy()
        self.config = deepcopy(value["config"])
        self.seed = int(value["seed"])
        self.gamma = float(self.config.get("gamma", 0.99))
        self.learning_rate = float(self.config.get("learning_rate", 2e-4))
        self.max_sequence_length = int(self.config.get("max_sequence_length", 80))
        self.burn_in = int(self.config.get("burn_in", 20))
        self.double_q = bool(self.config.get("double_q", False))
        self.updates, self.target_syncs = int(value["updates"]), int(value["target_syncs"])
        self.rng.bit_generator.state = deepcopy(value["rng"]["numpy"])
        torch.set_rng_state(value["rng"]["torch_cpu"].cpu())
        if self.device.type == "cuda" and "torch_cuda" in value["rng"]:
            states = value["rng"]["torch_cuda"]
            if len(states) != torch.cuda.device_count():
                raise ValueError("Checkpoint CUDA RNG device count does not match")
            torch.cuda.set_rng_state_all([state.cpu() for state in states])
        self.target_agent.eval()
        self.target_mixer.eval()
        return deepcopy(value["metadata"])
