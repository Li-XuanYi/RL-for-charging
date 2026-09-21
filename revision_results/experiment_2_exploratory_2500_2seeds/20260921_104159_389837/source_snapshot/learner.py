"""Recurrent QMIX with explicit requested/executed action histories.

Observations are already normalized by the environment:
    (physical_[SOC, V, K] - [0.5, 3.5, 308.0]) / [0.5, 1.0, 11.0].
Do not normalize them again here.  Action indices 0..15 denote 0..7.5 A.

With a safety filter, the *requested* action is the agent's decision and labels
the learned Q value.  The actual executed action is included in the next
recurrent input, so the history describes the transition that really occurred.
The filter is part of that method's environment transition function.

All episodes contain their actual final observation.  A time limit is a
truncation and must have done=False; goal and numerical failure are absorbing
terminations and must have done=True.  The caller controls target sync timing.
"""

from copy import deepcopy
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

try:
    from .vendor.NN import DRQN, QMIXNET
except ImportError:
    from vendor.NN import DRQN, QMIXNET


N_AGENTS = 3
N_ACTIONS = 16
N_FEATURES = 3
HIDDEN = 128
CHECKPOINT_VERSION = 1


def _finite_array(value, shape, name, dtype=np.float32):
    try:
        original = np.asarray(value)
        result = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} is not a numerical array") from exc
    if original.dtype.kind not in "biuf" or result.shape != shape:
        raise ValueError(f"{name} must be a numerical array with shape {shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains nonfinite or unrepresentable values")
    return result


def _boolean_array(value, shape, name):
    numeric = _finite_array(value, shape, name, dtype=np.float64)
    if not np.isin(numeric, (0, 1)).all():
        raise ValueError(f"{name} must contain only booleans or 0/1")
    return numeric.astype(bool)


def _indices(value, shape, name):
    numeric = _finite_array(value, shape, name, dtype=np.float64)
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} must contain integer action indices")
    if np.any(numeric < 0) or np.any(numeric >= N_ACTIONS):
        raise ValueError(f"{name} action indices must be in [0, {N_ACTIONS - 1}]")
    return numeric.astype(np.int64)


class Learner:
    def __init__(self, seed, device="cpu", learning_rate=2e-4, gamma=0.99):
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        if not 0 <= int(seed) < 2**63:
            raise ValueError("seed must be in [0, 2**63)")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be finite and in [0, 1]")
        self.seed = int(seed)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.device = torch.device(device)
        torch.manual_seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.conf = SimpleNamespace(
            drqn_hidden_dim=HIDDEN,
            qmix_hidden_dim=256,
            hyper_hidden_dim=64,
            n_agents=N_AGENTS,
            n_actions=N_ACTIONS,
            state_shape=N_AGENTS * N_FEATURES,
            two_hyper_layers=True,
        )
        self.agent = DRQN(N_FEATURES + N_ACTIONS + N_AGENTS, self.conf).to(self.device)
        self.mixer = QMIXNET(self.conf).to(self.device)
        self.target_agent = deepcopy(self.agent).eval().requires_grad_(False)
        self.target_mixer = deepcopy(self.mixer).eval().requires_grad_(False)
        self.params = list(self.agent.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.RMSprop(self.params, lr=self.learning_rate)
        self.updates = 0
        self.target_syncs = 0
        self.reset()

    def reset(self):
        """Start a new rollout, with no hidden state or previous action."""
        self.hidden = torch.zeros(N_AGENTS, HIDDEN, device=self.device)
        self.last = np.zeros((N_AGENTS, N_ACTIONS), dtype=np.float32)

    def choose(self, obs, mask, epsilon):
        obs = _finite_array(obs, (N_AGENTS, N_FEATURES), "observation")
        mask = _boolean_array(mask, (N_AGENTS, N_ACTIONS), "action mask")
        if not np.all(mask.any(axis=-1)):
            raise ValueError("Every real cell state must allow at least one action")
        if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be finite and in [0, 1]")
        inputs = np.concatenate(
            (obs, self.last, np.eye(N_AGENTS, dtype=np.float32)), axis=-1
        )
        with torch.no_grad():
            q_values, hidden = self.agent(
                torch.as_tensor(inputs, dtype=torch.float32, device=self.device),
                self.hidden,
            )
        if not torch.isfinite(q_values).all() or not torch.isfinite(hidden).all():
            raise FloatingPointError("Nonfinite network output during action selection")
        q_values = q_values.cpu().numpy()
        q_values[~mask] = -np.inf
        choices = q_values.argmax(axis=-1).astype(np.int64)
        # Greedy evaluations consume no random draw, including no unused
        # epsilon coin flip.  Fixed argmax tie breaking is intentional.
        if epsilon > 0:
            for cell in range(N_AGENTS):
                if epsilon == 1 or self.rng.random() < epsilon:
                    choices[cell] = int(self.rng.choice(np.flatnonzero(mask[cell])))
        self.hidden = hidden.detach()
        self.set_executed(choices)
        return choices

    def set_executed(self, indices):
        """Replace previous-action features with currents actually executed.

        Call this after every filtered transition, before the next choose().
        It changes no recurrent hidden state and consumes no random numbers.
        """
        actions = _indices(indices, (N_AGENTS,), "executed actions")
        self.last = np.eye(N_ACTIONS, dtype=np.float32)[actions]

    @staticmethod
    def _validate_episode(episode, number):
        prefix = f"episode {number}"
        required = ("obs", "masks", "actions", "executed_actions", "rewards", "done")
        if not isinstance(episode, dict) or any(key not in episode for key in required):
            raise ValueError(f"{prefix} requires fields {required}")
        actions_input = np.asarray(episode["actions"])
        if actions_input.ndim != 2 or actions_input.shape[1] != N_AGENTS:
            raise ValueError(f"{prefix} actions must have shape (T, {N_AGENTS})")
        count = actions_input.shape[0]
        if count == 0:
            raise ValueError(f"{prefix} contains no transitions")
        obs = _finite_array(episode["obs"], (count + 1, N_AGENTS, N_FEATURES), prefix + " observations")
        masks = _boolean_array(episode["masks"], (count + 1, N_AGENTS, N_ACTIONS), prefix + " masks")
        actions = _indices(episode["actions"], (count, N_AGENTS), prefix + " requested actions")
        executed = _indices(episode["executed_actions"], (count, N_AGENTS), prefix + " executed actions")
        rewards = _finite_array(episode["rewards"], (count,), prefix + " rewards")
        done = _boolean_array(episode["done"], (count,), prefix + " done flags")
        if done[:-1].any():
            raise ValueError(f"{prefix} contains transitions after an absorbing termination")
        if not masks.any(axis=-1).all():
            raise ValueError(f"{prefix}: every real observation must allow an action for each cell")
        selected_available = np.take_along_axis(masks[:-1], actions[..., None], axis=-1)
        if not selected_available.all():
            raise ValueError(f"{prefix} contains a requested action excluded by its policy mask")
        return count, obs, masks, actions, executed, rewards, done

    def train(self, episodes, max_sequence_length=80, burn_in=20):
        """One RMSprop update using one independent window per episode.

        Each learning window has at most max_sequence_length transitions.  Its
        start is uniformly sampled from all full-length windows of that
        episode.  Short episodes use their entire history.  Both GRUs consume
        the *complete* preceding history under no_grad, so a truncated window
        never starts with an invented zero hidden state.  burn_in controls
        the size of prefix-processing chunks; it never discards earlier history.

        Replay sampling and target sync are external.  Padding has zero loss
        weight.  Requested actions label Q; actual executed actions describe
        history.  Every learning window includes its true next observation.
        """
        for name, number in (("max_sequence_length", max_sequence_length), ("burn_in", burn_in)):
            if isinstance(number, bool) or not isinstance(number, (int, np.integer)) or number < 1:
                raise ValueError(f"{name} must be a positive integer")
        episodes = list(episodes)
        if not episodes:
            raise ValueError("Training requires at least one episode")
        checked = [self._validate_episode(ep, i) for i, ep in enumerate(episodes)]
        batch_size = len(checked)
        lengths = [min(ep[0], max_sequence_length) for ep in checked]
        starts = [int(self.rng.integers(0, ep[0] - count + 1)) if ep[0] > count else 0
                  for ep, count in zip(checked, lengths)]
        length = max(lengths)
        max_prefix = max(starts)
        obs = np.zeros((batch_size, length + 1, N_AGENTS, N_FEATURES), dtype=np.float32)
        actions = np.zeros((batch_size, length, N_AGENTS), dtype=np.int64)
        previous = np.zeros((batch_size, length + 1, N_AGENTS, N_ACTIONS), dtype=np.float32)
        rewards = np.zeros((batch_size, length, 1), dtype=np.float32)
        terminals = np.ones((batch_size, length, 1), dtype=bool)
        valid = np.zeros((batch_size, length, 1), dtype=np.float32)
        masks = np.zeros((batch_size, length + 1, N_AGENTS, N_ACTIONS), dtype=bool)
        masks[..., 0] = True  # Harmless valid action for internal padded states.
        input_size = N_FEATURES + N_ACTIONS + N_AGENTS
        prefix = np.zeros((batch_size, max_prefix, N_AGENTS, input_size), dtype=np.float32)
        action_eye = np.eye(N_ACTIONS, dtype=np.float32)
        agent_eye = np.eye(N_AGENTS, dtype=np.float32)
        for b, (_, ep_obs, ep_masks, requested, actual, reward, done) in enumerate(checked):
            start, count = starts[b], lengths[b]
            end = start + count
            obs[b, : count + 1] = ep_obs[start:end + 1]
            masks[b, : count + 1] = ep_masks[start:end + 1]
            actions[b, :count] = requested[start:end]
            previous[b, 1:count + 1] = action_eye[actual[start:end]]
            if start:
                previous[b, 0] = action_eye[actual[start - 1]]
                prefix[b, :start, :, :N_FEATURES] = ep_obs[:start]
                prefix[b, :start, :, -N_AGENTS:] = agent_eye
                if start > 1:
                    prefix[b, 1:start, :, N_FEATURES:N_FEATURES + N_ACTIONS] = action_eye[actual[:start - 1]]
            rewards[b, :count, 0] = reward[start:end]
            # Slicing a learning window does not itself terminate an episode.
            terminals[b, :count, 0] = done[start:end]
            valid[b, :count, 0] = 1
        obs, actions, previous, rewards, terminals, valid, masks, prefix = [
            torch.as_tensor(value, device=self.device)
            for value in (obs, actions, previous, rewards, terminals, valid, masks, prefix)
        ]
        states = obs.reshape(batch_size, length + 1, N_AGENTS * N_FEATURES)
        identity = torch.eye(N_AGENTS, device=self.device).expand(batch_size, length + 1, -1, -1)
        inputs = torch.cat((obs, previous, identity), dim=-1)
        hidden = torch.zeros(batch_size * N_AGENTS, HIDDEN, device=self.device)
        target_hidden = torch.zeros_like(hidden)
        online, targets = [], []
        self.agent.train()
        self.mixer.train()
        prefix_lengths = torch.as_tensor(starts, device=self.device)
        with torch.no_grad():
            for chunk_start in range(0, max_prefix, burn_in):
                for t in range(chunk_start, min(chunk_start + burn_in, max_prefix)):
                    x = prefix[:, t].reshape(batch_size * N_AGENTS, -1)
                    _, candidate = self.agent(x, hidden)
                    _, target_candidate = self.target_agent(x, target_hidden)
                    active = (prefix_lengths > t).repeat_interleave(N_AGENTS).unsqueeze(-1)
                    hidden = torch.where(active, candidate, hidden)
                    target_hidden = torch.where(active, target_candidate, target_hidden)
        for t in range(length + 1):
            x = inputs[:, t].reshape(batch_size * N_AGENTS, -1)
            q_values, hidden = self.agent(x, hidden)
            online.append(q_values.reshape(batch_size, N_AGENTS, N_ACTIONS))
            with torch.no_grad():
                target_values, target_hidden = self.target_agent(x, target_hidden)
                targets.append(target_values.reshape(batch_size, N_AGENTS, N_ACTIONS))
        online = torch.stack(online, dim=1)
        targets = torch.stack(targets, dim=1)
        chosen = online[:, :-1].gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        total = self.mixer(chosen, states[:, :-1])
        with torch.no_grad():
            # Each real and padded state has a nonempty mask.  Finite masking
            # also avoids (-inf * 0) propagation at terminal transitions.
            floor = torch.finfo(targets.dtype).min
            next_values = targets[:, 1:].masked_fill(~masks[:, 1:], floor).max(dim=-1).values
            target_total = self.target_mixer(next_values, states[:, 1:])
            continuation = torch.where(terminals, torch.zeros_like(target_total), target_total)
            td_target = rewards + self.gamma * continuation
        # Remove padded positions before squaring, rather than multiplying
        # their errors by zero after the calculation.
        real_error = (total - td_target).masked_select(valid.bool())
        loss = real_error.square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite recurrent QMIX loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, max_norm=10.0, error_if_nonfinite=True)
        self.optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in self.params):
            raise FloatingPointError("Nonfinite model parameter after RMSprop update")
        self.updates += 1
        return float(loss.detach().cpu())

    def sync(self):
        """Synchronize target networks only when explicitly requested by caller."""
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.target_agent.eval()
        self.target_mixer.eval()
        self.target_syncs += 1

    def get_rng_state(self):
        state = {
            "numpy_generator": deepcopy(self.rng.bit_generator.state),
            "torch_cpu": torch.get_rng_state().clone(),
        }
        if torch.cuda.is_initialized():
            state["torch_cuda"] = [value.clone() for value in torch.cuda.get_rng_state_all()]
        return state

    def set_rng_state(self, state):
        self.rng.bit_generator.state = deepcopy(state["numpy_generator"])
        torch.set_rng_state(state["torch_cpu"].cpu())
        if "torch_cuda" in state and self.device.type == "cuda":
            values = state["torch_cuda"]
            if len(values) != torch.cuda.device_count():
                raise ValueError("Checkpoint CUDA RNG states do not match the available device count")
            torch.cuda.set_rng_state_all([value.cpu() for value in values])

    def save(self, path, metadata):
        """Save learner, optimizer, rollout and RNG state; caller saves replay."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {
            "format_version": CHECKPOINT_VERSION,
            "config": {**vars(self.conf), "seed": self.seed, "gamma": self.gamma,
                       "learning_rate": self.learning_rate},
            "agent": self.agent.state_dict(),
            "mixer": self.mixer.state_dict(),
            "target_agent": self.target_agent.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng_state": self.get_rng_state(),
            "rollout_hidden": self.hidden.detach().cpu(),
            "rollout_last": self.last.copy(),
            "updates": self.updates,
            "target_syncs": self.target_syncs,
            "metadata": deepcopy(metadata),
        }
        temporary = path.with_name(path.name + ".tmp")
        torch.save(value, temporary)
        os.replace(temporary, path)

    def load(self, path):
        """Load a trusted locally generated checkpoint and return its metadata.

        weights_only=False is necessary for the NumPy RNG and optimizer state;
        never use this loader for checkpoints from an untrusted source.
        """
        value = torch.load(Path(path), map_location=self.device, weights_only=False)
        if value.get("format_version") != CHECKPOINT_VERSION:
            raise ValueError("Unsupported learner checkpoint format")
        config = value["config"]
        for name, expected in vars(self.conf).items():
            if config.get(name) != expected:
                raise ValueError(f"Checkpoint architecture mismatch: {name}")
        self.agent.load_state_dict(value["agent"])
        self.mixer.load_state_dict(value["mixer"])
        self.target_agent.load_state_dict(value["target_agent"])
        self.target_mixer.load_state_dict(value["target_mixer"])
        self.optimizer.load_state_dict(value["optimizer"])
        self.learning_rate = float(config["learning_rate"])
        self.gamma = float(config["gamma"])
        self.seed = int(config["seed"])
        self.updates = int(value["updates"])
        self.target_syncs = int(value["target_syncs"])
        self.target_agent.eval()
        self.target_mixer.eval()
        if any(not torch.isfinite(parameter).all() for network in
               (self.agent, self.mixer, self.target_agent, self.target_mixer)
               for parameter in network.parameters()):
            raise ValueError("Checkpoint contains nonfinite model parameters")
        hidden = value["rollout_hidden"].to(self.device)
        if hidden.shape != (N_AGENTS, HIDDEN) or not torch.isfinite(hidden).all():
            raise ValueError("Checkpoint contains malformed rollout hidden state")
        last = _finite_array(value["rollout_last"], (N_AGENTS, N_ACTIONS), "checkpoint previous actions")
        if not np.isin(last, (0, 1)).all() or not np.isin(last.sum(axis=-1), (0, 1)).all():
            raise ValueError("Checkpoint previous actions are not zero or one-hot vectors")
        self.hidden = hidden.detach()
        self.last = last.copy()
        self.set_rng_state(value["rng_state"])
        return deepcopy(value["metadata"])
