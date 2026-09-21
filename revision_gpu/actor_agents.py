"""Actor-critic comparison agents for the revision experiments.

This module is an implementation, not evidence that a method outperforms another.
It uses only NumPy and PyTorch already required by the reproduction environment.
No experiments or imports are performed by the code-writing workflow.

Protocol decisions that must accompany results:
* MAPPO is feedforward, has a shared local actor (SOC, V, T, agent identity),
  a centralized state-value critic, team GAE, and per-agent PPO clipping. It is
  not a recurrent-network ablation of the paper's recurrent QMIX.
* Discrete MAPPO uses the same 16 nonnegative current actions as discrete QMIX.
  Continuous MAPPO uses a Beta distribution over each available current range.
* SAC uses a centralized joint continuous actor and twin centralized Q critics.
  It belongs in the continuous comparison track, not a quantized discrete track.
* All losses condition on requested actions. An external, shared safety layer
  is part of the environment transition; executed actions must still be logged.
* Continuous densities include the physical-current change-of-variables term.
  SAC's target entropy is transformed by the same range scale and excludes
  dimensions forced to zero. Team log densities are summed, not averaged.
* A numerical or goal termination sets done=True. A time-limit truncation sets
  done=False and is bootstrapped from its actual final observation.

Episode interface: obs[T+1,N,3], masks[T+1,N,16], actions[T,N] (requested amps),
executed_actions[T,N], rewards[T], done[T], info[T] (dicts returned by act).
The caller must not update an on-policy learner until its collected episodes
are handed to learn(), and must discard those episodes after that update.
"""

from __future__ import annotations

import contextlib
import copy
import math
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Beta, Categorical, Normal


def _mlp(inputs: int, outputs: int, width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inputs, width), nn.Tanh(),
        nn.Linear(width, width), nn.Tanh(), nn.Linear(width, outputs),
    )


class ActorPolicy:
    """CPU/CUDA actor-critic policy with isolated checkpointed sampling streams."""

    def __init__(self, method, n_agents, seed, config):
        self.method = str(method).lower()
        if self.method not in {"mappo", "mappo_continuous", "sac"}:
            raise ValueError(f"Unsupported actor-critic method: {method}")
        self.n_agents = int(n_agents)
        if self.n_agents < 1:
            raise ValueError("n_agents must be positive")
        self.seed = int(seed)
        self.config = dict(config) if isinstance(config, Mapping) else dict(vars(config))
        self.device = torch.device(self.config.get("device", "cpu"))
        if self.device.type == 'cuda':
            if not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; use the GPU launcher/environment')
            self.device = torch.device('cuda', self.device.index if self.device.index is not None else torch.cuda.current_device())
        elif self.device.type != 'cpu':
            raise ValueError('Supported devices are cpu and cuda')
        self.action_kind = "discrete" if self.method == "mappo" else "continuous"
        self.learning_kind = "off_policy" if self.method == "sac" else "on_policy"
        self.n_actions = 16
        self.action_step = float(self.config.get("action_step", 0.5))
        self.max_current = self.action_step * (self.n_actions - 1)
        if not math.isfinite(self.action_step) or self.action_step <= 0:
            raise ValueError("action_step must be positive and finite")
        self.gamma = float(self.config.get("gamma", 0.99))
        self.lr = float(self.config.get("learning_rate", 3e-4))
        self.width = int(self.config.get("actor_hidden", 128))
        if not (0 <= self.gamma <= 1) or self.lr <= 0 or self.width < 1:
            raise ValueError("Invalid gamma, learning_rate, or actor_hidden")
        self.np_rng = np.random.default_rng(self.seed)
        self.torch_rng_state = torch.Generator(device="cpu").manual_seed(self.seed).get_state()
        self.cuda_rng_state = torch.Generator(device=self.device).manual_seed(self.seed).get_state() if self.device.type == 'cuda' else None
        self.update_count = 0
        self._identities = torch.eye(self.n_agents, dtype=torch.float32, device=self.device)
        with self._rng_scope():
            if self.method == "sac":
                self._init_sac()
            else:
                self._init_mappo()

    @contextlib.contextmanager
    def _rng_scope(self):
        # Restore both process streams after isolated sampling. Evaluation does
        # not consume these private streams. RNG byte tensors stay on the CPU.
        devices = [self.device.index] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self.torch_rng_state.cpu())
            if self.cuda_rng_state is not None:
                torch.cuda.set_rng_state(self.cuda_rng_state.cpu(), self.device)
            try:
                yield
            finally:
                self.torch_rng_state = torch.get_rng_state().clone()
                if self.device.type == 'cuda':
                    self.cuda_rng_state = torch.cuda.get_rng_state(self.device).clone()

    def _init_mappo(self):
        output = self.n_actions if self.action_kind == "discrete" else 2
        self.actor = _mlp(3 + self.n_agents, output, self.width).to(self.device)
        self.critic = _mlp(3 * self.n_agents, 1, self.width).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def _init_sac(self):
        # Available upper bounds are observable policy constraints, appended to
        # the joint physical observation for the actor and Q functions.
        central_inputs = 4 * self.n_agents
        self.actor = _mlp(central_inputs, 2 * self.n_agents, self.width).to(self.device)
        self.q1 = _mlp(central_inputs + self.n_agents, 1, self.width).to(self.device)
        self.q2 = _mlp(central_inputs + self.n_agents, 1, self.width).to(self.device)
        self.target_q1 = copy.deepcopy(self.q1).requires_grad_(False)
        self.target_q2 = copy.deepcopy(self.q2).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.lr
        )
        alpha = float(self.config.get("alpha", 0.2))
        if alpha <= 0 or not math.isfinite(alpha):
            raise ValueError("SAC alpha must be positive and finite")
        self.auto_alpha = bool(self.config.get("automatic_entropy_tuning", True))
        self.log_alpha = torch.tensor(math.log(alpha), requires_grad=self.auto_alpha, device=self.device)
        self.alpha_optimizer = (
            torch.optim.Adam([self.log_alpha], lr=float(self.config.get("alpha_learning_rate", self.lr)))
            if self.auto_alpha else None
        )
        self.tau = float(self.config.get("tau", 0.005))
        if not (0 < self.tau <= 1):
            raise ValueError("SAC tau must be in (0, 1]")

    def reset(self):
        """Feedforward policies have no hidden state to reset."""

    def set_executed(self, currents):
        """Accept the common runner interface; no recurrent context is stored."""
        values = np.asarray(currents, dtype=float)
        if values.shape != (self.n_agents,) or not np.isfinite(values).all():
            raise ValueError("Executed currents must be a finite N-vector")

    def sync(self, *args, **kwargs):
        """SAC has soft targets; MAPPO has no target network."""

    def parameter_count(self):
        """Count trainable actor/critic parameters, excluding duplicate targets."""
        modules = [self.actor, self.q1, self.q2] if self.method == "sac" else [self.actor, self.critic]
        count = sum(p.numel() for module in modules for p in module.parameters())
        return count + int(self.method == "sac" and self.auto_alpha)

    def _tensors(self, obs, masks):
        states = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        available = torch.as_tensor(np.asarray(masks), dtype=torch.bool, device=self.device)
        if states.shape[-2:] != (self.n_agents, 3):
            raise ValueError(f"Expected observations [...,{self.n_agents},3], got {tuple(states.shape)}")
        if available.shape != states.shape[:-1] + (self.n_actions,):
            raise ValueError("Action-mask dimensions do not match observations")
        if not bool(torch.isfinite(states).all()):
            raise ValueError("Observations contain a non-finite value")
        if not bool(available.any(dim=-1).all()):
            raise ValueError("Every agent must have at least one available action")
        return states, available

    def _bounds(self, masks):
        # The continuous action domain is a closed interval, not a collection
        # of quantized points. Reject masks with holes instead of interpreting
        # them as a misleading continuous safety guarantee.
        ids = torch.arange(self.n_actions, dtype=torch.long, device=self.device)
        last = torch.where(masks, ids, -1).amax(dim=-1)
        contiguous = ids <= last.unsqueeze(-1)
        if not bool((masks == contiguous).all()):
            raise ValueError("Continuous actors require a mask representing [0, maximum current]")
        return last.to(torch.float32) * self.action_step

    def _local_features(self, obs):
        ids = self._identities.expand(obs.shape[:-2] + self._identities.shape)
        return torch.cat([obs, ids], dim=-1)

    def _distribution(self, obs, masks):
        raw = self.actor(self._local_features(obs))
        if self.action_kind == "discrete":
            logits = raw.masked_fill(~masks, -1e9)
            return Categorical(logits=logits), masks.sum(dim=-1) > 1, None
        bounds = self._bounds(masks)
        concentration = torch.nn.functional.softplus(raw) + 1.0
        return Beta(concentration[..., 0], concentration[..., 1]), bounds > 0, bounds

    def _mappo_log_prob_entropy(self, distribution, actions, active, bounds):
        if self.action_kind == "discrete":
            indices = torch.round(actions / self.action_step).long()
            log_prob = distribution.log_prob(indices)
            entropy = distribution.entropy()
        else:
            safe_bounds = bounds.clamp_min(1e-6)
            unit = (actions / safe_bounds).clamp(1e-6, 1.0 - 1e-6)
            log_prob = distribution.log_prob(unit) - torch.log(safe_bounds)
            entropy = distribution.entropy() + torch.log(safe_bounds)
        return torch.where(active, log_prob, 0.0), torch.where(active, entropy, 0.0)

    def _sac_state(self, obs, masks):
        bounds = self._bounds(masks)
        return torch.cat([obs.flatten(start_dim=-2), bounds / self.max_current], dim=-1), bounds

    def _sac_sample(self, state, bounds, deterministic=False):
        raw = self.actor(state)
        mean, log_std = raw.chunk(2, dim=-1)
        # A bounded log standard deviation avoids overflow in Normal.rsample.
        log_std = log_std.clamp(-5.0, 2.0)
        distribution = Normal(mean, log_std.exp())
        latent = mean if deterministic else distribution.rsample()
        unit = torch.tanh(latent)
        active = bounds > 0
        safe_scale = torch.where(active, bounds / 2.0, torch.ones_like(bounds))
        actions = torch.where(active, (unit + 1.0) * bounds / 2.0, 0.0)
        # Stable log(1 - tanh(z)^2), without a biased near-boundary epsilon.
        log_jacobian = 2.0 * (math.log(2.0) - latent - torch.nn.functional.softplus(-2.0 * latent))
        per_agent_log_prob = distribution.log_prob(latent) - log_jacobian - safe_scale.log()
        per_agent_log_prob = torch.where(active, per_agent_log_prob, 0.0)
        log_prob = per_agent_log_prob.sum(dim=-1)
        # The usual normalized target of -1 per active dimension is expressed
        # in the same physical-current coordinates as log_prob.
        target_entropy = torch.where(active, -torch.ones_like(bounds) + safe_scale.log(), 0.0).sum(dim=-1)
        return actions, log_prob, target_entropy, active

    def act(self, obs, mask, training=False, epsilon=0.0):
        """Return requested currents in amps and on-policy collection metadata.

        epsilon is intentionally unused: actor-critic exploration follows its
        probability distribution, not the discrete Q-agent epsilon schedule.
        """
        del epsilon
        states, masks = self._tensors(obs, mask)
        if states.ndim != 2:
            raise ValueError("act expects one N-by-3 observation")
        with torch.no_grad():
            if self.method == "sac":
                central, bounds = self._sac_state(states, masks)
                if training:
                    with self._rng_scope():
                        actions, log_prob, _, _ = self._sac_sample(central, bounds)
                else:
                    actions, log_prob, _, _ = self._sac_sample(central, bounds, deterministic=True)
                info = {"log_prob": float(log_prob), "action_kind": "continuous"}
            else:
                distribution, active, bounds = self._distribution(states, masks)
                if self.action_kind == "discrete":
                    if training:
                        with self._rng_scope():
                            indices = distribution.sample()
                    else:
                        indices = distribution.logits.argmax(dim=-1)
                    actions = indices.to(torch.float32) * self.action_step
                else:
                    if training:
                        with self._rng_scope():
                            unit = distribution.sample()
                    else:
                        unit = distribution.mean
                    # Stored actions and density use the same clipped unit
                    # sample. This only guards floating-point Beta endpoints.
                    actions = unit.clamp(1e-6, 1.0 - 1e-6) * bounds
                log_probs, _ = self._mappo_log_prob_entropy(distribution, actions, active, bounds)
                value = self.critic(states.flatten()).squeeze(-1)
                info = {"log_probs": log_probs.cpu().numpy().copy(), "value": float(value),
                        "action_kind": self.action_kind}
                if bounds is not None:
                    info["normalized_action"] = (actions / bounds.clamp_min(1e-6)).cpu().numpy().copy()
        requested = actions.cpu().numpy().astype(np.float64, copy=True)
        if not np.isfinite(requested).all():
            raise FloatingPointError("Policy produced a non-finite action")
        return requested, info

    def _episode(self, episode):
        rewards = np.asarray(episode["rewards"], dtype=np.float32).reshape(-1)
        count = len(rewards)
        obs, masks = self._tensors(episode["obs"], episode["masks"])
        if obs.shape != (count + 1, self.n_agents, 3):
            raise ValueError("Episode must include the actual final next observation")
        actions = torch.as_tensor(np.asarray(episode["actions"]), dtype=torch.float32, device=self.device)
        done = torch.as_tensor(np.asarray(episode["done"]), dtype=torch.float32, device=self.device).reshape(-1)
        if actions.shape != (count, self.n_agents) or done.shape != (count,):
            raise ValueError("Episode action/done lengths do not match rewards")
        if not np.isfinite(rewards).all() or not bool(torch.isfinite(actions).all()):
            raise ValueError("Episode contains non-finite actions or rewards")
        if not bool(((done == 0) | (done == 1)).all()):
            raise ValueError("done must be boolean terminal flags")
        if count > 1 and bool((done[:-1] != 0).any()):
            raise ValueError("Episode continues after a terminal transition")
        if bool((actions < -1e-6).any()) or bool((actions > self.max_current + 1e-6).any()):
            raise ValueError("Requested current is outside the common action domain")
        if self.action_kind == "discrete":
            indices = torch.round(actions / self.action_step).long()
            if not bool(torch.isclose(actions, indices * self.action_step, atol=1e-5, rtol=0).all()):
                raise ValueError("Discrete MAPPO received a non-grid requested action")
            if not bool(masks[:-1].gather(-1, indices.unsqueeze(-1)).all()):
                raise ValueError("Requested discrete action was unavailable at collection")
        else:
            bounds = self._bounds(masks[:-1])
            if bool((actions > bounds + 1e-5).any()):
                raise ValueError("Requested continuous action exceeded its available range")
        return obs, masks, actions, torch.as_tensor(rewards, device=self.device), done

    def learn(self, episodes):
        episodes = list(episodes)
        if not episodes:
            return {"updates": 0, "loss": 0.0}
        return self._learn_sac(episodes) if self.method == "sac" else self._learn_mappo(episodes)

    @staticmethod
    def _finite_loss(loss, name):
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite {name}; run must be marked failed")

    def _learn_mappo(self, episodes):
        gae_lambda = float(self.config.get("gae_lambda", 0.95))
        if not 0 <= gae_lambda <= 1:
            raise ValueError("gae_lambda must be in [0, 1]")
        rows = {key: [] for key in ("obs", "masks", "actions", "log_probs", "values", "returns", "advantages")}
        for episode in episodes:
            obs, masks, actions, rewards, done = self._episode(episode)
            count = len(rewards)
            if not count:
                continue
            infos = episode.get("info", [])
            if len(infos) != count or any("log_probs" not in info or "value" not in info for info in infos):
                raise ValueError("MAPPO needs stored action log_probs and values from collection")
            old_log_probs = torch.as_tensor(np.asarray([info["log_probs"] for info in infos]), dtype=torch.float32, device=self.device)
            old_values = torch.as_tensor([float(info["value"]) for info in infos], dtype=torch.float32, device=self.device)
            if old_log_probs.shape != (count, self.n_agents):
                raise ValueError("MAPPO stored log_probs have incorrect dimensions")
            if not bool(torch.isfinite(old_log_probs).all() and torch.isfinite(old_values).all()):
                raise ValueError("MAPPO collection metadata must be finite")
            with torch.no_grad():
                final_value = self.critic(obs[-1].flatten()).reshape(())
            next_values = torch.cat([old_values[1:], final_value.reshape(1)])
            deltas = rewards + self.gamma * (1 - done) * next_values - old_values
            advantages = torch.empty(count, dtype=torch.float32, device=self.device)
            running = torch.tensor(0.0, device=self.device)
            for index in range(count - 1, -1, -1):
                running = deltas[index] + self.gamma * gae_lambda * (1 - done[index]) * running
                advantages[index] = running
            values = {"obs": obs[:-1], "masks": masks[:-1], "actions": actions,
                      "log_probs": old_log_probs, "values": old_values,
                      "returns": advantages + old_values, "advantages": advantages}
            for key, value in values.items():
                rows[key].append(value)
        if not rows["obs"]:
            return {"updates": 0, "loss": 0.0}
        data = {key: torch.cat(value, dim=0) for key, value in rows.items()}
        advantages = data["advantages"]
        # Scalar team advantage is standardized across transitions, then used
        # for each agent's local clipped likelihood-ratio objective.
        data["advantages"] = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        count = len(advantages)
        clip = float(self.config.get("clip_ratio", 0.2))
        epochs = int(self.config.get("ppo_epochs", 10))
        minibatch = int(self.config.get("minibatch_size", 256))
        entropy_coef = float(self.config.get("entropy_coef", 0.01))
        value_coef = float(self.config.get("value_loss_coef", 0.5))
        gradient_clip = float(self.config.get("ppo_gradient_clip", 0.5))
        if not (0 < clip < 1) or epochs < 1 or minibatch < 1 or gradient_clip <= 0:
            raise ValueError("Invalid PPO optimization configuration")
        stats = []
        for _ in range(epochs):
            order = self.np_rng.permutation(count)
            for start in range(0, count, minibatch):
                indices = torch.as_tensor(order[start:start + minibatch], dtype=torch.long, device=self.device)
                state = data["obs"][indices]
                distribution, active, bounds = self._distribution(state, data["masks"][indices])
                log_prob, entropy = self._mappo_log_prob_entropy(distribution, data["actions"][indices], active, bounds)
                log_ratio = log_prob - data["log_probs"][indices]
                # A guard at +/-20 prevents overflow while preserving the PPO
                # clipping regime; diagnostics retain the unclipped log ratio.
                ratio = torch.exp(log_ratio.clamp(-20, 20))
                advantage = data["advantages"][indices].unsqueeze(-1)
                surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
                active_float = active.to(torch.float32)
                denominator = active_float.sum().clamp_min(1)
                policy_loss = -(surrogate * active_float).sum() / denominator
                mean_entropy = (entropy * active_float).sum() / denominator
                actor_loss = policy_loss - entropy_coef * mean_entropy
                value = self.critic(state.flatten(start_dim=1)).squeeze(-1)
                value_error = (value - data["returns"][indices]).square()
                if bool(self.config.get("value_clip", False)):
                    old_value = data["values"][indices]
                    value_clipped = old_value + (value - old_value).clamp(-clip, clip)
                    value_error = torch.maximum(value_error, (value_clipped - data["returns"][indices]).square())
                critic_loss = 0.5 * value_error.mean()
                self._finite_loss(actor_loss, "MAPPO actor loss")
                self._finite_loss(critic_loss, "MAPPO critic loss")
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), gradient_clip, error_if_nonfinite=True)
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad(set_to_none=True)
                (value_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), gradient_clip, error_if_nonfinite=True)
                self.critic_optimizer.step()
                with torch.no_grad():
                    approximate_kl = (((ratio - 1) - log_ratio) * active_float).sum() / denominator
                    clip_fraction = (((ratio - 1).abs() > clip).float() * active_float).sum() / denominator
                stats.append([float(actor_loss.detach()), float(critic_loss.detach()),
                              float(mean_entropy.detach()), float(approximate_kl), float(clip_fraction)])
        self.update_count += len(stats)
        result = np.mean(stats, axis=0)
        return {"updates": len(stats), "transitions": count, "actor_loss": float(result[0]),
                "critic_loss": float(result[1]), "entropy": float(result[2]),
                "approximate_kl": float(result[3]), "clip_fraction": float(result[4]),
                "loss": float(result[0] + value_coef * result[1])}

    def _q_input(self, state, actions):
        return torch.cat([state, actions / self.max_current], dim=-1)

    def _learn_sac(self, episodes):
        rows = {key: [] for key in ("obs", "masks", "actions", "rewards", "done", "next_obs", "next_masks")}
        for episode in episodes:
            obs, masks, actions, rewards, done = self._episode(episode)
            if not len(rewards):
                continue
            values = {"obs": obs[:-1], "masks": masks[:-1], "actions": actions,
                      "rewards": rewards, "done": done, "next_obs": obs[1:], "next_masks": masks[1:]}
            for key, value in values.items():
                rows[key].append(value)
        if not rows["obs"]:
            return {"updates": 0, "loss": 0.0}
        data = {key: torch.cat(value, dim=0) for key, value in rows.items()}
        count = len(data["rewards"])
        batch_size = int(self.config.get("sac_batch_size", self.config.get("batch_size", 256)))
        steps = int(self.config.get("sac_gradient_steps", self.config.get("gradient_steps", 1)))
        gradient_clip = float(self.config.get("sac_gradient_clip", 10.0))
        if batch_size < 1 or steps < 1 or gradient_clip <= 0:
            raise ValueError("Invalid SAC optimization configuration")
        stats = []
        with self._rng_scope():
            for _ in range(steps):
                indices = self.np_rng.choice(count, size=min(batch_size, count), replace=False)
                ids = torch.as_tensor(indices, dtype=torch.long, device=self.device)
                state, bounds = self._sac_state(data["obs"][ids], data["masks"][ids])
                next_state, next_bounds = self._sac_state(data["next_obs"][ids], data["next_masks"][ids])
                alpha = self.log_alpha.exp().detach()
                with torch.no_grad():
                    next_actions, next_log_prob, _, _ = self._sac_sample(next_state, next_bounds)
                    next_input = self._q_input(next_state, next_actions)
                    next_q = torch.minimum(self.target_q1(next_input), self.target_q2(next_input)).squeeze(-1)
                    target = data["rewards"][ids] + self.gamma * (1 - data["done"][ids]) * (next_q - alpha * next_log_prob)
                q_input = self._q_input(state, data["actions"][ids])
                q1 = self.q1(q_input).squeeze(-1)
                q2 = self.q2(q_input).squeeze(-1)
                critic_loss = (q1 - target).square().mean() + (q2 - target).square().mean()
                self._finite_loss(critic_loss, "SAC critic loss")
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                critic_parameters = list(self.q1.parameters()) + list(self.q2.parameters())
                nn.utils.clip_grad_norm_(critic_parameters, gradient_clip, error_if_nonfinite=True)
                self.critic_optimizer.step()
                # Q gradients are unnecessary during the actor step, but
                # gradients through the requested action into Q are retained.
                for parameter in critic_parameters:
                    parameter.requires_grad_(False)
                try:
                    sampled, log_prob, target_entropy, active = self._sac_sample(state, bounds)
                    sampled_input = self._q_input(state, sampled)
                    q_actor = torch.minimum(self.q1(sampled_input), self.q2(sampled_input)).squeeze(-1)
                    actor_loss = (alpha * log_prob - q_actor).mean()
                    self._finite_loss(actor_loss, "SAC actor loss")
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), gradient_clip, error_if_nonfinite=True)
                    self.actor_optimizer.step()
                finally:
                    for parameter in critic_parameters:
                        parameter.requires_grad_(True)
                alpha_loss_value = 0.0
                active_samples = active.any(dim=-1)
                if self.auto_alpha and bool(active_samples.any()):
                    alpha_loss = -(self.log_alpha * (log_prob.detach() + target_entropy)[active_samples]).mean()
                    self._finite_loss(alpha_loss, "SAC temperature loss")
                    self.alpha_optimizer.zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    with torch.no_grad():
                        self.log_alpha.clamp_(-12.0, 5.0)
                    alpha_loss_value = float(alpha_loss.detach())
                with torch.no_grad():
                    for online, target_net in ((self.q1, self.target_q1), (self.q2, self.target_q2)):
                        for online_parameter, target_parameter in zip(online.parameters(), target_net.parameters()):
                            target_parameter.lerp_(online_parameter, self.tau)
                stats.append([float(actor_loss.detach()), float(critic_loss.detach()),
                              float(-log_prob.detach().mean()), float(self.log_alpha.detach().exp()), alpha_loss_value])
        self.update_count += len(stats)
        result = np.mean(stats, axis=0)
        return {"updates": len(stats), "transitions": count, "actor_loss": float(result[0]),
                "critic_loss": float(result[1]), "entropy": float(result[2]),
                "alpha": float(result[3]), "alpha_loss": float(result[4]),
                "loss": float(result[0] + result[1])}

    def save(self, path, metadata=None):
        """Persist weights, optimizers and private RNG state for local resume."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": 1, "method": self.method, "n_agents": self.n_agents,
                   "action_step": self.action_step, "config": self.config,
                   "actor": self.actor.state_dict(), "actor_optimizer": self.actor_optimizer.state_dict(),
                   "critic_optimizer": self.critic_optimizer.state_dict(),
                   "np_rng": self.np_rng.bit_generator.state, "torch_rng": self.torch_rng_state.cpu(),
                   "cuda_rng": self.cuda_rng_state.cpu() if self.cuda_rng_state is not None else None,
                   "training_device": str(self.device),
                   "update_count": self.update_count, "metadata": metadata or {}}
        if self.method == "sac":
            payload.update(q1=self.q1.state_dict(), q2=self.q2.state_dict(),
                           target_q1=self.target_q1.state_dict(), target_q2=self.target_q2.state_dict(),
                           log_alpha=self.log_alpha.detach().clone(), auto_alpha=self.auto_alpha,
                           alpha_optimizer=self.alpha_optimizer.state_dict() if self.alpha_optimizer else None)
        else:
            payload["critic"] = self.critic.state_dict()
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def load(self, path):
        """Load only a trusted checkpoint produced by this local experiment."""
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        if payload.get("schema") != 1 or payload.get("method") != self.method or payload.get("n_agents") != self.n_agents:
            raise ValueError("Checkpoint schema, method or agent count does not match this policy")
        if not math.isclose(float(payload.get("action_step", -1)), self.action_step):
            raise ValueError("Checkpoint current grid differs from the active protocol")
        if self.method == "sac" and bool(payload.get("auto_alpha")) != self.auto_alpha:
            raise ValueError("Checkpoint automatic-temperature setting differs")
        self.actor.load_state_dict(payload["actor"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        if self.method == "sac":
            for name in ("q1", "q2", "target_q1", "target_q2"):
                getattr(self, name).load_state_dict(payload[name])
            with torch.no_grad():
                self.log_alpha.copy_(payload["log_alpha"])
            if self.alpha_optimizer:
                self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        else:
            self.critic.load_state_dict(payload["critic"])
        self.np_rng.bit_generator.state = payload["np_rng"]
        self.torch_rng_state = payload["torch_rng"].cpu().clone()
        if self.device.type == 'cuda' and payload.get('cuda_rng') is not None:
            self.cuda_rng_state = payload['cuda_rng'].cpu().clone()
        self.update_count = int(payload["update_count"])
        return dict(payload.get("metadata", {}))
