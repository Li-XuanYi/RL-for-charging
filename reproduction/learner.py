"""QMIX training with original network classes and repaired trajectory/TD logic."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import torch
from vendor.NN import DRQN, QMIXNET
from battery import ACTIONS


class Learner:
    def __init__(self, seed, device="cpu"):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device)
        self.conf = SimpleNamespace(drqn_hidden_dim=128, qmix_hidden_dim=256,
            hyper_hidden_dim=64, n_agents=3, n_actions=len(ACTIONS), state_shape=9,
            two_hyper_layers=True)
        self.agent = DRQN(3 + len(ACTIONS) + 3, self.conf).to(self.device)
        self.mixer = QMIXNET(self.conf).to(self.device)
        self.target_agent = deepcopy(self.agent).eval()
        self.target_mixer = deepcopy(self.mixer).eval()
        self.params = list(self.agent.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.RMSprop(self.params, lr=2e-4)
        self.reset()

    def reset(self):
        self.hidden = torch.zeros(3, 128, device=self.device)
        self.last = np.zeros((3, len(ACTIONS)), dtype=np.float32)

    def choose(self, obs, mask, epsilon):
        inputs = np.concatenate([obs, self.last, np.eye(3)], axis=-1)
        with torch.no_grad():
            q, self.hidden = self.agent(torch.as_tensor(inputs, dtype=torch.float32, device=self.device), self.hidden)
        q = q.cpu().numpy()
        q[~mask] = -np.inf
        choices = [int(self.rng.choice(np.flatnonzero(mask[i]))) if self.rng.random() < epsilon
                   else int(q[i].argmax()) for i in range(3)]
        self.last = np.eye(len(ACTIONS), dtype=np.float32)[choices]
        return np.array(choices, dtype=np.int64)

    def train(self, episodes):
        batch_size = len(episodes)
        length = max(len(ep["actions"]) for ep in episodes)
        n = len(ACTIONS)
        obs = np.zeros((batch_size, length + 1, 3, 3), dtype=np.float32)
        acts = np.zeros((batch_size, length, 3), dtype=np.int64)
        rewards = np.zeros((batch_size, length, 1), dtype=np.float32)
        terminals = np.ones_like(rewards)
        valid = np.zeros_like(rewards)
        available = np.ones((batch_size, length + 1, 3, n), dtype=bool)
        for b, ep in enumerate(episodes):
            count = len(ep["actions"])
            obs[b, :count+1] = ep["obs"]
            acts[b, :count] = ep["actions"]
            rewards[b, :count, 0] = ep["rewards"]
            terminals[b, :count, 0] = ep["done"]
            valid[b, :count] = 1.
            available[b, :count+1] = ep["masks"]
        obs, acts, rewards, terminals, valid, available = [torch.as_tensor(a, device=self.device)
            for a in (obs, acts, rewards, terminals, valid, available)]
        states = obs.reshape(batch_size, length+1, 9)
        previous = torch.zeros(batch_size, length+1, 3, n, device=self.device)
        previous[:, 1:] = torch.nn.functional.one_hot(acts, n).float()
        ids = torch.eye(3, device=self.device).expand(batch_size, length+1, 3, 3)
        inputs = torch.cat([obs, previous, ids], -1)
        hidden = torch.zeros(batch_size * 3, 128, device=self.device)
        target_hidden = torch.zeros_like(hidden)
        online, targets = [], []
        # Burn both recurrent networks through the same complete history,
        # including the true final observation (not a duplicated previous one).
        for t in range(length+1):
            x = inputs[:, t].reshape(batch_size*3, -1)
            q, hidden = self.agent(x, hidden)
            online.append(q.reshape(batch_size, 3, n))
            with torch.no_grad():
                qt, target_hidden = self.target_agent(x, target_hidden)
                targets.append(qt.reshape(batch_size, 3, n))
        online = torch.stack(online, 1)
        targets = torch.stack(targets, 1)
        chosen = online[:, :-1].gather(-1, acts.unsqueeze(-1)).squeeze(-1)
        total = self.mixer(chosen, states[:, :-1])
        with torch.no_grad():
            next_q = targets[:, 1:].masked_fill(~available[:, 1:], -1e9).max(-1).values
            target_total = self.target_mixer(next_q, states[:, 1:])
            y = rewards + .99 * (1-terminals) * target_total
        loss = (((total-y)**2) * valid).sum() / valid.sum()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 10.)
        self.optimizer.step()
        return float(loss.item())

    def sync(self):
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def save(self, path, metadata):
        torch.save({"agent": self.agent.state_dict(), "mixer": self.mixer.state_dict(),
            "target_agent": self.target_agent.state_dict(), "target_mixer": self.target_mixer.state_dict(),
            "optimizer": self.optimizer.state_dict(), "metadata": metadata}, path)

    def load(self, path):
        value = torch.load(path, map_location=self.device, weights_only=False)
        self.agent.load_state_dict(value["agent"])
        self.mixer.load_state_dict(value["mixer"])
        self.target_agent.load_state_dict(value["target_agent"])
        self.target_mixer.load_state_dict(value["target_mixer"])
        self.optimizer.load_state_dict(value["optimizer"])
        self.reset()
        return value["metadata"]
