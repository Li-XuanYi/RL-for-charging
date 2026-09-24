"""Network definitions matching the submitted three-cell QMIX architecture.

The agent uses Linear(22,128)-ReLU-GRUCell(128,128)-Linear(128,16).
The monotonic mixing network has 256 hidden units and two-layer, 64-unit
hypernetworks.  These definitions are local to this experiment package.
"""

import torch
from torch import nn
from torch.nn import functional as F


class DRQN(nn.Module):
    def __init__(self, input_shape, conf):
        super().__init__()
        self.conf = conf
        self.fc1 = nn.Linear(input_shape, conf.drqn_hidden_dim)
        self.rnn = nn.GRUCell(conf.drqn_hidden_dim, conf.drqn_hidden_dim)
        self.fc2 = nn.Linear(conf.drqn_hidden_dim, conf.n_actions)

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        hidden = self.rnn(x, hidden_state.reshape(-1, self.conf.drqn_hidden_dim))
        return self.fc2(hidden), hidden


class QMIXNET(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.hyper_w1 = nn.Sequential(
            nn.Linear(conf.state_shape, conf.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(conf.hyper_hidden_dim, conf.n_agents * conf.qmix_hidden_dim),
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(conf.state_shape, conf.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(conf.hyper_hidden_dim, conf.qmix_hidden_dim),
        )
        self.hyper_b1 = nn.Linear(conf.state_shape, conf.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(conf.state_shape, conf.qmix_hidden_dim),
            nn.ReLU(),
            nn.Linear(conf.qmix_hidden_dim, 1),
        )

    def forward(self, q_values, states):
        episode_count = q_values.shape[0]
        q_values = q_values.reshape(-1, 1, self.conf.n_agents)
        states = states.reshape(-1, self.conf.state_shape)
        w1 = self.hyper_w1(states).abs().reshape(
            -1, self.conf.n_agents, self.conf.qmix_hidden_dim
        )
        b1 = self.hyper_b1(states).reshape(-1, 1, self.conf.qmix_hidden_dim)
        hidden = F.elu(torch.bmm(q_values, w1) + b1)
        w2 = self.hyper_w2(states).abs().reshape(-1, self.conf.qmix_hidden_dim, 1)
        b2 = self.hyper_b2(states).reshape(-1, 1, 1)
        return (torch.bmm(hidden, w2) + b2).reshape(episode_count, -1, 1)
