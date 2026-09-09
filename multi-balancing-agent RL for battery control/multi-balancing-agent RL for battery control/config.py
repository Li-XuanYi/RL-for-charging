"""Configuration for paper-aligned MBA-RL reproduction experiments."""

from __future__ import annotations

import torch


class Config:
    """Hold training, architecture, and reward settings.

    Defaults match Table II of the revised manuscript. Legacy settings such as
    800 episodes or a target interval of 200 must be requested explicitly and
    must not be mixed with the paper-aligned reproduction results.
    """

    def __init__(
        self,
        seed: int = 7,
        n_epochs: int = 1200,
        target_update_interval: int = 50,
        mixer: str = "qmix",
        episode_limit: int = 50,
        initialization_mode: str = "soc_consistent",
    ) -> None:
        if mixer not in {"qmix", "vdn", "dql"}:
            raise ValueError("mixer must be 'qmix', 'vdn', or 'dql'")
        if initialization_mode not in {"soc_consistent", "legacy_mixed"}:
            raise ValueError(
                "initialization_mode must be 'soc_consistent' or 'legacy_mixed'"
            )

        self.train = True
        self.seed = int(seed)
        self.cuda = True

        self.last_action = True
        self.reuse_network = True
        self.n_epochs = int(n_epochs)
        self.evaluate_epoch = 100
        self.evaluate_number = 1
        self.batch_size = 128
        self.buffer_size = 800
        self.gamma = 0.99
        self.grad_norm_clip = 10.0
        self.update_target_params = int(target_update_interval)
        self.episode_limit_config = int(episode_limit)
        self.initialization_mode = initialization_mode
        self.cell_parameter_overrides = None
        self.cell_parameter_source = "Chen2020 (unmodified)"

        self.beta = 0.02
        self.soc_ref = 0.90
        self.sample_time = 90
        self.max_voltage = 4.20
        self.max_temperature = 309.0
        self.time_penalty = -0.75
        self.balance_penalty_scale = -50.0
        self.voltage_penalty_scale = -20.0
        self.temperature_penalty_scale = -2.0

        self.load_model = False
        self.device = torch.device(
            "cuda" if self.cuda and torch.cuda.is_available() else "cpu"
        )

        self.mixer = mixer
        self.algorithm = mixer
        self.drqn_hidden_dim = 128
        self.qmix_hidden_dim = 256
        self.two_hyper_layers = True
        self.hyper_hidden_dim = 64
        self.model_dir = "./models/"
        self.optimizer = "RMS"
        self.learning_rate = 2e-4

        self.start_epsilon = 0.5
        self.end_epsilon = 0.0

    def set_env_info(self, env_info: dict[str, int]) -> None:
        self.n_actions = env_info["n_actions"]
        self.state_shape = env_info["state_shape"]
        self.obs_shape = env_info["obs_shape"]
        self.n_agents = env_info["n_agents"]
        self.episode_limit = env_info["episode_limit"]
