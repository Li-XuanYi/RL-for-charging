import unittest

import numpy as np
import torch

from config import Config
from dql import CentralDQL


def build_policy() -> CentralDQL:
    conf = Config(mixer="dql", episode_limit=2)
    conf.set_env_info(
        {
            "n_actions": 19,
            "state_shape": 9,
            "obs_shape": 3,
            "n_agents": 3,
            "episode_limit": 2,
        }
    )
    return CentralDQL(conf)


class CentralDQLTests(unittest.TestCase):
    def test_joint_action_round_trip(self):
        policy = build_policy()
        actions = torch.tensor([[0, 0, 0], [18, 18, 18], [1, 2, 3]])
        encoded = policy.encode_actions(actions)
        self.assertEqual(policy.decode_action(int(encoded[0])), [0, 0, 0])
        self.assertEqual(policy.decode_action(int(encoded[1])), [18, 18, 18])
        self.assertEqual(policy.decode_action(int(encoded[2])), [1, 2, 3])

    def test_joint_mask_is_cartesian_product(self):
        policy = build_policy()
        available = torch.zeros((1, 3, 19))
        available[0, 0, [0, 1]] = 1
        available[0, 1, [2, 3, 4]] = 1
        available[0, 2, [5, 6]] = 1
        joint = policy._joint_availability(available)
        self.assertEqual(int(joint.sum().item()), 2 * 3 * 2)
        selected = policy.encode_actions(torch.tensor([[1, 4, 6]]))
        self.assertEqual(float(joint[0, selected].item()), 1.0)

    def test_learning_step_returns_finite_loss(self):
        policy = build_policy()
        batch = {
            "s": np.zeros((2, 2, 9)),
            "s_": np.zeros((2, 2, 9)),
            "u": np.zeros((2, 2, 3, 1), dtype=np.int64),
            "r": np.ones((2, 2, 1)),
            "avail_u_": np.ones((2, 2, 3, 19)),
            "padded": np.zeros((2, 2, 1)),
            "terminated": np.zeros((2, 2, 1)),
        }
        loss = policy.learn(batch, max_episode_len=2, training_episode=1)
        self.assertTrue(np.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
