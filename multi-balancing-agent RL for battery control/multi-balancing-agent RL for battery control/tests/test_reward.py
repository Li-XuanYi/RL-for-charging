import math
import unittest

from reward import (
    balancing_reward,
    pack_reward,
    population_soc_std,
    safety_reward,
    terminal_status,
)


class RewardEquationTests(unittest.TestCase):
    def test_population_standard_deviation_matches_manuscript(self):
        socs = [0.3, 0.5, 0.7]
        expected = math.sqrt(((0.3 - 0.5) ** 2 + 0.0 + (0.7 - 0.5) ** 2) / 3)
        self.assertAlmostEqual(population_soc_std(socs), expected)

    def test_balance_penalty_is_zero_inside_threshold(self):
        reward, sigma = balancing_reward([0.90, 0.91, 0.92], beta=0.02)
        self.assertLessEqual(sigma, 0.02)
        self.assertEqual(reward, 0.0)

    def test_safety_penalty_matches_voltage_and_temperature_equations(self):
        total, voltage, temperature = safety_reward(
            [4.25, 4.10, 4.20],
            [310.0, 308.0, 309.0],
            max_voltage=4.2,
            max_temperature=309.0,
        )
        self.assertAlmostEqual(voltage, -1.0)
        self.assertAlmostEqual(temperature, -2.0)
        self.assertAlmostEqual(total, -3.0)

    def test_joint_terminal_requires_balance_and_reference_soc(self):
        done, _ = terminal_status([0.90, 0.91, 0.92], beta=0.02, soc_ref=0.90)
        self.assertTrue(done)
        not_done, _ = terminal_status(
            [0.88, 0.89, 0.90], beta=0.02, soc_ref=0.90
        )
        self.assertFalse(not_done)

    def test_pack_reward_contains_all_reported_components(self):
        reward, components = pack_reward(
            [0.90, 0.91, 0.92],
            [4.10, 4.10, 4.10],
            [300.0, 300.0, 300.0],
        )
        self.assertEqual(reward, -0.75)
        self.assertEqual(components["time"], -0.75)
        self.assertEqual(components["balance"], 0.0)
        self.assertEqual(components["safety"], 0.0)


if __name__ == "__main__":
    unittest.main()
