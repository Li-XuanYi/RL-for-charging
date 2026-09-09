import unittest

from main import normalize_parameter_keys


class ParameterKeyTests(unittest.TestCase):
    def test_legacy_diffusivity_names_are_migrated(self):
        normalized, migrations = normalize_parameter_keys(
            {
                "Positive electrode diffusivity [m2.s-1]": 5e-15,
                "Negative electrode diffusivity [m2.s-1]": 4e-14,
            }
        )
        self.assertEqual(
            normalized["Positive particle diffusivity [m2.s-1]"], 5e-15
        )
        self.assertEqual(
            normalized["Negative particle diffusivity [m2.s-1]"], 4e-14
        )
        self.assertEqual(len(migrations), 2)

    def test_conflicting_legacy_and_current_names_are_rejected(self):
        with self.assertRaises(ValueError):
            normalize_parameter_keys(
                {
                    "Positive electrode diffusivity [m2.s-1]": 5e-15,
                    "Positive particle diffusivity [m2.s-1]": 6e-15,
                }
            )


if __name__ == "__main__":
    unittest.main()
