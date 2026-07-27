import unittest


class ResultSchemaTests(unittest.TestCase):
    def test_diagnostic_fields_are_stable(self):
        baseline = {
            "accepted_samples": None,
            "acceptance_rate": None,
            "mean_uncertainty_weight": None,
            "classes_with_target_support": None,
        }
        adapted = {**baseline, "accepted_samples": 10, "acceptance_rate": 0.5}
        self.assertEqual(list(baseline), list(adapted))


if __name__ == "__main__":
    unittest.main()
