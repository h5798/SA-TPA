import unittest

import numpy as np

from src.methods import classify, normalize, run_method


class MethodTests(unittest.TestCase):
    def setUp(self):
        self.text = np.eye(3, dtype=np.float32)
        self.per_prompt = np.stack([self.text, self.text], axis=1)
        self.source = normalize(np.asarray([[1, .1, 0], [.9, 0, 0], [0, 1, .1], [0, .9, 0], [.1, 0, 1], [0, 0, .9]]))
        self.labels = np.asarray([0, 0, 1, 1, 2, 2])
        self.target = normalize(np.asarray([[1, .05, 0], [0, 1, .05], [.05, 0, 1]]))

    def test_probabilities_are_valid(self):
        probabilities = classify(self.target, self.text)
        np.testing.assert_allclose(probabilities.sum(1), 1.0, atol=1e-6)

    def test_zero_weights_equal_prompt_ensemble(self):
        baseline = run_method("prompt_ensemble", self.source, self.labels, self.target, self.text, self.per_prompt)
        adapted = run_method(
            "no_source_anchor", self.source, self.labels, self.target, self.text, self.per_prompt,
            alpha_source=0.0, alpha_target=0.0,
        )
        np.testing.assert_allclose(baseline.probabilities, adapted.probabilities, atol=1e-6)

    def test_satpa_is_finite(self):
        output = run_method("satpa", self.source, self.labels, self.target, self.text, self.per_prompt)
        self.assertTrue(np.isfinite(output.probabilities).all())
        self.assertEqual(output.probabilities.shape, (3, 3))

    def test_no_uncertainty_ablation_is_finite(self):
        output = run_method(
            "satpa_no_uncertainty", self.source, self.labels, self.target, self.text, self.per_prompt
        )
        self.assertTrue(np.isfinite(output.probabilities).all())

    def test_additional_baselines_are_valid(self):
        for method in ("t3a", "tip_adapter_source"):
            output = run_method(method, self.source, self.labels, self.target, self.text, self.per_prompt)
            np.testing.assert_allclose(output.probabilities.sum(1), 1.0, atol=1e-6)




if __name__ == "__main__":
    unittest.main()
