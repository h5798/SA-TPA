import unittest

import numpy as np

from src.methods import classify, normalize, run_method, sinkhorn_transport


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

    def test_agreement_variants_are_valid(self):
        for method in ("satpa_agreement", "adaptive_satpa"):
            output = run_method(method, self.source, self.labels, self.target, self.text, self.per_prompt)
            np.testing.assert_allclose(output.probabilities.sum(1), 1.0, atol=1e-6)
            self.assertIn("acceptance_rate", output.diagnostics)

    def test_adaptive_weights_are_bounded(self):
        output = run_method("adaptive_satpa", self.source, self.labels, self.target, self.text, self.per_prompt)
        self.assertGreaterEqual(output.diagnostics["min_source_weight"], 0.05 - 1e-7)
        self.assertLessEqual(output.diagnostics["max_source_weight"], 0.30 + 1e-7)
        self.assertGreaterEqual(output.diagnostics["min_target_weight"], 0.0)
        self.assertLessEqual(output.diagnostics["max_target_weight"], 0.10 + 1e-7)

    def test_iterative_variant_is_bounded(self):
        output = run_method(
            "iterative_satpa", self.source, self.labels, self.target, self.text, self.per_prompt,
            target_update_steps=3,
        )
        np.testing.assert_allclose(output.probabilities.sum(1), 1.0, atol=1e-6)
        self.assertGreaterEqual(output.diagnostics["rounds_completed"], 1)
        self.assertLessEqual(output.diagnostics["rounds_completed"], 3)

    def test_sinkhorn_matches_marginals(self):
        cost = np.asarray([[0.1, 0.9], [0.8, 0.2], [0.4, 0.5]], dtype=np.float64)
        mu = np.asarray([0.2, 0.3, 0.5])
        nu = np.asarray([0.6, 0.4])
        plan, diagnostics = sinkhorn_transport(cost, mu, nu, epsilon=0.05, max_iter=500)
        np.testing.assert_allclose(plan.sum(1), mu, atol=1e-6)
        np.testing.assert_allclose(plan.sum(0), nu, atol=1e-6)
        self.assertLessEqual(diagnostics["ot_column_marginal_error"], 1e-7)

    def test_spt_sa_is_valid(self):
        output = run_method("spt_sa", self.source, self.labels, self.target, self.text, self.per_prompt)
        np.testing.assert_allclose(output.probabilities.sum(1), 1.0, atol=1e-6)
        self.assertTrue(np.isfinite(output.probabilities).all())
        self.assertGreater(output.diagnostics["ot_min_effective_size"], 0.0)
        self.assertGreaterEqual(output.diagnostics["ot_classes_ess_below_3"], 0)
        self.assertLessEqual(output.diagnostics["ot_classes_ess_below_3"], 3)




if __name__ == "__main__":
    unittest.main()
