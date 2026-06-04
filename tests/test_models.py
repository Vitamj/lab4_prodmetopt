import unittest

import numpy as np

from src.models import HiddenLayerPerceptron, SingleLayerClassifier


class ModelTests(unittest.TestCase):
    def test_single_layer_shapes(self) -> None:
        model = SingleLayerClassifier(input_dim=2, seed=0)
        X = np.random.randn(5, 2)
        y_prob, cache = model.forward(X)
        grads = model.backward(X, np.array([0, 1, 0, 1, 1]), cache)
        self.assertEqual(y_prob.shape, (5, 1))
        self.assertEqual(grads["W"].shape, (2, 1))

    def test_hidden_layer_shapes(self) -> None:
        model = HiddenLayerPerceptron(input_dim=5, hidden_dim=8, seed=0)
        X = np.random.randn(7, 5)
        y_prob, cache = model.forward(X)
        grads = model.backward(X, np.array([0, 1, 0, 1, 1, 0, 1]), cache)
        self.assertEqual(y_prob.shape, (7, 1))
        self.assertEqual(grads["W1"].shape, (5, 8))
        self.assertEqual(grads["W2"].shape, (8, 1))


if __name__ == "__main__":
    unittest.main()
