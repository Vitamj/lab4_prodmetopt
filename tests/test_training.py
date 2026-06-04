import unittest

from src.data import build_datasets
from src.metrics import classification_metrics
from src.models import HiddenLayerPerceptron
from src.optimizers import AdamLikeOptimizer
from src.training import train_model


class TrainingTests(unittest.TestCase):
    def test_training_improves_on_moons(self) -> None:
        dataset = build_datasets(0)[0]
        model = HiddenLayerPerceptron(input_dim=2, hidden_dim=12, seed=0)
        before = classification_metrics(dataset.y_val, model.predict_proba(dataset.X_val))["accuracy"]
        training = train_model(
            model=model,
            optimizer=AdamLikeOptimizer(learning_rate=0.01),
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_val=dataset.X_val,
            y_val=dataset.y_val,
            epochs=80,
            batch_size=16,
            seed=0,
            patience=15,
        )
        after = classification_metrics(dataset.y_val, model.predict_proba(dataset.X_val))["accuracy"]
        self.assertGreater(after, before)
        self.assertGreaterEqual(training.best_val_accuracy, 0.8)


if __name__ == "__main__":
    unittest.main()
