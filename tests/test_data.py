import unittest

from src.data import build_datasets


class DataTests(unittest.TestCase):
    def test_dataset_shapes(self) -> None:
        moons, cls = build_datasets(0)
        self.assertEqual(moons.X_train.shape[0], 240)
        self.assertEqual(moons.X_val.shape[0], 80)
        self.assertEqual(moons.X_test.shape[0], 80)
        self.assertEqual(cls.X_train.shape[1], 5)


if __name__ == "__main__":
    unittest.main()
