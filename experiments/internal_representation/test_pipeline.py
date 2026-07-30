"""Fast regression tests that do not download or execute an LLM."""

import tempfile
import unittest
from collections import Counter

import numpy as np

from config import DataConfig, ProbeConfig
from dataset import prepare_dataset
from probing import train_probes


class PersonaPipelineTests(unittest.TestCase):
    def test_gender_is_label_and_messages_are_context(self):
        with tempfile.TemporaryDirectory() as directory:
            data = prepare_dataset(DataConfig(data_dir=directory, samples_per_group=2))
        self.assertEqual(Counter(data["labels"]["Gender"]), {"Female": 2, "Male": 2})
        self.assertTrue(all(text.startswith("User:") for text in data["conversations"]))
        self.assertTrue(all(data["last_user_questions"]))

    def test_probe_returns_layer_artifacts_and_control(self):
        rng = np.random.default_rng(42)
        labels = ["Female"] * 20 + ["Male"] * 20
        signal = np.array([-2.0] * 20 + [2.0] * 20)
        states = {
            layer: np.column_stack([signal + rng.normal(0, 0.1, 40), rng.normal(size=(40, 3))])
            for layer in (0, 1)
        }
        results, artifacts = train_probes(
            states, labels, "Gender", ProbeConfig(cv_folds=2), test_size=0.25
        )
        self.assertEqual(set(artifacts), {0, 1})
        self.assertEqual(len(results.results), 4)
        self.assertEqual({item.is_control for item in results.results}, {False, True})
        self.assertGreaterEqual(results.best_layer("Gender").accuracy, 0.9)


if __name__ == "__main__":
    unittest.main()
