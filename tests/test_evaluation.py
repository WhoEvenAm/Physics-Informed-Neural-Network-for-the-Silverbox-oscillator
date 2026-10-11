"""Checks for the assignment's continuous, independent test prediction rule."""

from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from src.silverbox_model import AccelerationModel, DT
from training_utils import TrainingConfig, free_run, save_checkpoint, load_checkpoint


class EvaluationTests(unittest.TestCase):
    def test_test_measurements_cannot_change_predictions(self):
        model = AccelerationModel()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()  # Constant velocity has a known trajectory.
        u_train, u_test = torch.zeros(5, 1), torch.zeros(3, 1)
        y_train = torch.arange(1, 6, dtype=torch.float32) * 0.01
        metrics, states = free_run(model, u_train, y_train, u_test, torch.zeros(3))
        changed_metrics, changed_states = free_run(model, u_train, y_train, u_test, torch.ones(3))
        torch.testing.assert_close(states, changed_states)
        self.assertNotEqual(metrics["test_rmse"], changed_metrics["test_rmse"])
        # Prediction at the first test sample continues the prior trajectory.
        expected_y = torch.arange(1, 9, dtype=torch.float32) * 0.01
        torch.testing.assert_close(states[:, 0], expected_y)

    def test_checkpoint_restores_weights_and_scaling(self):
        torch.manual_seed(10)
        model = AccelerationModel(
            hidden_sizes=(16, 8), input_scale=(0.05, 22, 0.02),
            acceleration_scale=10000,
        )
        inputs = torch.tensor([[0.04, 12.0, 0.01]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, model, TrainingConfig(hidden_sizes=(16, 8)), {})
            restored, metadata = load_checkpoint(path)
            torch.testing.assert_close(model(inputs), restored(inputs))
            self.assertEqual(metadata["sampling_interval"], DT)


if __name__ == "__main__":
    unittest.main()
