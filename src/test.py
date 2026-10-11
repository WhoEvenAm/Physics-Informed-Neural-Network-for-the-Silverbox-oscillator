"""Evaluate saved weights without training again.

Run: .venv/bin/python src/test.py
Add --no-show to save the plot without opening a window.
"""

import argparse
from pathlib import Path

from train import plot_test_predictions
from training_utils import data_tensors, free_run, load_checkpoint


def main():
    project = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=project / "output" / "best_model.pt")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    model, checkpoint = load_checkpoint(args.checkpoint)
    signals = data_tensors()
    metrics, states = free_run(model, *signals)
    print("Saved training settings:", checkpoint["config"])
    print(f"Full free-run training RMSE: {metrics['train_rmse']:.8f} V")
    print(f"Full free-run test RMSE: {metrics['test_rmse']:.8f} V")
    n = len(signals[1])
    plot_test_predictions(
        signals[3], states[n:, 0], metrics["test_rmse"], metrics["zero_test_rmse"],
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
