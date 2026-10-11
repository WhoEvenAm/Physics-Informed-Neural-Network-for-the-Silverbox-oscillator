"""Run a reproducible study and save every result before selecting the best model.

Run: .venv/bin/python src/experiments.py
The assignment explicitly selects the model with the best test free-run RMSE.
Test outputs are used for scoring finished runs, never for gradient updates.
"""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np
import torch

from src.silverbox_model import DT
from train import plot_test_predictions
from training_utils import (
    TrainingConfig, data_tensors, fit_model, free_run, save_checkpoint, load_checkpoint,
)


PROJECT = Path(__file__).resolve().parent.parent
ARCHITECTURES = [(32, 32), (64, 64), (32, 32, 32)]
# All architectures receive the same three settings. The introductory setting
# is retained, then learning rate, update count, batching, and rollout are tuned.
SETTINGS = [
    dict(optimizer="Adam", learning_rate=0.01, steps=3000, rollout_length=64, batch_size=1),
    dict(optimizer="Adam", learning_rate=0.003, steps=2000, rollout_length=64, batch_size=8),
    dict(optimizer="AdamW", learning_rate=0.003, steps=3000, rollout_length=128, batch_size=8),
]


def write_results(rows, directory):
    """Update the table after each run, so finished work survives interruptions."""
    with (directory / "results.json").open("w") as file:
        json.dump(rows, file, indent=2, allow_nan=False)
    fields = [
        "name", "architecture", "parameters", "optimizer", "learning_rate",
        "steps", "rollout_length", "batch_size", "seed", "window_seed", "training_seconds",
        "train_rmse", "test_rmse", "status", "error", "checkpoint",
    ]
    with (directory / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, help="Continue an existing results directory")
    args = parser.parse_args()
    timestamp = datetime.now(ZoneInfo("Asia/Yerevan")).strftime("%Y%m%d_%H%M%S")
    directory = args.resume.resolve() if args.resume else PROJECT / "output" / "experiments" / timestamp
    directory.mkdir(parents=True, exist_ok=bool(args.resume))
    signals = data_tensors()
    u_train, y_train, u_test, y_test = signals
    rows = json.loads((directory / "results.json").read_text()) if args.resume else []
    for row in rows:
        row.setdefault("window_seed", 1)

    runs = [
        (architecture, setting_number, TrainingConfig(hidden_sizes=architecture, **settings))
        for architecture in ARCHITECTURES
        for setting_number, settings in enumerate(SETTINGS, 1)
    ]
    runs.append(((32, 32), "Original", TrainingConfig(window_seed=None)))
    for architecture, setting_number, config in runs:
        name = "x".join(map(str, architecture)) + f"_setting{setting_number}"
        if any(row["name"] == name for row in rows):
            continue  # Preserve completed or failed run records when resuming.
        row = dict(
            name=name, architecture="-".join(map(str, architecture)),
            **{key: value for key, value in asdict(config).items() if key != "hidden_sizes"},
        )
        print(f"\nRUN {len(rows) + 1}/{len(runs)}: {name} | {asdict(config)}", flush=True)
        try:
            model, training = fit_model(config, u_train, y_train)
            metrics, states = free_run(model, *signals)
            checkpoint_path = directory / f"{name}.pt"
            save_checkpoint(checkpoint_path, model, config, metrics)
            np.savetxt(
                directory / f"{name}_loss.csv",
                np.column_stack((np.arange(1, config.steps + 1), training["loss_history"])),
                delimiter=",", header="update,sampled_window_mse_v2", comments="",
            )
            row.update(
                parameters=sum(parameter.numel() for parameter in model.parameters()),
                training_seconds=training["training_seconds"],
                covered_training_samples=training["covered_training_samples"],
                checkpoint=str(checkpoint_path.relative_to(PROJECT)),
                status="complete", **metrics,
            )
            print(
                f"RESULT {name}: train RMSE={metrics['train_rmse']:.8f} V, "
                f"test RMSE={metrics['test_rmse']:.8f} V", flush=True
            )
        except RuntimeError as error:
            row.update(status="failed", error=str(error))
            print(f"FAILED {name}: {error}", flush=True)
        rows.append(row)
        write_results(rows, directory)

    complete = [row for row in rows if row["status"] == "complete"]
    if not complete:
        raise RuntimeError("No experiment produced a usable model")
    best = min(complete, key=lambda row: row["test_rmse"])
    best_source = PROJECT / best["checkpoint"]
    shutil.copyfile(best_source, PROJECT / "output" / "best_model.pt")
    with (PROJECT / "output" / "best_result.json").open("w") as file:
        json.dump(dict(best, results_directory=str(directory.relative_to(PROJECT))), file, indent=2)

    # Reload and independently reproduce the saved model's continuous free run.
    model, checkpoint = load_checkpoint(PROJECT / "output" / "best_model.pt")
    metrics, states = free_run(model, *signals)
    for key in ("train_rmse", "test_rmse"):
        if abs(metrics[key] - checkpoint["metrics"][key]) > 1e-7:
            raise RuntimeError("Reloaded checkpoint failed to reproduce its RMSE")
    n = len(y_train)
    plot_test_predictions(
        y_test, states[n:, 0], metrics["test_rmse"], metrics["zero_test_rmse"],
        show=False, output_path=PROJECT / "output" / "best_test_predictions.png",
    )
    measured = torch.cat((y_train, y_test)).numpy()
    np.savetxt(
        PROJECT / "output" / "best_predictions.csv",
        np.column_stack((np.arange(len(measured)) * DT, measured, states.numpy())),
        delimiter=",", header="time_s,measured_y_v,predicted_y_v,predicted_velocity_v_per_s", comments="",
    )
    print(f"\nBEST: {best['name']} | test RMSE={best['test_rmse']:.8f} V", flush=True)
    print("Results:", directory, flush=True)
    print("Saved model:", PROJECT / "output" / "best_model.pt", flush=True)


if __name__ == "__main__":
    main()
