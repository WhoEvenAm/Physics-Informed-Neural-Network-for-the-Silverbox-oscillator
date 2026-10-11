# Silverbox oscillator: Task 5

A PyTorch neural acceleration model for the Silverbox oscillator, with RK4
simulation, architecture comparisons, and continuous free-run evaluation on
test data.

The selected network has two hidden layers of 64 neurons. Across ten completed
runs, it gives a training RMSE of **2.914 mV** and a test RMSE of **11.498 mV**.
It uses AdamW, learning rate 0.003, 3,000 updates, 128-sample windows, and batch
size 8. Evaluation starts at the beginning of the excerpt and carries the state
continuously into the test segment.

Read the [written report](output/pdf/task5_report.pdf), inspect the
[experiment table](output/experiments/20261011_220640/results.csv), or view the
[test prediction plot](output/best_test_predictions.png).

The assignment chooses the lowest test RMSE, so this test record also serves
for model selection. It is not a separate untouched final evaluation set.

Use Python 3.12 and the project's virtual environment. The existing installed
package versions are recorded in `requirements.txt`.

From the project folder, run the loader with:

```sh
source .venv/bin/activate
python src/data_loader.py
```

To recreate the environment if it is missing:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

VS Code is configured to use `.venv/bin/python`. If it previously selected a
different interpreter, run **Python: Select Interpreter** and choose that path,
then open a new terminal.

The loader reads `data/SNLS80mV.mat` relative to the project location, so it can
also be launched from another working directory. It selects 4,096 samples,
splits them into 3,072 training and 1,024 test samples, and centers both splits
using the means of the full 4,096-sample excerpt, as required by the homework.

To run the introductory training loop, run:

```sh
.venv/bin/python src/train.py
```

This script samples 64-point windows across the training record, using 3,000 Adam
updates at a learning rate of 0.01. Every window is contained in the training split.
It prints sampled-window losses and the first window's error before and after
training, along with the number of training samples covered by sampled windows.
Input and acceleration scales are RMS magnitudes estimated from training data.
The script also reports physical signal durations, training-loop elapsed time,
average time per optimizer update, and free-run evaluation elapsed time.
After training, it runs continuously through all 4,096 samples from the initial
state, without resetting at the test boundary. It reports separate full free-run
training and test RMSE values, with zero-prediction baselines for comparison.
It saves a new `output/test_predictions_<timestamp>.png` on each completed run,
then displays the plot. Close the plot window to let the script finish. Timestamps
use Yerevan time, and previous images are preserved. The plot shows measured and predicted test output,
a zoom of the first 0.15 seconds, and the prediction error over the full test
segment. The graph uses the same continuous free run as the reported test RMSE.
Each launch of `train.py` creates a new model.

Run the architecture/hyperparameter study with:

```sh
.venv/bin/python src/experiments.py
```

This tests hidden widths `(32, 32)`, `(64, 64)`, and `(32, 32, 32)` with three
training settings each, plus the original tutorial configuration: ten runs in
total. It saves configurations, loss histories, weights, timing, and continuous
train/test RMSE in a timestamped `output/experiments/`
folder. A separate seeded generator makes sampled windows independent of the
network architecture. The assignment selects the model with the smallest test
RMSE; measured test positions are never used for training updates.

The selected weights are saved as `output/best_model.pt`. Evaluate them without
retraining using:

```sh
.venv/bin/python src/test.py
```

Add `--no-show` to save the graph without opening a window. The saved checkpoint
contains its architecture, input/output scale buffers, weights, and settings.
The selected full trace and plot are also saved as `output/best_predictions.csv`
and `output/best_test_predictions.png`.

Run the evaluation/checkpoint checks with:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

To generate the written Task 5 PDF after completing the comparisons:

```sh
.venv/bin/python -m pip install -r requirements-report.txt
.venv/bin/python src/build_report.py --repository-url https://github.com/WhoEvenAm/Physics-Informed-Neural-Network-for-the-Silverbox-oscillator
```

The report is saved to `output/pdf/task5_report.pdf`. Without a repository URL,
it is clearly marked as a draft. Task 6's LangGraph system is a separate task.
