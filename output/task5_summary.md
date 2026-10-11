# Task 5: Silverbox oscillator

## Selected model

- Hidden widths: 64-64; parameters: 4481
- Settings: {'optimizer': 'AdamW', 'learning_rate': 0.003, 'steps': 3000, 'rollout_length': 128, 'batch_size': 8, 'seed': 0}
- Training free-run RMSE: 0.00291446 V
- Test free-run RMSE: 0.01149832 V

## Experiment comparison

| Architecture | Setting | Train RMSE (V) | Test RMSE (V) |
|---|---|---:|---:|
| 32-32 | 1 | 0.00750631 | 0.01961132 |
| 32-32 | 2 | 0.00502145 | 0.01678829 |
| 32-32 | 3 | 0.00265862 | 0.01490717 |
| 64-64 | 1 | 0.01653587 | 0.02005969 |
| 64-64 | 2 | 0.00475065 | 0.01633924 |
| 64-64 | 3 | 0.00291446 | 0.01149832 |
| 32-32-32 | 1 | 0.01598219 | 0.01981829 |
| 32-32-32 | 2 | 0.00639972 | 0.02112306 |
| 32-32-32 | 3 | 0.00406650 | 0.01696445 |
| 32-32 | Original | 0.01025929 | 0.01164837 |

Repository: https://github.com/WhoEvenAm/Physics-Informed-Neural-Network-for-the-Silverbox-oscillator
