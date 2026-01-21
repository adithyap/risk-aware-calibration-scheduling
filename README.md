# Risk-Aware Calibration Scheduling

Risk-aware calibration scheduling on CMAPSS-adapted turbofan data, using a Transformer with an integrated quantile regression head to produce safety-aware time-to-drift predictions and calibration priorities.

## Overview
- Adapt CMAPSS into a calibration setting with virtual thresholds, splice/stitch resets, and time-to-drift labels.
- Train sequence models (Transformer with quantile head, LSTM, CNN, TCN) and baselines (trees/boosting).
- Use quantile-triggered, risk-aware scheduling policies to balance violations and calibration cost.
- Generate plots, tables, and summaries for each CMAPSS subset (FD001–FD004).

## Key Components
- **Calibration adaptation**: Virtual thresholds per drift sensor, synthetic calibration resets, sawtooth TTD labels.
- **Models**: Quantile Transformer (pinball loss), LSTM with quantile head, CNN/TCN, tree/boosting baselines.
- **Scheduling**: Predictive policies trigger when lower-quantile TTD indicates violation risk; cost model supports calibration vs. violation trade-offs.
- **Outputs**: Metrics, policy costs, and plots per dataset.

## Running the pipeline
```bash
python3 calibration_scheduler.py
```
By default, this will process FD001–FD004 in sequence. Adjust configuration in `Config` (dataset selection, margins, costs, quantiles).

## Plots and artifacts
Figures used in the paper (moved to repo root for convenience):
- `drift_sensor_correlations.png` — monotonicity of sensors on FD001
- `threshold_example.png` — drift sensors and TTD labels
- `data_adaptation_flow.png` — adaptation pipeline (threshold, splice, TTD)
- `scheduler_flow.png` — risk-aware scheduling flow
- `lstm_training.png`, `transformer_training.png`, `cnn_training.png`, `tcn_training.png` — training curves
- `transformer_pred_vs_true.png`, `transformer_residuals.png` — prediction quality
- `priority_hist.png` — calibration priority distribution

## Citation
If you use this codebase, please cite the accompanying paper or reference this repository:
```
https://github.com/adithyap/risk-aware-calibration-scheduling
```
