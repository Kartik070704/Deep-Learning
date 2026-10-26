"""Calculate test metrics and diagnostic plots using saved predictions."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(actual, predicted):
    return {"MAE": float(mean_absolute_error(actual, predicted)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))), "R2": float(r2_score(actual, predicted))}


def run():
    out = Path("outputs"); out.mkdir(exist_ok=True)
    frame = pd.read_csv(out / "test_predictions.csv", parse_dates=["DateTime"])
    ann = metrics(frame.actual, frame.ann_prediction); baseline = metrics(frame.actual, frame.baseline_prediction)
    result = {"test_rows": len(frame), "unit": "kW", "ANN": ann, "Persistence_baseline": baseline,
              "ANN_lower_MAE": ann["MAE"] < baseline["MAE"], "ANN_lower_RMSE": ann["RMSE"] < baseline["RMSE"]}
    (out / "evaluation_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    actual, predicted = frame.actual.to_numpy(), frame.ann_prediction.to_numpy()
    fig, ax = plt.subplots(figsize=(7, 6)); ax.scatter(actual, predicted, alpha=.35, s=10)
    lo, hi = min(actual.min(), predicted.min()), max(actual.max(), predicted.max()); ax.plot([lo, hi], [lo, hi], "--", color="#334155")
    ax.set(xlabel="Actual active power (kW)", ylabel="Predicted active power (kW)", title="Actual vs predicted · test set"); fig.tight_layout(); fig.savefig(out/"actual_vs_predicted.png", dpi=160); plt.close(fig)
    hist = pd.read_csv(out/"training_history.csv"); fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(hist.epoch,hist.loss,label="Training"); ax.plot(hist.epoch,hist.val_loss,label="Validation"); ax.set(xlabel="Epoch",ylabel="MSE loss",title="Training and validation loss"); ax.legend(); fig.tight_layout(); fig.savefig(out/"training_validation_loss.png",dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4)); ax.hist(actual-predicted,bins=40,color="#0E9F9A",alpha=.85); ax.set(xlabel="Residual (actual − prediction, kW)",ylabel="Observations",title="Test residual distribution"); fig.tight_layout(); fig.savefig(out/"residual_distribution.png",dpi=160); plt.close(fig)
    sample=frame.head(min(500,len(frame))); fig,ax=plt.subplots(figsize=(11,4)); ax.plot(sample.DateTime,sample.actual,label="Actual",lw=1); ax.plot(sample.DateTime,sample.ann_prediction,label="ANN",lw=1); ax.set(xlabel="Time",ylabel="Active power (kW)",title="Sample chronological test interval"); ax.legend(); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(out/"sample_timeseries.png",dpi=160); plt.close(fig)
    write_report(result)
    return result


def write_report(result):
    s=json.loads(Path("data/dataset_summary.json").read_text(encoding="utf-8")); m=json.loads(Path("models/metadata.json").read_text(encoding="utf-8"))
    a,b=result["ANN"],result["Persistence_baseline"]
    report=f'''# EnergySense: Household Electricity Consumption Prediction Using an ANN

## Abstract
This study predicts Global_active_power one minute ahead using a dense TensorFlow/Keras MLP and real household electricity readings. It uses a chronological 50,000-row sample (or all valid rows if fewer), time-ordered train/validation/test partitions, and a previous-reading persistence baseline. Dataset facts and evaluation metrics below are computed by the project scripts.

## Introduction and objectives
The task is short-horizon regression on household load. The project demonstrates artificial neurons and activation functions (Unit I), feedforward propagation (Unit II), and loss, gradient descent and backpropagation (Unit III). It produces a reproducible training experiment, an ANN-versus-baseline evaluation, and a dashboard for exploration and predictions.

## Dataset
The supplied semicolon-delimited file has {s['raw_rows']:,} rows and nine original fields: Date, Time, Global_active_power (kW), Global_reactive_power (kW), Voltage (V), Global_intensity (A), and Sub_metering_1/2/3 (Wh). Missing `?` readings are converted to missing numeric values. There are {s['valid_complete_rows']:,} complete valid observations after removing invalid timestamps and incomplete measurements. The selected chronological subset has {s['selected_rows']:,} rows from {s['selected_start']} to {s['selected_end']}. The selected data file preserves exactly the nine source columns.

### Dataset summary
| Measurement | Mean | Std. dev. | Min | Max |
|---|---:|---:|---:|---:|
'''
    for col,v in s['statistics'].items(): report+=f"| {col} | {v['mean']:.4f} | {v['std']:.4f} | {v['min']:.4f} | {v['max']:.4f} |\n"
    report+=f'''
## Preprocessing and prediction horizon
Pandas reads the TXT directly, parses day-first Date and Time, and converts measurement columns to numeric values. Rows with missing values are dropped; no data are imputed or synthesized. The first 50,000 complete rows are kept chronologically. For target at time *t*, inputs are the six non-target electrical readings at *t−1*, Global_active_power at exact times *t−1* and *t−30*, and cyclic hour/weekday features known at *t*. The target horizon is one minute. Rows with missing exact lag timestamps are excluded. The target at *t* is not an input. The feature scaler is fitted on training rows only.

## ANN and learning
The MLP contains 12 inputs, Dense(64, ReLU), Dense(32, ReLU), and one linear output that predicts a residual correction to the previous-minute reading. A residual multiplier is calibrated from validation predictions only; the final estimate is previous active power plus the scaled correction. A dense neuron computes **z = wᵀx + b**. ReLU is **max(0,z)**. Feedforward propagation passes activations through the layers. MSE is **(1/n)Σ(yᵢ−ŷᵢ)²**. Backpropagation applies the chain rule to calculate each weight and bias gradient; Adam updates parameters. Early stopping monitors validation loss and restores the best weights.

## Experimental methodology and results
The split is chronological: 70% training, 15% validation, and 15% test. Training uses Adam, MSE loss, MAE metric, batch size 128, seed 42, and early stopping. The persistence baseline predicts the prior one-minute active-power reading at each test timestamp.

| Model | MAE (kW) | RMSE (kW) | R² |
|---|---:|---:|---:|
| ANN | {a['MAE']:.6f} | {a['RMSE']:.6f} | {a['R2']:.6f} |
| Persistence baseline | {b['MAE']:.6f} | {b['RMSE']:.6f} | {b['R2']:.6f} |

Test predictions: {result['test_rows']:,}. ANN lower MAE: **{result['ANN_lower_MAE']}**. ANN lower RMSE: **{result['ANN_lower_RMSE']}**. These are measured values on the held-out test interval.

## Diagnostic plots
![Actual versus predicted](../outputs/actual_vs_predicted.png)
![Training and validation loss](../outputs/training_validation_loss.png)
![Residual distribution](../outputs/residual_distribution.png)
![Sample time series](../outputs/sample_timeseries.png)

The scatter plot compares test estimates with observed targets; loss curves show training/validation behavior; residuals show signed errors; and the sample series illustrates the model over a chronological test segment. The baseline metrics determine comparative value; plots alone do not establish performance.

## Limitations, future work, and conclusion
The experiment uses an early chronological segment of a much longer dataset, so results apply to that interval. Exact-time lag matching reduces usable rows where readings are missing. Future work can evaluate more temporal cut points, tune regularization, and add transparent baselines. Performance should be judged from the reported held-out metrics; no generalization beyond this experiment is implied.

## References
1. Hebrail, G. and Berard, A. (2006). *Individual Household Electric Power Consumption*. UCI Machine Learning Repository.
2. Goodfellow, I., Bengio, Y. and Courville, A. (2016). *Deep Learning*. MIT Press.
3. Kingma, D. P. and Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR.
'''
    Path("reports").mkdir(exist_ok=True); Path("reports/project_report.md").write_text(report,encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run(),indent=2))
