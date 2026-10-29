# EnergySense

EnergySense is a CSM422 Deep Learning CA1 project that estimates household `Global_active_power` one minute ahead using a feedforward TensorFlow/Keras MLP. It uses the supplied real dataset, preserves time order, compares against a persistence baseline, and provides a light-themed Streamlit analytics dashboard.

## Project structure

```text
app.py                    Streamlit dashboard
preprocess.py             Dataset inspection and chronological sample creation
train_model.py            Keras MLP training with early stopping
evaluate.py               Holdout metrics, plots, and academic report
src/energy_sense.py       Shared loading and time-safe feature engineering
data/README.md            Local data artifact notes
models/                   Saved model, scaler, and metadata (after training)
outputs/                  Predictions, metrics, and plots (after evaluation)
reports/                  Generated academic report (after evaluation)
tests/                    Basic data and prediction checks
```

## Submission contents

The repository includes the trained model and scaler, model metadata, selected 50,000-row dataset, dataset summary, evaluation metrics and plots, test predictions, training history, and generated report. The original 132 MB source TXT is excluded; keep the supplied file in the project root if you need to rerun preprocessing from scratch. The selected dataset is included so training and evaluation can be rerun without that large source file.

## Dataset

Keep the provided `household_power_consumption.txt` in the project root. It is read directly with Pandas, is never modified, and is excluded from Git. Preprocessing writes `data/selected_observations.csv` with exactly nine original columns and up to the first 50,000 complete rows in chronological order. It also creates a compressed full-cleaned export and an inspection summary. No synthetic or duplicated observations are generated.

## Run locally on Windows

Use Python 3.10 or 3.11 in Command Prompt:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python preprocess.py
python train_model.py
python evaluate.py
python -m unittest discover -s tests -v
streamlit run app.py
```

The included results were generated from the supplied dataset and are available without rerunning training. The commands above reproduce them locally; `preprocess.py` needs the original TXT file, while `train_model.py` and `evaluate.py` can use the included `data/selected_observations.csv`.

The model uses the six non-target electrical readings exactly one minute earlier, active-power readings exactly one and 30 minutes earlier, and cyclic time features known at prediction time. Target rows without those exact earlier timestamps are removed. The first 70% of usable rows train, the next 15% validate, and the final 15% test. No random shuffle is used; scaling is fitted on training data only.

## Google Colab

Upload the project and the supplied TXT dataset, then run:

```python
%pip install -r requirements.txt
!python preprocess.py --input household_power_consumption.txt
!python train_model.py
!python evaluate.py
!python -m unittest discover -s tests -v
```

Download `models/`, `outputs/`, and `reports/project_report.md` before the Colab runtime ends.

## Streamlit Community Cloud

The project includes a light Streamlit theme in `.streamlit/config.toml`. To publish, push the source files to a GitHub repository you control, then create a Streamlit Community Cloud app with entrypoint `app.py`. The original raw file and full cleaned export are excluded; the selected sample and coursework evaluation artifacts are included for this submission. Before making the repository public, decide whether you want to publish those dataset-derived artifacts. The dashboard uses the included model, metrics, plots, and selected sample when they are present.

## Model and metrics

Architecture: 12 inputs → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, linear residual). The ANN estimates a correction added to the previous exact-minute active-power value, allowing a direct comparison with the persistence baseline. Compile with Adam, MSE loss, and MAE metric. Early stopping restores best validation weights. The evaluation script computes MAE, RMSE, and R² from actual held-out predictions, saves four plots, and writes the report with measured results.

The included held-out results are mixed: the ANN has lower RMSE and higher R² than persistence, but persistence has lower MAE. The report presents both results so the model is not described as better on every metric. See [the generated project report](reports/project_report.md) and [evaluation metrics](outputs/evaluation_metrics.json).
