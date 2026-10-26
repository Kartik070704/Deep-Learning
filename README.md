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

The project includes a light Streamlit theme in `.streamlit/config.toml`. To publish, push the source files to a GitHub repository you control, then create a Streamlit Community Cloud app with entrypoint `app.py`. Do not commit the original raw file or local measurement exports. The dashboard explains missing model or evaluation artifacts; provide the trained model and scaler through a secure deployment artifact path if the prediction page should be live. Keep dataset-derived metrics and report artifacts out of a public repository unless you decide to share them.

## Model and metrics

Architecture: 12 inputs → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, linear residual). The ANN estimates a correction added to the previous exact-minute active-power value, allowing a direct comparison with the persistence baseline. Compile with Adam, MSE loss, and MAE metric. Early stopping restores best validation weights. The evaluation script computes MAE, RMSE, and R² from actual held-out predictions, saves four plots, and writes the report with measured results.
