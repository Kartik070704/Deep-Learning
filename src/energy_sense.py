"""Shared loading, feature engineering, and prediction utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ORIGINAL_COLUMNS = [
    "Date", "Time", "Global_active_power", "Global_reactive_power", "Voltage",
    "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
]
MEASUREMENT_COLUMNS = ORIGINAL_COLUMNS[2:]
TARGET = "Global_active_power"
FEATURE_COLUMNS = [f"{col}_lag_1" for col in MEASUREMENT_COLUMNS if col != TARGET] + [
    "Global_active_power_lag_1", "Global_active_power_lag_30",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
]


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Read the supplied semicolon-delimited data without altering its source file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path, sep=";", na_values=["?"], low_memory=False)
    missing = set(ORIGINAL_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    frame = frame[ORIGINAL_COLUMNS].copy()
    frame["DateTime"] = pd.to_datetime(
        frame["Date"].astype(str) + " " + frame["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S", errors="coerce",
    )
    for col in MEASUREMENT_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop incomplete readings, order by time, and keep the first duplicate timestamp."""
    result = frame.dropna(subset=["DateTime", *MEASUREMENT_COLUMNS]).copy()
    result = result.sort_values("DateTime", kind="stable").drop_duplicates("DateTime", keep="first")
    result["Date"] = result["DateTime"].dt.strftime("%d/%m/%Y")
    result["Time"] = result["DateTime"].dt.strftime("%H:%M:%S")
    return result.reset_index(drop=True)


def make_supervised(clean: pd.DataFrame):
    """Build one-minute-ahead targets with strictly earlier exact-time measurements."""
    frame = clean.sort_values("DateTime", kind="stable").reset_index(drop=True)
    by_time = frame.set_index("DateTime")
    times = pd.DatetimeIndex(frame["DateTime"])
    prior_1 = times - pd.Timedelta(minutes=1)
    prior_30 = times - pd.Timedelta(minutes=30)
    x = pd.DataFrame(index=frame.index)
    for col in MEASUREMENT_COLUMNS:
        if col != TARGET:
            x[f"{col}_lag_1"] = by_time[col].reindex(prior_1).to_numpy()
    x["Global_active_power_lag_1"] = by_time[TARGET].reindex(prior_1).to_numpy()
    x["Global_active_power_lag_30"] = by_time[TARGET].reindex(prior_30).to_numpy()
    hour = times.hour + times.minute / 60.0
    weekday = times.dayofweek
    x["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    x["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    x["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    valid = x.notna().all(axis=1)
    return (x.loc[valid, FEATURE_COLUMNS].reset_index(drop=True),
            frame.loc[valid, TARGET].reset_index(drop=True),
            frame.loc[valid, "DateTime"].reset_index(drop=True))


def build_prediction_row(readings: Iterable[dict], prediction_time=None) -> pd.DataFrame:
    """Validate 30 consecutive minute readings (oldest first) for one next-minute estimate."""
    rows = list(readings)
    if len(rows) < 30:
        raise ValueError("Enter 30 consecutive minute readings, oldest first.")
    recent = pd.DataFrame(rows[-30:])
    absent = set(MEASUREMENT_COLUMNS) - set(recent.columns)
    if absent:
        raise ValueError(f"Missing required fields: {sorted(absent)}")
    recent = recent[MEASUREMENT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if recent.isna().any().any() or not np.isfinite(recent.to_numpy(dtype=float)).all():
        raise ValueError("Enter finite numeric readings for every required field.")
    row = {f"{col}_lag_1": float(recent.iloc[-1][col]) for col in MEASUREMENT_COLUMNS if col != TARGET}
    row["Global_active_power_lag_1"] = float(recent.iloc[-1][TARGET])
    row["Global_active_power_lag_30"] = float(recent.iloc[0][TARGET])
    timestamp = pd.Timestamp(prediction_time) if prediction_time else pd.Timestamp.now().ceil("min")
    hour = timestamp.hour + timestamp.minute / 60.0
    day = timestamp.dayofweek
    row.update(hour_sin=np.sin(2*np.pi*hour/24), hour_cos=np.cos(2*np.pi*hour/24),
               weekday_sin=np.sin(2*np.pi*day/7), weekday_cos=np.cos(2*np.pi*day/7))
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_next_power(model, scaler, readings, prediction_time=None, residual_scale: float = 1.0) -> float:
    rows = list(readings)
    features = build_prediction_row(rows, prediction_time)
    residual = np.asarray(model.predict(scaler.transform(features), verbose=0)).reshape(-1)
    prediction = residual_scale * residual + float(rows[-1][TARGET])
    if prediction.size != 1 or not np.isfinite(prediction[0]):
        raise ValueError("Model did not return one finite prediction.")
    return float(prediction[0])
