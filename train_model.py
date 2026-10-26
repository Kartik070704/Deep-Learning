"""Train a chronological TensorFlow/Keras MLP regression model."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.energy_sense import FEATURE_COLUMNS, TARGET, make_supervised

SEED = 42


def run(data_path=Path("data/selected_observations.csv"), model_dir=Path("models"), output_dir=Path("outputs"), epochs=100):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required; install project dependencies from requirements.txt.") from exc
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    frame = pd.read_csv(data_path)
    frame["DateTime"] = pd.to_datetime(frame.Date.astype(str) + " " + frame.Time.astype(str), format="%d/%m/%Y %H:%M:%S", errors="coerce")
    frame = frame.dropna(subset=["DateTime"])
    x, y, timestamps = make_supervised(frame)
    if len(x) < 100:
        raise ValueError(f"Only {len(x)} supervised observations are available.")
    n = len(x)
    train_end, val_end = int(n*.70), int(n*.85)
    x_train, x_val, x_test = x.iloc[:train_end], x.iloc[train_end:val_end], x.iloc[val_end:]
    y_train, y_val, y_test = y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:]
    scaler = StandardScaler().fit(x_train)
    scale = lambda part: scaler.transform(part).astype("float32")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(FEATURE_COLUMNS),)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="linear"),
    ], name="EnergySense_MLP")
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse", metrics=["mae"])
    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
    # Learn a correction to the persistence forecast; the final ANN prediction
    # is the previous active-power reading plus this learned residual.
    y_train_residual = y_train.to_numpy("float32") - x_train.Global_active_power_lag_1.to_numpy("float32")
    y_val_residual = y_val.to_numpy("float32") - x_val.Global_active_power_lag_1.to_numpy("float32")
    history = model.fit(scale(x_train), y_train_residual, validation_data=(scale(x_val), y_val_residual),
                        epochs=epochs, batch_size=128, shuffle=False, callbacks=[callback], verbose=2)
    model_dir.mkdir(parents=True, exist_ok=True); output_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "energy_sense.keras"); joblib.dump(scaler, model_dir / "feature_scaler.joblib")
    # Calibrate residual strength from validation only. This preserves the
    # persistence forecast if the ANN's correction is not useful.
    val_residual_prediction = model.predict(scale(x_val), verbose=0).reshape(-1)
    val_baseline = x_val.Global_active_power_lag_1.to_numpy()
    val_change = y_val.to_numpy() - val_baseline
    denominator = float(np.dot(val_residual_prediction, val_residual_prediction))
    residual_scale = (float(np.dot(val_residual_prediction, val_change)) / denominator) if denominator > 1e-12 else 0.0
    residual_scale = float(np.clip(residual_scale, 0.0, 1.5))
    residual_prediction = model.predict(scale(x_test), verbose=0).reshape(-1)
    test_baseline = x_test.Global_active_power_lag_1.to_numpy()
    prediction = test_baseline + residual_scale * residual_prediction
    pd.DataFrame({"DateTime": timestamps.iloc[val_end:].to_numpy(), "actual": y_test.to_numpy(),
                  "ann_prediction": prediction,
                  "baseline_prediction": x_test.Global_active_power_lag_1.to_numpy()}).to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame(history.history).to_csv(output_dir / "training_history.csv", index_label="epoch")
    metadata = {"seed": SEED, "features": FEATURE_COLUMNS, "target": TARGET, "selected_rows": int(len(frame)),
                "supervised_rows": n, "train_rows": len(x_train), "validation_rows": len(x_val), "test_rows": len(x_test),
                "split": "chronological 70/15/15", "prediction_horizon": "one minute; exact prior timestamps at 1 and 30 minutes",
                "output_form": "previous active power plus ANN-predicted residual",
                "validation_residual_scale": residual_scale,
                "epochs_ran": len(history.history["loss"]), "best_epoch": int(np.argmin(history.history["val_loss"])+1)}
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--data", type=Path, default=Path("data/selected_observations.csv")); parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args(); print(json.dumps(run(data_path=args.data, epochs=args.epochs), indent=2))
