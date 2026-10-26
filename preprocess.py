"""Inspect and prepare the supplied real household power measurements."""
import argparse
import json
from pathlib import Path

from src.energy_sense import MEASUREMENT_COLUMNS, ORIGINAL_COLUMNS, clean_data, load_raw_data


def run(input_path=Path("household_power_consumption.txt"), output_dir=Path("data"), sample_size=50_000):
    raw = load_raw_data(input_path)
    missing = raw[MEASUREMENT_COLUMNS].isna().sum()
    clean = clean_data(raw)
    selected = clean.head(sample_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean[[*ORIGINAL_COLUMNS, "DateTime"]].to_csv(output_dir / "cleaned_household_power_consumption.csv.gz", index=False, compression="gzip")
    selected[ORIGINAL_COLUMNS].to_csv(output_dir / "selected_observations.csv", index=False)
    summary = {
        "raw_rows": int(len(raw)), "raw_columns": ORIGINAL_COLUMNS,
        "raw_dtypes": {key: str(value) for key, value in raw[ORIGINAL_COLUMNS].dtypes.items()},
        "missing_counts": {key: int(value) for key, value in missing.items()},
        "valid_complete_rows": int(len(clean)), "selected_rows": int(len(selected)),
        "selection": f"first {sample_size} complete observations in chronological order, or fewer if unavailable",
        "selected_start": selected.DateTime.min().isoformat() if len(selected) else None,
        "selected_end": selected.DateTime.max().isoformat() if len(selected) else None,
        "statistics": clean[MEASUREMENT_COLUMNS].describe().to_dict(),
        "cleaning": "Invalid timestamps and rows with any missing/non-numeric measurement are excluded; no imputation or synthetic observations.",
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("household_power_consumption.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--sample-size", type=int, default=50_000)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output_dir, args.sample_size), indent=2))
