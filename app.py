"""EnergySense: an interactive household energy forecasting dashboard."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.energy_sense import FEATURE_COLUMNS, MEASUREMENT_COLUMNS, predict_next_power

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
SUMMARY_PATH = DATA_DIR / "dataset_summary.json"
METRICS_PATH = OUTPUT_DIR / "evaluation_metrics.json"
METADATA_PATH = MODEL_DIR / "metadata.json"

st.set_page_config(page_title="EnergySense", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#172b4d; --muted:#6c7d93; --teal:#087f78; --line:#e4ebf2; --paper:#f5f8fc; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:var(--paper); color:var(--ink); }
[data-testid="stSidebar"] { background:#fff; border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div { padding-top:1.5rem; }
.block-container { max-width:1440px; padding-top:2rem; padding-bottom:3rem; }
h1,h2,h3 { font-family:'Manrope',sans-serif; color:var(--ink); letter-spacing:-.035em; }
h1 { font-size:2.2rem !important; }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:16px; padding:18px 20px; box-shadow:0 5px 18px rgba(28,55,90,.035); }
[data-testid="stMetricLabel"] { color:var(--muted); font-weight:600; }
[data-testid="stMetricValue"] { color:var(--teal); font-family:'Manrope',sans-serif; }
[data-testid="stMetricDelta"] { font-size:.8rem; }
.stButton > button, .stDownloadButton > button { border-radius:10px; font-weight:650; border:1px solid #d8e4ec; }
.stButton > button[kind="primary"] { background:var(--teal); border-color:var(--teal); }
div[data-testid="stTabs"] button[aria-selected="true"] { color:var(--teal); }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
.eyebrow { color:var(--teal); font-size:.75rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }
.hero { background:linear-gradient(120deg,#e4f5f2 0%,#eef4ff 62%,#fff 100%); border:1px solid #dcece9; border-radius:22px; padding:28px 32px; margin:8px 0 24px; }
.hero h2 { margin:.35rem 0 .4rem; font-size:1.75rem; }
.hero p { color:#53677f; margin:0; max-width:720px; }
.panel { background:#fff; border:1px solid var(--line); border-radius:16px; padding:20px 22px; }
.small-muted { color:var(--muted); font-size:.9rem; }
.pill { display:inline-block; border-radius:999px; padding:5px 10px; background:#e8f5f3; color:#087f78; font-size:.78rem; font-weight:700; }
hr { border-color:var(--line); }
</style>
""", unsafe_allow_html=True)


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_table():
    path = DATA_DIR / "selected_observations.csv"
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
        if "Date" in frame and "Time" in frame:
            frame["Timestamp"] = pd.to_datetime(frame["Date"].astype(str) + " " + frame["Time"].astype(str), dayfirst=True, errors="coerce")
        return frame
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return None


@st.cache_resource(show_spinner="Loading the forecasting model…")
def load_model_bundle():
    model_path = MODEL_DIR / "energy_sense.keras"
    scaler_path = MODEL_DIR / "feature_scaler.joblib"
    if not model_path.exists() or not scaler_path.exists():
        return None
    import joblib
    from tensorflow import keras
    return keras.models.load_model(model_path), joblib.load(scaler_path)


summary = read_json(SUMMARY_PATH)
metrics = read_json(METRICS_PATH)
metadata = read_json(METADATA_PATH) or {}
selected = load_table()

with st.sidebar:
    st.markdown("<div style='font-family:Manrope;font-size:1.2rem;font-weight:800;color:#172b4d'>⚡ EnergySense</div>", unsafe_allow_html=True)
    st.caption("Household energy intelligence")
    st.markdown("---")
    page = st.radio("Navigate", ["Dashboard", "Prediction", "Data Explorer", "About"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<span class='pill'>CSM422 · Deep Learning</span>", unsafe_allow_html=True)
    st.markdown("<p class='small-muted' style='margin-top:12px'>One-minute-ahead household electricity forecasting.</p>", unsafe_allow_html=True)

if page == "Dashboard":
    st.markdown("<div class='eyebrow'>Energy overview</div>", unsafe_allow_html=True)
    st.title("Dashboard")
    st.markdown("<div class='hero'><div class='eyebrow'>EnergySense · Forecasting workspace</div><h2>See the patterns behind household power.</h2><p>Explore real meter readings, review a chronological neural-network evaluation, and estimate the next minute of active power.</p></div>", unsafe_allow_html=True)
    if summary:
        a,b,c,d = st.columns(4)
        a.metric("Source readings", f"{summary.get('raw_rows', 0):,}")
        b.metric("Complete readings", f"{summary.get('valid_complete_rows', 0):,}")
        c.metric("Model sample", f"{summary.get('selected_rows', 0):,}")
        d.metric("Input features", len(FEATURE_COLUMNS))
        if summary.get("selected_start") and summary.get("selected_end"):
            st.caption(f"Experiment window: {summary['selected_start']} — {summary['selected_end']}")
    else:
        a,b,c,d = st.columns(4)
        a.metric("Forecast horizon", "1 minute")
        b.metric("Model", "Feedforward ANN")
        c.metric("Input features", len(FEATURE_COLUMNS))
        d.metric("Evaluation split", "Chronological")
        st.info("Dataset summary is not bundled with this copy. The dashboard is ready; add the prepared summary and evaluation artifacts to show project results here.")

    left,right = st.columns([1.45,1])
    with left:
        st.subheader("Forecast performance")
        if metrics and "ANN" in metrics:
            m = metrics["ANN"]
            p = metrics.get("Persistence_baseline", {})
            chart = pd.DataFrame({"ANN": [m.get("MAE"),m.get("RMSE")], "Persistence baseline": [p.get("MAE"),p.get("RMSE")]}, index=["MAE (kW)","RMSE (kW)"])
            st.bar_chart(chart, color=["#0b8f86", "#a7b8cb"], height=260)
            st.caption(f"Test set: {metrics.get('test_rows', '—')} held-out predictions · lower error is better.")
        else:
            st.markdown("<div class='panel'><div class='eyebrow'>Evaluation</div><h3>Results will appear here</h3><p class='small-muted'>Run the preprocessing, training, and evaluation pipeline to populate metrics and charts.</p></div>", unsafe_allow_html=True)
    with right:
        st.subheader("Project snapshot")
        st.markdown("<div class='panel'><p><b>Target</b><br><span class='small-muted'>Global active power (kW)</span></p><hr><p><b>Network</b><br><span class='small-muted'>12 inputs → 64 → 32 → 1</span></p><hr><p><b>Benchmark</b><br><span class='small-muted'>Previous exact-minute reading</span></p></div>", unsafe_allow_html=True)
    if metrics:
        st.subheader("Evaluation visualizations")
        paths = [("actual_vs_predicted.png","Actual vs predicted"),("training_validation_loss.png","Training and validation loss"),("residual_distribution.png","Residual distribution"),("sample_timeseries.png","Chronological test sample")]
        available = [(OUTPUT_DIR/name,title) for name,title in paths if (OUTPUT_DIR/name).exists()]
        if available:
            cols = st.columns(2)
            for i,(path,title) in enumerate(available):
                with cols[i % 2]: st.image(str(path), caption=title, use_container_width=True)

elif page == "Prediction":
    st.markdown("<div class='eyebrow'>Interactive forecast</div>", unsafe_allow_html=True)
    st.title("Next-minute prediction")
    st.write("Enter the most recent 30 consecutive one-minute readings. EnergySense uses them to estimate the next minute’s household active power.")
    model_bundle = load_model_bundle()
    if model_bundle is None:
        st.warning("The trained model files are not available in this deployment yet. The prediction form will become active when `models/energy_sense.keras` and `models/feature_scaler.joblib` are included.")
    with st.expander("What should I enter?", expanded=False):
        st.write("Enter 30 consecutive rows in oldest-first order. Each row needs all six measurements. Active and reactive power are in kW, voltage in V, current in A, and sub-meter readings in Wh.")
        st.code(",".join(MEASUREMENT_COLUMNS) + "\n0.4,0.1,240,1.8,0,1,20", language="csv")
    upload = st.file_uploader("Or upload a CSV with 30 consecutive readings", type=["csv"], key="prediction_csv")
    if upload is not None:
        try:
            input_frame = pd.read_csv(upload)
            missing = set(MEASUREMENT_COLUMNS) - set(input_frame.columns)
            if missing:
                st.error("CSV is missing: " + ", ".join(sorted(missing)))
                input_frame = pd.DataFrame([{col: None for col in MEASUREMENT_COLUMNS} for _ in range(30)])
            else:
                input_frame = input_frame[MEASUREMENT_COLUMNS].tail(30)
                if len(input_frame) < 30:
                    st.warning(f"Your file has {len(input_frame)} rows; 30 are required.")
                    input_frame = input_frame.reindex(range(30))
        except Exception as exc:
            st.error(f"Could not read that CSV: {exc}")
            input_frame = pd.DataFrame([{col: None for col in MEASUREMENT_COLUMNS} for _ in range(30)])
    else:
        input_frame = pd.DataFrame([{col: None for col in MEASUREMENT_COLUMNS} for _ in range(30)])
    st.caption("Oldest reading at the top · newest reading at the bottom")
    readings = st.data_editor(input_frame, num_rows="fixed", use_container_width=True, hide_index=True, key="recent_readings", column_config={col: st.column_config.NumberColumn(col.replace("_", " "), format="%.3f") for col in MEASUREMENT_COLUMNS})
    prediction_time = st.text_input("Forecast timestamp (optional)", placeholder="2026-10-27 12:00")
    if st.button("Estimate next minute", type="primary", disabled=(model_bundle is None)):
        try:
            model, scaler = model_bundle
            estimate = predict_next_power(model, scaler, readings.to_dict(orient="records"), prediction_time or None, metadata.get("validation_residual_scale", 1.0))
            st.success(f"Estimated global active power: **{estimate:.3f} kW**")
            st.caption("Forecast based on the supplied recent readings and the trained EnergySense model.")
        except Exception as exc:
            st.error(f"Could not calculate a forecast: {exc}")

elif page == "Data Explorer":
    st.markdown("<div class='eyebrow'>Explore the measurements</div>", unsafe_allow_html=True)
    st.title("Data Explorer")
    st.write("Browse the original measurement fields, inspect the selected chronological sample, and download it for further analysis.")
    if selected is None:
        st.info("The selected observations are not included in this deployment. Add `data/selected_observations.csv` to enable the interactive table and charts.")
        if summary:
            with st.expander("Dataset inspection summary"):
                st.json(summary)
    else:
        numeric = [c for c in MEASUREMENT_COLUMNS if c in selected]
        if "Timestamp" in selected and selected["Timestamp"].notna().any():
            valid = selected["Timestamp"].dropna()
            date_range = st.slider("Date range", min_value=valid.min().to_pydatetime(), max_value=valid.max().to_pydatetime(), value=(valid.min().to_pydatetime(),valid.max().to_pydatetime()), format="YYYY-MM-DD")
            view = selected[selected["Timestamp"].between(pd.Timestamp(date_range[0]),pd.Timestamp(date_range[1]))]
        else:
            view = selected
        chosen = st.selectbox("Measurement to chart", numeric, index=numeric.index("Global_active_power") if "Global_active_power" in numeric else 0)
        x,y = st.columns([1.35,1])
        with x:
            st.subheader(chosen.replace("_"," "))
            if "Timestamp" in view:
                st.line_chart(view.set_index("Timestamp")[[chosen]], color="#0b8f86", height=300)
            else:
                st.line_chart(view[[chosen]], color="#0b8f86", height=300)
        with y:
            st.subheader("Quick statistics")
            st.dataframe(view[numeric].describe().T[["mean","std","min","max"]].round(3), use_container_width=True)
        st.subheader(f"Observations · {len(view):,} rows")
        st.dataframe(view.drop(columns=["Timestamp"], errors="ignore"), use_container_width=True, hide_index=True, height=360)
        st.download_button("Download filtered observations", view.drop(columns=["Timestamp"], errors="ignore").to_csv(index=False).encode("utf-8"), "energy_observations.csv", "text/csv")

else:
    st.markdown("<div class='eyebrow'>About this project</div>", unsafe_allow_html=True)
    st.title("About EnergySense")
    st.markdown("<div class='hero'><div class='eyebrow'>CSM422 · Deep Learning CA1</div><h2>Learning household electricity patterns.</h2><p>EnergySense is an educational forecasting project built around real household power measurements. It uses a small feedforward neural network to estimate global active power one minute ahead.</p></div>", unsafe_allow_html=True)
    one,two = st.columns(2)
    with one:
        st.subheader("Model workflow")
        st.markdown("<div class='panel'><b>01 · Prepare</b><p class='small-muted'>Parse and clean the source measurements, retain chronological order, and select complete observations.</p><b>02 · Learn</b><p class='small-muted'>Train a 12-feature, 64–32–1 multilayer perceptron with ReLU activations and early stopping.</p><b>03 · Compare</b><p class='small-muted'>Evaluate on a time-ordered holdout and compare with a persistence baseline.</p><b>04 · Forecast</b><p class='small-muted'>Use 30 recent readings to estimate active power for the next minute.</p></div>", unsafe_allow_html=True)
    with two:
        st.subheader("Responsible interpretation")
        st.markdown("<div class='panel'><p>Forecasts are estimates for coursework and exploration. They depend on complete, correctly ordered input readings and are not a substitute for a calibrated utility meter.</p><hr><p><b>Target</b><br><span class='small-muted'>Global active power, measured in kilowatts</span></p><p><b>Evaluation design</b><br><span class='small-muted'>Chronological train, validation, and test partitions</span></p></div>", unsafe_allow_html=True)
    st.subheader("Project artifacts")
    st.write("The dashboard reads prepared data from `data/`, model files from `models/`, and evaluation outputs from `outputs/`. To reproduce results, follow the setup steps in the repository README.")
