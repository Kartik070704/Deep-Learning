"""EnergySense · light-themed household energy analytics dashboard."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.energy_sense import FEATURE_COLUMNS, MEASUREMENT_COLUMNS, predict_next_power

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title="EnergySense",page_icon="⚡",layout="wide")
st.markdown("""<style>
.stApp{background:#F5F8FC;color:#172B4D} [data-testid="stSidebar"]{background:#EAF1F8;border-right:1px solid #DCE5EF}
[data-testid="stMetric"]{background:#FFFFFF;border:1px solid #DCE5EF;border-radius:14px;padding:15px;box-shadow:0 3px 12px #102a430b}
[data-testid="stMetricValue"]{color:#0E807B} h1,h2,h3{color:#183B56} .block-container{padding-top:2rem}
div[data-testid="stTabs"] button[aria-selected="true"]{color:#0E807B}
</style>""",unsafe_allow_html=True)
summary_path=ROOT/"data/dataset_summary.json"; metrics_path=ROOT/"outputs/evaluation_metrics.json"; metadata_path=ROOT/"models/metadata.json"
st.title("⚡ EnergySense")
st.caption("Household electricity analytics · One-minute-ahead ANN forecasting")
page=st.sidebar.radio("Workspace",["Overview","Dataset explorer","Preprocessing","ANN & training","Evaluation","Prediction"])
st.sidebar.markdown("---")
st.sidebar.caption("CSM422 · Deep Learning CA1")

if page=="Overview":
    st.header("Understand household energy, one minute ahead")
    st.write("EnergySense studies minute-level meter readings with an interpretable feedforward neural network. The experiment uses a chronological sample and compares the ANN with a persistence baseline.")
    if summary_path.exists():
        s=json.loads(summary_path.read_text(encoding="utf-8")); a,b,c=st.columns(3)
        a.metric("Source observations",f"{s['raw_rows']:,}"); b.metric("Complete valid readings",f"{s['valid_complete_rows']:,}"); c.metric("Experiment sample",f"{s['selected_rows']:,}")
        st.caption(f"Selected time span: {s['selected_start']} → {s['selected_end']}")
    else: st.info("Dataset summary is not present yet. Run `python preprocess.py` from the project directory.")
    st.subheader("Project at a glance")
    cols=st.columns(3)
    cols[0].markdown("**Target**  \nGlobal active power (kW)")
    cols[1].markdown("**Model**  \n64 → 32 neuron MLP · ReLU · Linear output")
    cols[2].markdown("**Evaluation**  \nChronological holdout · persistence baseline")
    st.markdown("**Course concepts:** artificial neurons and activations · feedforward propagation · loss, gradient updates, and backpropagation")
elif page=="Dataset explorer":
    st.header("Dataset explorer")
    if summary_path.exists():
        s=json.loads(summary_path.read_text(encoding="utf-8"))
        st.write(f"**{s['raw_rows']:,}** original rows · **{s['valid_complete_rows']:,}** complete observations · **{s['selected_rows']:,}** chronological sample")
        meanings=["Date","Time","Active power (kW), target","Reactive power (kW)","Voltage (V)","Current (A)","Sub-meter 1 (Wh)","Sub-meter 2 (Wh)","Sub-meter 3 (Wh)"]
        st.dataframe(pd.DataFrame({"Original field":s["raw_columns"],"Meaning":meanings}),use_container_width=True,hide_index=True)
        left,right=st.columns(2)
        with left: st.subheader("Missing source measurements"); st.dataframe(pd.Series(s["missing_counts"],name="Missing").to_frame(),use_container_width=True)
        with right: st.subheader("Numeric summary"); st.dataframe(pd.DataFrame(s["statistics"]).T,use_container_width=True)
        selected=ROOT/"data/selected_observations.csv"
        if selected.exists():
            df=pd.read_csv(selected); st.subheader("Selected real observations"); st.dataframe(df.head(100),use_container_width=True)
            st.download_button("Download selected data · 9 original columns",selected.read_bytes(),"selected_observations.csv","text/csv")
    else: st.info("Run preprocessing to generate the dataset summary and selected data preview.")
elif page=="Preprocessing":
    st.header("Preprocessing & feature design")
    st.markdown("The TXT file is loaded directly with Pandas using semicolons. `?` and non-numeric measurements become missing values. Invalid timestamps and incomplete rows are removed; complete measurements are not imputed. The valid readings are sorted by time, and the first 40,000 are selected.")
    st.markdown("**Forecast horizon:** target active power at minute *t*. The model receives the six other electrical measurements at *t−1*, active power at exact timestamps *t−1* and *t−30*, and cyclic encodings of hour and weekday. Rows without those exact past readings are excluded. The target at *t* is never an input.")
    if summary_path.exists():
        s=json.loads(summary_path.read_text(encoding="utf-8")); st.json({k:s[k] for k in ["raw_rows","raw_columns","raw_dtypes","missing_counts","valid_complete_rows","selected_rows","selected_start","selected_end","cleaning"]})
    else: st.info("Run `python preprocess.py` to inspect and prepare the data.")
elif page=="ANN & training":
    st.header("ANN architecture & learning")
    st.code(f"Input ({len(FEATURE_COLUMNS)} features) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Linear)")
    st.markdown("A layer computes **z = Wa + b** from its input activations, learned weights, and biases. ReLU applies **max(0,z)** in the hidden layers; the linear output estimates a correction to the persistence forecast. The final estimate adds that learned correction to the previous minute’s active power. Feedforward propagation passes each layer’s values onward. MSE, **(1/n)Σ(y−ŷ)²**, measures training error. Backpropagation applies the chain rule to find weight and bias gradients, and Adam updates the parameters. Early stopping restores the best validation weights.")
    if metadata_path.exists(): st.json(json.loads(metadata_path.read_text(encoding="utf-8")))
    else: st.warning("No trained model is available yet. Run `python preprocess.py`, then `python train_model.py`.")
elif page=="Evaluation":
    st.header("Held-out evaluation")
    if metrics_path.exists():
        m=json.loads(metrics_path.read_text(encoding="utf-8")); st.caption(f"Metrics from {m['test_rows']:,} held-out predictions · target unit kW")
        cols=st.columns(2)
        for idx,(label,key) in enumerate([("ANN","ANN"),("Previous-reading baseline","Persistence_baseline")]):
            with cols[idx]:
                st.subheader(label); a,b,c=st.columns(3); a.metric("MAE",f"{m[key]['MAE']:.4f}"); b.metric("RMSE",f"{m[key]['RMSE']:.4f}"); c.metric("R²",f"{m[key]['R2']:.4f}")
        st.info(f"ANN lower MAE: {m['ANN_lower_MAE']} · ANN lower RMSE: {m['ANN_lower_RMSE']}")
        for name,title in [("actual_vs_predicted.png","Actual versus predicted"),("training_validation_loss.png","Training and validation loss"),("residual_distribution.png","Residual distribution"),("sample_timeseries.png","Chronological test sample")]:
            path=ROOT/"outputs"/name
            if path.exists(): st.image(str(path),caption=title,use_container_width=True)
        predictions=ROOT/"outputs/test_predictions.csv"
        if predictions.exists(): st.download_button("Download test predictions",predictions.read_bytes(),"test_predictions.csv","text/csv")
    else: st.info("Evaluation is not available yet. Run preprocessing, training, then `python evaluate.py`.")
elif page=="Prediction":
    st.header("Predict next-minute electricity use")
    model_path=ROOT/"models/energy_sense.keras"; scaler_path=ROOT/"models/feature_scaler.joblib"
    if not model_path.exists() or not scaler_path.exists():
        st.info("The trained model is not generated yet. Run `python preprocess.py` and `python train_model.py` first.")
    else:
        st.write("Enter 30 consecutive readings in oldest-first order. Units: active/reactive power in kW, voltage in V, intensity in A, sub-meter readings in Wh. The output estimates active power for the following minute.")
        empty=pd.DataFrame([{c:None for c in MEASUREMENT_COLUMNS} for _ in range(30)])
        values=st.data_editor(empty,num_rows="fixed",use_container_width=True,key="recent-readings")
        time_text=st.text_input("Prediction time (optional)",placeholder="2026-10-27 12:00")
        if st.button("Estimate next minute",type="primary"):
            try:
                import joblib
                from tensorflow import keras
                model=keras.models.load_model(model_path); scaler=joblib.load(scaler_path)
                metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
                estimate=predict_next_power(model,scaler,values.to_dict(orient="records"),time_text or None,metadata.get("validation_residual_scale",1.0))
                st.success(f"Estimated Global_active_power: **{estimate:.3f} kW**")
            except Exception as exc: st.error(f"Prediction could not be calculated: {exc}")
