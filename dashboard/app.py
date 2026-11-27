

# STEP 38: MULTI-DATASET PREDICTIVE MAINTENANCE DASHBOARD

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st


# --------------------------------
# 1. Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="✈️",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "deployment"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

API_URL = "https://predictive-maintenance-rul-5zdy.onrender.com/predict"

DATASETS = ["FD001", "FD002", "FD003", "FD004"]

MODEL_MAPPING = {
    "FD001": "BiLSTM",
    "FD002": "BiLSTM",
    "FD003": "Transformer",
    "FD004": "Transformer"
}

COLUMNS = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


# --------------------------------
# 2. Dashboard Header
# --------------------------------

st.title("✈️ Predictive Maintenance Dashboard")

st.subheader(
    "Aircraft Engine Remaining Useful Life (RUL) Prediction"
)

st.markdown(
    """
    Monitor turbofan engine sensor readings and estimate
    Remaining Useful Life using trained deep learning models.

    **Supported datasets:** FD001, FD002, FD003 and FD004.
    """
)

st.divider()


# --------------------------------
# 3. Dataset Selection
# --------------------------------

st.header("Dataset Selection")

dataset = st.selectbox(
    "Choose NASA C-MAPSS Dataset",
    options=DATASETS
)

st.info(
    f"Selected model: {MODEL_MAPPING[dataset]}"
)


# --------------------------------
# 4. Load Dataset and Preprocessing
# --------------------------------

@st.cache_data
def load_data(dataset):

    # Load original NASA test data
    data = pd.read_csv(
        DATA_DIR / f"test_{dataset}.txt",
        sep=r"\s+",
        header=None
    )

    data = data.iloc[:, :26]
    data.columns = COLUMNS

    # Load dataset-specific preprocessing artifact
    preprocessing = joblib.load(
        ARTIFACTS_DIR / f"{dataset}_preprocessing.pkl"
    )

    constant_sensors = preprocessing["constant_sensors"]
    feature_columns = preprocessing["feature_columns"]

    # Remove constant sensors
    data = data.drop(
        columns=constant_sensors,
        errors="ignore"
    )

    return data, feature_columns


try:

    test_df, feature_columns = load_data(dataset)

except FileNotFoundError as error:

    st.error(f"Required file not found: {error}")
    st.stop()

except Exception as error:

    st.error(f"Dataset loading failed: {error}")
    st.stop()


# --------------------------------
# 5. Engine Selection
# --------------------------------

st.divider()

st.header("Engine Selection")

engine_ids = sorted(test_df["unit"].unique())

engine_id = st.selectbox(
    "Select Engine ID",
    options=engine_ids,
    index=0
)

engine_data = test_df[
    test_df["unit"] == engine_id
].sort_values("cycle")

if engine_data.empty:
    st.error("No sensor readings found for this engine.")
    st.stop()


# --------------------------------
# 6. Engine Information
# --------------------------------

st.divider()

st.header("Engine Information")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Dataset",
    dataset
)

col2.metric(
    "Engine ID",
    int(engine_id)
)

col3.metric(
    "Recorded Cycles",
    len(engine_data)
)

st.metric(
    "Latest Cycle",
    int(engine_data["cycle"].max())
)


# --------------------------------
# 7. Latest Sensor Readings
# --------------------------------

st.divider()

st.subheader("Latest Sensor Readings")

st.dataframe(
    engine_data.tail(5),
    use_container_width=True
)


# --------------------------------
# 8. Sensor Trend Visualization
# --------------------------------

st.divider()

st.subheader("Sensor Trends")

sensor_columns = [
    column
    for column in feature_columns
    if column.startswith("sensor_")
]

if sensor_columns:

    selected_sensor = st.selectbox(
        "Choose a sensor",
        options=sensor_columns
    )

    st.line_chart(
        engine_data.set_index("cycle")[selected_sensor]
    )

else:

    st.warning("No sensor columns available.")


# --------------------------------
# 9. RUL Prediction
# --------------------------------

st.divider()

st.header("Remaining Useful Life Prediction")

st.write(
    "Use the engine's latest 30 cycles to predict "
    "its remaining useful life."
)

st.caption(
    "For engines with fewer than 30 recorded cycles, "
    "the earliest available observation is repeated "
    "to create the required input window."
)


if st.button(
    "Predict Remaining Useful Life",
    type="primary"
):

    try:

        # Select features in exact training order
        sensor_window = engine_data[
            feature_columns
        ].to_numpy(dtype=np.float32)

        # Keep latest 30 cycles
        if len(sensor_window) >= 30:

            sensor_window = sensor_window[-30:]

        # Pad shorter histories
        else:

            padding = np.repeat(
                sensor_window[[0]],
                30 - len(sensor_window),
                axis=0
            )

            sensor_window = np.vstack(
                [padding, sensor_window]
            )

        # Prepare API request
        payload = {
            "dataset": dataset,
            "sensor_data": sensor_window.tolist()
        }

        # Send prediction request
        with st.spinner("Predicting Remaining Useful Life..."):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=60
            )

            response.raise_for_status()

            result = response.json()

        predicted_rul = result["predicted_rul"]

        # --------------------------------
        # 10. Display Prediction
        # --------------------------------

        st.success("Prediction completed successfully!")

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        col1.metric(
            "Predicted RUL",
            f"{predicted_rul} cycles"
        )

        col2.metric(
            "Dataset Model",
            MODEL_MAPPING[dataset]
        )

        # --------------------------------
        # 11. Maintenance Status
        # --------------------------------

        st.subheader("Engine Maintenance Status")

        if predicted_rul <= 30:

            st.error(
                "CRITICAL - Maintenance attention required"
            )

        elif predicted_rul <= 60:

            st.warning(
                "WARNING - Schedule maintenance inspection"
            )

        else:

            st.success(
                "NORMAL - Continue monitoring"
            )

        # --------------------------------
        # 12. RUL Progress
        # --------------------------------

        st.progress(
            min(max(predicted_rul / 125, 0.0), 1.0),
            text="Predicted RUL relative to the 125-cycle model cap"
        )

        st.caption(
            "Maintenance status thresholds are illustrative, "
            "not validated aviation maintenance limits."
        )

    except requests.exceptions.RequestException as error:

        st.error(f"API connection failed: {error}")

    except (ValueError, KeyError) as error:

        st.error(f"Prediction input or response error: {error}")

    except Exception as error:

        st.error(f"Prediction failed: {error}")
