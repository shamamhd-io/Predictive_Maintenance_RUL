

# STEP 35: TEST API ON ALL FOUR NASA DATASETS

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "CMAPSSData"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

API_URL = "http://127.0.0.1:8000/predict"

# Original NASA dataset column names
columns = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Test all four datasets
for dataset in ["FD001", "FD002", "FD003", "FD004"]:

    print(f"\nTesting {dataset}...")

    # Load test sensor data
    test_df = pd.read_csv(
        DATA_DIR / f"test_{dataset}.txt",
        sep=r"\s+",
        header=None
    )
    test_df.columns = columns

    # Load dataset-specific preprocessing information
    preprocessing = joblib.load(
        ARTIFACTS_DIR / f"{dataset}_preprocessing.pkl"
    )

    feature_columns = preprocessing["feature_columns"]

    # Select engine 1
    engine_data = test_df[
        test_df["unit"] == 1
    ].sort_values("cycle")

    # Select required features in the exact training order
    sensor_window = engine_data[feature_columns].to_numpy(
        dtype=np.float32
    )

    # Keep the latest 30 cycles
    if len(sensor_window) >= 30:
        sensor_window = sensor_window[-30:]

    # Pad shorter histories by repeating the earliest observation
    else:
        padding = np.repeat(
            sensor_window[[0]],
            30 - len(sensor_window),
            axis=0
        )

        sensor_window = np.vstack(
            [padding, sensor_window]
        )

    # Send request to FastAPI
    payload = {
        "dataset": dataset,
        "sensor_data": sensor_window.tolist()
    }

    response = requests.post(
        API_URL,
        json=payload,
        timeout=30
    )

    print("HTTP Status:", response.status_code)

    if response.status_code == 200:
        result = response.json()

        print(
            f"{dataset}: Predicted RUL = "
            f"{result['predicted_rul']} cycles"
        )
    else:
        print("Error:", response.text)
