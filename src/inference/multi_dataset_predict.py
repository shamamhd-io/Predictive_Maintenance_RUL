
# STEP 17: MULTI-DATASET RUL INFERENCE

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DEVICE = torch.device("cpu")


# BiLSTM architecture
class BiLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        prediction = self.fc(last_output)

        return prediction.squeeze(1)


# Transformer architecture
class TransformerRUL(nn.Module):

    def __init__(
        self,
        input_size,
        d_model=32,
        num_heads=2,
        num_layers=1,
        dropout=0.26,
        window_size=30
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_size,
            d_model
        )

        self.positional_encoding = nn.Parameter(
            torch.zeros(1, window_size, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        x = self.input_projection(x)

        x = x + self.positional_encoding

        output = self.transformer(x)

        last_output = output[:, -1, :]

        prediction = self.fc(last_output)

        return prediction.squeeze(1)

# STEP 18: DATASET CONFIGURATION

DATASET_CONFIG = {

    "FD001": {
        "model_type": "BiLSTM",
        "model_file": "FD001_bilstm.pth",
        "preprocessing_file": "FD001_preprocessing.pkl",
        "input_size": 18
    },

    "FD002": {
        "model_type": "BiLSTM",
        "model_file": "FD002_bilstm.pth",
        "preprocessing_file": "FD002_preprocessing.pkl",
        "input_size": 24
    },

    "FD003": {
        "model_type": "Transformer",
        "model_file": "FD003_transformer.pth",
        "preprocessing_file": "FD003_preprocessing.pkl",
        "input_size": 19
    },

    "FD004": {
        "model_type": "Transformer",
        "model_file": "FD004_transformer.pth",
        "preprocessing_file": "FD004_preprocessing.pkl",
        "input_size": 24
    }

}
# STEP 19: LOAD DATASET-SPECIFIC MODELS

def load_dataset_model(dataset):

    if dataset not in DATASET_CONFIG:
        raise ValueError(
            "Unsupported dataset. Choose FD001, FD002, FD003 or FD004."
        )

    config = DATASET_CONFIG[dataset]

    # Load preprocessing artifacts
    preprocessing_path = (
        ARTIFACTS_DIR / config["preprocessing_file"]
    )

    preprocessing = joblib.load(preprocessing_path)

    scaler = preprocessing["scaler"]

    feature_columns = preprocessing["feature_columns"]

    # Load the appropriate model architecture
    if config["model_type"] == "BiLSTM":

        model = BiLSTM(
            input_size=config["input_size"]
        )

    else:

        model = TransformerRUL(
            input_size=config["input_size"]
        )

    # Load trained weights
    model_path = (
        ARTIFACTS_DIR / "models" / config["model_file"]
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.to(DEVICE)

    model.eval()

    return model, scaler, feature_columns
# STEP 20: MULTI-DATASET RUL PREDICTION

def predict_rul_multi(dataset, sensor_window):

    # Load selected dataset model and preprocessing
    model, scaler, feature_columns = load_dataset_model(dataset)

    # Convert input into NumPy array
    sensor_window = np.asarray(
        sensor_window,
        dtype=np.float32
    )

    # Validate input shape
    expected_features = len(feature_columns)

    if sensor_window.shape != (30, expected_features):
        raise ValueError(
            f"{dataset} requires input shape "
            f"(30, {expected_features}). "
            f"Received {sensor_window.shape}."
        )

    # Validate sensor values
    if not np.isfinite(sensor_window).all():
        raise ValueError(
            "Input contains missing or invalid values."
        )

    # Apply dataset-specific scaling
    scaled_window = scaler.transform(
        pd.DataFrame(
            sensor_window,
            columns=feature_columns
        )
    )

    # Convert to PyTorch tensor
    input_tensor = torch.tensor(
        scaled_window,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    # Generate prediction
    with torch.no_grad():

        prediction = model(input_tensor).item()

    # Apply training target range
    prediction = float(
        np.clip(prediction, 0, 125)
    )

    return round(prediction, 2)
