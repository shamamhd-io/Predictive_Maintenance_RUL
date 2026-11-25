

# STEP 46: REUSABLE RUL PREDICTION MODULE

from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


# Select inference device
DEVICE = torch.device("cpu")


# Tuned Transformer architecture
class TunableTransformer(nn.Module):

    def __init__(
        self,
        d_model=32,
        num_heads=2,
        num_layers=1,
        dropout=0.2605961351078744
    ):
        super().__init__()

        self.input_projection = nn.Linear(18, d_model)

        self.positional_encoding = nn.Parameter(
            torch.zeros(1, 30, d_model)
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


# Load preprocessing artifacts
scaler = joblib.load(
    ARTIFACTS_DIR / "scaler.pkl"
)

feature_columns = joblib.load(
    ARTIFACTS_DIR / "feature_columns.pkl"
)


# Load trained model
model = TunableTransformer().to(DEVICE)

model.load_state_dict(
    torch.load(
        ARTIFACTS_DIR / "models" / "tuned_transformer_model.pth",
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()


# RUL prediction function
def predict_rul(sensor_window):

    sensor_window = np.asarray(
        sensor_window,
        dtype=np.float32
    )

    if sensor_window.shape != (30, 18):
        raise ValueError(
            "Input must contain exactly 30 cycles "
            "and 18 features per cycle."
        )

    if not np.isfinite(sensor_window).all():
        raise ValueError(
            "Input contains missing or invalid values."
        )
    import pandas as pd
    scaled_window = scaler.transform(
    pd.DataFrame(sensor_window, columns=feature_columns))


    input_tensor = torch.tensor(
        scaled_window,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        prediction = model(input_tensor).item()

    prediction = float(np.clip(prediction, 0, 125))

    return round(prediction, 2)
