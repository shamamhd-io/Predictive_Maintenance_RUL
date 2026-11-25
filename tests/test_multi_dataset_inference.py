

import numpy as np
import pytest

from src.inference.multi_dataset_predict import (
    DATASET_CONFIG,
    load_dataset_model,
    predict_rul_multi
)


@pytest.mark.parametrize(
    "dataset",
    ["FD001", "FD002", "FD003", "FD004"]
)
def test_model_loading(dataset):

    model, scaler, feature_columns = load_dataset_model(dataset)

    assert model is not None
    assert scaler is not None
    assert len(feature_columns) == DATASET_CONFIG[dataset]["input_size"]


@pytest.mark.parametrize(
    "dataset",
    ["FD001", "FD002", "FD003", "FD004"]
)
def test_prediction_output(dataset):

    input_size = DATASET_CONFIG[dataset]["input_size"]

    sensor_window = np.zeros(
        (30, input_size),
        dtype=np.float32
    )

    prediction = predict_rul_multi(
        dataset,
        sensor_window
    )

    assert isinstance(prediction, float)
    assert 0 <= prediction <= 125


def test_invalid_input_shape():

    with pytest.raises(ValueError):

        predict_rul_multi(
            "FD001",
            np.zeros((20, 18), dtype=np.float32)
        )


def test_invalid_dataset():

    with pytest.raises(ValueError):

        predict_rul_multi(
            "FD005",
            np.zeros((30, 18), dtype=np.float32)
        )
