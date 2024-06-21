from pathlib import Path
from .ensemble import (
    AggregatorEnsemble,
    CalibratedAggregatorClassificationModel,
    SklearnPredictionCallibration,
    AggregatorLike,
)
from .model import (
    init_cdh1_model,
)
import pandas as pd

from .env_var import CHECKPOINT_CALIBRATION_FILE_PATH, ENSEMBLE_MODEL_PATH, ARTIFACTS_DIR_PATH


def build_agata_ensemble(metadata_path: str) -> AggregatorLike:
    metadata = pd.read_csv(metadata_path)
    agata_models = [load_calibrated_aggregator(data) for _, data in metadata.iterrows()]
    return AggregatorEnsemble(*agata_models)


def load_calibrated_aggregator(metadata: pd.Series) -> AggregatorLike:
    model = init_cdh1_model(str(Path(ARTIFACTS_DIR_PATH, metadata.checkpoint_path)))
    calibrated_model = calibrate_agata(
        model,
        metadata.linear_regression_slope,
        metadata.linear_regression_intercept,
        metadata.sigmoid_a,
        metadata.sigmoid_b,
    )
    return calibrated_model


def calibrate_agata(
    base_agata: AggregatorLike,
    lr_slope: float,
    lr_intercept: float,
    sigmoid_a: float,
    sigmoid_b: float,
) -> AggregatorLike:
    calibrated_agata = CalibratedAggregatorClassificationModel(
        base_agata,
        SklearnPredictionCallibration(lr_slope, lr_intercept, sigmoid_a, sigmoid_b),
    )
    return calibrated_agata


if __name__ == "__main__":
    agata_ensemble_model = build_agata_ensemble(CHECKPOINT_CALIBRATION_FILE_PATH)
    model = agata_ensemble_model.to_torchscript(file_path=ENSEMBLE_MODEL_PATH, method='script')
    print(f'Saved torchscripted ensemble model to "{ENSEMBLE_MODEL_PATH}".')
    print(f'Model Structure: \n {model}')
