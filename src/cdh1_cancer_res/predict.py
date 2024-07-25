from typing import List

import pandas as pd
import torch
from torch import Tensor

from .data import load_dataset_df, load_filtered_embeddings
from .env_var import DATASET_CSV_PATH, ENSEMBLE_MODEL_PATH
from .model import ModelOutput


def predict(model: torch.nn.Module, embeddings: List[Tensor]) -> List[ModelOutput]:
    """Runs prediction on model with batchsize 1.

    Returns:
        List[ModelOutput]
    """
    model.eval()

    outs = []

    with torch.no_grad():
        for x_0 in embeddings:
            x_1 = x_0.unsqueeze(0)
            out = model(x_1, None)
            outs.append(out)

    return outs


def gather_cdh1_prediction_values(prediction_outputs: List[ModelOutput]) -> List[float]:
    predictions = []

    for output in prediction_outputs:
        prediction = output.heads_activations['label_0'].item()
        predictions.append(prediction)

    return predictions


def gather_cdh1_embeddings(prediction_outputs: List[ModelOutput]) -> List[Tensor]:
    embeddings = []

    for output in prediction_outputs:
        # accumulate across attention heads.
        raw_embeddings = output.backbone_embedding.sum(0)

        # remove batch dimension since we ran predict with batchsize 1
        raw_embeddings = raw_embeddings.squeeze(0)

        # permute dimensions from (features, tiles) to (tiles, features)
        tile_embedding = raw_embeddings.permute(1, 0)
        embeddings.append(tile_embedding)

    return embeddings


def print_predictions(
    dataset: pd.DataFrame, prediction_values: List[float], predction_threshold: float
) -> None:
    print(
        'Prediction results: \nindex,\tslide_name,\tground_truth,\tprediction,\tbinarized_prediction'
    )
    for (i, row), pred in zip(dataset.iterrows(), prediction_values):
        print(
            f'{i}\t{row.slide_name}\t{row.cdh1_ground_truth}\t\t{round(pred, 4)}\t\t{int(pred>=predction_threshold)}'
        )


if __name__ == '__main__':
    # Loads full gt dataset worth of embeddings.
    dataset_df = load_dataset_df(DATASET_CSV_PATH)
    metadata, embeddings = load_filtered_embeddings(
        dataset_df=dataset_df,
        target_columns=['p_4', 'p_5', 'p_6'],
        threshold=0.1,
    )

    model = torch.jit.load(ENSEMBLE_MODEL_PATH)

    outputs = predict(model, embeddings)
    predictions = gather_cdh1_prediction_values(outputs)
    embeddings = gather_cdh1_embeddings(outputs)

    print_predictions(dataset_df, predictions, 0.7568)
