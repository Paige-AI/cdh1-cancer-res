import pandas as pd
import torch
from pathlib import Path
from typing import List, Tuple
from torch import Tensor
from .env_var import EMBEDDINGS_PATH


def load_dataset_df(dataset_csv_path: str) -> pd.DataFrame:
    return pd.read_csv(dataset_csv_path)


def load_filtered_embeddings(
    *, dataset_df: pd.DataFrame, target_columns: List[str], threshold: float
) -> Tuple[List[pd.DataFrame], List[Tensor]]:
    """Loads embeddings corresponding to a slide provided by the dataset dataframe.
    Filters the tile-level slide embeddings based on metadata columns."""
    metadata_dfs = []
    embeddings = []
    for slide_name in dataset_df.slide_name:
        metadata, embedding = load_tile_embedding(slide_name)
        pdf, etensor = tile_embedding_filter(metadata, embedding, target_columns, threshold)
        metadata_dfs.append(pdf)
        embeddings.append(etensor)
    return metadata_dfs, embeddings


def load_tile_embedding(file_stem: str) -> Tuple[pd.DataFrame, Tensor]:
    filename = Path(EMBEDDINGS_PATH) / file_stem
    df = pd.read_csv(f'{filename}.csv')
    tensor = torch.load(f'{filename}.pt')
    return df, tensor


def tile_embedding_filter(
    metadata: pd.DataFrame, embedding: Tensor, target_columns: List[str], threshold: float
) -> Tuple[pd.DataFrame, Tensor]:
    """Filter tile-embeddings based on feature extractor cancer prediction scores."""
    filtered_indices = metadata.index[metadata[target_columns].max(1) > threshold].tolist()
    selected_df = metadata[metadata.index.isin(filtered_indices)]
    selected_tensor = torch.index_select(embedding, 0, torch.tensor(filtered_indices))
    return selected_df, selected_tensor
