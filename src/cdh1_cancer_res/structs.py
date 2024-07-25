from typing import Dict, NamedTuple

from torch import Tensor


class ModelOutput(NamedTuple):
    """Torch scriptable model output."""

    backbone_embedding: Tensor
    heads_activations: Dict[str, Tensor]
    attn_scores: Tensor
