from pathlib import Path
from typing import Dict, Optional, Tuple, Type, Union

import torch
import pandas as pd
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.nn import functional as F

from .env_var import (
    CHECKPOINT_CALIBRATION_FILE_PATH,
    CHECKPOINT_FILE_EXTENSION,
    TORCHSCRIPTED_CHECKPOINT_FILE_EXTENSION,
    ARTIFACTS_DIR_PATH,
)
from .structs import ModelOutput


class Attention(nn.Module):
    def __init__(
        self,
        in_features: int,
        reduce: bool = True,
        pad_value: Union[float, int] = -1e16,
    ) -> None:
        """Vanilla learned attention module

        Args:
            in_features (int): Number of features in input.
            reduce (bool, optional): Sum attended features to single
                vector. Defaults to True.
            pad_value (float or int, optional): Value in masks that
                represent padding positions. Defaults to -1e16.
        """
        super().__init__()
        self.att = nn.Conv1d(in_features, 1, kernel_size=1, bias=False)
        self.reduce = reduce
        self.pad_value = pad_value

    def forward(
        self, query: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of Attention

        Args:
            query (torch.Tensor): Tensor of query vectors in
                (batch, features, sequence) ordering
            value (torch.Tensor): Tensor of value vectors in
                (batch, features, sequence) ordering
            mask (torch.Tensor): Tensor representing sequences that
                should be masked out from softmax calculation (i.e.
                when there is padding that wants to be ignored).
                Defaults to None.

        Returns:
            tuple(torch.Tensor, torch.Tensor): The output of attended
                sequence vectors and the non-normalized attention scores.
        """
        att_mask = self.att(query)
        if mask is not None:
            att_mask[mask == self.pad_value] = -1e16

        normalized_att_mask = F.softmax(att_mask, dim=-1)
        output = value * normalized_att_mask

        if self.reduce:
            output = output.sum(-1)

        return output, att_mask


class AggregatorClassificationModel(LightningModule):
    """Defines a two layer, feed forward, 'Aggregator with Attention' model architecture.
    It's fit for reading in a checkpoint file for inference only and not meant to be used for training.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels1: int,
        out_channels2: int,
        reduce_attention: bool = True,
        num_attn_heads: int = 1,
        activation: Type[nn.Module] = nn.ReLU,
        num_classes: Tuple[int, ...] = (1,),
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels1, kernel_size=1)
        self.conv2 = nn.Conv1d(out_channels1, out_channels2, kernel_size=1)
        self.attention = nn.ModuleList(
            [
                Attention(
                    out_channels1,
                    reduce=reduce_attention,
                )
                for _ in range(num_attn_heads)
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.activation = activation()
        self.num_attn_heads = num_attn_heads
        self.num_classes = num_classes
        self.classifiers = nn.ModuleList([nn.Linear(out_channels2, nc) for nc in self.num_classes])
        self.sigmoid = nn.Sigmoid()
        self._reduce_attention = reduce_attention

    def forward(self, x: Tensor, padding_masks: Optional[Tensor]) -> ModelOutput:
        """Runs a forward pass.

        Args:
            x: Batch of group embeddings. Expected to be in the shape of (batch, features, sequence)
            Beware! torch.nn.Conv1d requires features to be in the second dimension,
            which is different to linear layer implementation of Agata. Refs:
             - https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
             - https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            padding_masks: Mask that indicates which indices in a sequence are padding and which are
                valid. Expected to be in the shape of (batch, sequence).

        Returns: A dictionary containing output tensors for the aggregated embedding, output
        head logits and activations, and attention scores.
        """
        # Receives: x.shape -> (batch, features, sequence), a.k.a (B, S, F).
        # Expects: (B, F, S). Hence, we transpose.
        x_0 = x.permute(0, 2, 1)
        x_1, x_2 = self.forward_features(x_0)

        # x_3.shape -> (B, F), attn.shape -> (B, S, n_attention_queries)
        x_3, attn = self.apply_attention(key=x_1, value=x_2, padding_masks=padding_masks)

        x_4 = x_3.sum(dim=0)

        if self._reduce_attention is False:
            x_4 = x_4.sum(dim=-1)

        heads_activations = self._forward_output_heads(x_4)

        return ModelOutput(
            backbone_embedding=x_3,
            heads_activations=heads_activations,
            attn_scores=attn,
        )

    def forward_features(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Runs the layers in the network that come before attention.

        .. note::
            this API is named as such in order to achieve consistency with timm architectures,
            which all posess a `forward_features` method.
        """
        x_1 = self.conv1(x)
        x_1 = self.activation(x_1)

        x_2 = self.conv2(x_1)
        x_2 = self.activation(x_2)
        return x_1, x_2

    def apply_attention(
        self, key: Tensor, value: Tensor, padding_masks: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """Applies simple dot attention.
        x_3 is accumulated for each attention head. Hence, it's of shape (Heads, Batch, Sequence, Features).
        Sequence is bigger than 1 if the attention implemention doesn't reduce across the sequence dimension.
        """
        attn_masks = []
        x_3 = []
        for attn_block in self.attention:
            x_3_tmp, attn_mask = attn_block(key, value, padding_masks)
            x_3.append(x_3_tmp)
            attn_masks.append(attn_mask)

        return torch.stack(x_3), torch.stack(attn_masks)

    def _forward_output_heads(self, backbone_embedding: Tensor) -> Dict[str, Tensor]:
        """Forward pass for output heads.

        Args:
            backbone_embedding: Shared embedding to be distributed to output heads.

        Returns: Dictionaries of label names mapped to head logits/activations.
        """
        heads_activations: Dict[str, Tensor] = {}
        for i, classifier in enumerate(self.classifiers):
            out = classifier(backbone_embedding)
            heads_activations[f'label_{i}'] = self.sigmoid(out)

        return heads_activations


def init_cdh1_model(
    checkpoint_path: str,
    extract_tile_level_backbone_embeddings: bool = False,
) -> AggregatorClassificationModel:
    """Initializes the aggregator classification model architecture with weights for inference."""
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    agg = AggregatorClassificationModel(
        in_channels=512,
        out_channels1=512,
        out_channels2=512,
        num_classes=(1, 1, 1, 1),
        num_attn_heads=4,
        activation=torch.nn.PReLU,
        reduce_attention=extract_tile_level_backbone_embeddings,
    )
    agg.load_state_dict(checkpoint['state_dict'])
    return agg


if __name__ == '__main__':
    """Script module in two possible versions based on the extract_tile_level_backbone_embeddings boolean:
    1. if False, return aggregated prediction and aggregated slide embedding, or
    2. if True, return aggregated prediction and tile-level embeddings."""
    checkpoint_df = pd.read_csv(CHECKPOINT_CALIBRATION_FILE_PATH)
    checkpoint_path = str(Path(ARTIFACTS_DIR_PATH, checkpoint_df.checkpoint_path[0]))
    nh_agg = init_cdh1_model(
        checkpoint_path,
        extract_tile_level_backbone_embeddings=False,
    )
    path = checkpoint_path.replace(
        CHECKPOINT_FILE_EXTENSION, TORCHSCRIPTED_CHECKPOINT_FILE_EXTENSION
    )
    agg_scripted = nh_agg.to_torchscript(file_path=path, method='script')
    print(f'Created torchscripted checkpoint at {path}')
