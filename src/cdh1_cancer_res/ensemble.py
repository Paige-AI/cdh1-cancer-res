from typing import Dict, Optional, Protocol

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn

from .structs import ModelOutput


class AggregatorLike(Protocol):
    """Required methods in order to work with calibration and ensemble classes."""

    def __call__(self, x: Tensor, padding_masks: Optional[Tensor]) -> ModelOutput: ...

    def forward(self, x: Tensor, padding_masks: Optional[Tensor]) -> ModelOutput: ...


class NaiveLR(nn.Module):
    """Parameterized using predifined slope and intercept scalars.
    This is useful for rescaling of prediction values with known scaling weights.
    """

    def __init__(self, slope: float, intercept: float) -> None:
        super().__init__()
        self._m = torch.tensor(slope, requires_grad=False)
        self._b = torch.tensor(intercept, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._m * x + self._b
        return outputs


class SklearnPredictionCallibration(nn.Module):
    """Combines the CallibratedClassifierCV calculations when the base_estimator is a logistic regression,
    i.e. fuses ShiftedSigmoid(NaiveLR(x)). Assumes all parameters have been pre-determined using scikit-learn version 1.1.0
    [https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html].
    """

    def __init__(self, slope: float, intercept: float, sigmoid_a: float, sigmoid_b: float) -> None:
        super().__init__()
        scaling = sigmoid_a * slope
        offset = sigmoid_a * intercept + sigmoid_b
        self.shift = NaiveLR(slope=scaling, intercept=offset)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        shifted_input = self.shift(x)
        # The -1 is needed in order to match the calibration of the scikit-learn implementation.
        return self.sig(-shifted_input)


class CalibratedAggregatorClassificationModel(nn.Module):
    def __init__(
        self,
        aggregator: AggregatorLike,
        calibrator: nn.Module,
    ) -> None:
        super().__init__()
        self.aggregator = aggregator
        self.calibrator = calibrator

    def forward(self, x: Tensor, padding_masks: Optional[Tensor]) -> ModelOutput:
        model_output = self.aggregator(x, padding_masks)

        calibrated_activations: Dict[str, Tensor] = {}
        for label, activations in model_output.heads_activations.items():
            calibrated_activations[label] = self.calibrator(activations)

        return ModelOutput(
            backbone_embedding=model_output.backbone_embedding,
            heads_activations=calibrated_activations,
            attn_scores=model_output.attn_scores,
        )


class AggregatorEnsemble(LightningModule):
    def __init__(self, *models: AggregatorLike) -> None:
        super().__init__()
        self._models = nn.ModuleList(models)

    def forward(self, x: Tensor, padding_masks: Optional[Tensor]) -> ModelOutput:
        """Runs forward pass on all models and collects their output. The collected outputs are stacked and averaged.

        Assumptions:
          1. all ensembled models have the same output keys,
          2. all ensembled models prediction are of the same length.
        """
        backbone_embedding: Optional[Tensor] = None
        heads_activations: Optional[Dict[str, Tensor]] = None
        attn_scores: Optional[Tensor] = None
        n = torch.tensor(0)  # sum count
        for model in self._models:
            out = model(x, padding_masks)
            backbone_embedding = self._rolling_sum_tensor(
                backbone_embedding, out.backbone_embedding
            )
            heads_activations = self._rolling_sum_dict(heads_activations, out.heads_activations)
            attn_scores = self._rolling_sum_tensor(attn_scores, out.attn_scores)
            # increment sum count
            n = n + 1

        n_float = n.float()
        ave_backbone_embedding = backbone_embedding / n_float
        ave_head_activations = self._divide_dict(heads_activations, n_float)
        ave_attn_scores = attn_scores / n_float

        return ModelOutput(
            backbone_embedding=ave_backbone_embedding,
            heads_activations=ave_head_activations,
            attn_scores=ave_attn_scores,
        )

    def _rolling_sum_dict(
        self, d_out: Optional[Dict[str, Tensor]], d_in: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """Instead of stacking tensors and taking the mean, we compute the average by accumulating a rolling sum.
        This approach is more memory efficient.
        """
        if d_out is None:
            # init rolling sum
            return d_in

        itermediate_outputs: Dict[str, Tensor] = {}
        for dict_key in d_out.keys():
            itermediate_outputs[dict_key] = self._rolling_sum_tensor(
                d_out[dict_key], d_in[dict_key]
            )

        return itermediate_outputs

    @staticmethod
    def _rolling_sum_tensor(t_out: Optional[Tensor], t_in: Tensor) -> Tensor:
        if t_out is None:
            # init rolling sum
            return t_in
        t_out = t_out + t_in
        return t_out

    @staticmethod
    def _divide_dict(d_in: Dict[str, Tensor], n: Tensor) -> Dict[str, Tensor]:
        d_out: Dict[str, Tensor] = {}
        for dict_key in d_in.keys():
            d_out[dict_key] = d_in[dict_key] / n

        return d_out
