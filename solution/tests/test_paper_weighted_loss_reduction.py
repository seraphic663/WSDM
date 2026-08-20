import os
import unittest
from unittest.mock import patch

import torch

from FlagEmbedding.finetune.embedder.decoder_only.base.modeling import (
    BiDecoderOnlyEmbedderModel,
)


class PaperWeightedLossReductionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = BiDecoderOnlyEmbedderModel.__new__(
            BiDecoderOnlyEmbedderModel
        )
        torch.nn.Module.__init__(self.model)
        self.model.cross_entropy = torch.nn.CrossEntropyLoss(
            reduction="mean"
        )
        self.model.cross_entropy_none = torch.nn.CrossEntropyLoss(
            reduction="none"
        )
        self.scores = torch.tensor(
            [[2.0, 0.0], [0.0, 1.0]], dtype=torch.float32
        )
        self.targets = torch.tensor([0, 1], dtype=torch.long)
        self.weights = torch.tensor([0.5, 1.5], dtype=torch.float32)

    def test_paper_mode_is_mean_of_weighted_sample_losses(self) -> None:
        sample_losses = self.model.cross_entropy_none(
            self.scores, self.targets
        )
        expected = (sample_losses * self.weights).mean()
        with patch.dict(
            os.environ, {"LRAT_WEIGHT_REDUCTION": "paper_mean"}
        ):
            actual = self.model.compute_loss(
                self.scores, self.targets, self.weights
            )
        torch.testing.assert_close(actual, expected)

    def test_default_preserves_existing_normalized_weight_sum(self) -> None:
        sample_losses = self.model.cross_entropy_none(
            self.scores, self.targets
        )
        expected = (
            sample_losses * self.weights
        ).sum() / self.weights.sum()
        with patch.dict(os.environ, {}, clear=True):
            actual = self.model.compute_loss(
                self.scores, self.targets, self.weights
            )
        torch.testing.assert_close(actual, expected)

    def test_unknown_reduction_is_rejected(self) -> None:
        with patch.dict(
            os.environ, {"LRAT_WEIGHT_REDUCTION": "unknown"}
        ):
            with self.assertRaisesRegex(
                ValueError, "unsupported LRAT_WEIGHT_REDUCTION"
            ):
                self.model.compute_loss(
                    self.scores, self.targets, self.weights
                )


if __name__ == "__main__":
    unittest.main()
