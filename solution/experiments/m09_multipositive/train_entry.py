#!/usr/bin/env python3
"""FlagEmbedding entry point with an in-process multi-positive objective."""

from __future__ import annotations

import math
import os
import random
from typing import Sequence

import torch


def multi_positive_loss(
    scores: torch.Tensor,
    positive_counts: Sequence[int] | torch.Tensor,
    group_size: int,
    reweight_rates: Sequence[float] | torch.Tensor | None = None,
) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D tensor")
    counts = torch.as_tensor(positive_counts, device=scores.device, dtype=torch.long).view(-1)
    if counts.numel() != scores.size(0):
        raise ValueError("positive_counts must contain one value per query")
    if group_size <= 1 or scores.size(1) < scores.size(0) * group_size:
        raise ValueError("scores are incompatible with the query group size")
    if (counts < 1).any() or (counts >= group_size).any():
        raise ValueError("each positive count must be in [1, group_size)")

    row_indexes = torch.arange(scores.size(0), device=scores.device)
    offsets = row_indexes * group_size
    columns = torch.arange(scores.size(1), device=scores.device).unsqueeze(0)
    positive_mask = (columns >= offsets.unsqueeze(1)) & (
        columns < (offsets + counts).unsqueeze(1)
    )
    positive_scores = scores.masked_fill(~positive_mask, -torch.inf)
    per_query = torch.logsumexp(scores, dim=1) - torch.logsumexp(positive_scores, dim=1)
    if not torch.isfinite(per_query).all():
        raise ValueError("multi-positive loss produced non-finite values")

    if reweight_rates is None:
        return per_query.mean()
    weights = torch.as_tensor(reweight_rates, device=scores.device, dtype=scores.dtype).view(-1)
    if weights.numel() != per_query.numel():
        raise ValueError("reweight_rates must contain one value per query")
    if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("reweight_rates must be finite, non-negative, and have a positive sum")
    return (per_query * weights).sum() / weights.sum().clamp_min(1e-12)


def choose_passage_indexes(
    positive_count: int,
    negative_count: int,
    train_group_size: int,
    max_positives: int,
    rng: random.Random | None = None,
) -> tuple[list[int], list[int]]:
    if positive_count < 1 or negative_count < 1:
        raise ValueError("each row needs at least one positive and one negative")
    if train_group_size <= 1:
        raise ValueError("train_group_size must be greater than one")
    if max_positives < 1:
        raise ValueError("max_positives must be positive")
    sampler = rng if rng is not None else random
    used_positive_count = min(positive_count, max_positives, train_group_size - 1)
    positive_indexes = sampler.sample(range(positive_count), used_positive_count)
    needed_negatives = train_group_size - used_positive_count
    negative_pool = list(range(negative_count))
    if negative_count < needed_negatives:
        copies = math.ceil(needed_negatives / negative_count)
        negative_indexes = sampler.sample(negative_pool * copies, needed_negatives)
    else:
        negative_indexes = sampler.sample(negative_pool, needed_negatives)
    return positive_indexes, negative_indexes


def install_multi_positive_patches(max_positives: int) -> None:
    from FlagEmbedding.abc.finetune.embedder import AbsEmbedderCollator, AbsEmbedderTrainDataset
    from FlagEmbedding.finetune.embedder.decoder_only.base.modeling import (
        BiDecoderOnlyEmbedderModel,
        EmbedderOutput,
    )

    original_collator_call = AbsEmbedderCollator.__call__

    def dataset_getitem(self, item):
        data = self.dataset[item]
        train_group_size = self.args.train_group_size
        query = data["query"]
        reweight_rate = data.get("reweight_rate", 1)
        if self.args.query_instruction_for_retrieval is not None:
            query = self.args.query_instruction_format.format(
                data["prompt"] if "prompt" in data else self.args.query_instruction_for_retrieval,
                query,
            )

        if not isinstance(data["pos"], list) or not isinstance(data["neg"], list):
            raise TypeError("pos and neg must be lists")
        positive_indexes, negative_indexes = choose_passage_indexes(
            len(data["pos"]),
            len(data["neg"]),
            train_group_size,
            max_positives,
        )
        passages = [self._shuffle_text(data["pos"][index]) for index in positive_indexes]
        passages.extend(data["neg"][index] for index in negative_indexes)

        teacher_scores = None
        if self.args.knowledge_distillation:
            if len(positive_indexes) != 1:
                raise ValueError("multi-positive mode does not support knowledge distillation")
            teacher_scores = [data["pos_scores"][positive_indexes[0]]]
            teacher_scores.extend(data["neg_scores"][index] for index in negative_indexes)
        if self.args.passage_instruction_for_retrieval is not None:
            passages = [
                self.args.passage_instruction_format.format(
                    self.args.passage_instruction_for_retrieval,
                    passage,
                )
                for passage in passages
            ]
        return query, passages, teacher_scores, reweight_rate, len(positive_indexes)

    def collator_call(self, features):
        positive_counts = [feature[4] for feature in features]
        value = original_collator_call(self, [feature[:4] for feature in features])
        value["positive_counts"] = positive_counts
        return value

    def in_batch_loss(
        self,
        q_reps,
        p_reps,
        teacher_targets=None,
        reweight_rates=None,
        positive_counts=None,
        compute_score_func=None,
        **kwargs,
    ):
        if teacher_targets is not None:
            raise ValueError("multi-positive mode does not support knowledge distillation")
        group_size = p_reps.size(0) // q_reps.size(0)
        scores = (
            self.compute_score(q_reps, p_reps)
            if compute_score_func is None
            else compute_score_func(q_reps, p_reps, **kwargs)
        )
        loss = multi_positive_loss(scores, positive_counts, group_size, reweight_rates)
        return scores, loss

    def cross_device_loss(
        self,
        q_reps,
        p_reps,
        teacher_targets=None,
        reweight_rates=None,
        positive_counts=None,
        compute_score_func=None,
        **kwargs,
    ):
        if teacher_targets is not None:
            raise ValueError("multi-positive mode does not support knowledge distillation")
        group_size = p_reps.size(0) // q_reps.size(0)
        cross_q_reps = self._dist_gather_tensor(q_reps)
        cross_p_reps = self._dist_gather_tensor(p_reps)
        scores = (
            self.compute_score(cross_q_reps, cross_p_reps)
            if compute_score_func is None
            else compute_score_func(cross_q_reps, cross_p_reps, **kwargs)
        )
        local_counts = torch.as_tensor(positive_counts, device=q_reps.device, dtype=torch.long).view(-1)
        if local_counts.numel() != q_reps.size(0):
            raise ValueError("each local query must have exactly one positive count")
        cross_counts = self._dist_gather_tensor(local_counts)
        cross_weights = None
        if reweight_rates is not None:
            local_weights = torch.as_tensor(
                reweight_rates,
                device=q_reps.device,
                dtype=scores.dtype,
            ).view(-1)
            if local_weights.numel() != q_reps.size(0):
                raise ValueError("each local query must have exactly one reweight_rate")
            cross_weights = self._dist_gather_tensor(local_weights)
        loss = multi_positive_loss(scores, cross_counts, group_size, cross_weights)
        return scores, loss

    def forward(
        self,
        queries=None,
        passages=None,
        teacher_scores=None,
        reweight_rates=None,
        positive_counts=None,
        no_in_batch_neg_flag=False,
    ):
        if teacher_scores is not None:
            raise ValueError("multi-positive mode does not support knowledge distillation")
        if no_in_batch_neg_flag:
            raise ValueError("multi-positive mode requires in-batch negatives")
        q_reps = self.encode(queries)
        p_reps = self.encode(passages)
        loss = None
        if self.training:
            loss_fn = cross_device_loss if self.negatives_cross_device else in_batch_loss
            _, loss = loss_fn(
                self,
                q_reps,
                p_reps,
                reweight_rates=reweight_rates,
                positive_counts=positive_counts,
            )
        return EmbedderOutput(loss=loss)

    AbsEmbedderTrainDataset.__getitem__ = dataset_getitem
    AbsEmbedderCollator.__call__ = collator_call
    BiDecoderOnlyEmbedderModel._compute_in_batch_neg_loss = in_batch_loss
    BiDecoderOnlyEmbedderModel._compute_cross_device_neg_loss = cross_device_loss
    BiDecoderOnlyEmbedderModel.forward = forward


def main() -> None:
    from transformers import HfArgumentParser
    from FlagEmbedding.finetune.embedder.decoder_only.base import (
        DecoderOnlyEmbedderDataArguments,
        DecoderOnlyEmbedderModelArguments,
        DecoderOnlyEmbedderRunner,
        DecoderOnlyEmbedderTrainingArguments,
    )

    max_positives = int(os.environ.get("LRAT_MAX_POSITIVES_PER_QUERY", "3"))
    install_multi_positive_patches(max_positives)
    parser = HfArgumentParser(
        (
            DecoderOnlyEmbedderModelArguments,
            DecoderOnlyEmbedderDataArguments,
            DecoderOnlyEmbedderTrainingArguments,
        )
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    runner = DecoderOnlyEmbedderRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )
    runner.run()


if __name__ == "__main__":
    main()
