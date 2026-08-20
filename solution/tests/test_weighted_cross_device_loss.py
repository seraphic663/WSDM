#!/usr/bin/env python3
"""Numerical checks for LRAT weighted local and cross-device loss."""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from FlagEmbedding.finetune.embedder.decoder_only.base.modeling import (
    BiDecoderOnlyEmbedderModel,
)


def build_model(*, cross_device: bool) -> BiDecoderOnlyEmbedderModel:
    return BiDecoderOnlyEmbedderModel(
        nn.Identity(),
        negatives_cross_device=cross_device,
        temperature=1.0,
        sentence_pooling_method="last_token",
        normalize_embeddings=False,
    )


def test_local_weighted_loss() -> None:
    model = build_model(cross_device=False)
    scores = torch.tensor([[4.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([0, 0])
    per_sample = F.cross_entropy(scores, targets, reduction="none")

    uniform = model.compute_loss(scores, targets, [1.0, 1.0])
    weighted = model.compute_loss(scores, targets, [1.0, 5.0])
    expected = (per_sample * torch.tensor([1.0, 5.0])).sum() / 6.0
    torch.testing.assert_close(weighted, expected)
    assert weighted > uniform

    for bad_weights in ([1.0], [1.0, float("nan")], [1.0, -1.0], [0.0, 0.0]):
        try:
            model.compute_loss(scores, targets, bad_weights)
        except ValueError:
            pass
        else:
            raise AssertionError(f"bad weights were accepted: {bad_weights}")


def test_base_model_checkpoint_loading() -> None:
    base_model = nn.Linear(2, 2, bias=False)
    model = BiDecoderOnlyEmbedderModel(base_model)
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = model.load_state_dict({"weight": expected.clone()}, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    torch.testing.assert_close(base_model.weight, expected)

    wrapper_state = {"model.weight": expected.add(1)}
    model.load_state_dict(wrapper_state, strict=True)
    torch.testing.assert_close(base_model.weight, expected.add(1))


def test_distributed_weighted_loss() -> None:
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    assert dist.get_world_size() == 2

    model = build_model(cross_device=True)
    if rank == 0:
        q_reps = torch.tensor([[1.0, 0.0]], requires_grad=True)
        p_reps = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]], requires_grad=True
        )
        weights = [1.0]
    else:
        q_reps = torch.tensor([[0.0, 1.0]], requires_grad=True)
        p_reps = torch.tensor(
            [[0.0, 1.0], [1.0, 0.0]], requires_grad=True
        )
        weights = [3.0]

    scores, loss = model._compute_cross_device_neg_loss(
        q_reps,
        p_reps,
        reweight_rates=weights,
    )
    targets = torch.tensor([0, 2])
    per_sample = F.cross_entropy(scores, targets, reduction="none")
    expected = (per_sample * torch.tensor([1.0, 3.0])).sum() / 4.0
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert q_reps.grad is not None and torch.isfinite(q_reps.grad).all()
    assert p_reps.grad is not None and torch.isfinite(p_reps.grad).all()

    gathered_loss = [torch.zeros_like(loss) for _ in range(2)]
    dist.all_gather(gathered_loss, loss.detach())
    torch.testing.assert_close(gathered_loss[0], gathered_loss[1])
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()
    if args.distributed:
        test_distributed_weighted_loss()
        if int(os.environ["RANK"]) == 0:
            print("distributed weighted loss: ok")
    else:
        test_local_weighted_loss()
        test_base_model_checkpoint_loading()
        print("local weighted loss: ok")


if __name__ == "__main__":
    main()
