#!/usr/bin/env python3
"""FlagEmbedding entrypoint using a collision-free global query order.

This isolated entrypoint leaves the upstream Trainer untouched. It relies on
Transformers/Accelerate's default ``split_batches=False`` behavior, where
consecutive per-device batches are assigned to consecutive ranks.
"""

from __future__ import annotations

import logging

from transformers import HfArgumentParser

from FlagEmbedding.finetune.embedder.decoder_only.base import (
    DecoderOnlyEmbedderDataArguments,
    DecoderOnlyEmbedderModelArguments,
    DecoderOnlyEmbedderRunner,
    DecoderOnlyEmbedderTrainer,
    DecoderOnlyEmbedderTrainingArguments,
)
from solution.src.query_collision_sampler import QueryCollisionFreeSampler


logger = logging.getLogger(__name__)


class QueryCollisionFreeTrainer(DecoderOnlyEmbedderTrainer):
    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if dataset is None or not hasattr(dataset, "dataset"):
            raise TypeError("collision-free training requires AbsEmbedderTrainDataset.dataset")
        if self.args.group_by_length:
            raise ValueError("collision-free query sampling is incompatible with group_by_length")
        if not self.args.dataloader_drop_last:
            raise ValueError("collision-free distributed training requires dataloader_drop_last=True")
        if getattr(self.accelerator, "split_batches", False):
            raise ValueError("collision-free sampler expects Accelerate split_batches=False")
        queries = dataset.dataset["query"]
        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        sampler = QueryCollisionFreeSampler(
            queries,
            per_device_batch_size=self.args.per_device_train_batch_size,
            num_replicas=self.args.world_size,
            seed=seed,
        )
        if self.args.process_index == 0:
            logger.info("collision-free sampler audit: %s", sampler.audit())
        return sampler


class QueryCollisionFreeRunner(DecoderOnlyEmbedderRunner):
    def load_trainer(self) -> QueryCollisionFreeTrainer:
        if self.data_args.same_dataset_within_batch:
            raise ValueError("collision-free entrypoint does not support same_dataset_within_batch")
        return QueryCollisionFreeTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )


def main() -> None:
    parser = HfArgumentParser(
        (
            DecoderOnlyEmbedderModelArguments,
            DecoderOnlyEmbedderDataArguments,
            DecoderOnlyEmbedderTrainingArguments,
        )
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    runner = QueryCollisionFreeRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )
    runner.run()


if __name__ == "__main__":
    main()
