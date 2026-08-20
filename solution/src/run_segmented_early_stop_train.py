#!/usr/bin/env python3
"""Pause FlagEmbedding training at a predeclared optimizer step for dev eval."""

from __future__ import annotations

import os
from pathlib import Path

from transformers import HfArgumentParser, TrainerCallback

from FlagEmbedding.finetune.embedder.decoder_only.base import (
    DecoderOnlyEmbedderDataArguments,
    DecoderOnlyEmbedderModelArguments,
    DecoderOnlyEmbedderRunner,
    DecoderOnlyEmbedderTrainer,
    DecoderOnlyEmbedderTrainingArguments,
)
from solution.src.query_collision_sampler import QueryCollisionFreeSampler
from solution.src.segment_control import should_pause, write_pause_marker


class PauseAtStepCallback(TrainerCallback):
    def __init__(self, target_step: int, output_dir: Path) -> None:
        self.target_step = target_step
        self.marker = output_dir / "PAUSED_AT_STEP"
        self.paused_step: int | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        if self.target_step <= state.global_step:
            raise ValueError(
                f"STOP_AFTER_STEP={self.target_step} must exceed resumed global_step={state.global_step}"
            )
        if args.process_index == 0:
            self.marker.unlink(missing_ok=True)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if should_pause(state.global_step, state.max_steps, self.target_step):
            control.should_save = True
            control.should_training_stop = True
            self.paused_step = state.global_step
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.paused_step is not None and args.process_index == 0:
            write_pause_marker(
                self.marker,
                global_step=self.paused_step,
                max_steps=state.max_steps,
                target_step=self.target_step,
            )
        return control


class SegmentedTrainer(DecoderOnlyEmbedderTrainer):
    pass


class SegmentedCollisionFreeTrainer(DecoderOnlyEmbedderTrainer):
    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if dataset is None or not hasattr(dataset, "dataset"):
            raise TypeError("collision-free training requires AbsEmbedderTrainDataset.dataset")
        if self.args.group_by_length or not self.args.dataloader_drop_last:
            raise ValueError("collision-free training requires no length grouping and drop_last=True")
        if getattr(self.accelerator, "split_batches", False):
            raise ValueError("collision-free sampler expects Accelerate split_batches=False")
        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        return QueryCollisionFreeSampler(
            dataset.dataset["query"],
            per_device_batch_size=self.args.per_device_train_batch_size,
            num_replicas=self.args.world_size,
            seed=seed,
        )


class SegmentedRunner(DecoderOnlyEmbedderRunner):
    def load_trainer(self):
        if self.data_args.same_dataset_within_batch:
            raise ValueError("segmented entrypoint does not support same_dataset_within_batch")
        trainer_class = (
            SegmentedCollisionFreeTrainer
            if os.environ.get("COLLISION_FREE_QUERIES", "0") == "1"
            else SegmentedTrainer
        )
        trainer = trainer_class(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )
        target = int(os.environ["STOP_AFTER_STEP"])
        if target > self.training_args.max_steps:
            raise ValueError("STOP_AFTER_STEP must not exceed the fixed MAX_STEPS")
        trainer.add_callback(PauseAtStepCallback(target, Path(self.training_args.output_dir)))
        return trainer


def main() -> None:
    parser = HfArgumentParser(
        (
            DecoderOnlyEmbedderModelArguments,
            DecoderOnlyEmbedderDataArguments,
            DecoderOnlyEmbedderTrainingArguments,
        )
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    SegmentedRunner(model_args=model_args, data_args=data_args, training_args=training_args).run()


if __name__ == "__main__":
    main()
