#!/usr/bin/env python3
"""Deterministic distributed sampler with no normalized-query batch collisions.

Every rank builds the same global microbatches, then takes its rank-local
slice.  Consequently cross-device in-batch negatives see at most one source
row for each normalized query.  The sampler is intentionally independent of
Transformers so its scheduling contract can be tested without GPU libraries.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import heapq
import json
import random
import re
from pathlib import Path
from typing import Iterable, Iterator, Sequence


WS = re.compile(r"\s+")


def norm_query(value: str) -> str:
    return WS.sub(" ", value.strip()).casefold()


def build_global_batches(
    queries: Sequence[str], *, global_batch_size: int, seed: int, epoch: int
) -> tuple[list[list[int]], list[int]]:
    """Return maximum-size full batches and indices that cannot be scheduled.

    A largest-group-first heap prevents a repeated query group from being
    stranded at the end.  Randomized tie breakers and per-group index shuffles
    are deterministic for ``seed + epoch``.
    """
    if global_batch_size < 1:
        raise ValueError("global_batch_size must be positive")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    rng = random.Random(seed + epoch)
    groups: dict[str, list[int]] = collections.defaultdict(list)
    for index, query in enumerate(queries):
        if not isinstance(query, str):
            raise TypeError(f"query {index} is not a string")
        groups[norm_query(query)].append(index)
    for indices in groups.values():
        rng.shuffle(indices)

    heap: list[tuple[int, float, str]] = [
        (-len(indices), rng.random(), key) for key, indices in groups.items()
    ]
    heapq.heapify(heap)
    batches: list[list[int]] = []
    while len(heap) >= global_batch_size:
        chosen = [heapq.heappop(heap) for _ in range(global_batch_size)]
        batch: list[int] = []
        for _, _, key in chosen:
            batch.append(groups[key].pop())
        rng.shuffle(batch)
        batches.append(batch)
        for _, _, key in chosen:
            if groups[key]:
                heapq.heappush(heap, (-len(groups[key]), rng.random(), key))

    dropped: list[int] = []
    while heap:
        _, _, key = heapq.heappop(heap)
        dropped.extend(groups[key])
    return batches, dropped


class QueryCollisionFreeSampler:
    """Sampler ordered for Transformers Trainer + Accelerate batch sharding.

    The sampler yields a single identical global order on every rank.  A
    DataLoader first chunks it into per-device batches; Accelerate's default
    ``BatchSamplerShard(split_batches=False)`` then sends consecutive local
    batches to consecutive ranks.  Do not pre-slice this sampler by rank.
    """

    def __init__(
        self,
        queries: Sequence[str],
        *,
        per_device_batch_size: int,
        num_replicas: int,
        seed: int,
    ) -> None:
        if per_device_batch_size < 1 or num_replicas < 1:
            raise ValueError("batch size and num_replicas must be positive")
        self.queries = queries
        self.per_device_batch_size = per_device_batch_size
        self.num_replicas = num_replicas
        self.seed = seed
        self.epoch = 0
        self._cached_epoch: int | None = None
        self._cached_schedule: tuple[list[list[int]], list[int]] | None = None

    @property
    def global_batch_size(self) -> int:
        return self.per_device_batch_size * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        if epoch != self._cached_epoch:
            self._cached_schedule = None

    def _schedule(self) -> tuple[list[list[int]], list[int]]:
        if self._cached_schedule is None or self._cached_epoch != self.epoch:
            self._cached_schedule = build_global_batches(
                self.queries,
                global_batch_size=self.global_batch_size,
                seed=self.seed,
                epoch=self.epoch,
            )
            self._cached_epoch = self.epoch
        return self._cached_schedule

    def __iter__(self) -> Iterator[int]:
        batches, _ = self._schedule()
        for batch in batches:
            yield from batch

    def __len__(self) -> int:
        batches, _ = self._schedule()
        return len(batches) * self.global_batch_size

    def audit(self) -> dict:
        batches, dropped = self._schedule()
        collisions = 0
        for batch in batches:
            keys = [norm_query(self.queries[index]) for index in batch]
            collisions += len(keys) - len(set(keys))
        return {
            "rows": len(self.queries),
            "normalized_query_groups": len({norm_query(query) for query in self.queries}),
            "global_batch_size": self.global_batch_size,
            "full_global_batches": len(batches),
            "scheduled_rows": len(batches) * self.global_batch_size,
            "dropped_rows": len(dropped),
            "query_collisions": collisions,
            "seed": self.seed,
            "epoch": self.epoch,
        }


def read_training_queries(source: Path, split_manifest: Path | None) -> list[str]:
    excluded_hashes: set[str] = set()
    if split_manifest is not None:
        manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
        for label in ("dev", "test"):
            excluded_hashes.update(item["normalized_query_sha256"] for item in manifest[label]["provenance"])
    queries = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            query = row.get("query")
            if not isinstance(query, str):
                raise ValueError(f"line {line_number}: query must be a string")
            key_hash = hashlib.sha256(norm_query(query).encode()).hexdigest()
            if key_hash not in excluded_hashes:
                queries.append(query)
    return queries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--num-replicas", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    queries = read_training_queries(args.source, args.split_manifest)
    sampler = QueryCollisionFreeSampler(
        queries,
        per_device_batch_size=args.per_device_batch_size,
        num_replicas=args.num_replicas,
        seed=args.seed,
    )
    sampler.set_epoch(args.epoch)
    result = sampler.audit()
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
