import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("query_collision_sampler", ROOT / "src/query_collision_sampler.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_global_batches_have_no_query_collision_and_drop_only_remainder() -> None:
    queries = [f"query {index}" for index in range(20)] + ["same", " SAME ", "same"]
    batches, dropped = MODULE.build_global_batches(queries, global_batch_size=4, seed=7, epoch=0)
    assert len(batches) == len(queries) // 4
    assert len(dropped) == len(queries) % 4
    flattened = [index for batch in batches for index in batch] + dropped
    assert sorted(flattened) == list(range(len(queries)))
    for batch in batches:
        keys = [MODULE.norm_query(queries[index]) for index in batch]
        assert len(keys) == len(set(keys))


def test_accelerate_style_batch_shards_reconstruct_global_batches() -> None:
    queries = [f"q{index}" for index in range(16)]
    sampler = MODULE.QueryCollisionFreeSampler(
        queries, per_device_batch_size=2, num_replicas=2, seed=11
    )
    order = list(sampler)
    local_batches = [order[offset : offset + 2] for offset in range(0, len(order), 2)]
    rank_batches = [local_batches[rank::2] for rank in range(2)]
    for step in range(len(rank_batches[0])):
        global_batch = rank_batches[0][step] + rank_batches[1][step]
        keys = [MODULE.norm_query(queries[index]) for index in global_batch]
        assert len(keys) == len(set(keys)) == 4
    assert sorted(order) == list(range(16))


def test_epoch_changes_order_but_is_deterministic() -> None:
    queries = [f"q{index}" for index in range(24)]
    sampler = MODULE.QueryCollisionFreeSampler(
        queries, per_device_batch_size=2, num_replicas=2, seed=3
    )
    epoch0 = list(sampler)
    assert epoch0 == list(sampler)
    sampler.set_epoch(1)
    assert epoch0 != list(sampler)
